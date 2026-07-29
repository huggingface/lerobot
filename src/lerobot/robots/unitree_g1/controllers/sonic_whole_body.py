#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SONIC decoder whole-body controller for the Unitree G1 (token-only).

Pure-Python/ONNX re-implementation of the *decode* half of NVIDIA's SONIC deploy stack.
The encoder is intentionally absent: a token-output VLA (e.g. ``nepyope/sonic_walk``)
supplies the 64-D latent ``motion_token`` directly each tick, and the SONIC **decoder**
maps ``token + recent proprioception history`` to a residual action that is scaled and
added onto ``DEFAULT_ANGLES`` to produce 50 Hz joint-position targets for the robot's PD
controller.

Index spaces: joints exist in two orderings — **IsaacLab** (policy/training order) and
**MuJoCo** (deploy order). ``ISAACLAB_TO_MUJOCO`` / ``MUJOCO_TO_ISAACLAB`` (in g1_utils)
convert between them. Quaternions are scalar-first ``(w, x, y, z)``.
"""

from __future__ import annotations

import logging

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from ..g1_utils import (
    ISAACLAB_TO_MUJOCO,
    MOTOR_ARMATURE,
    MUJOCO_TO_ISAACLAB,
    NATURAL_FREQ,
    G1_29_JointIndex,
    compute_pd_gains,
    get_gravity_orientation,
    lowstate_to_obs,
    make_ort_session_options,
)

logger = logging.getLogger(__name__)

# ── Constants (hardware-validated; see the NVIDIA SONIC deploy reference) ──────
CONTROL_DT = 0.02  # 50 Hz control period (s)
TOKEN_DIM = 64  # decoder latent size

# Nominal standing pose (rad), 29 joints in IsaacLab order. Decoder actions are residuals
# added on top of this.
DEFAULT_ANGLES = np.array(
    [
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        0.0, 0.0, 0.0,
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
        0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    ],
    dtype=np.float32,
)

# Per-motor torque limits (N·m), used only for SONIC's residual-action scaling. The
# armature / bandwidth constants and the PD-gain formula are shared (see g1_utils).
EFFORT = {"5020": 25.0, "7520_14": 88.0, "7520_22": 139.0, "4010": 5.0}


def _action_scale(k):
    """Per-motor residual-action scale (maps policy output to joint-angle delta)."""
    return 0.25 * EFFORT[k] / (MOTOR_ARMATURE[k] * NATURAL_FREQ**2)


# Per-joint motor model (IsaacLab order): legs, waist, then arms. Single source of truth
# for both ACTION_SCALE and compute_kp_kd().
MOTOR_MODELS = (
    ["7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020"] * 2
    + ["7520_14", "5020", "5020"]
    + ["5020", "5020", "5020", "5020", "5020", "4010", "4010"] * 2
)
ACTION_SCALE = np.array([_action_scale(k) for k in MOTOR_MODELS], dtype=np.float32)  # (29,) IsaacLab


def _to_mujoco(a):
    """Apply the ``MUJOCO_TO_ISAACLAB`` gather to a 29-vector (deploy-order reorder).

    NOTE: this returns ``a[MUJOCO_TO_ISAACLAB]``. The ``_mj`` suffixes and the exact
    permutation direction are a fixed convention validated against the deployed SONIC ONNX
    policy (the decoder consumes vectors in this order). Do not "correct" the table or
    rename toward the opposite direction without re-validating on hardware.
    """
    return a[MUJOCO_TO_ISAACLAB]


DEFAULT_ANGLES_MUJOCO = _to_mujoco(DEFAULT_ANGLES)


# Ankle + waist joint indices (IsaacLab order) that get a x2 stiffness/damping factor.
_SONIC_DOUBLE = {4, 5, 10, 11, 13, 14}


def compute_kp_kd():
    """SONIC per-joint PD gains (kp, kd), (29,) float32 in IsaacLab joint order."""
    return compute_pd_gains(MOTOR_MODELS, _SONIC_DOUBLE)


# Action-feature prefix for the latent-token interface (see _extract_token_from_action).
TOKEN_ACTION_PREFIX = "motion_token"
# Proprio-state prefix for the token interface: the robot echoes the last commanded token
# here so ``lerobot-rollout`` aggregates it into a 64-D ``observation.state``.
TOKEN_STATE_PREFIX = "motion_token_state"


def token_action_key(i: int) -> str:
    """Action-dict key for the i-th component of the 64-D SONIC latent token.

    The ``.pos`` suffix is required so the value flows through ``lerobot-rollout``, which
    only routes ``.pos`` scalar features onto the policy action vector.
    """
    return f"{TOKEN_ACTION_PREFIX}.{i}.pos"


def token_state_key(i: int) -> str:
    """Observation key for the i-th component of the 64-D SONIC latent token state."""
    return f"{TOKEN_STATE_PREFIX}.{i}.pos"


# Startup blend duration: over the first control ticks, linearly interpolate every joint
# from the robot's initial measured pose into the policy's commanded target, so control
# eases in without a snap on the first command.
INIT_RAMP_S = 3.0

# Neutral ("zero pose") SONIC token, held by token_mode until the first real token arrives.
# Captured from the encoder's own output while the robot stood idle in sim: the encoder is
# an FSQ bottleneck (~5 bit/dim, Div(16)), so its tokens live on the 1/16 grid. We store the
# integer FSQ codes and rescale by 1/16, giving an exact on-grid token -- unlike the literal
# all-zero token, which is off the learned manifold and decodes to a slightly goofy stance.
# This one decodes to a stable, natural standing pose.
_NEUTRAL_TOKEN_CODES = np.array(
    [-1, 3, 1, -1, 1, -3, 6, 1, 1, 1, -2, -4, -2, 0, -3, -1,
     2, -1, -3, -5, 3, 1, 1, -4, -1, -1, 1, -7, 0, 1, 2, -2,
     5, -2, -2, -4, 0, -1, 3, -1, 0, -5, -1, 0, -4, 0, 0, -1,
     -1, 2, -2, 1, 3, 3, 1, 0, 0, 6, 0, -7, 3, 0, 2, -2],
    dtype=np.float32,
)
NEUTRAL_TOKEN = _NEUTRAL_TOKEN_CODES / 16.0  # FSQ Div(16): integer codes -> on-grid token


def _extract_token_from_action(action: dict | None) -> np.ndarray | None:
    """Reassemble a dense (64,) latent token from ``motion_token.{i}`` keys, or None.

    The token-only interface: the caller supplies the 64-D encoder latent directly (e.g. a
    token-output VLA's action), which the decoder consumes with the encoder bypassed.
    Requires the full dense token; a partial one is ignored (returns None).
    """
    if not action:
        return None
    keys = [token_action_key(i) for i in range(TOKEN_DIM)]
    if any(key not in action for key in keys):
        return None
    return np.fromiter((float(action[key]) for key in keys), dtype=np.float32, count=TOKEN_DIM)


class SonicDecoder:
    """Runs the SONIC decoder ONNX model and owns the proprioception history.

    Each tick it appends the latest robot state to 10-frame history buffers, then maps the
    supplied 64-D ``token`` + that history to a residual action added onto
    ``DEFAULT_ANGLES``. The encoder is bypassed entirely (token supplied by the policy).
    """

    def __init__(self, decoder):
        self.decoder = decoder
        self.decoder_input = decoder.get_inputs()[0].name
        dec_dim = int(decoder.get_inputs()[0].shape[1])
        if dec_dim != 994:
            raise RuntimeError(f"Unexpected decoder input dim {dec_dim} (expected 994)")
        self.token = np.zeros(TOKEN_DIM, np.float32)
        self.last_action_mj = np.zeros(29, np.float32)
        self.h_q_mj = [np.zeros(29, np.float32)] * 10
        self.h_dq_mj = [np.zeros(29, np.float32)] * 10
        self.h_ang = [np.zeros(3, np.float32)] * 10
        self.h_act_mj = [np.zeros(29, np.float32)] * 10
        self.h_quat = [np.array([1, 0, 0, 0], np.float32)] * 10

    def reset(self):
        """Clear the token and 10-frame proprioception history.

        ``UnitreeG1.reset()`` relies on this so the first decoder outputs of a new episode
        are not contaminated by the previous episode's state.
        """
        self.token = np.zeros(TOKEN_DIM, np.float32)
        self.last_action_mj = np.zeros(29, np.float32)
        self.h_q_mj = [np.zeros(29, np.float32)] * 10
        self.h_dq_mj = [np.zeros(29, np.float32)] * 10
        self.h_ang = [np.zeros(3, np.float32)] * 10
        self.h_act_mj = [np.zeros(29, np.float32)] * 10
        self.h_quat = [np.array([1, 0, 0, 0], np.float32)] * 10

    def update_history(self, q, dq, ang, quat):
        """Push the latest proprioception (pos/vel/gyro/orientation) into the 10-frame buffers."""
        quat = quat / (np.linalg.norm(quat) + 1e-8)
        q_mj = _to_mujoco(q)
        dq_mj = _to_mujoco(dq)
        self.h_q_mj = [q_mj - DEFAULT_ANGLES_MUJOCO] + self.h_q_mj[:-1]
        self.h_dq_mj = [dq_mj] + self.h_dq_mj[:-1]
        self.h_ang = [ang.copy()] + self.h_ang[:-1]
        self.h_act_mj = [self.last_action_mj.copy()] + self.h_act_mj[:-1]
        self.h_quat = [quat.copy()] + self.h_quat[:-1]

    def build_decoder_obs(self):
        """Assemble the 994-D decoder input: token + 10-frame proprioception history + gravity."""
        obs = np.zeros(994, np.float32)
        off = 0
        obs[off : off + 64] = self.token
        off += 64
        for h, sz in [
            (list(reversed(self.h_ang)), 3),
            (list(reversed(self.h_q_mj)), 29),
            (list(reversed(self.h_dq_mj)), 29),
            (list(reversed(self.h_act_mj)), 29),
        ]:
            for f in range(10):
                obs[off : off + sz] = h[f]
                off += sz
        for q in reversed(self.h_quat):
            obs[off : off + 3] = get_gravity_orientation(q)
            off += 3
        assert off == 994, f"Decoder obs mismatch: {off}"
        return obs

    def step(self, robot_obs, token, debug=False):
        """One control tick: read robot obs, decode the supplied token -> joint targets.

        Args:
            robot_obs: dict with ``<joint>.q``/``.dq`` and ``imu.*`` fields.
            token: 64-D latent supplied by the policy (encoder bypassed).
            debug: log action/delta norms.

        Returns:
            dict of ``<joint>.q`` target positions (rad) in IsaacLab joint order.
        """
        self.token = np.asarray(token, np.float32)
        jnames = [m.name for m in G1_29_JointIndex]
        q = np.array(
            [
                robot_obs.get(f"{n}.q", DEFAULT_ANGLES[m.value])
                for m, n in zip(G1_29_JointIndex, jnames, strict=False)
            ],
            np.float32,
        )
        dq = np.array([robot_obs.get(f"{n}.dq", 0.0) for n in jnames], np.float32)
        quat = np.array(
            [
                robot_obs.get("imu.quat.w", 1),
                robot_obs.get("imu.quat.x", 0),
                robot_obs.get("imu.quat.y", 0),
                robot_obs.get("imu.quat.z", 0),
            ],
            np.float32,
        )
        ang = np.array([robot_obs.get(f"imu.gyro.{a}", 0) for a in "xyz"], np.float32)
        self.update_history(q, dq, ang, quat)
        action_mj = (
            self.decoder.run(None, {self.decoder_input: self.build_decoder_obs().reshape(1, -1)})[0]
            .squeeze()
            .astype(np.float32)
        )
        self.last_action_mj = action_mj.copy()
        target = DEFAULT_ANGLES + action_mj[ISAACLAB_TO_MUJOCO] * ACTION_SCALE
        if debug:
            delta = target - q
            logger.debug(
                "token_norm=%.4f action_norm=%.4f delta_max=%.4f delta_rms=%.4f",
                np.linalg.norm(self.token),
                np.linalg.norm(action_mj),
                np.max(np.abs(delta)),
                np.sqrt(np.mean(delta**2)),
            )
        return {f"{m.name}.q": float(target[m.value]) for m in G1_29_JointIndex}


class SonicRuntime:
    """Loads the SONIC decoder ONNX model and owns the decode controller.

    Token-only deploy: the encoder is bypassed; each tick the decoder consumes a 64-D
    latent token supplied directly by the policy.
    """

    def __init__(self):
        decoder_path = hf_hub_download(repo_id="nvidia/GEAR-SONIC", filename="model_decoder.onnx")

        so = make_ort_session_options()
        decoder_sess = ort.InferenceSession(decoder_path, sess_options=so)

        self.kp, self.kd = compute_kp_kd()
        self.controller = SonicDecoder(decoder_sess)

    @property
    def pipeline(self):
        return self.controller

    def reset(self):
        self.controller.reset()

    def shutdown(self):
        pass


class SonicWholeBodyController:
    """Full-body SONIC controller for UnitreeG1's background controller thread."""

    control_dt = CONTROL_DT
    full_body = True

    def __init__(self):
        logger.info("Loading SONIC whole-body controller...")
        self._runtime = SonicRuntime()
        self.kp = self._runtime.kp
        self.kd = self._runtime.kd
        self.controller = self._runtime.controller

        # Startup blend: ease from the robot's initial pose into the first commanded policy
        # targets over INIT_RAMP_S (captured on the first control tick).
        self._init_ramp_steps = max(1, round(INIT_RAMP_S / CONTROL_DT))
        self._init_step = 0
        self._start_pose: dict[str, float] = {}

        # Token-interface state. ``token_mode`` is set True by the robot when the deploy is
        # token-driven (``UnitreeG1Config.sonic_token_action``): the controller then holds a
        # stable *neutral* token until the first real token arrives, and afterwards holds the
        # *last* token received between ticks (the async controller runs ~50 Hz while a token
        # VLA streams ~30 Hz). This lives here (not in the entry-point script) so it applies
        # uniformly to run_g1_server, lerobot-rollout and the sim replays.
        self.token_mode = False
        self._last_token: np.ndarray | None = None

        logger.info("SONIC ready (decoder, 64-D token command path)")

    def _startup_blend(self, obs: dict, out: dict) -> dict:
        """Ease into policy control at startup: for the first ``INIT_RAMP_S`` seconds,
        interpolate between the robot's pose captured on the first tick and the policy's
        live commanded target, so the handoff has no snap.

        ``out`` is the policy's ``<joint>.q`` target dict for this tick; the blend ratio
        climbs 0->1 over the ramp, after which the raw policy target passes through.
        """
        if self._init_step >= self._init_ramp_steps or not out:
            return out
        if self._init_step == 0:
            # Capture the robot's actual pose as the interpolation start point.
            self._start_pose = {
                f"{m.name}.q": float(obs.get(f"{m.name}.q", DEFAULT_ANGLES[m.value]))
                for m in G1_29_JointIndex
            }
        self._init_step += 1
        ratio = min(1.0, self._init_step / self._init_ramp_steps)
        blended = {
            k: self._start_pose.get(k, float(tgt)) * (1.0 - ratio) + float(tgt) * ratio
            for k, tgt in out.items()
        }
        if self._init_step >= self._init_ramp_steps:
            logger.info("SONIC startup blend complete -> full policy control")
        return blended

    def run_step(self, action: dict, lowstate) -> dict:
        if lowstate is None:
            return {}
        obs = lowstate_to_obs(lowstate)

        # Token-only interface (token-output VLA): a dense 64-D ``motion_token.{i}`` command
        # is decoded directly, encoder bypassed.
        token = _extract_token_from_action(action)
        if token is not None:
            self._last_token = token
        elif self._last_token is None and self.token_mode:
            # Token-driven deploy, but no token has arrived yet: hold the captured neutral
            # token (NEUTRAL_TOKEN), which the decoder maps to a stable, natural standing pose.
            self._last_token = NEUTRAL_TOKEN.copy()
        if self._last_token is None:
            # No token yet and not in token_mode: hold (keep last target).
            return {}
        # Either a fresh token this tick or the last one received (held between the ~30 Hz
        # token stream and the ~50 Hz control loop).
        return self._startup_blend(obs, self.controller.step(obs, self._last_token))

    def reset(self):
        self._runtime.reset()
        self._init_step = 0  # re-run the startup blend after a reset
        self._start_pose = {}
        # Drop the held token so token_mode re-seeds the neutral token after a reset.
        self._last_token = None

    def shutdown(self):
        self._runtime.shutdown()
