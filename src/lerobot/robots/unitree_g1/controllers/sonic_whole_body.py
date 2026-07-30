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
added onto the standing pose (``default_angles``) to produce 50 Hz joint-position targets
for the robot's PD controller.

Index spaces: joints exist in two orderings — **IsaacLab** (policy/training order) and
**MuJoCo** (deploy order). ``ISAACLAB_TO_MUJOCO`` / ``MUJOCO_TO_ISAACLAB`` (in g1_utils)
convert between them. Quaternions are scalar-first ``(w, x, y, z)``.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import onnx
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from ..g1_utils import (
    ISAACLAB_TO_MUJOCO,
    MUJOCO_TO_ISAACLAB,
    G1_29_JointIndex,
    get_gravity_orientation,
)

logger = logging.getLogger(__name__)

# ── Constants (hardware-validated; see the NVIDIA SONIC deploy reference) ──────
CONTROL_DT = 0.02  # 50 Hz control period (s)
TOKEN_DIM = 64  # decoder latent size

# SONIC decoder checkpoint: NVIDIA's decoder ONNX re-packaged with its deploy constants
# (kp/kd PD gains, the standing pose default_angles, and the residual action_scale) embedded
# in the ONNX metadata; see upload_sonic_decoder.py for provisioning. The runtime loads the
# model *and* all of these straight from the checkpoint (the Holosoma convention), so no
# motor-physics math happens at deploy time.
DEFAULT_SONIC_REPO_ID = "lerobot/sonic_decoder"
DECODER_FILENAME = "model_decoder.onnx"
DECODER_INPUT_DIM = 994  # token(64) + 10-frame proprio history + gravity


def load_sonic_decoder(repo_id: str = DEFAULT_SONIC_REPO_ID):
    """Load the SONIC decoder ONNX and its baked-in deploy constants from the checkpoint.

    Returns ``(decoder_session, kp, kd, default_angles, action_scale, neutral_token)``. The
    gains/pose/scale are (29,) float32 in IsaacLab joint order and ``neutral_token`` is the
    (64,) float32 idle latent -- all read from the ONNX ``metadata_props`` rather than
    recomputed/hardcoded at deploy time (mirrors ``holosoma_locomotion.load_policy``).
    """
    decoder_path = hf_hub_download(repo_id=repo_id, filename=DECODER_FILENAME)
    so = ort.SessionOptions()
    so.log_severity_level = 3  # quiet ORT logs
    session = ort.InferenceSession(decoder_path, sess_options=so)
    dec_dim = int(session.get_inputs()[0].shape[1])
    if dec_dim != DECODER_INPUT_DIM:
        raise RuntimeError(f"Unexpected decoder input dim {dec_dim} (expected {DECODER_INPUT_DIM})")

    meta = {p.key: p.value for p in onnx.load(decoder_path, load_external_data=False).metadata_props}
    required = ("kp", "kd", "default_angles", "action_scale", "neutral_token")
    missing = [k for k in required if k not in meta]
    if missing:
        raise ValueError(
            f"SONIC decoder ONNX at {repo_id} is missing metadata {missing}; "
            "re-run upload_sonic_decoder.py to (re)provision the checkpoint."
        )
    arr = {k: np.array(json.loads(meta[k]), dtype=np.float32) for k in required}
    logger.info("Loaded SONIC deploy constants from %s (%d joints)", repo_id, len(arr["kp"]))
    return session, arr["kp"], arr["kd"], arr["default_angles"], arr["action_scale"], arr["neutral_token"]


# Action-feature prefix for the latent-token interface (see _extract_token_from_action).
TOKEN_ACTION_PREFIX = "motion_token"  # nosec B105 - feature-key prefix, not a secret
# Proprio-state prefix for the token interface: the robot echoes the last commanded token
# here so ``lerobot-rollout`` aggregates it into a 64-D ``observation.state``.
TOKEN_STATE_PREFIX = "motion_token_state"  # nosec B105 - feature-key prefix, not a secret


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
    supplied 64-D ``token`` + that history to a residual action added onto ``default_angles``.
    The encoder is bypassed entirely (token supplied by the policy). ``default_angles`` and
    ``action_scale`` are (29,) float32 in IsaacLab order, loaded from the checkpoint.
    """

    def __init__(self, decoder, default_angles, action_scale):
        self.decoder = decoder
        self.decoder_input = decoder.get_inputs()[0].name
        self.default_angles = np.asarray(default_angles, np.float32)
        self.action_scale = np.asarray(action_scale, np.float32)
        self.default_angles_mj = self.default_angles[MUJOCO_TO_ISAACLAB]
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
        # Reorder IsaacLab-order state into the MuJoCo order the decoder consumes. This
        # permutation direction is validated against the deployed SONIC ONNX; don't flip it.
        q_mj = q[MUJOCO_TO_ISAACLAB]
        dq_mj = dq[MUJOCO_TO_ISAACLAB]
        self.h_q_mj = [q_mj - self.default_angles_mj] + self.h_q_mj[:-1]
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
                robot_obs.get(f"{n}.q", self.default_angles[m.value])
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
        target = self.default_angles + action_mj[ISAACLAB_TO_MUJOCO] * self.action_scale
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
        decoder_sess, self.kp, self.kd, default_angles, action_scale, neutral_token = load_sonic_decoder()
        self.default_angles = default_angles
        self.neutral_token = neutral_token
        self.controller = SonicDecoder(decoder_sess, default_angles, action_scale)

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
        self._default_angles = self._runtime.default_angles
        self._neutral_token = self._runtime.neutral_token

        # Startup blend: ease from the robot's initial pose into the first commanded policy
        # targets over INIT_RAMP_S (captured on the first control tick).
        self._init_ramp_steps = max(1, round(INIT_RAMP_S / CONTROL_DT))
        self._init_step = 0
        self._start_pose: dict[str, float] = {}

        # Token-interface state. ``token_mode`` is set True by the robot whenever a SONIC
        # whole-body controller is selected (token-driven deploy): the controller then holds a
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
                f"{m.name}.q": float(obs.get(f"{m.name}.q", self._default_angles[m.value]))
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

    def run_step(self, action: dict, obs: dict) -> dict:
        if not obs:
            return {}

        # Token-only interface (token-output VLA): a dense 64-D ``motion_token.{i}`` command
        # is decoded directly, encoder bypassed.
        token = _extract_token_from_action(action)
        if token is not None:
            self._last_token = token
        elif self._last_token is None and self.token_mode:
            # Token-driven deploy, but no token has arrived yet: hold the checkpoint's neutral
            # token, which the decoder maps to a stable, natural standing pose.
            self._last_token = self._neutral_token.copy()
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
