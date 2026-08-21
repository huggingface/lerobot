#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
    NUM_MOTORS,
    G1_29_JointArmIndex,
    G1_29_JointIndex,
    get_gravity_orientation,
    make_ort_session_options,
)
from ..unitree_g1 import RobotController

logger = logging.getLogger(__name__)

CONTROL_DT = 0.02  # 50 Hz control period (s)
TOKEN_DIM = 64  # decoder latent size
HISTORY_LEN = 10  # proprioception frames the decoder conditions on

# Latent-token feature-key prefixes: action carries the token, obs echoes it back.
TOKEN_ACTION_PREFIX = "motion_token"  # nosec B105 - feature-key prefix, not a secret
TOKEN_STATE_PREFIX = "motion_token_state"  # nosec B105 - feature-key prefix, not a secret

# SONIC decoder checkpoint. Deploy constants (kp/kd, default_angles, action_scale,
# neutral_token) are baked into the ONNX metadata; see upload_sonic_decoder.py.
DEFAULT_SONIC_REPO_ID = "lerobot/sonic_decoder"
# token + HISTORY_LEN frames of (angular velocity, joint pos, joint vel, last action) + gravity
DECODER_INPUT_DIM = TOKEN_DIM + HISTORY_LEN * (3 + 3 * NUM_MOTORS) + HISTORY_LEN * 3  # 994

# Decoder filename mapping: the full decoder (default) or NVIDIA's distilled low-latency one.
POLICY_FILES = {
    "default": "model_decoder.onnx",
    "low_latency": "low_latency/model_decoder.onnx",
}


def load_policy(
    repo_id: str = DEFAULT_SONIC_REPO_ID,
    policy_type: str = "default",
) -> tuple[ort.InferenceSession, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the SONIC decoder and its baked-in deploy constants from ONNX metadata.

    Args:
        repo_id: Hugging Face Hub repo ID
        policy_type: Either "default" (full decoder) or "low_latency" (distilled)

    Returns:
        (decoder, kp, kd, default_angles, action_scale, neutral_token) tuple. The gains/pose/
        scale are (29,) float32 in IsaacLab joint order; neutral_token is the (64,) idle latent.
    """
    if policy_type not in POLICY_FILES:
        raise ValueError(f"Unknown policy type: {policy_type}. Choose from: {list(POLICY_FILES.keys())}")

    filename = POLICY_FILES[policy_type]
    logger.info(f"Loading {policy_type.upper()} SONIC decoder from: {repo_id}/{filename}")
    decoder_path = hf_hub_download(repo_id=repo_id, filename=filename)

    # Cap the thread pool: onboard, this decoder steps at 50 Hz in the same process as the
    # ZMQ camera server (capture + JPEG encode). Default session options let ORT grab every
    # core on the NX, which starves the camera thread (stale frames) and jitters the control
    # loop (limping gait). It is a small MLP, so 1 thread is enough and lowest-latency.
    session_options = make_ort_session_options(intra_op_num_threads=1, inter_op_num_threads=1)
    decoder = ort.InferenceSession(decoder_path, sess_options=session_options)
    logger.info(f"Decoder loaded: {decoder.get_inputs()[0].shape} → {decoder.get_outputs()[0].shape}")

    # Extract deploy constants from ONNX metadata
    model = onnx.load(decoder_path, load_external_data=False)
    metadata = {prop.key: prop.value for prop in model.metadata_props}

    required = ("kp", "kd", "default_angles", "action_scale", "neutral_token")
    missing = [k for k in required if k not in metadata]
    if missing:
        raise ValueError(f"ONNX model must contain {list(required)} in metadata (missing {missing})")

    arr = {k: np.array(json.loads(metadata[k]), dtype=np.float32) for k in required}
    logger.info(f"Loaded SONIC deploy constants from ONNX ({len(arr['kp'])} joints)")
    return decoder, arr["kp"], arr["kd"], arr["default_angles"], arr["action_scale"], arr["neutral_token"]


class SonicWholeBodyController(RobotController):
    """Full-body SONIC decoder controller for UnitreeG1's background controller thread.

    Token-only deploy (encoder bypassed): each tick it appends the latest robot state to
    10-frame history buffers, then maps the policy-supplied 64-D token + that history to a
    residual action added onto ``default_angles`` -> 50 Hz joint-position targets.
    """

    control_dt = CONTROL_DT
    # Whole-body: the arms are part of the decoder's output, so its gains apply to them too.
    controlled_joints = tuple(m.value for m in G1_29_JointIndex)
    # Class attributes, not instance ones, so the laptop client can read the schema by name
    # without building a controller -- which would load the decoder and imply it runs there.
    # 64-D latent-token action space; rollout maps the policy's 64-D output onto these keys.
    action_ft = {f"{TOKEN_ACTION_PREFIX}.{i}.pos": float for i in range(TOKEN_DIM)}
    # 64-D token proprio state, aggregated by rollout into observation.state (last token).
    observation_ft = {f"{TOKEN_STATE_PREFIX}.{i}.pos": float for i in range(TOKEN_DIM)}

    def __init__(self, policy_type: str = "default"):
        self.decoder, self.kp, self.kd, self.default_angles, self.action_scale, self.neutral_token = (
            load_policy(policy_type=policy_type)
        )
        self.decoder_input = self.decoder.get_inputs()[0].name
        self.default_angles_mj = self.default_angles[MUJOCO_TO_ISAACLAB]

        self.reset()
        logger.info("SonicWholeBodyController initialized")

    def reset(self) -> None:
        """Reset internal state for a new episode: held token and proprioception history."""
        self.last_action_mj = np.zeros(NUM_MOTORS, np.float32)
        self.h_q_mj = [np.zeros(NUM_MOTORS, np.float32) for _ in range(HISTORY_LEN)]
        self.h_dq_mj = [np.zeros(NUM_MOTORS, np.float32) for _ in range(HISTORY_LEN)]
        self.h_ang = [np.zeros(3, np.float32) for _ in range(HISTORY_LEN)]
        self.h_act_mj = [np.zeros(NUM_MOTORS, np.float32) for _ in range(HISTORY_LEN)]
        self.h_quat = [np.array([1, 0, 0, 0], np.float32) for _ in range(HISTORY_LEN)]
        self._last_token = None  # neutral token is re-seeded on the first tick

    def observation_state(self) -> dict[str, float]:
        """Echo the last decoded token as ``observation.state`` so a token-output VLA closes
        the loop on its own previous token."""
        token = self._last_token if self._last_token is not None else np.zeros(TOKEN_DIM, dtype=np.float32)
        return {f"{TOKEN_STATE_PREFIX}.{i}.pos": float(v) for i, v in enumerate(token)}

    def run_step(self, action: dict, lowstate) -> dict:
        """Decode one control tick into absolute joint-position targets.

        Args:
            action: Latest action snapshot. All 64 ``motion_token.{i}.pos`` keys must be
                present to update the latent; any other content (joystick axes, a partial
                chunk) leaves the previously held token in place.
            lowstate: Unitree lowstate carrying joint positions/velocities and IMU state.

        Returns:
            Absolute joint targets keyed ``<joint>.q`` for all 29 joints: ``default_angles``
            plus the decoder's residual, scaled by ``action_scale``. Empty until the first
            lowstate arrives, which onboard is a real tick or two after the loop starts.
        """
        if lowstate is None:
            return {}

        # Token: reassemble the dense 64-D latent from motion_token.{i}.pos (all keys required);
        # else hold the last one (neutral until the first real token, which decodes to a stand).
        keys = [f"{TOKEN_ACTION_PREFIX}.{i}.pos" for i in range(TOKEN_DIM)]
        if action and all(k in action for k in keys):
            self._last_token = np.fromiter(
                (float(action[k]) for k in keys), dtype=np.float32, count=TOKEN_DIM
            )
        elif self._last_token is None:
            self._last_token = self.neutral_token.copy()

        # Read proprioception from lowstate (IsaacLab joint order).
        q = np.array([lowstate.motor_state[m.value].q for m in G1_29_JointIndex], np.float32)
        dq = np.array([lowstate.motor_state[m.value].dq for m in G1_29_JointIndex], np.float32)
        quat = np.array(lowstate.imu_state.quaternion, np.float32)  # (w, x, y, z)
        quat = quat / (np.linalg.norm(quat) + 1e-8)
        ang = np.array(lowstate.imu_state.gyroscope, np.float32)

        # Push into the 10-frame history (newest first). The decoder consumes MuJoCo joint
        # order, so reorder q/dq via MUJOCO_TO_ISAACLAB (validated against the ONNX; don't flip).
        self.h_q_mj = [q[MUJOCO_TO_ISAACLAB] - self.default_angles_mj] + self.h_q_mj[:-1]
        self.h_dq_mj = [dq[MUJOCO_TO_ISAACLAB]] + self.h_dq_mj[:-1]
        self.h_ang = [ang] + self.h_ang[:-1]
        self.h_act_mj = [self.last_action_mj.copy()] + self.h_act_mj[:-1]
        self.h_quat = [quat] + self.h_quat[:-1]

        # Assemble the 994-D decoder input: token + oldest->newest history + gravity.
        obs = np.zeros(DECODER_INPUT_DIM, np.float32)
        obs[:TOKEN_DIM] = self._last_token
        off = TOKEN_DIM
        for hist, sz in (
            (self.h_ang, 3),
            (self.h_q_mj, NUM_MOTORS),
            (self.h_dq_mj, NUM_MOTORS),
            (self.h_act_mj, NUM_MOTORS),
        ):
            for frame in reversed(hist):
                obs[off : off + sz] = frame
                off += sz
        for hquat in reversed(self.h_quat):
            obs[off : off + 3] = get_gravity_orientation(hquat)
            off += 3

        # Decode -> residual action (MuJoCo order) added onto the standing pose.
        action_mj = (
            self.decoder.run(None, {self.decoder_input: obs.reshape(1, -1)})[0].squeeze().astype(np.float32)
        )
        self.last_action_mj = action_mj.copy()
        target = self.default_angles + action_mj[ISAACLAB_TO_MUJOCO] * self.action_scale
        return {f"{m.name}.q": float(target[m.value]) for m in G1_29_JointIndex}


class SonicLowerBodyController(SonicWholeBodyController):
    """SONIC holding the robot up, with the arms handed over to the caller.

    Same decoder, same 50 Hz loop, but only legs and waist are published; motors 15-28 are
    left for ``send_action`` to drive. That lets a manipulation policy own the arms while
    balance stays where it belongs. No token is consumed, so the decoder runs on
    ``neutral_token`` throughout, which decodes to a stand -- and that makes a stalled policy
    stream safe by construction: the arms hold their last command and the robot keeps
    standing, rather than continuing to execute a stale intent the way a walk token would.

    The decoder still conditions on its own action history, so the arm commands it did not
    produce are folded back into that history. Otherwise it would predict against ten frames
    of arm actions it never took while feeling the resulting motion in its proprioception,
    and the mismatch would show up as leg compensation for arm movement it cannot explain.
    """

    # Legs and waist: everything below the first arm joint.
    controlled_joints = tuple(range(G1_29_JointArmIndex.kLeftShoulderPitch))
    # Fold externally commanded arm targets into the decoder's action history.
    arm_feedback = True
    # Arm joint targets, not a 64-D token: this controller is not driven by a token VLA.
    action_ft = {f"{m.name}.q": float for m in G1_29_JointArmIndex}
    # No observation_ft, so the robot advertises its 29 joint positions rather than a token
    # echo -- the retargeting layer needs the real arm angles to run FK.
    observation_ft = None

    def __init__(self, policy_type: str = "default"):
        super().__init__(policy_type)
        owned = set(self.controlled_joints)
        self._lower_body_keys = [f"{m.name}.q" for m in G1_29_JointIndex if m.value in owned]
        self._arm_slots = [(f"{m.name}.q", m.value) for m in G1_29_JointArmIndex]
        self._inv_action_scale = np.where(self.action_scale != 0, 1.0 / self.action_scale, 0.0)
        logger.info("SonicLowerBodyController initialized (legs+waist; arms free)")

    def observation_state(self) -> dict[str, float]:
        """No token echo: the robot falls back to publishing real joint positions."""
        return {}

    def run_step(self, action: dict, lowstate) -> dict:
        if self.arm_feedback:
            self._absorb_arm_command(action)
        targets = super().run_step(action, lowstate)
        return {key: targets[key] for key in self._lower_body_keys if key in targets}

    def _absorb_arm_command(self, action: dict) -> None:
        """Rewrite the arm entries of the decoder's last action to match what was commanded.

        ``run_step`` stores its own output as ``last_action_mj`` (a residual in MuJoCo order,
        scaled by ``action_scale`` and added onto ``default_angles``). Inverting that mapping
        turns a commanded arm angle back into the residual the decoder would have had to emit
        to produce it.
        """
        if not action:
            return
        commanded = [(motor, action[key]) for key, motor in self._arm_slots if key in action]
        if not commanded:
            return
        # Fancy indexing copies, so this reorders into IsaacLab space without aliasing.
        residual = self.last_action_mj[ISAACLAB_TO_MUJOCO]
        for motor, value in commanded:
            residual[motor] = (float(value) - self.default_angles[motor]) * self._inv_action_scale[motor]
        self.last_action_mj = residual[MUJOCO_TO_ISAACLAB]
