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
    G1_29_JointIndex,
    get_gravity_orientation,
)

logger = logging.getLogger(__name__)

CONTROL_DT = 0.02  # 50 Hz control period (s)
TOKEN_DIM = 64  # decoder latent size

# Latent-token feature-key prefixes: action carries the token, obs echoes it back.
TOKEN_ACTION_PREFIX = "motion_token"  # nosec B105 - feature-key prefix, not a secret
TOKEN_STATE_PREFIX = "motion_token_state"  # nosec B105 - feature-key prefix, not a secret

# SONIC decoder checkpoint. Deploy constants (kp/kd, default_angles, action_scale,
# neutral_token) are baked into the ONNX metadata; see upload_sonic_decoder.py.
DEFAULT_SONIC_REPO_ID = "lerobot/sonic_decoder"
DECODER_INPUT_DIM = 994  # token(64) + 10-frame proprio history + gravity

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

    decoder = ort.InferenceSession(decoder_path)
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


class SonicWholeBodyController:
    """Full-body SONIC decoder controller for UnitreeG1's background controller thread.

    Token-only deploy (encoder bypassed): each tick it appends the latest robot state to
    10-frame history buffers, then maps the policy-supplied 64-D token + that history to a
    residual action added onto ``default_angles`` -> 50 Hz joint-position targets.
    """

    control_dt = CONTROL_DT

    def __init__(self, policy_type: str = "default"):
        self.decoder, self.kp, self.kd, self.default_angles, self.action_scale, self.neutral_token = (
            load_policy(policy_type=policy_type)
        )
        self.decoder_input = self.decoder.get_inputs()[0].name
        self.default_angles_mj = self.default_angles[MUJOCO_TO_ISAACLAB]

        # 64-D latent-token action space; rollout maps the policy's 64-D output onto these keys.
        self.action_ft = {f"{TOKEN_ACTION_PREFIX}.{i}.pos": float for i in range(TOKEN_DIM)}
        # 64-D token proprio state, aggregated by rollout into observation.state (last token).
        self.observation_ft = {f"{TOKEN_STATE_PREFIX}.{i}.pos": float for i in range(TOKEN_DIM)}

        self.reset()
        logger.info("SonicWholeBodyController initialized")

    def reset(self) -> None:
        """Reset internal state for a new episode: held token and 10-frame history buffers."""
        self.last_action_mj = np.zeros(29, np.float32)
        self.h_q_mj = [np.zeros(29, np.float32)] * 10
        self.h_dq_mj = [np.zeros(29, np.float32)] * 10
        self.h_ang = [np.zeros(3, np.float32)] * 10
        self.h_act_mj = [np.zeros(29, np.float32)] * 10
        self.h_quat = [np.array([1, 0, 0, 0], np.float32)] * 10
        self._last_token = None  # neutral token is re-seeded on the first tick

    def observation_state(self) -> dict[str, float]:
        """Echo the last decoded token as ``observation.state`` so a token-output VLA closes
        the loop on its own previous token."""
        token = self._last_token if self._last_token is not None else np.zeros(TOKEN_DIM, dtype=np.float32)
        return {f"{TOKEN_STATE_PREFIX}.{i}.pos": float(v) for i, v in enumerate(token)}

    def run_step(self, action: dict, lowstate) -> dict:
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
        for hist, sz in ((self.h_ang, 3), (self.h_q_mj, 29), (self.h_dq_mj, 29), (self.h_act_mj, 29)):
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
