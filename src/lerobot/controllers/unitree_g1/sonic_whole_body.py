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

import logging

import numpy as np
import onnxruntime as ort
import torch

from lerobot.robots.unitree_g1.g1_utils import (
    ISAACLAB_TO_MUJOCO,
    MUJOCO_TO_ISAACLAB,
    NUM_MOTORS,
    G1_29_JointIndex,
    get_gravity_orientation,
)

from ..controller import RobotController
from ..history import FrameHistory
from ..hub import load_onnx_policy
from ..processors import make_controller_processors
from .proprio import read_proprio

logger = logging.getLogger(__name__)

CONTROL_DT = 0.02  # 50 Hz control period (s)
TOKEN_DIM = 64  # decoder latent size
HISTORY_LEN = 10  # proprioception frames the decoder conditions on

# Latent-token feature-key prefixes: action carries the token, obs echoes it back under the
# same keys (like an arm's ``<joint>.pos``), so rollout can hold and home to an observed token.
TOKEN_ACTION_PREFIX = "motion_token"  # nosec B105 - feature-key prefix, not a secret
TOKEN_STATE_PREFIX = TOKEN_ACTION_PREFIX

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

    required = ("kp", "kd", "default_angles", "action_scale", "neutral_token")
    decoder, arr = load_onnx_policy(repo_id, POLICY_FILES[policy_type], metadata_keys=required)
    return decoder, arr["kp"], arr["kd"], arr["default_angles"], arr["action_scale"], arr["neutral_token"]


class SonicWholeBodyController(RobotController):
    """Full-body SONIC decoder controller for UnitreeG1's background controller thread.

    Token-only deploy (encoder bypassed): each tick it appends the latest robot state to
    10-frame history buffers, then maps the policy-supplied 64-D token + that history to a
    residual action added onto ``default_angles`` -> 50 Hz joint-position targets.
    """

    control_dt = CONTROL_DT
    # Re-sending the last token keeps the decoder balancing, unlike a held joystick velocity.
    supports_position_hold = True

    def __init__(self, policy_type: str = "default"):
        self.decoder, self.kp, self.kd, self.default_angles, self.action_scale, self.neutral_token = (
            load_policy(policy_type=policy_type)
        )
        self.decoder_input = self.decoder.get_inputs()[0].name

        # The decoder consumes MuJoCo joint order: q/dq are reordered via MUJOCO_TO_ISAACLAB and its
        # residual back via ISAACLAB_TO_MUJOCO (validated against the ONNX; don't flip).
        self.preprocessor, self.postprocessor = make_controller_processors(
            observation_stats={"observation.joint_pos": (self.default_angles, np.ones(NUM_MOTORS))},
            action_stats=(self.default_angles, self.action_scale),
            observation_order={
                "observation.joint_pos": MUJOCO_TO_ISAACLAB,
                "observation.joint_vel": MUJOCO_TO_ISAACLAB,
            },
            action_order=ISAACLAB_TO_MUJOCO,
        )

        # 64-D latent-token action space; rollout maps the policy's 64-D output onto these keys.
        self.action_ft = {f"{TOKEN_ACTION_PREFIX}.{i}.pos": float for i in range(TOKEN_DIM)}
        # 64-D token proprio state, aggregated by rollout into observation.state (last token).
        self.observation_ft = {f"{TOKEN_STATE_PREFIX}.{i}.pos": float for i in range(TOKEN_DIM)}

        # 10-frame proprioception history, in MuJoCo joint order.
        self.h_q_mj = FrameHistory(HISTORY_LEN, np.zeros(NUM_MOTORS, np.float32))
        self.h_dq_mj = FrameHistory(HISTORY_LEN, np.zeros(NUM_MOTORS, np.float32))
        self.h_ang = FrameHistory(HISTORY_LEN, np.zeros(3, np.float32))
        self.h_act_mj = FrameHistory(HISTORY_LEN, np.zeros(NUM_MOTORS, np.float32))
        self.h_quat = FrameHistory(HISTORY_LEN, np.array([1, 0, 0, 0], np.float32))

        self.reset()
        logger.info("SonicWholeBodyController initialized")

    def reset(self) -> None:
        """Reset internal state for a new episode: held token and proprioception history."""
        self.last_action_mj: np.ndarray = np.zeros(NUM_MOTORS, np.float32)
        for history in (self.h_q_mj, self.h_dq_mj, self.h_ang, self.h_act_mj, self.h_quat):
            history.reset()
        self._last_token: np.ndarray | None = None  # neutral token is re-seeded on the first tick

    def observation_state(self) -> dict[str, float]:
        """Echo the last decoded token as ``observation.state`` so a token-output VLA closes
        the loop on its own previous token. Before the first tick this is the neutral token the
        decoder is about to hold; an all-zeros token decodes to a bent-over pose, not a stand."""
        token = self._last_token if self._last_token is not None else self.neutral_token
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
            plus the decoder's residual, scaled by ``action_scale``.
        """
        # Token: reassemble the dense 64-D latent from motion_token.{i}.pos (all keys required);
        # else hold the last one (neutral until the first real token, which decodes to a stand).
        keys = [f"{TOKEN_ACTION_PREFIX}.{i}.pos" for i in range(TOKEN_DIM)]
        if action and all(k in action for k in keys):
            self._last_token = np.fromiter(
                (float(action[k]) for k in keys), dtype=np.float32, count=TOKEN_DIM
            )
        elif self._last_token is None:
            self._last_token = self.neutral_token.copy()

        # Read proprioception from lowstate (IsaacLab joint order) into the decoder's MuJoCo order.
        proprio = read_proprio(lowstate)
        mj = self.preprocessor({"observation.joint_pos": proprio.q, "observation.joint_vel": proprio.dq})
        quat = proprio.quat / (np.linalg.norm(proprio.quat) + 1e-8)

        self.h_q_mj.push(mj["observation.joint_pos"])
        self.h_dq_mj.push(mj["observation.joint_vel"])
        self.h_ang.push(proprio.ang_vel)
        self.h_act_mj.push(self.last_action_mj)
        self.h_quat.push(quat)

        # Assemble the 994-D decoder input: token + oldest->newest history + gravity.
        gravity = [get_gravity_orientation(hquat) for hquat in self.h_quat]
        obs: np.ndarray = np.concatenate(
            [
                self._last_token,
                self.h_ang.stacked(),
                self.h_q_mj.stacked(),
                self.h_dq_mj.stacked(),
                self.h_act_mj.stacked(),
                *gravity,
            ]
        ).astype(np.float32)

        # Decode -> residual action (MuJoCo order) added onto the standing pose.
        action_mj = (
            self.decoder.run(None, {self.decoder_input: obs.reshape(1, -1)})[0].squeeze().astype(np.float32)
        )
        self.last_action_mj = action_mj.copy()
        target = self.postprocessor(torch.from_numpy(action_mj)).numpy()
        return {f"{m.name}.q": float(target[m.value]) for m in G1_29_JointIndex}
