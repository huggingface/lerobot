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

import logging
from collections import deque

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from lerobot.robots.unitree_g1.g1_utils import G1_29_JointIndex, get_gravity_orientation

logger = logging.getLogger(__name__)


GROOT_DEFAULT_ANGLES = np.zeros(29, dtype=np.float32)
GROOT_DEFAULT_ANGLES[[0, 6]] = -0.1  # Hip pitch
GROOT_DEFAULT_ANGLES[[3, 9]] = 0.3  # Knee
GROOT_DEFAULT_ANGLES[[4, 10]] = -0.2  # Ankle pitch

MISSING_JOINTS = []
G1_MODEL = "g1_23"  # Or "g1_29"
if G1_MODEL == "g1_23":
    MISSING_JOINTS = [12, 14, 20, 21, 27, 28]  # Waist yaw/pitch, wrist pitch/yaw

# Control parameters
CONTROL_DT = 0.02  # 50Hz
ACTION_SCALE = 0.25
ANG_VEL_SCALE: float = 0.25
DOF_POS_SCALE: float = 1.0
DOF_VEL_SCALE: float = 0.05
CMD_SCALE: list = [2.0, 2.0, 0.25]


DEFAULT_GROOT_REPO_ID = "nepyope/GR00T-WholeBodyControl_g1"


def load_groot_policies(
    repo_id: str = DEFAULT_GROOT_REPO_ID,
) -> tuple[ort.InferenceSession, ort.InferenceSession]:
    """Load GR00T dual-policy system (Balance + Walk) from the hub.

    Args:
        repo_id: Hugging Face Hub repository ID containing the ONNX policies.
    """
    logger.info(f"Loading GR00T dual-policy system from the hub ({repo_id})...")

    # Download ONNX policies from Hugging Face Hub
    balance_path = hf_hub_download(
        repo_id=repo_id,
        filename="GR00T-WholeBodyControl-Balance.onnx",
    )
    walk_path = hf_hub_download(
        repo_id=repo_id,
        filename="GR00T-WholeBodyControl-Walk.onnx",
    )

    # Load ONNX policies
    policy_balance = ort.InferenceSession(balance_path)
    policy_walk = ort.InferenceSession(walk_path)

    logger.info("GR00T policies loaded successfully")

    return policy_balance, policy_walk


class GrootLocomotionController:
    """GR00T lower-body locomotion controller for the Unitree G1."""

    control_dt = CONTROL_DT  # Expose for unitree_g1.py

    def __init__(self):
        # Load policies
        self.policy_balance, self.policy_walk = load_groot_policies()

        self.cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # vx, vy, theta_dot

        # Robot state
        self.groot_qj_all = np.zeros(29, dtype=np.float32)
        self.groot_dqj_all = np.zeros(29, dtype=np.float32)
        self.groot_action = np.zeros(15, dtype=np.float32)
        self.groot_obs_single = np.zeros(86, dtype=np.float32)
        self.groot_obs_history = deque(maxlen=6)
        self.groot_obs_stacked = np.zeros(516, dtype=np.float32)
        self.groot_height_cmd = 0.74  # Default base height
        self.groot_orientation_cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Input to GR00T is 6 frames (6*86D=516)
        for _ in range(6):
            self.groot_obs_history.append(np.zeros(86, dtype=np.float32))

        logger.info("GrootLocomotionController initialized")

    def run_step(self, action: dict, lowstate) -> dict:
        """Run one step of the locomotion controller.

        Args:
            action: Action dict containing remote.lx/ly/rx/ry and buttons
            lowstate: Robot lowstate containing motor positions/velocities and IMU

        Returns:
            Action dict for lower body joints (0-14)
        """
        if lowstate is None:
            return {}

        # Get buttons from action
        buttons = [int(action.get(f"remote.button.{i}", 0)) for i in range(16)]
        if buttons[0]:  # R1 - raise waist
            self.groot_height_cmd += 0.001
            self.groot_height_cmd = np.clip(self.groot_height_cmd, 0.50, 1.00)
        if buttons[4]:  # R2 - lower waist
            self.groot_height_cmd -= 0.001
            self.groot_height_cmd = np.clip(self.groot_height_cmd, 0.50, 1.00)

        self.cmd[0] = action.get("remote.ly", 0.0)  # Forward/backward
        self.cmd[1] = action.get("remote.lx", 0.0) * -1  # Left/right (negated)
        self.cmd[2] = action.get("remote.rx", 0.0) * -1  # Rotation rate (negated)

        # Get joint positions and velocities from lowstate
        for motor in G1_29_JointIndex:
            idx = motor.value
            self.groot_qj_all[idx] = lowstate.motor_state[idx].q
            self.groot_dqj_all[idx] = lowstate.motor_state[idx].dq

        # Adapt observation for g1_23dof
        for idx in MISSING_JOINTS:
            self.groot_qj_all[idx] = 0.0
            self.groot_dqj_all[idx] = 0.0

        # Scale joint positions and velocities
        qj_obs = self.groot_qj_all.copy()
        dqj_obs = self.groot_dqj_all.copy()

        # Express IMU data in gravity frame of reference
        quat = lowstate.imu_state.quaternion
        ang_vel = np.array(lowstate.imu_state.gyroscope, dtype=np.float32)
        gravity_orientation = get_gravity_orientation(quat)

        # Scale joint positions and velocities before policy inference
        qj_obs = (qj_obs - GROOT_DEFAULT_ANGLES) * DOF_POS_SCALE
        dqj_obs = dqj_obs * DOF_VEL_SCALE
        ang_vel_scaled = ang_vel * ANG_VEL_SCALE

        # Build single frame observation
        self.groot_obs_single[:3] = self.cmd * np.array(CMD_SCALE)
        self.groot_obs_single[3] = self.groot_height_cmd
        self.groot_obs_single[4:7] = self.groot_orientation_cmd
        self.groot_obs_single[7:10] = ang_vel_scaled
        self.groot_obs_single[10:13] = gravity_orientation
        self.groot_obs_single[13:42] = qj_obs
        self.groot_obs_single[42:71] = dqj_obs
        self.groot_obs_single[71:86] = self.groot_action  # 15D previous actions

        # Add to history and stack observations (6 frames × 86D = 516D)
        self.groot_obs_history.append(self.groot_obs_single.copy())

        # Stack all 6 frames into 516D vector
        for i, obs_frame in enumerate(self.groot_obs_history):
            start_idx = i * 86
            end_idx = start_idx + 86
            self.groot_obs_stacked[start_idx:end_idx] = obs_frame

        cmd_magnitude = np.linalg.norm(self.cmd)
        selected_policy = (
            self.policy_balance if cmd_magnitude < 0.05 else self.policy_walk
        )  # Balance/standing policy for small commands, walking policy for movement commands

        # Run policy inference
        ort_inputs = {selected_policy.get_inputs()[0].name: np.expand_dims(self.groot_obs_stacked, axis=0)}
        ort_outs = selected_policy.run(None, ort_inputs)
        self.groot_action = ort_outs[0].squeeze()

        # Transform action back to target joint positions
        target_dof_pos_15 = GROOT_DEFAULT_ANGLES[:15] + self.groot_action * ACTION_SCALE

        # Build action dict (only first 15 joints for GR00T)
        action_dict = {}
        for i in range(15):
            motor_name = G1_29_JointIndex(i).name
            action_dict[f"{motor_name}.q"] = float(target_dof_pos_15[i])

        # Zero out missing joints for g1_23dof
        for joint_idx in MISSING_JOINTS:
            motor_name = G1_29_JointIndex(joint_idx).name
            action_dict[f"{motor_name}.q"] = 0.0

        return action_dict
