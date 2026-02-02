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

"""
Unitree RL Locomotion Controller for G1

This runs the motion.pt TorchScript policy for 12-DOF leg locomotion.
Policy from: https://huggingface.co/nepyope/unitree_rl_locomotion
"""

import logging

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from lerobot.robots.unitree_g1.g1_utils import G1_29_JointIndex

logger = logging.getLogger(__name__)

# Default repository and filename
DEFAULT_REPO_ID = "nepyope/unitree_rl_locomotion"
DEFAULT_POLICY_FILE = "motion.pt"

# Control parameters
CONTROL_DT = 0.02  # 50Hz
ACTION_SCALE = 0.25
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
CMD_SCALE = np.array([2.0, 2.0, 0.25])
MAX_CMD = np.array([0.8, 0.5, 1.57])
GAIT_PERIOD = 0.8

# 12-DOF leg joints
NUM_LEG_JOINTS = 12
LEG_JOINT_INDICES = list(range(12))  # 0-11
DEFAULT_LEG_ANGLES = np.array(
    [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0],
    dtype=np.float32
)

# Observation size: 9 + 12*3 + 2 = 47
NUM_OBS = 9 + NUM_LEG_JOINTS * 3 + 2


def _load_policy(
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = DEFAULT_POLICY_FILE,
) -> torch.jit.ScriptModule:
    """Load TorchScript locomotion policy from HuggingFace Hub."""
    logger.info(f"Loading policy from: {repo_id}/{filename}")
    policy_path = hf_hub_download(repo_id=repo_id, filename=filename)
    policy = torch.jit.load(policy_path)
    policy.eval()
    logger.info("TorchScript policy loaded successfully")
    return policy


def _get_gravity_orientation(quaternion) -> np.ndarray:
    """Get gravity orientation from quaternion [w, x, y, z]."""
    qw, qx, qy, qz = quaternion
    gravity = np.zeros(3, dtype=np.float32)
    gravity[0] = 2 * (-qz * qx + qw * qy)
    gravity[1] = -2 * (qz * qy + qw * qx)
    gravity[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity


class UnitreeRLLocomotionController:
    """Unitree RL Locomotion Controller for G1 (12-DOF legs).
    
    Compatible interface with GrootLocomotionController and HolosomaLocomotionController.
    """

    def __init__(self):
        # Load policy
        self.policy = _load_policy()

        self.cmd = np.zeros(3, dtype=np.float32)

        # State buffers
        self.qj = np.zeros(NUM_LEG_JOINTS, dtype=np.float32)
        self.dqj = np.zeros(NUM_LEG_JOINTS, dtype=np.float32)
        self.obs = np.zeros(NUM_OBS, dtype=np.float32)
        self.last_action = np.zeros(NUM_LEG_JOINTS, dtype=np.float32)

        # Counter for phase calculation
        self.counter = 0

        logger.info(f"UnitreeRLLocomotionController initialized: {NUM_LEG_JOINTS} DOF, {NUM_OBS}D obs")

    def run_step(self, action: dict, lowstate) -> dict:
        """Run one step of the locomotion controller.

        Args:
            action: Action dict from teleoperator containing remote.lx/ly/rx/ry
            lowstate: Robot lowstate containing motor positions/velocities and IMU

        Returns:
            Action dict for leg joints (0-11) and waist (12-14)
        """
        if lowstate is None:
            return {}

        self.counter += 1

        # Get command from remote controller in action (with deadzone)
        ly = action.get("remote.ly", 0.0)
        lx = action.get("remote.lx", 0.0)
        rx = action.get("remote.rx", 0.0)
        ly = ly if abs(ly) > 0.1 else 0.0
        lx = lx if abs(lx) > 0.1 else 0.0
        rx = rx if abs(rx) > 0.1 else 0.0
        self.cmd[:] = [ly, -lx, -rx]

        # Get leg joint positions and velocities from lowstate
        for i in range(NUM_LEG_JOINTS):
            self.qj[i] = lowstate.motor_state[i].q
            self.dqj[i] = lowstate.motor_state[i].dq

        # Get IMU data
        quat = lowstate.imu_state.quaternion  # [w, x, y, z]
        ang_vel = np.array(lowstate.imu_state.gyroscope, dtype=np.float32)

        # Get gravity orientation
        gravity = _get_gravity_orientation(quat)

        # Scale observations
        qj_obs = (self.qj - DEFAULT_LEG_ANGLES) * DOF_POS_SCALE
        dqj_obs = self.dqj * DOF_VEL_SCALE
        ang_vel_s = ang_vel * ANG_VEL_SCALE

        # Calculate gait phase
        phase = (self.counter * CONTROL_DT % GAIT_PERIOD) / GAIT_PERIOD
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)

        # Build observation vector (47D)
        # [ang_vel(3), gravity(3), cmd(3), qj_obs(12), dqj_obs(12), last_action(12), sin(1), cos(1)]
        self.obs[0:3] = ang_vel_s
        self.obs[3:6] = gravity
        self.obs[6:9] = self.cmd * CMD_SCALE * MAX_CMD
        self.obs[9:21] = qj_obs
        self.obs[21:33] = dqj_obs
        self.obs[33:45] = self.last_action
        self.obs[45] = sin_phase
        self.obs[46] = cos_phase

        # Run policy inference (TorchScript)
        with torch.no_grad():
            obs_tensor = torch.from_numpy(self.obs).unsqueeze(0).float()
            policy_action = self.policy(obs_tensor).numpy().squeeze()

        self.last_action = policy_action.copy()

        # Transform action to target joint positions
        target_pos = DEFAULT_LEG_ANGLES + policy_action * ACTION_SCALE

        # Build action dict
        action_dict = {}

        # Leg joints (0-11)
        for i in range(NUM_LEG_JOINTS):
            motor_name = G1_29_JointIndex(i).name
            action_dict[f"{motor_name}.q"] = float(target_pos[i])

        # Hold waist at zero (12-14)
        for i in range(12, 15):
            motor_name = G1_29_JointIndex(i).name
            action_dict[f"{motor_name}.q"] = 0.0

        return action_dict
