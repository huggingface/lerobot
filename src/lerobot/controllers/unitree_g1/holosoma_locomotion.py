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

import numpy as np
import onnxruntime as ort
import torch

from lerobot.robots.unitree_g1.g1_utils import (
    REMOTE_AXES,
    REMOTE_KEYS,
    G1_29_JointArmIndex,
    G1_29_JointIndex,
)

from ..controller import RobotController
from ..hub import load_onnx_policy
from ..processors import make_controller_processors
from .proprio import read_proprio

logger = logging.getLogger(__name__)

DEFAULT_ANGLES: np.ndarray = np.zeros(29, dtype=np.float32)
DEFAULT_ANGLES[[0, 6]] = -0.312  # Hip pitch
DEFAULT_ANGLES[[3, 9]] = 0.669  # Knee
DEFAULT_ANGLES[[4, 10]] = -0.363  # Ankle pitch
DEFAULT_ANGLES[[15, 22]] = 0.2  # Shoulder pitch
DEFAULT_ANGLES[16] = 0.2  # Left shoulder roll
DEFAULT_ANGLES[23] = -0.2  # Right shoulder roll
DEFAULT_ANGLES[[18, 25]] = 0.6  # Elbow

# Control parameters
ACTION_SCALE = 0.25
CONTROL_DT = 0.005  # 200Hz
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
GAIT_PERIOD = 0.5


DEFAULT_HOLOSOMA_REPO_ID = "nepyope/holosoma_locomotion"

# Policy filename mapping
POLICY_FILES = {
    "fastsac": "fastsac_g1_29dof.onnx",
    "ppo": "ppo_g1_29dof.onnx",
}


def load_policy(
    repo_id: str = DEFAULT_HOLOSOMA_REPO_ID,
    policy_type: str = "fastsac",
) -> tuple[ort.InferenceSession, np.ndarray, np.ndarray]:
    """Load Holosoma locomotion policy and extract KP/KD from metadata.

    Args:
        repo_id: Hugging Face Hub repo ID
        policy_type: Either "fastsac" (default) or "ppo"

    Returns:
        (policy, kp, kd) tuple
    """
    if policy_type not in POLICY_FILES:
        raise ValueError(f"Unknown policy type: {policy_type}. Choose from: {list(POLICY_FILES.keys())}")

    policy, constants = load_onnx_policy(repo_id, POLICY_FILES[policy_type], metadata_keys=("kp", "kd"))
    return policy, constants["kp"], constants["kd"]


class HolosomaLocomotionController(RobotController):
    """Holosoma lower-body locomotion controller for Unitree G1."""

    control_dt = CONTROL_DT
    # Joystick velocities and buttons; the arms stay the caller's.
    action_ft = dict.fromkeys(REMOTE_KEYS, float)
    passthrough = True

    def __init__(self):
        # Load policy and gains
        self.policy, self.kp, self.kd = load_policy()

        self.default_angles = DEFAULT_ANGLES
        self.cmd = np.zeros(3, dtype=np.float32)

        # Robot state
        self.obs = np.zeros(100, dtype=np.float32)
        self.last_action = np.zeros(29, dtype=np.float32)

        # Gait phase
        self.phase = np.array([[0.0, np.pi]], dtype=np.float32)
        self.phase_dt = 2 * np.pi / ((1.0 / CONTROL_DT) * GAIT_PERIOD)
        self.is_standing = True

        # `scale * x + offset` in. The policy acts on all 29 joints; the leg and waist targets (0-14)
        # come back as `ACTION_SCALE * a + default`.
        self.preprocessor, self.postprocessor = make_controller_processors(
            observation_affine={
                "observation.joint_pos": (DOF_POS_SCALE, -DEFAULT_ANGLES * DOF_POS_SCALE),
                "observation.joint_vel": (np.full(29, DOF_VEL_SCALE), 0.0),
                "observation.ang_vel": (np.full(3, ANG_VEL_SCALE), 0.0),
            },
            action_affine=(ACTION_SCALE, DEFAULT_ANGLES[:15]),
            action_order=range(15),
        )

        logger.info("HolosomaLocomotionController initialized")

    def reset(self) -> None:
        """Reset internal state for a new episode."""
        self.cmd[:] = 0.0
        self.obs[:] = 0.0
        self.last_action[:] = 0.0
        self.phase = np.array([[0.0, np.pi]], dtype=np.float32)
        self.is_standing = True

    def run_step(self, action: dict, lowstate) -> dict:
        """Run one step of the locomotion controller.

        Args:
            action: Action dict containing remote.lx/ly/rx/ry
            lowstate: Robot lowstate containing motor positions/velocities and IMU

        Returns:
            Action dict for lower body joints (0-14)
        """
        if lowstate is None:
            return {}

        lx, ly, rx, _ry = (action.get(k, 0.0) for k in REMOTE_AXES)
        ly = ly if abs(ly) > 0.1 else 0.0
        lx = lx if abs(lx) > 0.1 else 0.0
        rx = rx if abs(rx) > 0.1 else 0.0
        ly = np.clip(ly, -0.3, 0.3)
        lx = np.clip(lx, -0.3, 0.3)
        self.cmd[:] = [ly, -lx, -rx]

        proprio = read_proprio(lowstate)
        qj, dqj = proprio.q, proprio.dq

        # Hide arm positions from policy (show DEFAULT_ANGLES instead)
        # This prevents policy from reacting to teleop arm movements
        for arm_joint in G1_29_JointArmIndex:
            qj[arm_joint.value] = DEFAULT_ANGLES[arm_joint.value]
            dqj[arm_joint.value] = 0.0

        obs = self.preprocessor(
            {
                "observation.joint_pos": qj,
                "observation.joint_vel": dqj,
                "observation.ang_vel": proprio.ang_vel,
            }
        )

        # Update gait phase
        if np.linalg.norm(self.cmd[:2]) < 0.01 and abs(self.cmd[2]) < 0.01:
            self.phase[0, :] = np.pi
            self.is_standing = True
        elif self.is_standing:
            self.phase = np.array([[0.0, np.pi]], dtype=np.float32)
            self.is_standing = False
        else:
            self.phase = np.fmod(self.phase + self.phase_dt + np.pi, 2 * np.pi) - np.pi

        sin_ph = np.sin(self.phase[0])
        cos_ph = np.cos(self.phase[0])

        # Build observations
        self.obs[0:29] = self.last_action
        self.obs[29:32] = obs["observation.ang_vel"]
        self.obs[32] = self.cmd[2]
        self.obs[33:35] = self.cmd[:2]
        self.obs[35:37] = cos_ph
        self.obs[37:66] = obs["observation.joint_pos"]
        self.obs[66:95] = obs["observation.joint_vel"]
        self.obs[95:98] = proprio.gravity
        self.obs[98:100] = sin_ph

        # Run policy inference
        ort_in = {self.policy.get_inputs()[0].name: self.obs.reshape(1, -1).astype(np.float32)}
        raw_action = self.policy.run(None, ort_in)[0].squeeze()
        policy_action = np.clip(raw_action, -100.0, 100.0)
        self.last_action = policy_action.copy()

        # Transform action back to target joint positions (first 15 joints only)
        target = self.postprocessor(torch.as_tensor(policy_action)).numpy()

        action_dict = {}
        for i in range(15):
            motor_name = G1_29_JointIndex(i).name
            action_dict[f"{motor_name}.q"] = float(target[i])

        return action_dict
