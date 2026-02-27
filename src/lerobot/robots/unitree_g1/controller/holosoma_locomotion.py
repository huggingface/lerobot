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

import json
import logging

import numpy as np
import onnx
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from lerobot.robots.unitree_g1.g1_utils import G1_29_JointIndex, get_gravity_orientation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_ANGLES = np.zeros(29, dtype=np.float32)
DEFAULT_ANGLES[[0, 6]] = -0.312  # Hip pitch
DEFAULT_ANGLES[[3, 9]] = 0.669  # Knee
DEFAULT_ANGLES[[4, 10]] = -0.363  # Ankle pitch
DEFAULT_ANGLES[[15, 22]] = 0.2  # Shoulder pitch
DEFAULT_ANGLES[16] = 0.2  # Left shoulder roll
DEFAULT_ANGLES[23] = -0.2  # Right shoulder roll
DEFAULT_ANGLES[[18, 25]] = 0.6  # Elbow

# Control parameters
ACTION_SCALE = 0.25
CONTROL_DT = 0.005  # 50Hz
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

    filename = POLICY_FILES[policy_type]
    logger.info(f"Loading {policy_type.upper()} policy from: {repo_id}/{filename}")
    policy_path = hf_hub_download(repo_id=repo_id, filename=filename)

    policy = ort.InferenceSession(policy_path)
    logger.info(f"Policy loaded: {policy.get_inputs()[0].shape} → {policy.get_outputs()[0].shape}")

    # Extract KP/KD from ONNX metadata
    model = onnx.load(policy_path)
    metadata = {prop.key: prop.value for prop in model.metadata_props}

    if "kp" not in metadata or "kd" not in metadata:
        raise ValueError("ONNX model must contain 'kp' and 'kd' in metadata")

    kp = np.array(json.loads(metadata["kp"]), dtype=np.float32)
    kd = np.array(json.loads(metadata["kd"]), dtype=np.float32)
    logger.info(f"Loaded KP/KD from ONNX ({len(kp)} joints)")

    return policy, kp, kd


class HolosomaLocomotionController:
    """Holosoma lower-body locomotion controller for Unitree G1."""

    control_dt = CONTROL_DT  # Expose for unitree_g1.py

    def __init__(self):
        # Load policy and gains
        self.policy, self.kp, self.kd = load_policy()

        self.cmd = np.zeros(3, dtype=np.float32)

        # Robot state
        self.qj = np.zeros(29, dtype=np.float32)
        self.dqj = np.zeros(29, dtype=np.float32)
        self.obs = np.zeros(100, dtype=np.float32)
        self.last_action = np.zeros(29, dtype=np.float32)

        # Gait phase
        self.phase = np.array([[0.0, np.pi]], dtype=np.float32)
        self.phase_dt = 2 * np.pi / ((1.0 / CONTROL_DT) * GAIT_PERIOD)
        self.is_standing = True

        logger.info("HolosomaLocomotionController initialized")

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

        # Get command from action (with deadzone, vx/vy capped at 30%)
        ly = action.get("remote.ly", 0.0)
        lx = action.get("remote.lx", 0.0)
        rx = action.get("remote.rx", 0.0)
        ly = ly if abs(ly) > 0.1 else 0.0
        lx = lx if abs(lx) > 0.1 else 0.0
        rx = rx if abs(rx) > 0.1 else 0.0
        ly = np.clip(ly, -0.3, 0.3)
        lx = np.clip(lx, -0.3, 0.3)
        self.cmd[:] = [ly, -lx, -rx]

        # Get joint positions and velocities from lowstate
        for motor in G1_29_JointIndex:
            idx = motor.value
            self.qj[idx] = lowstate.motor_state[idx].q
            self.dqj[idx] = lowstate.motor_state[idx].dq

        # Hide arm positions from policy (show DEFAULT_ANGLES instead)
        # This prevents policy from reacting to teleop arm movements
        for idx in range(15, 29):
            self.qj[idx] = DEFAULT_ANGLES[idx]
            self.dqj[idx] = 0.0

        # Express IMU data in gravity frame of reference
        quat = lowstate.imu_state.quaternion
        ang_vel = np.array(lowstate.imu_state.gyroscope, dtype=np.float32)
        gravity = get_gravity_orientation(quat)

        # Scale joint positions and velocities before policy inference
        qj_obs = (self.qj - DEFAULT_ANGLES) * DOF_POS_SCALE
        dqj_obs = self.dqj * DOF_VEL_SCALE
        ang_vel_s = ang_vel * ANG_VEL_SCALE

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
        self.obs[29:32] = ang_vel_s
        self.obs[32] = self.cmd[2]
        self.obs[33:35] = self.cmd[:2]
        self.obs[35:37] = cos_ph
        self.obs[37:66] = qj_obs
        self.obs[66:95] = dqj_obs
        self.obs[95:98] = gravity
        self.obs[98:100] = sin_ph

        # Run policy inference
        ort_in = {self.policy.get_inputs()[0].name: self.obs.reshape(1, -1).astype(np.float32)}
        raw_action = self.policy.run(None, ort_in)[0].squeeze()
        policy_action = np.clip(raw_action, -100.0, 100.0)
        self.last_action = policy_action.copy()

        # Transform action back to target joint positions
        target = DEFAULT_ANGLES + policy_action * ACTION_SCALE

        # Build action dict for all 29 joints (teleop will override arms)
        action_dict = {}
        for motor in G1_29_JointIndex:
            action_dict[f"{motor.name}.q"] = float(target[motor.value])

        return action_dict
