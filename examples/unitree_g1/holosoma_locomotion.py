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

import argparse
import json
import logging
import time

import numpy as np
import onnx
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config
from lerobot.robots.unitree_g1.g1_utils import G1_29_JointIndex
from lerobot.robots.unitree_g1.unitree_g1 import UnitreeG1

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

MISSING_JOINTS = []
G1_MODEL = "g1_23"  # Or "g1_29"
if G1_MODEL == "g1_23":
    MISSING_JOINTS = [12, 14, 20, 21, 27, 28]  # Waist yaw/pitch, wrist pitch/yaw

# Control parameters
ACTION_SCALE = 0.25
CONTROL_DT = 0.02  # 50Hz
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
GAIT_PERIOD = 1.0


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
    """Holosoma whole-body locomotion controller for Unitree G1."""

    def __init__(self, policy, robot, kp: np.ndarray, kd: np.ndarray):
        self.policy = policy
        self.robot = robot

        # Override robot's PD gains with policy gains
        self.robot.kp = kp
        self.robot.kd = kd

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

    def run_step(self):
        # Get current observation
        obs = self.robot.get_observation()

        if not obs:
            return

        # Get command from remote controller
        ly = obs["remote.ly"] if abs(obs["remote.ly"]) > 0.1 else 0.0
        lx = obs["remote.lx"] if abs(obs["remote.lx"]) > 0.1 else 0.0
        rx = obs["remote.rx"] if abs(obs["remote.rx"]) > 0.1 else 0.0
        self.cmd[:] = [ly, -lx, -rx]

        # Get joint positions and velocities
        for motor in G1_29_JointIndex:
            name = motor.name
            idx = motor.value
            self.qj[idx] = obs[f"{name}.q"]
            self.dqj[idx] = obs[f"{name}.dq"]

        # Adapt observation for g1_23dof
        for idx in MISSING_JOINTS:
            self.qj[idx] = 0.0
            self.dqj[idx] = 0.0

        # Express IMU data in gravity frame of reference
        quat = [obs["imu.quat.w"], obs["imu.quat.x"], obs["imu.quat.y"], obs["imu.quat.z"]]
        ang_vel = np.array([obs["imu.gyro.x"], obs["imu.gyro.y"], obs["imu.gyro.z"]], dtype=np.float32)
        gravity = self.robot.get_gravity_orientation(quat)

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
        action = np.clip(raw_action, -100.0, 100.0)
        self.last_action = action.copy()

        # Transform action back to target joint positions
        target = DEFAULT_ANGLES + action * ACTION_SCALE

        # Build action dict
        action_dict = {}
        for motor in G1_29_JointIndex:
            action_dict[f"{motor.name}.q"] = float(target[motor.value])

        # Zero out missing joints for g1_23dof
        for joint_idx in MISSING_JOINTS:
            motor_name = G1_29_JointIndex(joint_idx).name
            action_dict[f"{motor_name}.q"] = 0.0

        # Send action to robot
        self.robot.send_action(action_dict)


def run(repo_id: str = DEFAULT_HOLOSOMA_REPO_ID, policy_type: str = "fastsac") -> None:
    """Main function to run the Holosoma locomotion controller.

    Args:
        repo_id: Hugging Face Hub repository ID for Holosoma policies.
        policy_type: Policy type to use ('fastsac' or 'ppo').
    """
    # Load policy and gains
    policy, kp, kd = load_policy(repo_id=repo_id, policy_type=policy_type)

    # Initialize robot
    config = UnitreeG1Config()
    robot = UnitreeG1(config)
    robot.connect()

    holosoma_controller = HolosomaLocomotionController(policy, robot, kp, kd)

    try:
        robot.reset(CONTROL_DT, DEFAULT_ANGLES)

        logger.info("Use joystick: LY=fwd/back, LX=left/right, RX=rotate")
        logger.info("Press Ctrl+C to stop")

        # Run step
        while not robot._shutdown_event.is_set():
            start_time = time.time()
            holosoma_controller.run_step()
            elapsed = time.time() - start_time
            sleep_time = max(0, CONTROL_DT - elapsed)
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        logger.info("Stopping locomotion...")
    finally:
        if robot.is_connected:
            robot.disconnect()
        logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Holosoma Locomotion Controller for Unitree G1")
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_HOLOSOMA_REPO_ID,
        help=f"Hugging Face Hub repo ID for Holosoma policies (default: {DEFAULT_HOLOSOMA_REPO_ID})",
    )
    parser.add_argument(
        "--policy",
        type=str,
        choices=["fastsac", "ppo"],
        default="fastsac",
        help="Policy type to use: 'fastsac' (default) or 'ppo'",
    )
    args = parser.parse_args()

    run(repo_id=args.repo_id, policy_type=args.policy)
