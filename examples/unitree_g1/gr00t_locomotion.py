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
import logging
import time
from collections import deque

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config
from lerobot.robots.unitree_g1.g1_utils import G1_29_JointIndex
from lerobot.robots.unitree_g1.unitree_g1 import UnitreeG1

logging.basicConfig(level=logging.INFO)
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
ACTION_SCALE = 0.25
CONTROL_DT = 0.02  # 50Hz
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

    def __init__(self, policy_balance, policy_walk, robot, config):
        self.policy_balance = policy_balance
        self.policy_walk = policy_walk
        self.robot = robot
        self.config = config

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

    def run_step(self):
        # Get current observation
        obs = self.robot.get_observation()

        if not obs:
            return

        # Get command from remote controller
        if obs["remote.buttons"][0]:  # R1 - raise waist
            self.groot_height_cmd += 0.001
            self.groot_height_cmd = np.clip(self.groot_height_cmd, 0.50, 1.00)
        if obs["remote.buttons"][4]:  # R2 - lower waist
            self.groot_height_cmd -= 0.001
            self.groot_height_cmd = np.clip(self.groot_height_cmd, 0.50, 1.00)

        self.cmd[0] = obs["remote.ly"]  # Forward/backward
        self.cmd[1] = obs["remote.lx"] * -1  # Left/right
        self.cmd[2] = obs["remote.rx"] * -1  # Rotation rate

        # Get joint positions and velocities from flat dict
        for motor in G1_29_JointIndex:
            name = motor.name
            idx = motor.value
            self.groot_qj_all[idx] = obs[f"{name}.q"]
            self.groot_dqj_all[idx] = obs[f"{name}.dq"]

        # Adapt observation for g1_23dof
        for idx in MISSING_JOINTS:
            self.groot_qj_all[idx] = 0.0
            self.groot_dqj_all[idx] = 0.0

        # Scale joint positions and velocities
        qj_obs = self.groot_qj_all.copy()
        dqj_obs = self.groot_dqj_all.copy()

        # Express IMU data in gravity frame of reference
        quat = [obs["imu.quat.w"], obs["imu.quat.x"], obs["imu.quat.y"], obs["imu.quat.z"]]
        ang_vel = np.array([obs["imu.gyro.x"], obs["imu.gyro.y"], obs["imu.gyro.z"]], dtype=np.float32)
        gravity_orientation = self.robot.get_gravity_orientation(quat)

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

        # Send action to robot
        self.robot.send_action(action_dict)


def run(repo_id: str = DEFAULT_GROOT_REPO_ID) -> None:
    """Main function to run the GR00T locomotion controller.

    Args:
        repo_id: Hugging Face Hub repository ID for GR00T policies.
    """
    # Load policies
    policy_balance, policy_walk = load_groot_policies(repo_id=repo_id)

    # Initialize robot
    config = UnitreeG1Config()
    robot = UnitreeG1(config)

    robot.connect()

    # Initialize gr00T locomotion controller
    groot_controller = GrootLocomotionController(
        policy_balance=policy_balance,
        policy_walk=policy_walk,
        robot=robot,
        config=config,
    )

    try:
        robot.reset(CONTROL_DT, GROOT_DEFAULT_ANGLES)

        logger.info("Use joystick: LY=fwd/back, LX=left/right, RX=rotate, R1=raise waist, R2=lower waist")
        logger.info("Press Ctrl+C to stop")

        # Run step
        while not robot._shutdown_event.is_set():
            start_time = time.time()
            groot_controller.run_step()
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
    parser = argparse.ArgumentParser(description="GR00T Locomotion Controller for Unitree G1")
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_GROOT_REPO_ID,
        help=f"Hugging Face Hub repo ID for GR00T policies (default: {DEFAULT_GROOT_REPO_ID})",
    )
    args = parser.parse_args()

    run(repo_id=args.repo_id)
