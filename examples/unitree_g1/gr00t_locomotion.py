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
Example: GR00T Locomotion with Pre-loaded Policies

This example demonstrates the NEW pattern for loading GR00T policies externally
and passing them to the robot class.
"""

import argparse
import logging
import threading
import time
from collections import deque

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config
from lerobot.robots.unitree_g1.unitree_g1 import UnitreeG1

logger = logging.getLogger(__name__)

GROOT_DEFAULT_ANGLES = np.zeros(29, dtype=np.float32)
GROOT_DEFAULT_ANGLES[[0, 6]] = -0.1  # hip pitch
GROOT_DEFAULT_ANGLES[[3, 9]] = 0.3  # knee
GROOT_DEFAULT_ANGLES[[4, 10]] = -0.2  # ankle pitch

MISSING_JOINTS = []
G1_MODEL = "g1_23"  # or "g1_29"
if G1_MODEL == "g1_23":
    MISSING_JOINTS = [12, 14, 20, 21, 27, 28]  # waist yaw/pitch, wrist pitch/yaw

LOCOMOTION_ACTION_SCALE = 0.25

LOCOMOTION_CONTROL_DT = 0.02

ANG_VEL_SCALE: float = 0.25
DOF_POS_SCALE: float = 1.0
DOF_VEL_SCALE: float = 0.05
CMD_SCALE: list = [2.0, 2.0, 0.25]


DEFAULT_GROOT_REPO_ID = "nepyope/GR00T-WholeBodyControl_g1"


def load_groot_policies(
    repo_id: str = DEFAULT_GROOT_REPO_ID,
) -> tuple[ort.InferenceSession, ort.InferenceSession]:
    """Load GR00T dual-policy system (Balance + Walk) from Hugging Face Hub.

    Args:
        repo_id: Hugging Face Hub repository ID containing the ONNX policies.
    """
    logger.info(f"Loading GR00T dual-policy system from Hugging Face Hub ({repo_id})...")

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
    """
    Handles GR00T-style locomotion control for the Unitree G1 robot.

    This controller manages:
    - Dual-policy system (Balance + Walk)
    - 29-joint observation processing
    - 15D action output (legs + waist)
    - Policy inference and motor command generation
    """

    def __init__(self, policy_balance, policy_walk, robot, config):
        self.policy_balance = policy_balance
        self.policy_walk = policy_walk
        self.robot = robot
        self.config = config

        self.locomotion_cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # vx, vy, theta_dot

        # GR00T-specific state
        self.groot_qj_all = np.zeros(29, dtype=np.float32)
        self.groot_dqj_all = np.zeros(29, dtype=np.float32)
        self.groot_action = np.zeros(15, dtype=np.float32)
        self.groot_obs_single = np.zeros(86, dtype=np.float32)
        self.groot_obs_history = deque(maxlen=6)
        self.groot_obs_stacked = np.zeros(516, dtype=np.float32)
        self.groot_height_cmd = 0.74  # Default base height
        self.groot_orientation_cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # input to gr00t is 6 frames (6*86D=516)
        for _ in range(6):
            self.groot_obs_history.append(np.zeros(86, dtype=np.float32))

        # Thread management
        self.locomotion_running = False
        self.locomotion_thread = None

        logger.info("GrootLocomotionController initialized")

    def groot_locomotion_run(self):
        # get current observation
        robot_state = self.robot.get_observation()

        if robot_state is None:
            return

        # get command from remote controller
        if robot_state.wireless_remote is not None:
            self.robot.remote_controller.set(robot_state.wireless_remote)
            if self.robot.remote_controller.button[0]:  # R1 - raise waist
                self.groot_height_cmd += 0.001
                self.groot_height_cmd = np.clip(self.groot_height_cmd, 0.50, 1.00)
            if self.robot.remote_controller.button[4]:  # R2 - lower waist
                self.groot_height_cmd -= 0.001
                self.groot_height_cmd = np.clip(self.groot_height_cmd, 0.50, 1.00)
        else:
            self.robot.remote_controller.lx = 0.0
            self.robot.remote_controller.ly = 0.0
            self.robot.remote_controller.rx = 0.0
            self.robot.remote_controller.ry = 0.0

        self.locomotion_cmd[0] = self.robot.remote_controller.ly  # forward/backward
        self.locomotion_cmd[1] = self.robot.remote_controller.lx * -1  # left/right
        self.locomotion_cmd[2] = self.robot.remote_controller.rx * -1  # rotation rate

        for i in range(29):
            self.groot_qj_all[i] = robot_state.motor_state[i].q
            self.groot_dqj_all[i] = robot_state.motor_state[i].dq

        # adapt observation for g1_23dof
        for idx in MISSING_JOINTS:
            self.groot_qj_all[idx] = 0.0
            self.groot_dqj_all[idx] = 0.0

        # Scale joint positions and velocities
        qj_obs = self.groot_qj_all.copy()
        dqj_obs = self.groot_dqj_all.copy()

        # express imu data in gravity frame of reference
        quat = robot_state.imu_state.quaternion
        ang_vel = np.array(robot_state.imu_state.gyroscope, dtype=np.float32)
        gravity_orientation = self.robot.get_gravity_orientation(quat)

        # scale joint positions and velocities before policy inference
        qj_obs = (qj_obs - GROOT_DEFAULT_ANGLES) * DOF_POS_SCALE
        dqj_obs = dqj_obs * DOF_VEL_SCALE
        ang_vel_scaled = ang_vel * ANG_VEL_SCALE

        # build single frame observation
        self.groot_obs_single[:3] = self.locomotion_cmd * np.array(CMD_SCALE)
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

        # Run policy inference (ONNX) with 516D stacked observation

        cmd_magnitude = np.linalg.norm(self.locomotion_cmd)

        selected_policy = (
            self.policy_balance if cmd_magnitude < 0.05 else self.policy_walk
        )  # balance/standing policy for small commands, walking policy for movement commands

        # run policy inference
        ort_inputs = {selected_policy.get_inputs()[0].name: np.expand_dims(self.groot_obs_stacked, axis=0)}
        ort_outs = selected_policy.run(None, ort_inputs)
        self.groot_action = ort_outs[0].squeeze()

        # transform action back to target joint positions
        target_dof_pos_15 = GROOT_DEFAULT_ANGLES[:15] + self.groot_action * LOCOMOTION_ACTION_SCALE

        # command motors
        for i in range(15):
            motor_idx = i
            self.robot.msg.motor_cmd[motor_idx].q = target_dof_pos_15[i]
            self.robot.msg.motor_cmd[motor_idx].qd = 0
            self.robot.msg.motor_cmd[motor_idx].kp = self.robot.kp[motor_idx]
            self.robot.msg.motor_cmd[motor_idx].kd = self.robot.kd[motor_idx]
            self.robot.msg.motor_cmd[motor_idx].tau = 0

        # adapt action for g1_23dof
        for joint_idx in MISSING_JOINTS:
            self.robot.msg.motor_cmd[joint_idx].q = 0.0
            self.robot.msg.motor_cmd[joint_idx].qd = 0
            self.robot.msg.motor_cmd[joint_idx].kp = self.robot.kp[joint_idx]
            self.robot.msg.motor_cmd[joint_idx].kd = self.robot.kd[joint_idx]
            self.robot.msg.motor_cmd[joint_idx].tau = 0

        # send action to robot
        self.robot.send_action(self.robot.msg)

    def _locomotion_thread_loop(self):
        """Background thread that runs the locomotion policy at specified rate."""
        logger.info("Locomotion thread started")
        while self.locomotion_running:
            start_time = time.time()
            try:
                self.groot_locomotion_run()
            except Exception as e:
                logger.error(f"Error in locomotion loop: {e}")

            # Sleep to maintain control rate
            elapsed = time.time() - start_time
            sleep_time = max(0, LOCOMOTION_CONTROL_DT - elapsed)
            time.sleep(sleep_time)
        logger.info("Locomotion thread stopped")

    def start_locomotion_thread(self):
        if self.locomotion_running:
            logger.warning("Locomotion thread already running")
            return

        logger.info("Starting locomotion control thread...")
        self.locomotion_running = True
        self.locomotion_thread = threading.Thread(target=self._locomotion_thread_loop, daemon=True)
        self.locomotion_thread.start()

        logger.info("Locomotion control thread started!")

    def stop_locomotion_thread(self):
        if not self.locomotion_running:
            return

        logger.info("Stopping locomotion control thread...")
        self.locomotion_running = False
        if self.locomotion_thread:
            self.locomotion_thread.join(timeout=2.0)
        logger.info("Locomotion control thread stopped")

    def reset_robot(self):
        """Move robot legs to default standing position over 2 seconds (arms are not moved)."""
        total_time = 3.0
        num_step = int(total_time / self.robot.control_dt)

        # Only control legs, not arms (first 12 joints)
        default_pos = GROOT_DEFAULT_ANGLES  # First 12 values are leg angles
        dof_size = len(default_pos)

        # Get current lowstate
        robot_state = self.robot.get_observation()

        # Record the current leg positions
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = robot_state.motor_state[i].q

        # Move legs to default pos
        for i in range(num_step):
            alpha = i / num_step
            for motor_idx in range(dof_size):
                target_pos = default_pos[motor_idx]
                self.robot.msg.motor_cmd[motor_idx].q = (
                    init_dof_pos[motor_idx] * (1 - alpha) + target_pos * alpha
                )
                self.robot.msg.motor_cmd[motor_idx].qd = 0
                self.robot.msg.motor_cmd[motor_idx].kp = self.robot.kp[motor_idx]
                self.robot.msg.motor_cmd[motor_idx].kd = self.robot.kd[motor_idx]
                self.robot.msg.motor_cmd[motor_idx].tau = 0
            self.robot.msg.crc = self.robot.crc.Crc(self.robot.msg)
            self.robot.lowcmd_publisher.Write(self.robot.msg)
            time.sleep(self.robot.control_dt)
        logger.info("Reached default position (legs only)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GR00T Locomotion Controller for Unitree G1")
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_GROOT_REPO_ID,
        help=f"Hugging Face Hub repo ID for GR00T policies (default: {DEFAULT_GROOT_REPO_ID})",
    )
    args = parser.parse_args()

    # load policies
    policy_balance, policy_walk = load_groot_policies(repo_id=args.repo_id)

    # initialize robot
    config = UnitreeG1Config()
    robot = UnitreeG1(config)

    # initialize gr00t locomotion controller
    groot_controller = GrootLocomotionController(
        policy_balance=policy_balance,
        policy_walk=policy_walk,
        robot=robot,
        config=config,
    )

    # reset legs and start locomotion thread
    try:
        groot_controller.reset_robot()
        groot_controller.start_locomotion_thread()

        # log status
        logger.info("Robot initialized with GR00T locomotion policies")
        logger.info("Locomotion controller running in background thread")
        logger.info("Press Ctrl+C to stop")

        # keep robot alive
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping locomotion...")
        groot_controller.stop_locomotion_thread()
        print("Done!")
