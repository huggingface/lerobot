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
Example: Unitree RL 12-DOF Legs-Only Locomotion (TorchScript)

This example demonstrates loading a 12-DOF legs-only locomotion policy
(TorchScript .pt format) and running it on the Unitree G1 robot.

Key characteristics:
- Single TorchScript policy (.pt)
- 47D observations, 12D actions (legs only)
- Phase-based gait timing
- Arms and waist held at fixed positions
"""

import argparse
import logging
import threading
import time

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from scipy.spatial.transform import Rotation as R

from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config
from lerobot.robots.unitree_g1.unitree_g1 import UnitreeG1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 12-DOF leg joint configuration
# Joint order: [L_hip_pitch, L_hip_roll, L_hip_yaw, L_knee, L_ankle_pitch, L_ankle_roll,
#               R_hip_pitch, R_hip_roll, R_hip_yaw, R_knee, R_ankle_pitch, R_ankle_roll]
LEG_JOINT_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# Default leg angles for standing
DEFAULT_LEG_ANGLES = np.array([
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,   # left leg
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,   # right leg
], dtype=np.float32)

# KP/KD for leg joints
LEG_KPS = np.array([150, 150, 150, 300, 40, 40, 150, 150, 150, 300, 40, 40], dtype=np.float32)
LEG_KDS = np.array([6, 6, 6, 4, 2, 2, 6, 6, 6, 4, 2, 2], dtype=np.float32)

# Waist configuration (held at zero)
WAIST_JOINT_INDICES = [12, 13, 14]  # yaw, roll, pitch
WAIST_KPS = np.array([250, 250, 250], dtype=np.float32)
WAIST_KDS = np.array([5, 5, 5], dtype=np.float32)

# Arm configuration (indices 15-28, held at initial position)
ARM_JOINT_INDICES = list(range(15, 29))
ARM_KPS = np.array([80, 80, 80, 80, 40, 40, 40,   # left arm (shoulder + wrist)
                   80, 80, 80, 80, 40, 40, 40], dtype=np.float32)  # right arm
ARM_KDS = np.array([3, 3, 3, 3, 1.5, 1.5, 1.5,
                   3, 3, 3, 3, 1.5, 1.5, 1.5], dtype=np.float32)

# Control parameters
LOCOMOTION_CONTROL_DT = 0.02  # 50Hz control rate
LOCOMOTION_ACTION_SCALE = 0.25
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)
MAX_CMD = np.array([0.8, 0.5, 1.57], dtype=np.float32)  # max vx, vy, yaw_rate

# Gait parameters
GAIT_PERIOD = 0.8  # seconds

DEFAULT_REPO_ID = "nepyope/unitree_rl_locomotion"


def load_torchscript_policy(
    repo_id: str = DEFAULT_REPO_ID,
    filename: str = "motion.pt",
) -> torch.jit.ScriptModule:
    """Load TorchScript locomotion policy from Hugging Face Hub.

    Args:
        repo_id: Hugging Face Hub repository ID containing the policy.
        filename: Policy filename (default: motion.pt).
    """
    logger.info(f"Loading TorchScript policy from Hugging Face Hub ({repo_id}/{filename})...")

    policy_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
    )

    policy = torch.jit.load(policy_path)
    policy.eval()

    logger.info("TorchScript policy loaded successfully")

    return policy


class UnitreeRLLocomotionController:
    """
    Handles 12-DOF legs-only locomotion control for the Unitree G1 robot.

    This controller manages:
    - Single TorchScript policy
    - 47D observations (single frame)
    - 12D action output (legs only)
    - Arms and waist held at fixed positions
    - Phase-based gait timing
    """

    def __init__(self, policy, robot, config):
        self.policy = policy
        self.robot = robot
        self.config = config

        # Velocity commands (vx, vy, yaw_rate)
        self.locomotion_cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # State variables (12 DOF legs)
        self.qj = np.zeros(12, dtype=np.float32)
        self.dqj = np.zeros(12, dtype=np.float32)
        self.locomotion_action = np.zeros(12, dtype=np.float32)
        self.locomotion_obs = np.zeros(47, dtype=np.float32)

        # Initial arm positions (captured on reset)
        self.initial_arm_positions = np.zeros(14, dtype=np.float32)

        # Counter for phase calculation
        self.counter = 0

        # Thread management
        self.locomotion_running = False
        self.locomotion_thread = None

        logger.info("UnitreeRLLocomotionController initialized")
        logger.info("  Observation dim: 47, Action dim: 12 (legs only)")

    def locomotion_run(self):
        """12-DOF legs-only locomotion policy loop."""
        self.counter += 1

        if self.counter == 1:
            print("\n" + "=" * 60)
            print("🚀 RUNNING UNITREE RL 12-DOF LOCOMOTION POLICY")
            print("   47D observations → 12D actions (legs only)")
            print("   Arms and waist held at fixed positions")
            print("=" * 60 + "\n")

        # Get current observation
        robot_state = self.robot.get_observation()
        if robot_state is None:
            return

        # Get command from remote controller
        if robot_state.wireless_remote is not None:
            self.robot.remote_controller.set(robot_state.wireless_remote)
        else:
            self.robot.remote_controller.lx = 0.0
            self.robot.remote_controller.ly = 0.0
            self.robot.remote_controller.rx = 0.0
            self.robot.remote_controller.ry = 0.0

        self.locomotion_cmd[0] = self.robot.remote_controller.ly       # forward/backward
        self.locomotion_cmd[1] = self.robot.remote_controller.lx * -1  # left/right (inverted)
        self.locomotion_cmd[2] = self.robot.remote_controller.rx * -1  # yaw (inverted)

        # Get leg joint positions and velocities (12 DOF)
        for i, motor_idx in enumerate(LEG_JOINT_INDICES):
            self.qj[i] = robot_state.motor_state[motor_idx].q
            self.dqj[i] = robot_state.motor_state[motor_idx].dq

        # Get IMU data
        quat = robot_state.imu_state.quaternion
        ang_vel = np.array(robot_state.imu_state.gyroscope, dtype=np.float32)

        # Scale observations
        gravity_orientation = self.robot.get_gravity_orientation(quat)
        qj_obs = (self.qj - DEFAULT_LEG_ANGLES) * DOF_POS_SCALE
        dqj_obs = self.dqj * DOF_VEL_SCALE
        ang_vel_scaled = ang_vel * ANG_VEL_SCALE

        # Calculate phase
        count = self.counter * LOCOMOTION_CONTROL_DT
        phase = (count % GAIT_PERIOD) / GAIT_PERIOD
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)

        # Build 47D observation vector
        # [0:3]   - angular velocity (scaled)
        # [3:6]   - gravity orientation
        # [6:9]   - velocity command (scaled)
        # [9:21]  - joint positions (12D, relative to default)
        # [21:33] - joint velocities (12D, scaled)
        # [33:45] - previous actions (12D)
        # [45]    - sin_phase
        # [46]    - cos_phase
        self.locomotion_obs[0:3] = ang_vel_scaled
        self.locomotion_obs[3:6] = gravity_orientation
        self.locomotion_obs[6:9] = self.locomotion_cmd * CMD_SCALE * MAX_CMD
        self.locomotion_obs[9:21] = qj_obs
        self.locomotion_obs[21:33] = dqj_obs
        self.locomotion_obs[33:45] = self.locomotion_action
        self.locomotion_obs[45] = sin_phase
        self.locomotion_obs[46] = cos_phase

        # Run policy inference (TorchScript)
        obs_tensor = torch.from_numpy(self.locomotion_obs).unsqueeze(0).float()
        with torch.no_grad():
            action_tensor = self.policy(obs_tensor)
        self.locomotion_action = action_tensor.squeeze().numpy()

        # Transform action to target joint positions
        target_leg_pos = DEFAULT_LEG_ANGLES + self.locomotion_action * LOCOMOTION_ACTION_SCALE

        # Debug logging (first 3 iterations)
        if self.counter <= 3:
            print(f"\n[Unitree RL Debug #{self.counter}]")
            print(f"  Phase: {phase:.3f} (sin={sin_phase:.3f}, cos={cos_phase:.3f})")
            print(f"  Cmd (vx, vy, yaw): ({self.locomotion_cmd[0]:.2f}, {self.locomotion_cmd[1]:.2f}, {self.locomotion_cmd[2]:.2f})")
            print(f"  Action range: [{self.locomotion_action.min():.3f}, {self.locomotion_action.max():.3f}]")

        # Send commands to LEG motors (0-11)
        for i, motor_idx in enumerate(LEG_JOINT_INDICES):
            self.robot.msg.motor_cmd[motor_idx].q = target_leg_pos[i]
            self.robot.msg.motor_cmd[motor_idx].qd = 0
            self.robot.msg.motor_cmd[motor_idx].kp = LEG_KPS[i]
            self.robot.msg.motor_cmd[motor_idx].kd = LEG_KDS[i]
            self.robot.msg.motor_cmd[motor_idx].tau = 0

        # Hold WAIST motors at zero (12, 13, 14)
        for i, motor_idx in enumerate(WAIST_JOINT_INDICES):
            self.robot.msg.motor_cmd[motor_idx].q = 0.0
            self.robot.msg.motor_cmd[motor_idx].qd = 0
            self.robot.msg.motor_cmd[motor_idx].kp = WAIST_KPS[i]
            self.robot.msg.motor_cmd[motor_idx].kd = WAIST_KDS[i]
            self.robot.msg.motor_cmd[motor_idx].tau = 0

        # Hold ARM motors at initial position (15-28)
        for i, motor_idx in enumerate(ARM_JOINT_INDICES):
            self.robot.msg.motor_cmd[motor_idx].q = self.initial_arm_positions[i]
            self.robot.msg.motor_cmd[motor_idx].qd = 0
            self.robot.msg.motor_cmd[motor_idx].kp = ARM_KPS[i]
            self.robot.msg.motor_cmd[motor_idx].kd = ARM_KDS[i]
            self.robot.msg.motor_cmd[motor_idx].tau = 0

        # Send command
        self.robot.send_action(self.robot.msg)

    def _locomotion_thread_loop(self):
        """Background thread that runs the locomotion policy at specified rate."""
        logger.info("Locomotion thread started")
        while self.locomotion_running:
            start_time = time.time()
            try:
                self.locomotion_run()
            except Exception as e:
                logger.error(f"Error in locomotion loop: {e}")
                import traceback
                traceback.print_exc()

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
        """Move legs to default standing position over 2 seconds (arms are captured and held)."""
        logger.info("Moving legs to default position...")

        total_time = 2.0
        num_step = int(total_time / self.robot.control_dt)

        # Get current state
        robot_state = self.robot.get_observation()

        # Capture initial arm positions (to hold during locomotion)
        for i, motor_idx in enumerate(ARM_JOINT_INDICES):
            self.initial_arm_positions[i] = robot_state.motor_state[motor_idx].q
        logger.info(f"Captured initial arm positions: {self.initial_arm_positions[:4]}...")

        # Record current leg positions
        init_leg_pos = np.zeros(12, dtype=np.float32)
        for i, motor_idx in enumerate(LEG_JOINT_INDICES):
            init_leg_pos[i] = robot_state.motor_state[motor_idx].q

        # Interpolate legs to default position
        for step in range(num_step):
            alpha = step / num_step

            # Interpolate leg positions
            for i, motor_idx in enumerate(LEG_JOINT_INDICES):
                target_pos = DEFAULT_LEG_ANGLES[i]
                self.robot.msg.motor_cmd[motor_idx].q = (
                    init_leg_pos[i] * (1 - alpha) + target_pos * alpha
                )
                self.robot.msg.motor_cmd[motor_idx].qd = 0
                self.robot.msg.motor_cmd[motor_idx].kp = LEG_KPS[i]
                self.robot.msg.motor_cmd[motor_idx].kd = LEG_KDS[i]
                self.robot.msg.motor_cmd[motor_idx].tau = 0

            # Hold waist at zero
            for i, motor_idx in enumerate(WAIST_JOINT_INDICES):
                self.robot.msg.motor_cmd[motor_idx].q = 0.0
                self.robot.msg.motor_cmd[motor_idx].qd = 0
                self.robot.msg.motor_cmd[motor_idx].kp = WAIST_KPS[i]
                self.robot.msg.motor_cmd[motor_idx].kd = WAIST_KDS[i]
                self.robot.msg.motor_cmd[motor_idx].tau = 0

            # Hold arms at initial position
            for i, motor_idx in enumerate(ARM_JOINT_INDICES):
                self.robot.msg.motor_cmd[motor_idx].q = self.initial_arm_positions[i]
                self.robot.msg.motor_cmd[motor_idx].qd = 0
                self.robot.msg.motor_cmd[motor_idx].kp = ARM_KPS[i]
                self.robot.msg.motor_cmd[motor_idx].kd = ARM_KDS[i]
                self.robot.msg.motor_cmd[motor_idx].tau = 0

            self.robot.msg.crc = self.robot.crc.Crc(self.robot.msg)
            self.robot.lowcmd_publisher.Write(self.robot.msg)
            time.sleep(self.robot.control_dt)

        logger.info("Reached default leg position")

        # Hold position for 2 seconds
        logger.info("Holding default position for 2 seconds...")
        hold_time = 2.0
        num_hold_steps = int(hold_time / self.robot.control_dt)

        for _ in range(num_hold_steps):
            # Hold legs at default
            for i, motor_idx in enumerate(LEG_JOINT_INDICES):
                self.robot.msg.motor_cmd[motor_idx].q = DEFAULT_LEG_ANGLES[i]
                self.robot.msg.motor_cmd[motor_idx].qd = 0
                self.robot.msg.motor_cmd[motor_idx].kp = LEG_KPS[i]
                self.robot.msg.motor_cmd[motor_idx].kd = LEG_KDS[i]
                self.robot.msg.motor_cmd[motor_idx].tau = 0

            # Hold waist at zero
            for i, motor_idx in enumerate(WAIST_JOINT_INDICES):
                self.robot.msg.motor_cmd[motor_idx].q = 0.0
                self.robot.msg.motor_cmd[motor_idx].qd = 0
                self.robot.msg.motor_cmd[motor_idx].kp = WAIST_KPS[i]
                self.robot.msg.motor_cmd[motor_idx].kd = WAIST_KDS[i]
                self.robot.msg.motor_cmd[motor_idx].tau = 0

            # Hold arms at initial position
            for i, motor_idx in enumerate(ARM_JOINT_INDICES):
                self.robot.msg.motor_cmd[motor_idx].q = self.initial_arm_positions[i]
                self.robot.msg.motor_cmd[motor_idx].qd = 0
                self.robot.msg.motor_cmd[motor_idx].kp = ARM_KPS[i]
                self.robot.msg.motor_cmd[motor_idx].kd = ARM_KDS[i]
                self.robot.msg.motor_cmd[motor_idx].tau = 0

            self.robot.msg.crc = self.robot.crc.Crc(self.robot.msg)
            self.robot.lowcmd_publisher.Write(self.robot.msg)
            time.sleep(self.robot.control_dt)

        logger.info("Ready to start locomotion!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unitree RL 12-DOF Locomotion Controller for Unitree G1")
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face Hub repo ID for policy (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="motion.pt",
        help="Policy filename (default: motion.pt)",
    )
    args = parser.parse_args()

    # Load policy
    policy = load_torchscript_policy(repo_id=args.repo_id, filename=args.filename)

    # Initialize robot
    config = UnitreeG1Config()
    robot = UnitreeG1(config)

    # Initialize locomotion controller
    locomotion_controller = UnitreeRLLocomotionController(
        policy=policy,
        robot=robot,
        config=config,
    )

    # Reset robot and start locomotion thread
    try:
        locomotion_controller.reset_robot()
        locomotion_controller.start_locomotion_thread()

        # Log status
        logger.info("Robot initialized with Unitree RL locomotion policy")
        logger.info("Locomotion controller running in background thread")
        logger.info("Use remote controller to command velocity:")
        logger.info("  Left stick Y: forward/backward")
        logger.info("  Left stick X: left/right")
        logger.info("  Right stick X: rotate")
        logger.info("Press Ctrl+C to stop")

        # Keep robot alive
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping locomotion...")
        locomotion_controller.stop_locomotion_thread()
        print("Done!")

