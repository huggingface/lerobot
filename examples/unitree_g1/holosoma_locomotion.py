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
Example: Holosoma 29-DOF Whole-Body Locomotion

This example demonstrates loading Amazon/Holosoma 29-DOF whole-body locomotion
policies and running them on the Unitree G1 robot.

Key differences from GR00T:
- Single policy (not dual Balance/Walk)
- 100D observations, 29D actions (all joints)
- Phase-based gait with standing/walking modes
"""

import argparse
import logging
import threading
import time

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from scipy.spatial.transform import Rotation as R

from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config
from lerobot.robots.unitree_g1.unitree_g1 import UnitreeG1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default joint angles from holosoma (29 DOF)
# fmt: off
HOLOSOMA_DEFAULT_ANGLES = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # left leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # right leg
    0.0, 0.0, 0.0,                          # waist (yaw, roll, pitch)
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,     # left arm (shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw)
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,    # right arm
], dtype=np.float32)

# KP/KD values from holosoma (tuned for G1 hardware)
HOLOSOMA_KP = np.array([
    40.179238471,   # left_hip_pitch
    99.098427777,   # left_hip_roll
    40.179238471,   # left_hip_yaw
    99.098427777,   # left_knee
    28.501246196,   # left_ankle_pitch
    28.501246196,   # left_ankle_roll
    40.179238471,   # right_hip_pitch
    99.098427777,   # right_hip_roll
    40.179238471,   # right_hip_yaw
    99.098427777,   # right_knee
    28.501246196,   # right_ankle_pitch
    28.501246196,   # right_ankle_roll
    40.179238471,   # waist_yaw
    28.501246196,   # waist_roll
    28.501246196,   # waist_pitch
    14.250623098,   # left_shoulder_pitch
    14.250623098,   # left_shoulder_roll
    14.250623098,   # left_shoulder_yaw
    14.250623098,   # left_elbow
    14.250623098,   # left_wrist_roll
    16.778327481,   # left_wrist_pitch
    16.778327481,   # left_wrist_yaw
    14.250623098,   # right_shoulder_pitch
    14.250623098,   # right_shoulder_roll
    14.250623098,   # right_shoulder_yaw
    14.250623098,   # right_elbow
    14.250623098,   # right_wrist_roll
    16.778327481,   # right_wrist_pitch
    16.778327481,   # right_wrist_yaw
], dtype=np.float32)

HOLOSOMA_KD = np.array([
    2.557889765,    # left_hip_pitch
    6.308801854,    # left_hip_roll
    2.557889765,    # left_hip_yaw
    6.308801854,    # left_knee
    1.814445687,    # left_ankle_pitch
    1.814445687,    # left_ankle_roll
    2.557889765,    # right_hip_pitch
    6.308801854,    # right_hip_roll
    2.557889765,    # right_hip_yaw
    6.308801854,    # right_knee
    1.814445687,    # right_ankle_pitch
    1.814445687,    # right_ankle_roll
    2.557889765,    # waist_yaw
    1.814445687,    # waist_roll
    1.814445687,    # waist_pitch
    0.907222843,    # left_shoulder_pitch
    0.907222843,    # left_shoulder_roll
    0.907222843,    # left_shoulder_yaw
    0.907222843,    # left_elbow
    0.907222843,    # left_wrist_roll
    1.068141502,    # left_wrist_pitch
    1.068141502,    # left_wrist_yaw
    0.907222843,    # right_shoulder_pitch
    0.907222843,    # right_shoulder_roll
    0.907222843,    # right_shoulder_yaw
    0.907222843,    # right_elbow
    0.907222843,    # right_wrist_roll
    1.068141502,    # right_wrist_pitch
    1.068141502,    # right_wrist_yaw
], dtype=np.float32)
# fmt: on

# G1 model configuration
G1_MODEL = "g1_23"  # or "g1_29"
MISSING_JOINTS = []
if G1_MODEL == "g1_23":
    # Joints that G1 23-DOF doesn't have (freeze these)
    # 12: waist_yaw, 14: waist_pitch
    # 20: left_wrist_pitch, 21: left_wrist_yaw
    # 27: right_wrist_pitch, 28: right_wrist_yaw
    MISSING_JOINTS = [12, 14, 20, 21, 27, 28]

# Control parameters
LOCOMOTION_CONTROL_DT = 0.02  # 50Hz control rate
LOCOMOTION_ACTION_SCALE = 0.25
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05

# Gait parameters
GAIT_PERIOD = 1.0  # seconds

DEFAULT_HOLOSOMA_REPO_ID = "nepyope/holosoma_locomotion"


def load_holosoma_policy(
    repo_id: str = DEFAULT_HOLOSOMA_REPO_ID,
    policy_name: str = "fastsac",
    local_path: str | None = None
) -> ort.InferenceSession:
    """Load Holosoma 29-DOF locomotion policy from Hugging Face Hub.

    Args:
        repo_id: Hugging Face Hub repository ID containing the ONNX policies.
        policy_name: Policy variant to load ("fastsac" or "ppo").
    """
    if local_path is not None:
        logger.info(f"Loading policy from local path: {local_path}")
        policy_path = local_path
    filename_map = {
        "fastsac": "fastsac_g1_29dof.onnx",
        "ppo": "ppo_g1_29dof.onnx",
    }

    if policy_name not in filename_map:
        raise ValueError(f"Unknown policy_name: {policy_name}. Must be one of {list(filename_map.keys())}")

    logger.info(f"Loading Holosoma {policy_name} policy from Hugging Face Hub ({repo_id})...")

    policy_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename_map[policy_name],
    )

    policy = ort.InferenceSession(policy_path)

    logger.info(f"Holosoma {policy_name} policy loaded successfully")
    logger.info(f"  Input: {policy.get_inputs()[0].name}, shape: {policy.get_inputs()[0].shape}")
    logger.info(f"  Output: {policy.get_outputs()[0].name}, shape: {policy.get_outputs()[0].shape}")

    return policy


class HolosomaLocomotionController:
    """
    Handles Holosoma-style 29-DOF whole-body locomotion control for the Unitree G1 robot.

    This controller manages:
    - Single ONNX policy (FastSAC or PPO)
    - 100D observations (single frame)
    - 29D action output (all joints: legs + waist + arms)
    - Phase-based gait with standing/walking modes
    """

    def __init__(self, policy, robot, config):
        self.policy = policy
        self.robot = robot
        self.config = config

        # Velocity commands (vx, vy, yaw_rate)
        self.locomotion_cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # State variables (29 DOF)
        self.qj = np.zeros(29, dtype=np.float32)
        self.dqj = np.zeros(29, dtype=np.float32)
        self.locomotion_action = np.zeros(29, dtype=np.float32)
        self.locomotion_obs = np.zeros(100, dtype=np.float32)

        # Phase state for gait (2D: left foot, right foot)
        self.phase = np.zeros((1, 2), dtype=np.float32)
        self.phase[0, 0] = 0.0      # left foot starts at 0
        self.phase[0, 1] = np.pi    # right foot starts at π (opposite phase)
        self.phase_dt = 2 * np.pi / (50.0 * GAIT_PERIOD)  # 50Hz control rate
        self.is_standing = False

        # Store last unscaled action for observation (policy expects previous actions)
        self.last_unscaled_action = np.zeros(29, dtype=np.float32)

        # Counter for logging
        self.counter = 0

        # Thread management
        self.locomotion_running = False
        self.locomotion_thread = None

        logger.info("HolosomaLocomotionController initialized")
        logger.info(f"  Observation dim: 100, Action dim: 29")
        logger.info(f"  Missing joints (G1 23-DOF): {MISSING_JOINTS}")

    def holosoma_locomotion_run(self):
        """29-DOF whole-body locomotion policy loop - controls ALL 29 joints."""
        self.counter += 1

        if self.counter == 1:
            print("\n" + "=" * 60)
            print("🚀 RUNNING HOLOSOMA 29-DOF LOCOMOTION POLICY")
            print("   100D observations → 29D actions (all joints)")
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

        # Apply deadzone (0.1) like holosoma does
        ly = self.robot.remote_controller.ly if abs(self.robot.remote_controller.ly) > 0.1 else 0.0
        lx = self.robot.remote_controller.lx if abs(self.robot.remote_controller.lx) > 0.1 else 0.0
        rx = self.robot.remote_controller.rx if abs(self.robot.remote_controller.rx) > 0.1 else 0.0

        self.locomotion_cmd[0] = ly       # forward/backward
        self.locomotion_cmd[1] = -lx      # left/right (inverted)
        self.locomotion_cmd[2] = -rx      # yaw (inverted)

        # Get ALL 29 joint positions and velocities
        for i in range(29):
            self.qj[i] = robot_state.motor_state[i].q
            self.dqj[i] = robot_state.motor_state[i].dq

        # Get IMU data
        quat = robot_state.imu_state.quaternion
        ang_vel = np.array(robot_state.imu_state.gyroscope, dtype=np.float32)

        # Transform IMU from torso to pelvis frame (if using torso IMU)
        #waist_yaw = robot_state.motor_state[12].q
        #waist_yaw_omega = robot_state.motor_state[12].dq
        #quat, ang_vel = self._transform_imu_data(waist_yaw, waist_yaw_omega, quat, ang_vel)

        # Zero out observations for joints missing in G1 23-DOF
        for joint_idx in MISSING_JOINTS:
            self.qj[joint_idx] = 0.0
            self.dqj[joint_idx] = 0.0

        # Create observation with correct scaling factors
        gravity_orientation = self.robot.get_gravity_orientation(quat)
        qj_obs = (self.qj - HOLOSOMA_DEFAULT_ANGLES) * DOF_POS_SCALE
        dqj_obs = self.dqj * DOF_VEL_SCALE
        ang_vel_scaled = ang_vel * ANG_VEL_SCALE

        # Update phase using holosoma's method
        cmd_norm = np.linalg.norm(self.locomotion_cmd[:2])
        ang_cmd_norm = np.abs(self.locomotion_cmd[2])

        if cmd_norm < 0.01 and ang_cmd_norm < 0.01:
            # Standing still - both feet at π
            self.phase[0, :] = np.pi * np.ones(2)
            self.is_standing = True
        elif self.is_standing:
            # Resuming walking from standing - reset phase to initial state
            self.phase = np.array([[0.0, np.pi]], dtype=np.float32)
            self.is_standing = False
        else:
            # Walking - update phase
            phase_tp1 = self.phase + self.phase_dt
            self.phase = np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi

        # Compute sin/cos phase for both feet
        sin_phase = np.sin(self.phase[0, :])  # shape (2,)
        cos_phase = np.cos(self.phase[0, :])  # shape (2,)

        # Build 100D observation vector (components in ALPHABETICAL order!)
        # Joints within each 29D component stay in motor index order (0-28)
        self.locomotion_obs[0:29] = self.last_unscaled_action   # 1. actions (previous UNSCALED, ×1.0)
        self.locomotion_obs[29:32] = ang_vel_scaled              # 2. base_ang_vel (×0.25)
        self.locomotion_obs[32] = self.locomotion_cmd[2]         # 3. command_ang_vel (yaw, ×1.0)
        self.locomotion_obs[33:35] = self.locomotion_cmd[:2]     # 4. command_lin_vel (vx, vy, ×1.0)
        self.locomotion_obs[35:37] = cos_phase                   # 5. cos_phase (2D: left, right)
        self.locomotion_obs[37:66] = qj_obs                      # 6. dof_pos (relative, ×1.0)
        self.locomotion_obs[66:95] = dqj_obs                     # 7. dof_vel (×0.05)
        self.locomotion_obs[95:98] = gravity_orientation         # 8. projected_gravity (×1.0)
        self.locomotion_obs[98:100] = sin_phase                  # 9. sin_phase (2D: left, right)

        # Run policy inference (ONNX)
        obs_input = self.locomotion_obs.reshape(1, -1).astype(np.float32)
        ort_inputs = {self.policy.get_inputs()[0].name: obs_input}
        ort_outs = self.policy.run(None, ort_inputs)

        # Post-process ONNX output: clip to ±100, then scale by 0.25
        raw_action = ort_outs[0].squeeze()
        clipped_action = np.clip(raw_action, -100.0, 100.0)

        # Zero out actions for joints missing in G1 23-DOF
        for joint_idx in MISSING_JOINTS:
            clipped_action[joint_idx] = 0.0

        self.last_unscaled_action = clipped_action.copy()  # Store UNSCALED for next obs
        self.locomotion_action = clipped_action * LOCOMOTION_ACTION_SCALE  # Scale by 0.25 for motors

        # Debug logging (first 3 iterations)
        if self.counter <= 3:
            print(f"\n[Holosoma Debug #{self.counter}]")
            print(f"  Phase (left, right): ({self.phase[0, 0]:.3f}, {self.phase[0, 1]:.3f})")
            print(f"  Cmd (vx, vy, yaw): ({self.locomotion_cmd[0]:.2f}, {self.locomotion_cmd[1]:.2f}, {self.locomotion_cmd[2]:.2f})")
            print(f"  Raw action range: [{raw_action.min():.3f}, {raw_action.max():.3f}]")
            print(f"  Scaled action range: [{self.locomotion_action.min():.3f}, {self.locomotion_action.max():.3f}]")

        # Transform action to target joint positions (ALL 29 joints)
        target_dof_pos = HOLOSOMA_DEFAULT_ANGLES + self.locomotion_action

        # Send commands to ALL 29 motors
        for i in range(29):
            self.robot.msg.motor_cmd[i].q = target_dof_pos[i]
            self.robot.msg.motor_cmd[i].qd = 0
            self.robot.msg.motor_cmd[i].kp = HOLOSOMA_KP[i]
            self.robot.msg.motor_cmd[i].kd = HOLOSOMA_KD[i]
            self.robot.msg.motor_cmd[i].tau = 0

        # Send command
        self.robot.send_action(self.robot.msg)

    def _locomotion_thread_loop(self):
        """Background thread that runs the locomotion policy at specified rate."""
        logger.info("Locomotion thread started")
        while self.locomotion_running:
            start_time = time.time()
            try:
                self.holosoma_locomotion_run()
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
        """Move all 29 joints to default standing position over 3 seconds."""
        logger.info("Moving all 29 joints to default position...")

        total_time = 3.0
        num_step = int(total_time / self.robot.control_dt)
        default_pos = HOLOSOMA_DEFAULT_ANGLES

        # Get current state
        robot_state = self.robot.get_observation()

        # Record current positions
        init_dof_pos = np.zeros(29, dtype=np.float32)
        for i in range(29):
            init_dof_pos[i] = robot_state.motor_state[i].q

        # Interpolate to target
        for i in range(num_step):
            alpha = i / num_step
            for motor_idx in range(29):
                target_pos = default_pos[motor_idx]
                self.robot.msg.motor_cmd[motor_idx].q = (
                    init_dof_pos[motor_idx] * (1 - alpha) + target_pos * alpha
                )
                self.robot.msg.motor_cmd[motor_idx].qd = 0
                self.robot.msg.motor_cmd[motor_idx].kp = HOLOSOMA_KP[motor_idx]
                self.robot.msg.motor_cmd[motor_idx].kd = HOLOSOMA_KD[motor_idx]
                self.robot.msg.motor_cmd[motor_idx].tau = 0

            self.robot.msg.crc = self.robot.crc.Crc(self.robot.msg)
            self.robot.lowcmd_publisher.Write(self.robot.msg)
            time.sleep(self.robot.control_dt)

        logger.info("Reached default position (all 29 joints)")

        # Hold position for 2 seconds
        logger.info("Holding default position for 2 seconds...")
        hold_time = 2.0
        num_hold_steps = int(hold_time / self.robot.control_dt)

        for _ in range(num_hold_steps):
            for motor_idx in range(29):
                self.robot.msg.motor_cmd[motor_idx].q = default_pos[motor_idx]
                self.robot.msg.motor_cmd[motor_idx].qd = 0
                self.robot.msg.motor_cmd[motor_idx].kp = HOLOSOMA_KP[motor_idx]
                self.robot.msg.motor_cmd[motor_idx].kd = HOLOSOMA_KD[motor_idx]
                self.robot.msg.motor_cmd[motor_idx].tau = 0

            self.robot.msg.crc = self.robot.crc.Crc(self.robot.msg)
            self.robot.lowcmd_publisher.Write(self.robot.msg)
            time.sleep(self.robot.control_dt)

        logger.info("Ready to start locomotion!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Holosoma 29-DOF Locomotion Controller for Unitree G1")
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_HOLOSOMA_REPO_ID,
        help=f"Hugging Face Hub repo ID for Holosoma policies (default: {DEFAULT_HOLOSOMA_REPO_ID})",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="fastsac",
        choices=["fastsac", "ppo"],
        help="Policy variant to load (default: fastsac)",
    )
    parser.add_argument(
    "--local-path",
    type=str,
    default=None,
    help="Path to local ONNX file (overrides --repo-id and --policy)",
)
    args = parser.parse_args()

    # Load policy
    policy = load_holosoma_policy(repo_id=args.repo_id, policy_name=args.policy)

    # Initialize robot
    config = UnitreeG1Config()
    robot = UnitreeG1(config)

    # Initialize holosoma locomotion controller
    holosoma_controller = HolosomaLocomotionController(
        policy=policy,
        robot=robot,
        config=config,
    )

    # Reset robot and start locomotion thread
    try:
        holosoma_controller.reset_robot()
        holosoma_controller.start_locomotion_thread()

        # Log status
        logger.info(f"Robot initialized with Holosoma {args.policy} locomotion policy")
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
        holosoma_controller.stop_locomotion_thread()
        print("Done!")

