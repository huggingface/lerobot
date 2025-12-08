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
Example: Holosoma Whole-Body Locomotion (23-DOF and 29-DOF)

This example demonstrates loading Holosoma whole-body locomotion policies
and running them on the Unitree G1 robot.

Supports both:
- 23-DOF native policies (82D observations, 23D actions)
- 29-DOF policies (100D observations, 29D actions)
"""

import argparse
import logging
import threading
import time

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config
from lerobot.robots.unitree_g1.unitree_g1 import UnitreeG1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# 29-DOF Configuration
# =============================================================================
# fmt: off
HOLOSOMA_29DOF_DEFAULT_ANGLES = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # left leg
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # right leg
    0.0, 0.0, 0.0,                          # waist (yaw, roll, pitch)
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,     # left arm
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,    # right arm
], dtype=np.float32)

HOLOSOMA_29DOF_KP = np.array([
    40.179238471, 99.098427777, 40.179238471, 99.098427777, 28.501246196, 28.501246196,  # left leg
    40.179238471, 99.098427777, 40.179238471, 99.098427777, 28.501246196, 28.501246196,  # right leg
    40.179238471, 28.501246196, 28.501246196,  # waist
    14.250623098, 14.250623098, 14.250623098, 14.250623098, 14.250623098, 16.778327481, 16.778327481,  # left arm
    14.250623098, 14.250623098, 14.250623098, 14.250623098, 14.250623098, 16.778327481, 16.778327481,  # right arm
], dtype=np.float32)

HOLOSOMA_29DOF_KD = np.array([
    2.557889765, 6.308801854, 2.557889765, 6.308801854, 1.814445687, 1.814445687,  # left leg
    2.557889765, 6.308801854, 2.557889765, 6.308801854, 1.814445687, 1.814445687,  # right leg
    2.557889765, 1.814445687, 1.814445687,  # waist
    0.907222843, 0.907222843, 0.907222843, 0.907222843, 0.907222843, 1.068141502, 1.068141502,  # left arm
    0.907222843, 0.907222843, 0.907222843, 0.907222843, 0.907222843, 1.068141502, 1.068141502,  # right arm
], dtype=np.float32)

# =============================================================================
# 23-DOF Configuration (native G1-23: no waist_roll/pitch, no wrist_pitch/yaw)
# Derived from 29-DOF Holosoma values
# =============================================================================
# Joint order: 6 left leg, 6 right leg, 1 waist_yaw, 5 left arm, 5 right arm
HOLOSOMA_23DOF_DEFAULT_ANGLES = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # left leg (from 29-DOF)
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # right leg (from 29-DOF)
    0.0,                                    # waist_yaw only (from 29-DOF)
    0.2, 0.2, 0.0, 0.6, 0.0,               # left arm first 5 joints (from 29-DOF)
    0.2, -0.2, 0.0, 0.6, 0.0,              # right arm first 5 joints (from 29-DOF)
], dtype=np.float32)

HOLOSOMA_23DOF_KP = np.array([
    40.179238471, 99.098427777, 40.179238471, 99.098427777, 28.501246196, 28.501246196,  # left leg
    40.179238471, 99.098427777, 40.179238471, 99.098427777, 28.501246196, 28.501246196,  # right leg
    40.179238471,                                                                         # waist_yaw
    14.250623098, 14.250623098, 14.250623098, 14.250623098, 14.250623098,                 # left arm
    14.250623098, 14.250623098, 14.250623098, 14.250623098, 14.250623098,                 # right arm
], dtype=np.float32)

HOLOSOMA_23DOF_KD = np.array([
    2.557889765, 6.308801854, 2.557889765, 6.308801854, 1.814445687, 1.814445687,  # left leg
    2.557889765, 6.308801854, 2.557889765, 6.308801854, 1.814445687, 1.814445687,  # right leg
    2.557889765,                                                                    # waist_yaw
    0.907222843, 0.907222843, 0.907222843, 0.907222843, 0.907222843,               # left arm
    0.907222843, 0.907222843, 0.907222843, 0.907222843, 0.907222843,               # right arm
], dtype=np.float32)

# Maps 23-DOF policy index → 29-DOF motor index
# 23-DOF: legs(0-11), waist_yaw(12), L_arm(13-17), R_arm(18-22)
# 29-DOF: legs(0-11), waist(12-14), L_arm(15-21), R_arm(22-28)
DOF_23_TO_MOTOR_MAP = [
    0, 1, 2, 3, 4, 5,       # left leg → motor 0-5
    6, 7, 8, 9, 10, 11,     # right leg → motor 6-11
    12,                      # waist_yaw → motor 12
    15, 16, 17, 18, 19,     # left arm (skip wrist_pitch/yaw) → motor 15-19
    22, 23, 24, 25, 26,     # right arm (skip wrist_pitch/yaw) → motor 22-26
]
# fmt: on

# Control parameters
LOCOMOTION_CONTROL_DT = 0.02  # 50Hz
LOCOMOTION_ACTION_SCALE = 0.25
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
GAIT_PERIOD = 1.0

DEFAULT_HOLOSOMA_REPO_ID = "nepyope/holosoma_locomotion"


def load_holosoma_policy(
    repo_id: str = DEFAULT_HOLOSOMA_REPO_ID,
    policy_name: str = "fastsac",
    local_path: str | None = None,
) -> tuple[ort.InferenceSession, int]:
    """Load Holosoma policy and detect observation dimension.

    Returns:
        (policy, obs_dim) tuple where obs_dim is 82 (23-DOF) or 100 (29-DOF)
    """
    if local_path is not None:
        logger.info(f"Loading policy from local path: {local_path}")
        policy_path = local_path
    else:
        logger.info(f"Loading policy from Hugging Face Hub: {repo_id}")
        policy_path = hf_hub_download(repo_id=repo_id, filename=f"{policy_name}_g1_29dof.onnx")

    policy = ort.InferenceSession(policy_path)

    # Detect observation dimension from model input shape
    input_shape = policy.get_inputs()[0].shape
    obs_dim = input_shape[1] if len(input_shape) > 1 else input_shape[0]

    logger.info(f"Policy loaded successfully")
    logger.info(f"  Input: {policy.get_inputs()[0].name}, shape: {input_shape} → obs_dim={obs_dim}")
    logger.info(f"  Output: {policy.get_outputs()[0].name}, shape: {policy.get_outputs()[0].shape}")

    return policy, obs_dim


class HolosomaLocomotionController:
    """
    Handles Holosoma whole-body locomotion for Unitree G1.
    Supports both 23-DOF (82D obs) and 29-DOF (100D obs) policies.
    """

    def __init__(self, policy, robot, config, obs_dim: int = 100):
        self.policy = policy
        self.robot = robot
        self.config = config
        self.obs_dim = obs_dim

        # Detect policy type from observation dimension
        self.is_23dof = (obs_dim == 82)
        self.num_dof = 23 if self.is_23dof else 29

        # Velocity commands
        self.locomotion_cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # State variables sized for policy type
        self.qj = np.zeros(self.num_dof, dtype=np.float32)
        self.dqj = np.zeros(self.num_dof, dtype=np.float32)
        self.locomotion_action = np.zeros(self.num_dof, dtype=np.float32)
        self.locomotion_obs = np.zeros(obs_dim, dtype=np.float32)
        self.last_unscaled_action = np.zeros(self.num_dof, dtype=np.float32)

        # Select config based on DOF
        if self.is_23dof:
            self.default_angles = HOLOSOMA_23DOF_DEFAULT_ANGLES
            self.kp = HOLOSOMA_23DOF_KP
            self.kd = HOLOSOMA_23DOF_KD
            self.motor_map = DOF_23_TO_MOTOR_MAP
        else:
            self.default_angles = HOLOSOMA_29DOF_DEFAULT_ANGLES
            self.kp = HOLOSOMA_29DOF_KP
            self.kd = HOLOSOMA_29DOF_KD
            self.motor_map = list(range(29))  # Identity map for 29-DOF

        # Phase state for gait
        self.phase = np.zeros((1, 2), dtype=np.float32)
        self.phase[0, 0] = 0.0
        self.phase[0, 1] = np.pi
        self.phase_dt = 2 * np.pi / (50.0 * GAIT_PERIOD)
        self.is_standing = False

        self.counter = 0
        self.locomotion_running = False
        self.locomotion_thread = None

        logger.info(f"HolosomaLocomotionController initialized")
        logger.info(f"  Mode: {'23-DOF (82D obs)' if self.is_23dof else '29-DOF (100D obs)'}")
        logger.info(f"  Action dim: {self.num_dof}")

    def holosoma_locomotion_run(self):
        """Main locomotion loop - handles both 23-DOF and 29-DOF."""
        self.counter += 1

        if self.counter == 1:
            print("\n" + "=" * 60)
            print(f"🚀 RUNNING HOLOSOMA {self.num_dof}-DOF LOCOMOTION POLICY")
            print(f"   {self.obs_dim}D observations → {self.num_dof}D actions")
            print("=" * 60 + "\n")

        robot_state = self.robot.get_observation()
        if robot_state is None:
            return

        # Remote controller
        if robot_state.wireless_remote is not None:
            self.robot.remote_controller.set(robot_state.wireless_remote)
        else:
            self.robot.remote_controller.lx = 0.0
            self.robot.remote_controller.ly = 0.0
            self.robot.remote_controller.rx = 0.0
            self.robot.remote_controller.ry = 0.0

        # Deadzone
        ly = self.robot.remote_controller.ly if abs(self.robot.remote_controller.ly) > 0.1 else 0.0
        lx = self.robot.remote_controller.lx if abs(self.robot.remote_controller.lx) > 0.1 else 0.0
        rx = self.robot.remote_controller.rx if abs(self.robot.remote_controller.rx) > 0.1 else 0.0

        self.locomotion_cmd[0] = ly
        self.locomotion_cmd[1] = -lx
        self.locomotion_cmd[2] = -rx

        # Read joint states using motor map
        for i in range(self.num_dof):
            motor_idx = self.motor_map[i]
            self.qj[i] = robot_state.motor_state[motor_idx].q
            self.dqj[i] = robot_state.motor_state[motor_idx].dq

        # IMU
        quat = robot_state.imu_state.quaternion
        ang_vel = np.array(robot_state.imu_state.gyroscope, dtype=np.float32)
        gravity_orientation = self.robot.get_gravity_orientation(quat)

        # Scale observations
        qj_obs = (self.qj - self.default_angles) * DOF_POS_SCALE
        dqj_obs = self.dqj * DOF_VEL_SCALE
        ang_vel_scaled = ang_vel * ANG_VEL_SCALE

        # Phase update
        cmd_norm = np.linalg.norm(self.locomotion_cmd[:2])
        ang_cmd_norm = np.abs(self.locomotion_cmd[2])

        if cmd_norm < 0.01 and ang_cmd_norm < 0.01:
            self.phase[0, :] = np.pi * np.ones(2)
            self.is_standing = True
        elif self.is_standing:
            self.phase = np.array([[0.0, np.pi]], dtype=np.float32)
            self.is_standing = False
        else:
            phase_tp1 = self.phase + self.phase_dt
            self.phase = np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi

        sin_phase = np.sin(self.phase[0, :])
        cos_phase = np.cos(self.phase[0, :])

        # Build observation (format depends on DOF)
        if self.is_23dof:
            # 82D: [23 actions, 3 ang_vel, 1 cmd_yaw, 2 cmd_lin, 2 cos, 23 pos, 23 vel, 3 grav, 2 sin]
            self.locomotion_obs[0:23] = self.last_unscaled_action
            self.locomotion_obs[23:26] = ang_vel_scaled
            self.locomotion_obs[26] = self.locomotion_cmd[2]
            self.locomotion_obs[27:29] = self.locomotion_cmd[:2]
            self.locomotion_obs[29:31] = cos_phase
            self.locomotion_obs[31:54] = qj_obs
            self.locomotion_obs[54:77] = dqj_obs
            self.locomotion_obs[77:80] = gravity_orientation
            self.locomotion_obs[80:82] = sin_phase
        else:
            # 100D: [29 actions, 3 ang_vel, 1 cmd_yaw, 2 cmd_lin, 2 cos, 29 pos, 29 vel, 3 grav, 2 sin]
            self.locomotion_obs[0:29] = self.last_unscaled_action
            self.locomotion_obs[29:32] = ang_vel_scaled
            self.locomotion_obs[32] = self.locomotion_cmd[2]
            self.locomotion_obs[33:35] = self.locomotion_cmd[:2]
            self.locomotion_obs[35:37] = cos_phase
            self.locomotion_obs[37:66] = qj_obs
            self.locomotion_obs[66:95] = dqj_obs
            self.locomotion_obs[95:98] = gravity_orientation
            self.locomotion_obs[98:100] = sin_phase

        # Policy inference
        obs_input = self.locomotion_obs.reshape(1, -1).astype(np.float32)
        ort_inputs = {self.policy.get_inputs()[0].name: obs_input}
        ort_outs = self.policy.run(None, ort_inputs)

        raw_action = ort_outs[0].squeeze()
        clipped_action = np.clip(raw_action, -100.0, 100.0)

        self.last_unscaled_action = clipped_action.copy()
        self.locomotion_action = clipped_action * LOCOMOTION_ACTION_SCALE

        # Debug
        if self.counter <= 3:
            print(f"\n[Holosoma Debug #{self.counter}]")
            print(f"  Phase: ({self.phase[0, 0]:.3f}, {self.phase[0, 1]:.3f})")
            print(f"  Cmd: ({self.locomotion_cmd[0]:.2f}, {self.locomotion_cmd[1]:.2f}, {self.locomotion_cmd[2]:.2f})")
            print(f"  Action range: [{raw_action.min():.3f}, {raw_action.max():.3f}]")

        # Compute target positions
        target_dof_pos = self.default_angles + self.locomotion_action

        # Send commands to motors via motor map
        for i in range(self.num_dof):
            motor_idx = self.motor_map[i]
            self.robot.msg.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.robot.msg.motor_cmd[motor_idx].qd = 0
            self.robot.msg.motor_cmd[motor_idx].kp = self.kp[i]
            self.robot.msg.motor_cmd[motor_idx].kd = self.kd[i]
            self.robot.msg.motor_cmd[motor_idx].tau = 0

        # For 23-DOF: zero out missing joints (waist_roll/pitch, wrist_pitch/yaw)
        if self.is_23dof:
            missing_motors = [13, 14, 20, 21, 27, 28]  # waist_roll, waist_pitch, wrist_pitch/yaw
            for motor_idx in missing_motors:
                self.robot.msg.motor_cmd[motor_idx].q = 0.0
                self.robot.msg.motor_cmd[motor_idx].qd = 0
                self.robot.msg.motor_cmd[motor_idx].kp = 40.0
                self.robot.msg.motor_cmd[motor_idx].kd = 2.0
                self.robot.msg.motor_cmd[motor_idx].tau = 0

        self.robot.send_action(self.robot.msg)

    def _locomotion_thread_loop(self):
        logger.info("Locomotion thread started")
        while self.locomotion_running:
            start_time = time.time()
            try:
                self.holosoma_locomotion_run()
            except Exception as e:
                logger.error(f"Error in locomotion loop: {e}")
                import traceback
                traceback.print_exc()

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
        """Move joints to default position."""
        logger.info(f"Moving {self.num_dof} joints to default position...")

        total_time = 3.0
        num_step = int(total_time / self.robot.control_dt)

        robot_state = self.robot.get_observation()

        # Record current positions
        init_dof_pos = np.zeros(self.num_dof, dtype=np.float32)
        for i in range(self.num_dof):
            motor_idx = self.motor_map[i]
            init_dof_pos[i] = robot_state.motor_state[motor_idx].q

        # Interpolate to target
        for step in range(num_step):
            alpha = step / num_step
            for i in range(self.num_dof):
                motor_idx = self.motor_map[i]
                target = self.default_angles[i]
                self.robot.msg.motor_cmd[motor_idx].q = init_dof_pos[i] * (1 - alpha) + target * alpha
                self.robot.msg.motor_cmd[motor_idx].qd = 0
                self.robot.msg.motor_cmd[motor_idx].kp = self.kp[i]
                self.robot.msg.motor_cmd[motor_idx].kd = self.kd[i]
                self.robot.msg.motor_cmd[motor_idx].tau = 0

            # Zero missing joints for 23-DOF
            if self.is_23dof:
                for motor_idx in [13, 14, 20, 21, 27, 28]:
                    self.robot.msg.motor_cmd[motor_idx].q = 0.0
                    self.robot.msg.motor_cmd[motor_idx].qd = 0
                    self.robot.msg.motor_cmd[motor_idx].kp = 40.0
                    self.robot.msg.motor_cmd[motor_idx].kd = 2.0
                    self.robot.msg.motor_cmd[motor_idx].tau = 0

            self.robot.msg.crc = self.robot.crc.Crc(self.robot.msg)
            self.robot.lowcmd_publisher.Write(self.robot.msg)
            time.sleep(self.robot.control_dt)

        logger.info(f"Reached default position ({self.num_dof} joints)")

        # Hold for 2 seconds
        logger.info("Holding default position for 2 seconds...")
        hold_steps = int(2.0 / self.robot.control_dt)
        for _ in range(hold_steps):
            for i in range(self.num_dof):
                motor_idx = self.motor_map[i]
                self.robot.msg.motor_cmd[motor_idx].q = self.default_angles[i]
                self.robot.msg.motor_cmd[motor_idx].qd = 0
                self.robot.msg.motor_cmd[motor_idx].kp = self.kp[i]
                self.robot.msg.motor_cmd[motor_idx].kd = self.kd[i]
                self.robot.msg.motor_cmd[motor_idx].tau = 0

            if self.is_23dof:
                for motor_idx in [13, 14, 20, 21, 27, 28]:
                    self.robot.msg.motor_cmd[motor_idx].q = 0.0
                    self.robot.msg.motor_cmd[motor_idx].qd = 0
                    self.robot.msg.motor_cmd[motor_idx].kp = 40.0
                    self.robot.msg.motor_cmd[motor_idx].kd = 2.0
                    self.robot.msg.motor_cmd[motor_idx].tau = 0

            self.robot.msg.crc = self.robot.crc.Crc(self.robot.msg)
            self.robot.lowcmd_publisher.Write(self.robot.msg)
            time.sleep(self.robot.control_dt)

        logger.info("Ready to start locomotion!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Holosoma Locomotion Controller for Unitree G1")
    parser.add_argument("--repo-id", type=str, default=DEFAULT_HOLOSOMA_REPO_ID)
    parser.add_argument("--policy", type=str, default="fastsac", choices=["fastsac", "ppo"])
    parser.add_argument("--local-path", type=str, default=None, help="Path to local ONNX file")
    args = parser.parse_args()

    # Load policy and detect dimensions
    policy, obs_dim = load_holosoma_policy(
        repo_id=args.repo_id,
        policy_name=args.policy,
        local_path=args.local_path,
    )

    # Initialize robot
    config = UnitreeG1Config()
    robot = UnitreeG1(config)

    # Initialize controller with detected obs_dim
    controller = HolosomaLocomotionController(
        policy=policy,
        robot=robot,
        config=config,
        obs_dim=obs_dim,
    )

    try:
        controller.reset_robot()
        controller.start_locomotion_thread()

        logger.info(f"Robot initialized with Holosoma {'23-DOF' if obs_dim == 82 else '29-DOF'} policy")
        logger.info("Use remote controller: LY=fwd/back, LX=left/right, RX=rotate")
        logger.info("Press Ctrl+C to stop")

        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping locomotion...")
        controller.stop_locomotion_thread()
        print("Done!")
