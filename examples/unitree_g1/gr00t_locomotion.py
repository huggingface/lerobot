#!/usr/bin/env python

"""
Example: GR00T Locomotion with Pre-loaded Policies

This example demonstrates the NEW pattern for loading GR00T policies externally
and passing them to the robot class.
"""

import logging
import threading
import time
from collections import deque

import numpy as np
import onnxruntime as ort
import torch

from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config
from lerobot.robots.unitree_g1.unitree_g1 import UnitreeG1

logger = logging.getLogger(__name__)

GROOT_DEFAULT_ANGLES = np.array(
    [
        -0.1,
        0.0,
        0.0,
        0.3,
        -0.2,
        0.0,  # left leg
        -0.1,
        0.0,
        0.0,
        0.3,
        -0.2,
        0.0,  # right leg
        0.0,
        0.0,
        0.0,  # waist
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,  # left arm (zeroed)
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,  # right arm (zeroed)
    ],
    dtype=np.float32,
)

JOINTS_TO_ZERO = [12, 14, 20, 21, 27, 28]  # waist yaw/pitch, wrist pitch/yaw
PROBLEMATIC_JOINTS = [12, 14, 20, 21, 27, 28]

LOCOMOTION_ACTION_SCALE = 0.25

LOCOMOTION_CONTROL_DT = 0.02


ANG_VEL_SCALE: float = 0.25
DOF_POS_SCALE: float = 1.0
DOF_VEL_SCALE: float = 0.05
CMD_SCALE: list = [2.0, 2.0, 0.25]


def load_groot_policies() -> tuple:
    """Load GR00T dual-policy system (Balance + Walk) from ONNX files."""
    logger.info("Loading GR00T dual-policy system...")

    # Load ONNX policies
    policy_balance = ort.InferenceSession(
        "examples/unitree_g1/locomotion/GR00T-WholeBodyControl-Balance.onnx"
    )
    policy_walk = ort.InferenceSession("examples/unitree_g1/locomotion/GR00T-WholeBodyControl-Walk.onnx")

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
        """
        Initialize the GR00T locomotion controller.

        Args:
            policy_balance: ONNX InferenceSession for balance/standing policy
            policy_walk: ONNX InferenceSession for walking policy
            robot: Reference to the UnitreeG1 robot instance
            config: UnitreeG1Config object with locomotion parameters
        """
        self.policy_balance = policy_balance
        self.policy_walk = policy_walk
        self.robot = robot
        self.config = config

        self.locomotion_cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # vx, vy, yaw_rate

        # GR00T-specific state
        self.groot_qj_all = np.zeros(29, dtype=np.float32)
        self.groot_dqj_all = np.zeros(29, dtype=np.float32)
        self.groot_action = np.zeros(15, dtype=np.float32)
        self.groot_obs_single = np.zeros(86, dtype=np.float32)
        self.groot_obs_history = deque(maxlen=6)
        self.groot_obs_stacked = np.zeros(516, dtype=np.float32)
        self.groot_height_cmd = 0.74  # Default base height
        self.groot_orientation_cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Initialize history with zeros
        for _ in range(6):
            self.groot_obs_history.append(np.zeros(86, dtype=np.float32))

        # Thread management
        self.locomotion_running = False
        self.locomotion_thread = None

        logger.info("GrootLocomotionController initialized")

    def groot_locomotion_run(self):
        # Get current obs
        robot_state = self.robot.get_observation()
        if robot_state is None:
            return

        # Update remote controller from lowstate
        if robot_state.wireless_remote is not None:
            self.robot.remote_controller.set(robot_state.wireless_remote)

            # R1/R2 buttons for height control on real robot (button indices 0 and 4)
            if self.robot.remote_controller.button[0]:  # R1 - raise height
                self.groot_height_cmd += 0.001  # Small increment per timestep (~0.05m per second at 50Hz)
                self.groot_height_cmd = np.clip(self.groot_height_cmd, 0.50, 1.00)
            if self.robot.remote_controller.button[4]:  # R2 - lower height
                self.groot_height_cmd -= 0.001  # Small decrement per timestep
                self.groot_height_cmd = np.clip(self.groot_height_cmd, 0.50, 1.00)
        else:
            # Default to zero commands if no remote data
            self.robot.remote_controller.lx = 0.0
            self.robot.remote_controller.ly = 0.0
            self.robot.remote_controller.rx = 0.0
            self.robot.remote_controller.ry = 0.0

        # Get ALL 29 joint positions and velocities
        for i in range(29):
            self.groot_qj_all[i] = robot_state.motor_state[i].q
            self.groot_dqj_all[i] = robot_state.motor_state[i].dq

        # Get IMU data
        quat = robot_state.imu_state.quaternion
        ang_vel = np.array(robot_state.imu_state.gyroscope, dtype=np.float32)

        gravity_orientation = self.robot.get_gravity_orientation(quat)

        # Zero out specific joints in observation
        for idx in JOINTS_TO_ZERO:
            self.groot_qj_all[idx] = 0.0
            self.groot_dqj_all[idx] = 0.0

        # Scale joint positions and velocities
        qj_obs = self.groot_qj_all.copy()
        dqj_obs = self.groot_dqj_all.copy()

        qj_obs = (qj_obs - GROOT_DEFAULT_ANGLES) * DOF_POS_SCALE
        dqj_obs = dqj_obs * DOF_VEL_SCALE
        ang_vel_scaled = ang_vel * ANG_VEL_SCALE

        # Get velocity commands (keyboard or remote)
        if not self.robot.simulation_mode:
            self.locomotion_cmd[0] = self.robot.remote_controller.ly
            self.locomotion_cmd[1] = self.robot.remote_controller.lx * -1
            self.locomotion_cmd[2] = self.robot.remote_controller.rx * -1

        # Build 86D single frame observation (GR00T format)
        self.groot_obs_single[:3] = self.locomotion_cmd * np.array(CMD_SCALE)
        self.groot_obs_single[3] = self.groot_height_cmd
        self.groot_obs_single[4:7] = self.groot_orientation_cmd
        self.groot_obs_single[7:10] = ang_vel_scaled
        self.groot_obs_single[10:13] = gravity_orientation
        self.groot_obs_single[13:42] = qj_obs  # 29D joint positions
        self.groot_obs_single[42:71] = dqj_obs  # 29D joint velocities
        self.groot_obs_single[71:86] = self.groot_action  # 15D previous actions

        # Add to history and stack observations (6 frames × 86D = 516D)
        self.groot_obs_history.append(self.groot_obs_single.copy())

        # Stack all 6 frames into 516D vector
        for i, obs_frame in enumerate(self.groot_obs_history):
            start_idx = i * 86
            end_idx = start_idx + 86
            self.groot_obs_stacked[start_idx:end_idx] = obs_frame

        # Run policy inference (ONNX) with 516D stacked observation
        obs_tensor = torch.from_numpy(self.groot_obs_stacked).unsqueeze(0)

        # Select appropriate policy based on command magnitude (dual-policy system)
        cmd_magnitude = np.linalg.norm(self.locomotion_cmd)
        if cmd_magnitude < 0.05:
            # Use balance/standing policy for small commands
            selected_policy = self.policy_balance
        else:
            # Use walking policy for movement commands
            selected_policy = self.policy_walk

        ort_inputs = {selected_policy.get_inputs()[0].name: obs_tensor.cpu().numpy()}
        ort_outs = selected_policy.run(None, ort_inputs)
        self.groot_action = ort_outs[0].squeeze()

        # Zero out waist actions (yaw=12, roll=13, pitch=14) - only use leg actions (0-11)
        self.groot_action[12] = 0.0  # Waist yaw
        self.groot_action[13] = 0.0  # Waist roll
        self.groot_action[14] = 0.0  # Waist pitch

        # Transform action to target joint positions (15D: legs + waist)
        target_dof_pos_15 = GROOT_DEFAULT_ANGLES[:15] + self.groot_action * LOCOMOTION_ACTION_SCALE

        # Send commands to LEG motors (0-11)
        for i in range(12):
            motor_idx = i
            self.robot.msg.motor_cmd[motor_idx].q = target_dof_pos_15[i]
            self.robot.msg.motor_cmd[motor_idx].qd = 0
            self.robot.msg.motor_cmd[motor_idx].kp = self.robot.kp[motor_idx]
            self.robot.msg.motor_cmd[motor_idx].kd = self.robot.kd[motor_idx]
            self.robot.msg.motor_cmd[motor_idx].tau = 0

        # Send WAIST commands - but SKIP waist yaw (12) and waist pitch (14)
        # Only send waist roll (13)
        waist_roll_idx = 13
        waist_roll_action_idx = 13
        self.robot.msg.motor_cmd[waist_roll_idx].q = target_dof_pos_15[waist_roll_action_idx]
        self.robot.msg.motor_cmd[waist_roll_idx].qd = 0
        self.robot.msg.motor_cmd[waist_roll_idx].kp = self.robot.kp[waist_roll_idx]
        self.robot.msg.motor_cmd[waist_roll_idx].kd = self.robot.kd[waist_roll_idx]
        self.robot.msg.motor_cmd[waist_roll_idx].tau = 0

        # Zero out the problematic joints (waist yaw, waist pitch, wrist pitch/yaw)
        for joint_idx in PROBLEMATIC_JOINTS:
            self.robot.msg.motor_cmd[joint_idx].q = 0.0
            self.robot.msg.motor_cmd[joint_idx].qd = 0
            self.robot.msg.motor_cmd[joint_idx].kp = self.robot.kp[joint_idx]
            self.robot.msg.motor_cmd[joint_idx].kd = self.robot.kd[joint_idx]
            self.robot.msg.motor_cmd[joint_idx].tau = 0

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
        """Start the background locomotion control thread."""
        if self.locomotion_running:
            logger.warning("Locomotion thread already running")
            return

        logger.info("Starting locomotion control thread...")
        self.locomotion_running = True
        self.locomotion_thread = threading.Thread(target=self._locomotion_thread_loop, daemon=True)
        self.locomotion_thread.start()
        logger.info("Locomotion control thread started!")

    def stop_locomotion_thread(self):
        """Stop the background locomotion control thread."""
        if not self.locomotion_running:
            return

        logger.info("Stopping locomotion control thread...")
        self.locomotion_running = False
        if self.locomotion_thread:
            self.locomotion_thread.join(timeout=2.0)
        logger.info("Locomotion control thread stopped")

    def init_groot_locomotion(self):
        """Initialize GR00T-style locomotion for ONNX policies (29 DOF, 15D actions)."""
        logger.info("Starting GR00T locomotion initialization...")

        # Reset legs to default position
        self.robot.reset_legs()

        # Wait 3 seconds
        time.sleep(3.0)

        # Start locomotion policy thread
        logger.info("Starting GR00T locomotion policy control...")
        self.start_locomotion_thread()


if __name__ == "__main__":
    # 1. Load policies externally (separate from robot initialization)
    policy_balance, policy_walk = load_groot_policies()

    # 2. Create config (no locomotion_control=True since we're using external controller)
    config = UnitreeG1Config()

    # 3. Initialize robot
    robot = UnitreeG1(config)

    # 4. Create GR00T locomotion controller with loaded policies
    groot_controller = GrootLocomotionController(
        policy_balance=policy_balance,
        policy_walk=policy_walk,
        robot=robot,
        config=config,
    )

    # 5. Initialize and start locomotion
    groot_controller.init_groot_locomotion()

    # Robot is now ready with locomotion control!
    print("Robot initialized with GR00T locomotion policies")
    print("Locomotion controller running in background thread")
    print("Press Ctrl+C to stop")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping locomotion...")
        groot_controller.stop_locomotion_thread()
        print("Done!")
