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
import time
from functools import cached_property
from typing import Any, Dict

import numpy as np
import pinocchio as pin

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.damiao import DamiaoMotorsBus
from lerobot.motors.damiao.tables import MotorType
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_openarms_follower import OpenArmsFollowerConfig

logger = logging.getLogger(__name__)


class OpenArmsFollower(Robot):
    """
    OpenArms Follower Robot which uses CAN bus communication to control 7 DOF arm with a gripper.
    The arm uses Damiao motors in MIT control mode.
    """

    config_class = OpenArmsFollowerConfig
    name = "openarms_follower"

    def __init__(self, config: OpenArmsFollowerConfig):
        super().__init__(config)
        self.config = config
        
        norm_mode_body = MotorNormMode.DEGREES # Always use degrees for Damiao motors
        motors = {}
        
        # Right arm
        for motor_name, (send_id, recv_id, motor_type_str) in config.motor_config.items():
            prefixed_name = f"right_{motor_name}"
            motor = Motor(send_id, motor_type_str, norm_mode_body)
            motor.recv_id = recv_id
            motor.motor_type = getattr(MotorType, motor_type_str.upper().replace("-", "_"))
            motors[prefixed_name] = motor
        
        # Left arm (offset IDs by 8)
        for motor_name, (send_id, recv_id, motor_type_str) in config.motor_config.items():
            prefixed_name = f"left_{motor_name}"
            motor = Motor(send_id + 0x08, motor_type_str, norm_mode_body)
            motor.recv_id = recv_id + 0x08
            motor.motor_type = getattr(MotorType, motor_type_str.upper().replace("-", "_"))
            motors[prefixed_name] = motor
        
        # Initialize the Damiao motors bus
        self.bus = DamiaoMotorsBus(
            port=self.config.port,
            motors=motors,
            calibration=self.calibration,
            can_interface=self.config.can_interface,
        )
        
        # Initialize cameras
        self.cameras = make_cameras_from_configs(config.cameras)
        
        # Initialize Pinocchio robot model for dynamics (optional)
        self.pin_robot = None
        try:
            # Try to load URDF if available
            # TODO: Add OpenArms URDF file to repository
            self.pin_robot = pin.RobotWrapper.BuildFromURDF("urdf/openarms.urdf", "urdf")
            logger.info("Loaded OpenArms URDF for dynamics computation")
        except Exception as e:
            logger.warning(f"Could not load URDF for dynamics: {e}. Gravity compensation will not be available.")

    @property
    def _motors_ft(self) -> Dict[str, type]:
        """Motor features for observation and action spaces."""
        features = {}
        for motor in self.bus.motors:
            features[f"{motor}.pos"] = float
            features[f"{motor}.vel"] = float
            features[f"{motor}.torque"] = float
        return features

    @property
    def _cameras_ft(self) -> Dict[str, tuple]:
        """Camera features for observation space."""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> Dict[str, type | tuple]:
        """Combined observation features from motors and cameras."""
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> Dict[str, type]:
        """Action features (motor positions only)."""
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        """Check if robot is connected."""
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to the robot and optionally calibrate.
        
        We assume that at connection time, the arm is in a safe rest position,
        and torque can be safely disabled to run calibration if needed.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Connect to CAN bus
        self.bus.connect()
        
        # Run calibration if needed
        if not self.is_calibrated and calibrate:
            logger.info(
                "No calibration found or calibration mismatch. Running calibration..."
            )
            self.calibrate()
        
        # Connect cameras
        for cam in self.cameras.values():
            cam.connect()
        
        # Configure motors
        self.configure()
        
        # Optionally set zero position
        if self.config.zero_position_on_connect:
            logger.info("Setting current position as zero...")
            self.bus.set_zero_position()
        
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """Check if robot is calibrated."""
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        """
        Run calibration procedure for OpenArms robot.
        
        The calibration procedure:
        1. Disable torque
        2. Ask user to position arm in hanging position with gripper closed
        3. Set this as zero position
        4. Record range of motion for each joint
        5. Save calibration
        """
        if self.calibration:
            # Ask user whether to use existing calibration
            user_input = input(
                f"Press ENTER to use existing calibration for {self.id}, "
                f"or type 'c' and press ENTER to run new calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Using existing calibration for {self.id}")
                self.bus.write_calibration(self.calibration)
                return
        
        logger.info(f"\nRunning calibration for {self}")
        
        # Disable torque for manual positioning
        self.bus.disable_torque()
        time.sleep(0.1)
        
        # Step 1: Set zero position
        input(
            "\nCalibration Step 1: Zero Position\n"
            "Position the arm in the following configuration:\n"
            "  - Arm hanging straight down\n"
            "  - Gripper closed\n"
            "Press ENTER when ready..."
        )
        
        # Set current position as zero for all motors
        self.bus.set_zero_position()
        logger.info("Zero position set.")
        
        # Step 2: Record range of motion
        print(
            "\nCalibration Step 2: Range of Motion\n"
            "Move each joint through its full range of motion.\n"
            "The system will record min/max positions.\n"
            "Press ENTER when done..."
        )
        
        # Record ranges
        range_mins, range_maxes = self.bus.record_ranges_of_motion()
        
        # Create calibration data (ranges are already in degrees)
        self.calibration = {}
        for motor_name, motor in self.bus.motors.items():
            self.calibration[motor_name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,  # Normal direction
                homing_offset=0,  # Already set via set_zero_position
                range_min=range_mins.get(motor_name, -180.0),  # Default -180 degrees
                range_max=range_maxes.get(motor_name, 180.0),   # Default +180 degrees
            )
        
        # Special handling for gripper range
        if "gripper" in self.calibration:
            gripper_cal = self.calibration["gripper"]
            gripper_range = abs(gripper_cal.range_max - gripper_cal.range_min)
            if gripper_range < 5.0:  # If gripper wasn't moved much (less than 5 degrees)
                # Set default gripper range in degrees
                gripper_cal.range_min = 0.0
                gripper_cal.range_max = 90.0  # 90 degrees for full gripper motion
        
        # Write calibration to motors and save to file
        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        
        print(f"\nCalibration complete and saved to {self.calibration_fpath}")
        
        # Re-enable torque
        self.bus.enable_torque()

    def configure(self) -> None:
        """Configure motors with appropriate settings."""
        with self.bus.torque_disabled():
            # Configure all motors
            self.bus.configure_motors()
            
            # Set specific parameters for gripper if present
            if "gripper" in self.bus.motors:
                # Gripper uses lower gains to avoid damage
                # These will be applied during MIT control commands
                pass  # Parameters are set during control commands

    def setup_motors(self) -> None:
        raise NotImplementedError("Motor ID configuration is typically done via manufacturer tools for CAN motors.")

    def get_observation(self) -> Dict[str, Any]:
        """Get current observation from robot including position, velocity, and torque."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        obs_dict = {}
        
        # Read motor positions, velocities, and torques
        start = time.perf_counter()
        positions = self.bus.sync_read("Present_Position")
        velocities = self.bus.sync_read("Present_Velocity")
        torques = self.bus.sync_read("Present_Torque")
        
        for motor in self.bus.motors:
            obs_dict[f"{motor}.pos"] = positions.get(motor, 0.0)
            obs_dict[f"{motor}.vel"] = velocities.get(motor, 0.0)
            obs_dict[f"{motor}.torque"] = torques.get(motor, 0.0)
        
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")
        
        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")
        
        return obs_dict

    def send_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send action command to robot.
        
        The action magnitude may be clipped based on safety limits.
        
        Args:
            action: Dictionary with motor positions
            
        Returns:
            The action actually sent (potentially clipped)
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        # Extract motor positions from action
        goal_pos = {
            key.removesuffix(".pos"): val 
            for key, val in action.items() 
            if key.endswith(".pos")
        }
        
        # Apply safety limits if configured
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {
                key: (g_pos, present_pos[key]) 
                for key, g_pos in goal_pos.items()
            }
            goal_pos = ensure_safe_goal_position(
                goal_present_pos, 
                self.config.max_relative_target
            )
        
        # Prepare MIT control commands for each motor
        for motor_name, position_degrees in goal_pos.items():
            # Use different gains for gripper
            if motor_name == "gripper":
                kp = self.config.position_kp * 0.5  # Lower gain for gripper
                kd = self.config.position_kd * 0.5
            else:
                kp = self.config.position_kp
                kd = self.config.position_kd
            
            # Send MIT control command (position is in degrees)
            self.bus._mit_control(
                motor_name,
                kp=kp,
                kd=kd,
                position_degrees=position_degrees,
                velocity_deg_per_sec=0.0,
                torque=0.0
            )
        
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def disconnect(self):
        """Disconnect from robot."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        # Disconnect from CAN bus
        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        
        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()
        
        logger.info(f"{self} disconnected.")
    
    def _deg_to_rad(self, deg: Dict[str, float | int]) -> Dict[str, float]:
        """Convert degrees to radians for all motors."""
        return {m: np.deg2rad(float(v)) for m, v in deg.items()}
    
    def _gravity_from_q(self, q_rad: Dict[str, float]) -> Dict[str, float]:
        """
        Compute g(q) [N·m] for all joints in the robot.
        The order of joints in the URDF matches self.bus.motors.
        
        Args:
            q_rad: Dictionary mapping motor names to positions in radians
            
        Returns:
            Dictionary mapping motor names to gravity torques in N·m
            
        Raises:
            RuntimeError: If URDF model is not loaded
        """
        if self.pin_robot is None:
            raise RuntimeError(
                "Cannot compute gravity: URDF model not loaded. "
                "Ensure urdf/openarms.urdf exists and is valid."
            )
        
        # Build position vector in the order of motors
        q = np.zeros(self.pin_robot.model.nq)
        for i, motor_name in enumerate(self.bus.motors):
            q[i] = q_rad[motor_name]
        
        # Compute generalized gravity vector
        g = pin.computeGeneralizedGravity(self.pin_robot.model, self.pin_robot.data, q)
        
        # Map back to motor names
        return {motor_name: float(g[i]) for i, motor_name in enumerate(self.bus.motors)}
    
