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
from typing import Dict

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.damiao import DamiaoMotorsBus
from lerobot.motors.damiao.tables import MotorType
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_openarms_leader import OpenArmsLeaderConfig

logger = logging.getLogger(__name__)


class OpenArmsLeader(Teleoperator):
    """
    OpenArms Leader/Teleoperator Arm with Damiao motors.
    
    This teleoperator uses CAN bus communication to read positions from 
    Damiao motors that are manually moved (torque disabled).
    """

    config_class = OpenArmsLeaderConfig
    name = "openarms_leader"

    def __init__(self, config: OpenArmsLeaderConfig):
        super().__init__(config)
        self.config = config
        
        
        norm_mode_body = MotorNormMode.DEGREES # Always use degrees for Damiao motors
        motors = {}
        
        # Right arm (original IDs)
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

    @property
    def action_features(self) -> Dict[str, type]:
        """Features produced by this teleoperator."""
        features = {}
        for motor in self.bus.motors:
            features[f"{motor}.pos"] = float
            features[f"{motor}.vel"] = float
            features[f"{motor}.torque"] = float
        return features

    @property
    def feedback_features(self) -> Dict[str, type]:
        """Feedback features (not implemented for OpenArms)."""
        return {}

    @property
    def is_connected(self) -> bool:
        """Check if teleoperator is connected."""
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to the teleoperator.
        
        For manual control, we disable torque after connecting so the
        arm can be moved by hand.
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
        
        # Configure for manual control
        self.configure()
        
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """Check if teleoperator is calibrated."""
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        """
        Run calibration procedure for OpenArms leader.
        
        The calibration procedure:
        1. Disable torque (if not already disabled)
        2. Ask user to position arm in zero position (hanging with gripper closed)
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
        
        # Ensure torque is disabled for manual positioning
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
        
        # Write calibration and save to file
        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        
        print(f"\nCalibration complete and saved to {self.calibration_fpath}")

    def configure(self) -> None:
        """
        Configure motors for manual teleoperation.
        
        For manual control, we disable torque so the arm can be moved by hand.
        """
        if self.config.manual_control:
            # Disable torque for manual control
            logger.info("Disabling torque for manual control...")
            self.bus.disable_torque()
        else:
            # Configure motors normally
            self.bus.configure_motors()

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

    def send_feedback(self, feedback: Dict[str, float]) -> None:
        raise NotImplementedError("Feedback is not yet implemented for OpenArms leader.")

    def disconnect(self) -> None:
        """Disconnect from teleoperator."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        # For manual control, ensure torque is disabled before disconnecting
        if self.config.manual_control:
            try:
                self.bus.disable_torque()
            except Exception as e:
                logger.warning(f"Failed to disable torque during disconnect: {e}")
        
        # Disconnect from CAN bus
        self.bus.disconnect(disable_torque=False)  # Already disabled above if needed
        
        logger.info(f"{self} disconnected.")

