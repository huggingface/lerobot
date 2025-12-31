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

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_openarms_mini import OpenArmsMiniConfig

logger = logging.getLogger(__name__)


class OpenArmsMini(Teleoperator):
    """
    OpenArms Mini Teleoperator with dual Feetech-based arms (8 motors per arm).
    Each arm has 7 joints plus a gripper, using the same DOF as OpenArms.
    """

    config_class = OpenArmsMiniConfig
    name = "openarms_mini"

    def __init__(self, config: OpenArmsMiniConfig):
        super().__init__(config)
        self.config = config
        
        # Use degrees mode for all motors
        norm_mode_body = MotorNormMode.DEGREES
        
        # Right arm motors (8 motors: joint_1 to joint_7 + gripper)
        motors_right = {
            "joint_1": Motor(1, "sts3215", norm_mode_body),
            "joint_2": Motor(2, "sts3215", norm_mode_body),
            "joint_3": Motor(3, "sts3215", norm_mode_body),
            "joint_4": Motor(4, "sts3215", norm_mode_body),
            "joint_5": Motor(5, "sts3215", norm_mode_body),
            "joint_6": Motor(6, "sts3215", norm_mode_body),
            "joint_7": Motor(7, "sts3215", norm_mode_body),
            "gripper": Motor(8, "sts3215", MotorNormMode.RANGE_0_100),
        }
        
        # Left arm motors (8 motors: joint_1 to joint_7 + gripper)
        # Note: Left arm uses IDs 11-18 to avoid conflicts if on same bus
        motors_left = {
            "joint_1": Motor(1, "sts3215", norm_mode_body),
            "joint_2": Motor(2, "sts3215", norm_mode_body),
            "joint_3": Motor(3, "sts3215", norm_mode_body),
            "joint_4": Motor(4, "sts3215", norm_mode_body),
            "joint_5": Motor(5, "sts3215", norm_mode_body),
            "joint_6": Motor(6, "sts3215", norm_mode_body),
            "joint_7": Motor(7, "sts3215", norm_mode_body),
            "gripper": Motor(8, "sts3215", MotorNormMode.RANGE_0_100),
        }
        
        # Initialize Feetech motor buses for each arm
        self.bus_right = FeetechMotorsBus(
            port=self.config.port_right,
            motors=motors_right,
            calibration={k.replace("right_", ""): v for k, v in (self.calibration or {}).items() if k.startswith("right_")},
        )
        
        self.bus_left = FeetechMotorsBus(
            port=self.config.port_left,
            motors=motors_left,
            calibration={k.replace("left_", ""): v for k, v in (self.calibration or {}).items() if k.startswith("left_")},
        )

    @property
    def action_features(self) -> dict[str, type]:
        """Action features include positions for all motors (16 total: 8 per arm)."""
        features = {}
        # Right arm motors
        for motor in self.bus_right.motors:
            features[f"right_{motor}.pos"] = float
        # Left arm motors
        for motor in self.bus_left.motors:
            features[f"left_{motor}.pos"] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        """No feedback features for now."""
        return {}

    @property
    def is_connected(self) -> bool:
        """Check if both arms are connected."""
        return self.bus_right.is_connected and self.bus_left.is_connected

    def connect(self, calibrate: bool = True) -> None:
        """Connect to both arms and optionally calibrate."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Connect to both buses
        logger.info(f"Connecting right arm on {self.config.port_right}...")
        self.bus_right.connect()
        logger.info(f"Connecting left arm on {self.config.port_left}...")
        self.bus_left.connect()
        
        # Calibrate if requested (always prompt user)
        if calibrate:
            self.calibrate()

        # Configure motors
        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """Check if both arms are calibrated."""
        return self.bus_right.is_calibrated and self.bus_left.is_calibrated

    def calibrate(self) -> None:
        """
        Run calibration procedure for OpenArms Mini.
        
        The calibration procedure:
        1. Disable torque
        2. Ask user to position arms in hanging position with grippers closed
        3. Set this as zero position
        4. Set fixed range of -90° to +90° for all joints (0-100 for gripper)
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
                # Split calibration for each bus
                cal_right = {k.replace("right_", ""): v for k, v in self.calibration.items() if k.startswith("right_")}
                cal_left = {k.replace("left_", ""): v for k, v in self.calibration.items() if k.startswith("left_")}
                self.bus_right.write_calibration(cal_right)
                self.bus_left.write_calibration(cal_left)
                return

        logger.info(f"\nRunning calibration for {self}")
        
        # Calibrate each arm separately
        self._calibrate_arm("right", self.bus_right)
        self._calibrate_arm("left", self.bus_left)
        
        self._save_calibration()
        print(f"\nCalibration complete and saved to {self.calibration_fpath}")

    def _calibrate_arm(self, arm_name: str, bus: FeetechMotorsBus) -> None:
        """Calibrate a single arm with Feetech motors."""
        logger.info(f"\n=== Calibrating {arm_name.upper()} arm ===")
        
        # Disable torque for manual positioning
        bus.disable_torque()
        
        # Set Phase to 12 for all motors
        logger.info(f"Setting Phase to 12 for all motors in {arm_name.upper()} arm...")
        for motor in bus.motors:
            bus.write("Phase", motor, 12)
        logger.info(f"Phase set to 12 for all motors in {arm_name.upper()} arm")
        
        # Set all motors to position mode
        for motor in bus.motors:
            bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
        
        # Step 1: Set zero position
        input(
            f"\nCalibration: Zero Position ({arm_name.upper()} arm)\n"
            "Position the arm in the following configuration:\n"
            "  - Arm hanging straight down\n"
            "  - Gripper closed\n"
            "Press ENTER when ready..."
        )
        
        # Set current position as zero (half-turn homing)
        homing_offsets = bus.set_half_turn_homings()
        logger.info(f"{arm_name.capitalize()} arm zero position set.")
        
        # Step 2: Set ranges for joints and gripper
        print(f"\nSetting motor ranges for {arm_name.upper()} arm\n")
        
        # Create calibration data with full motor ranges
        if self.calibration is None:
            self.calibration = {}
        
        # Get motor resolution
        motor_resolution = bus.model_resolution_table[list(bus.motors.values())[0].model]
        max_res = motor_resolution - 1
        
        for motor_name, motor in bus.motors.items():
            # Prefix motor name with arm name for storage
            prefixed_name = f"{arm_name}_{motor_name}"
            
            if motor_name == "gripper":
                # Interactive calibration for gripper
                input(
                    f"\nGripper Calibration ({arm_name.upper()} arm)\n"
                    f"Step 1: CLOSE the gripper fully\n"
                    f"Press ENTER when gripper is closed..."
                )
                closed_pos = bus.read("Present_Position", motor_name, normalize=False)
                logger.info(f"  Gripper closed position recorded: {closed_pos}")
                
                input(
                    f"\nStep 2: OPEN the gripper fully\n"
                    f"Press ENTER when gripper is fully open..."
                )
                open_pos = bus.read("Present_Position", motor_name, normalize=False)
                logger.info(f"  Gripper open position recorded: {open_pos}")
                
                # For RANGE_0_100: range_min maps to 0 (closed), range_max maps to 100 (open)
                # If gripper motor direction is reversed (closed > open), we need to swap and use drive_mode=1
                if closed_pos < open_pos:
                    # Normal direction: 0=closed, 100=open
                    range_min = int(closed_pos)
                    range_max = int(open_pos)
                    drive_mode = 0
                else:
                    # Reversed direction: swap so min < max, and set drive_mode=1 to reverse
                    range_min = int(open_pos)
                    range_max = int(closed_pos)
                    drive_mode = 1
                
                logger.info(f"  {prefixed_name}: range set to [{range_min}, {range_max}] (0=closed, 100=open, drive_mode={drive_mode})")
            else:
                # Use full motor range for joint motors (will use degrees normalization)
                range_min = 0
                range_max = max_res
                drive_mode = 0  # Normal direction for non-gripper motors
                logger.info(f"  {prefixed_name}: range set to [0, {max_res}] (full motor range)")
            
            self.calibration[prefixed_name] = MotorCalibration(
                id=motor.id,
                drive_mode=drive_mode,
                homing_offset=homing_offsets[motor_name],
                range_min=range_min,
                range_max=range_max,
            )
        
        # Write calibration to this arm's motors
        cal_for_bus = {k.replace(f"{arm_name}_", ""): v for k, v in self.calibration.items() if k.startswith(f"{arm_name}_")}
        bus.write_calibration(cal_for_bus)

    def configure(self) -> None:
        """Configure motors with appropriate settings."""
        # Configure right arm
        self.bus_right.disable_torque()
        self.bus_right.configure_motors()
        for motor in self.bus_right.motors:
            self.bus_right.write("Operating_Mode", motor, OperatingMode.POSITION.value)
        
        # Configure left arm
        self.bus_left.disable_torque()
        self.bus_left.configure_motors()
        for motor in self.bus_left.motors:
            self.bus_left.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    def setup_motors(self) -> None:
        """Setup motor IDs for both arms."""
        print("\nSetting up RIGHT arm motors...")
        for motor in reversed(self.bus_right.motors):
            input(f"Connect the controller board to the RIGHT '{motor}' motor only and press enter.")
            self.bus_right.setup_motor(motor)
            print(f"RIGHT '{motor}' motor id set to {self.bus_right.motors[motor].id}")
        
        print("\nSetting up LEFT arm motors...")
        for motor in reversed(self.bus_left.motors):
            input(f"Connect the controller board to the LEFT '{motor}' motor only and press enter.")
            self.bus_left.setup_motor(motor)
            print(f"LEFT '{motor}' motor id set to {self.bus_left.motors[motor].id}")

    def get_action(self) -> dict[str, float]:
        """Get current action from both arms (read positions from all motors)."""
        start = time.perf_counter()
        
        # Motors to flip (invert direction) - different for each arm
        right_motors_to_flip = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5"]
        left_motors_to_flip = ["joint_1", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"]
        
        # Joint mapping: teleop joint -> robot joint (swap 6 and 7)
        joint_map = {"joint_6": "joint_7", "joint_7": "joint_6"}
        
        # Read positions from both arms
        right_positions = self.bus_right.sync_read("Present_Position")
        left_positions = self.bus_left.sync_read("Present_Position")
        
        # Combine into single action dict with arm prefixes, flip and map joints
        action = {}
        for motor, val in right_positions.items():
            robot_joint = joint_map.get(motor, motor)
            if motor in right_motors_to_flip:
                action[f"right_{robot_joint}.pos"] = -val
            else:
                action[f"right_{robot_joint}.pos"] = val
        for motor, val in left_positions.items():
            robot_joint = joint_map.get(motor, motor)
            if motor in left_motors_to_flip:
                action[f"left_{robot_joint}.pos"] = -val
            else:
                action[f"left_{robot_joint}.pos"] = val
        
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """
        Send position feedback to teleoperator motors.
        
        Args:
            feedback: Dictionary with motor positions (e.g., "right_joint_1.pos", "left_joint_2.pos")
        """
        # Motors to flip (invert direction) -> matches get_action()
        right_motors_to_flip = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5"]
        left_motors_to_flip = ["joint_1", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"]
        
        # Joint mapping: robot joint -> teleop joint (swap 6 and 7)
        joint_map = {"joint_6": "joint_7", "joint_7": "joint_6"}
        
        # Split feedback by arm and flip motors as needed
        right_positions = {}
        left_positions = {}
        
        for key, val in feedback.items():
            if key.endswith(".pos"):
                motor_name = key.removesuffix(".pos")
                if motor_name.startswith("right_"):
                    base_name = motor_name.removeprefix("right_")
                    # Apply joint mapping (swap 6↔7)
                    teleop_joint = joint_map.get(base_name, base_name)
                    if teleop_joint in right_motors_to_flip:
                        right_positions[teleop_joint] = -val
                    else:
                        right_positions[teleop_joint] = val
                elif motor_name.startswith("left_"):
                    base_name = motor_name.removeprefix("left_")
                    # Apply joint mapping (swap 6↔7)
                    teleop_joint = joint_map.get(base_name, base_name)
                    if teleop_joint in left_motors_to_flip:
                        left_positions[teleop_joint] = -val
                    else:
                        left_positions[teleop_joint] = val
        
        # Write positions to both arms
        if right_positions:
            self.bus_right.sync_write("Goal_Position", right_positions)
        if left_positions:
            self.bus_left.sync_write("Goal_Position", left_positions)
    
    def enable_torque(self) -> None:
        """Enable torque on both arms (for position tracking)."""
        self.bus_right.enable_torque()
        self.bus_left.enable_torque()
    
    def disable_torque(self) -> None:
        """Disable torque on both arms (for free movement)."""
        self.bus_right.disable_torque()
        self.bus_left.disable_torque()

    def disconnect(self) -> None:
        """Disconnect from both arms."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Disconnect both buses
        self.bus_right.disconnect()
        self.bus_left.disconnect()
        logger.info(f"{self} disconnected.")

