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

import json
import logging
from typing import Any

from lerobot.common.constants import HF_LEROBOT_CALIBRATION, TELEOPERATORS
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.common.motors.feetech import FeetechMotorsBus, OperatingMode

from ..teleoperator import Teleoperator
from .config_bimanual_teleop import BimanualTeleopConfig

logger = logging.getLogger(__name__)


class BimanualTeleop(Teleoperator):
    """
    Bimanual Teleoperator consisting of two SO-101 Leader Arms.
    """

    config_class = BimanualTeleopConfig
    name = "bimanual_teleop"

    def __init__(self, config: BimanualTeleopConfig):
        super().__init__(config)
        self.config = config

        if config.left_id and config.right_id:
            self._load_and_merge_arm_calibrations()

        # Motor normalization mode
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100

        # Initialize left leader bus
        self.left_bus = FeetechMotorsBus(
            port=self.config.left_port,
            motors={
                "left_shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "left_shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "left_elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "left_wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "left_wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "left_gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

        # Initialize right leader bus
        self.right_bus = FeetechMotorsBus(
            port=self.config.right_port,
            motors={
                "right_shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "right_shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "right_elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "right_wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "right_wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "right_gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

    def _load_and_merge_arm_calibrations(self):
        """Load individual arm calibrations and merge them with appropriate key prefixes."""
        self.calibration = {}
        single_arm_calibration_dir = HF_LEROBOT_CALIBRATION / TELEOPERATORS / "so101_leader"

        # Load and map left arm calibration
        left_calib_path = single_arm_calibration_dir / f"{self.config.left_id}.json"
        if left_calib_path.is_file():
            with open(left_calib_path) as f:
                left_calib_data = json.load(f)
            for key, value in left_calib_data.items():
                self.calibration[f"left_{key}"] = MotorCalibration(**value)
        else:
            logger.warning(f"Calibration file not found for left arm: {left_calib_path}")

        # Load and map right arm calibration
        right_calib_path = single_arm_calibration_dir / f"{self.config.right_id}.json"
        if right_calib_path.is_file():
            with open(right_calib_path) as f:
                right_calib_data = json.load(f)
            for key, value in right_calib_data.items():
                self.calibration[f"right_{key}"] = MotorCalibration(**value)
        else:
            logger.warning(f"Calibration file not found for right arm: {right_calib_path}")

    @property
    def action_features(self) -> dict[str, type]:
        left_motors = {f"{motor}.pos": float for motor in self.left_bus.motors}
        right_motors = {f"{motor}.pos": float for motor in self.right_bus.motors}
        return {**left_motors, **right_motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        # No haptic feedback for this teleoperator
        return {}

    @property
    def is_connected(self) -> bool:
        return self.left_bus.is_connected and self.right_bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.left_bus.connect()
        self.right_bus.connect()
        
        # Leaders are assumed to be pre-calibrated
        # if not self.is_calibrated and calibrate:
        #     self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        # Both leaders must be calibrated
        # For now, we assume they are pre-calibrated
        return self.left_bus.is_calibrated and self.right_bus.is_calibrated

    def calibrate(self) -> None:
        """Calibrate both leader arms sequentially and save individual files."""
        logger.info(f"\nRunning calibration of {self}")
        self.calibration = {}

        # Calibrate left leader
        logger.info("Calibrating LEFT leader arm...")
        self._calibrate_arm(self.left_bus, "left")

        # Calibrate right leader
        logger.info("Calibrating RIGHT leader arm...")
        self._calibrate_arm(self.right_bus, "right")

        # Save separate calibration files for each arm
        self._save_individual_arm_calibrations()
        print("Individual teleoperator arm calibration files saved.")

    def _calibrate_arm(self, bus: FeetechMotorsBus, arm_prefix: str) -> None:
        """Calibrate a single leader arm and store calibration data with a prefix."""
        bus.disable_torque()
        for motor in bus.motors:
            bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move {arm_prefix.upper()} leader arm to the middle of its range of motion and press ENTER....")
        homing_offsets = bus.set_half_turn_homings()

        print(
            f"Move all {arm_prefix.upper()} leader arm joints sequentially through their entire ranges "
            "of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = bus.record_ranges_of_motion()

        # Update calibration for this arm, adding a prefix to the keys
        for motor, m in bus.motors.items():
            motor_name_without_prefix = motor.removeprefix(f"{arm_prefix}_")
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        bus.write_calibration(self.calibration)

    def _save_individual_arm_calibrations(self):
        """Save separate JSON calibration files for each teleoperator arm."""
        single_arm_calibration_dir = HF_LEROBOT_CALIBRATION / TELEOPERATORS / "so101_leader"
        single_arm_calibration_dir.mkdir(parents=True, exist_ok=True)

        left_calib_data = {}
        right_calib_data = {}

        for key, calib_obj in self.calibration.items():
            if key.startswith("left_"):
                motor_name = key.removeprefix("left_")
                left_calib_data[motor_name] = calib_obj.to_dict()
            elif key.startswith("right_"):
                motor_name = key.removeprefix("right_")
                right_calib_data[motor_name] = calib_obj.to_dict()

        if self.config.left_id:
            left_path = single_arm_calibration_dir / f"{self.config.left_id}.json"
            with open(left_path, "w") as f:
                json.dump(left_calib_data, f, indent=4)
            print(f"Saved left teleoperator arm calibration to {left_path}")

        if self.config.right_id:
            right_path = single_arm_calibration_dir / f"{self.config.right_id}.json"
            with open(right_path, "w") as f:
                json.dump(right_calib_data, f, indent=4)
            print(f"Saved right teleoperator arm calibration to {right_path}")

    def configure(self) -> None:
        # Enable torque for both arms
        self.left_bus.enable_torque()
        self.right_bus.enable_torque()
        # Set operating mode to position control
        for motor in self.left_bus.motors:
            self.left_bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
        for motor in self.right_bus.motors:
            self.right_bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} not connected.")

        # Read positions from both leader arms
        left_action = self.left_bus.sync_read("Present_Position")
        right_action = self.right_bus.sync_read("Present_Position")
        
        # Combine actions into a single dictionary
        action = {f"{motor}.pos": val for motor, val in left_action.items()}
        action.update({f"{motor}.pos": val for motor, val in right_action.items()})
        
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        # No haptic feedback implemented
        pass

    def disconnect(self) -> None:
        if self.left_bus.is_connected:
            self.left_bus.disconnect()
        if self.right_bus.is_connected:
            self.right_bus.disconnect()
        logger.info(f"{self} disconnected.") 