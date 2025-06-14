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
from typing import Any

from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.motors import Motor, MotorNormMode
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
        # This teleoperator assumes pre-calibrated leader arms.
        logger.info(f"{self} assumes pre-calibrated leader arms. Skipping calibration.")
        pass

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