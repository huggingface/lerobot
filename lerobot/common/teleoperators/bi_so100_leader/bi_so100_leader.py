#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.common.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)

from ..teleoperator import Teleoperator
from .config_bi_so100_leader import BiSO100LeaderConfig

logger = logging.getLogger(__name__)


class BiSO100Leader(Teleoperator):
    """
    Bimanual SO-100 Leader Arms - manages two SO-100 leader arms for bimanual teleoperation
    """

    config_class = BiSO100LeaderConfig
    name = "bi_so100_leader"

    def __init__(self, config: BiSO100LeaderConfig):
        super().__init__(config)
        self.config = config

        # Create left arm bus
        self.left_bus = FeetechMotorsBus(
            port=self.config.left_port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
                "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
                "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
                "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
                "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self._get_left_calibration(),
        )

        # Create right arm bus
        self.right_bus = FeetechMotorsBus(
            port=self.config.right_port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
                "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
                "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
                "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
                "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self._get_right_calibration(),
        )

    def _get_left_calibration(self):
        """Load calibration for left arm from existing so100_leader calibration"""
        from pathlib import Path

        import draccus

        left_calibration_path = (
            Path("/home/*/.cache/huggingface/lerobot/calibration/teleoperators/so100_leader")
            / f"{self.config.left_id}.json"
        )
        if left_calibration_path.exists():
            try:
                with open(left_calibration_path) as f, draccus.config_type("json"):
                    return draccus.load(dict[str, MotorCalibration], f)
            except Exception as e:
                logger.warning(f"Failed to load left arm calibration: {e}")
        return None

    def _get_right_calibration(self):
        """Load calibration for right arm from existing so100_leader calibration"""
        from pathlib import Path

        import draccus

        right_calibration_path = (
            Path("/home/*/.cache/huggingface/lerobot/calibration/teleoperators/so100_leader")
            / f"{self.config.right_id}.json"
        )
        if right_calibration_path.exists():
            try:
                with open(right_calibration_path) as f, draccus.config_type("json"):
                    return draccus.load(dict[str, MotorCalibration], f)
            except Exception as e:
                logger.warning(f"Failed to load right arm calibration: {e}")
        return None

    @property
    def action_features(self) -> dict[str, type]:
        action_ft = {}
        for motor in self.left_bus.motors:
            action_ft[f"left_{motor}.pos"] = float
        for motor in self.right_bus.motors:
            action_ft[f"right_{motor}.pos"] = float
        return action_ft

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.left_bus.is_connected and self.right_bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Connect both buses
        self.left_bus.connect()
        self.right_bus.connect()

        if not self.is_calibrated and calibrate:
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        left_calibrated = self.left_bus.is_calibrated and self._get_left_calibration() is not None
        right_calibrated = self.right_bus.is_calibrated and self._get_right_calibration() is not None
        return left_calibrated and right_calibrated

    def calibrate(self) -> None:
        logger.info(f"\nLoading existing calibrations for {self}")

        # Load calibrations from existing files
        left_calibration = self._get_left_calibration()
        right_calibration = self._get_right_calibration()

        if left_calibration is None:
            raise ValueError(
                f"No calibration found for left arm (ID: {self.config.left_id}). "
                "Please ensure calibration exists at /home/*/.cache/huggingface/lerobot/calibration/teleoperators/so100_leader/"
            )

        if right_calibration is None:
            raise ValueError(
                f"No calibration found for right arm (ID: {self.config.right_id}). "
                "Please ensure calibration exists at /home/*/.cache/huggingface/lerobot/calibration/teleoperators/so100_leader/"
            )

        # Write calibrations to both buses
        self.left_bus.write_calibration(left_calibration)
        self.right_bus.write_calibration(right_calibration)

        logger.info("Successfully loaded calibrations for both arms")

    def configure(self) -> None:
        # Configure left arm
        self.left_bus.disable_torque()
        self.left_bus.configure_motors()
        for motor in self.left_bus.motors:
            self.left_bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        # Configure right arm
        self.right_bus.disable_torque()
        self.right_bus.configure_motors()
        for motor in self.right_bus.motors:
            self.right_bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    def setup_motors(self) -> None:
        print("Setting up left arm motors:")
        for motor in reversed(self.left_bus.motors):
            input(f"Connect the controller board to the left '{motor}' motor only and press enter.")
            self.left_bus.setup_motor(motor)
            print(f"Left '{motor}' motor id set to {self.left_bus.motors[motor].id}")

        print("Setting up right arm motors:")
        for motor in reversed(self.right_bus.motors):
            input(f"Connect the controller board to the right '{motor}' motor only and press enter.")
            self.right_bus.setup_motor(motor)
            print(f"Right '{motor}' motor id set to {self.right_bus.motors[motor].id}")

    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()

        # Read from both arms
        left_action = self.left_bus.sync_read("Present_Position")
        right_action = self.right_bus.sync_read("Present_Position")

        # Combine actions with prefixes
        action = {}
        for motor, val in left_action.items():
            action[f"left_{motor}.pos"] = val
        for motor, val in right_action.items():
            action[f"right_{motor}.pos"] = val

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            DeviceNotConnectedError(f"{self} is not connected.")

        self.left_bus.disconnect()
        self.right_bus.disconnect()
        logger.info(f"{self} disconnected.")
