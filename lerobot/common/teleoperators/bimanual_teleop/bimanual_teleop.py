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
from lerobot.lerobot.common.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
from lerobot.lerobot.common.teleoperators.so101_leader.so101_leader import SO101Leader

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

        left_config = SO101LeaderConfig(
            port=self.config.left_port,
            id=self.config.left_id,
        )
        # Initialize left leader bus
        self.left_arm = SO101Leader(left_config)

        right_config = SO101LeaderConfig(
            port=self.config.right_port,
            id=self.config.right_id,
        )
        self.right_arm = SO101Leader(right_config)

    @property
    def action_features(self) -> dict[str, type]:
        left_motors = {f"left_{key}": value for key, value in self.left_arm.action_features.items()}
        right_motors = {f"right_{key}": value for key, value in self.right_arm.action_features.items()}
        return {**left_motors, **right_motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        # No haptic feedback for this teleoperator
        return {}

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.left_arm.connect()
        self.right_arm.connect()

        # Leaders are assumed to be pre-calibrated
        # if not self.is_calibrated and calibrate:
        #     self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        # Both leaders must be calibrated
        # For now, we assume they are pre-calibrated
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        """Calibrate both leader arms sequentially and save individual files."""
        pass
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
        self.left_arm.configure()
        self.right_arm.configure()

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} not connected.")

        left_action = self.left_arm.get_action()
        right_action = self.right_arm.get_action()

        prefixed_left_action = {f"left_{key}": value for key, value in left_action.items()}
        prefixed_right_action = {f"right_{key}": value for key, value in right_action.items()}

        return {**prefixed_left_action, **prefixed_right_action}

    def disconnect(self) -> None:
        if self.left_arm.is_connected:
            self.left_arm.disconnect()
        if self.right_arm.is_connected:
            self.right_arm.disconnect()
        logger.info(f"{self} disconnected.")
