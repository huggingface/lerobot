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
from functools import cached_property

from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader

from ..teleoperator import Teleoperator
from .config_bi_so101_leader import BiSO101LeaderConfig

logger = logging.getLogger(__name__)


class BiSO101Leader(Teleoperator):
    """
    [Bimanual SO-101 Leader Arms](https://github.com/TheRobotStudio/SO-ARM100) designed by TheRobotStudio
    This bimanual leader arm uses two SO-101 leader arms for teleoperation of dual-arm robots.
    """

    config_class = BiSO101LeaderConfig
    name = "bi_so101_leader"

    def __init__(self, config: BiSO101LeaderConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = SO101LeaderConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=self.calibration_dir,  # Use parent's calibration_dir
            port=config.left_arm_port,
        )

        right_arm_config = SO101LeaderConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=self.calibration_dir,  # Use parent's calibration_dir
            port=config.right_arm_port,
        )

        self.left_arm = SO101Leader(left_arm_config)
        self.right_arm = SO101Leader(right_arm_config)

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"left_{motor}.pos": float for motor in self.left_arm.bus.motors} | {
            f"right_{motor}.pos": float for motor in self.right_arm.bus.motors
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    def get_action(self) -> dict[str, float]:
        action_dict = {}

        # Get left action and add prefix - optimized
        left_action = self.left_arm.get_action()
        for key, value in left_action.items():
            action_dict[f"left_{key}"] = value

        # Get right action and add prefix - optimized
        right_action = self.right_arm.get_action()
        for key, value in right_action.items():
            action_dict[f"right_{key}"] = value

        return action_dict

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # Optimized feedback parsing - single pass through dictionary
        left_feedback = {}
        right_feedback = {}
        
        for key, value in feedback.items():
            if key.startswith("left_"):
                left_feedback[key[5:]] = value  # Remove "left_" prefix (5 chars)
            elif key.startswith("right_"):
                right_feedback[key[6:]] = value  # Remove "right_" prefix (6 chars)

        if left_feedback:
            self.left_arm.send_feedback(left_feedback)
        if right_feedback:
            self.right_arm.send_feedback(right_feedback)

    def disconnect(self) -> None:
        self.left_arm.disconnect()
        self.right_arm.disconnect()