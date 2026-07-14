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

from lerobot.types import RobotAction
from lerobot.utils.bimanual import BimanualMixin
from lerobot.utils.decorators import check_if_not_connected

from ..so_leader import SOLeader, SOLeaderTeleopConfig
from ..teleoperator import Teleoperator
from .config_bi_so_leader import BiSOLeaderConfig

logger = logging.getLogger(__name__)


class BiSOLeader(BimanualMixin, Teleoperator):
    """
    [Bimanual SO Leader Arms](https://github.com/TheRobotStudio/SO-ARM100) designed by TheRobotStudio
    """

    config_class = BiSOLeaderConfig
    name = "bi_so_leader"

    def __init__(self, config: BiSOLeaderConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = SOLeaderTeleopConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_config.port,
        )

        right_arm_config = SOLeaderTeleopConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_config.port,
        )

        self.left_arm = SOLeader(left_arm_config)
        self.right_arm = SOLeader(right_arm_config)

    @cached_property
    def action_features(self) -> dict[str, type]:
        left_arm_features = self.left_arm.action_features
        right_arm_features = self.right_arm.action_features

        return {
            **{f"left_{k}": v for k, v in left_arm_features.items()},
            **{f"right_{k}": v for k, v in right_arm_features.items()},
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        # Bimanual teleop has feedback (can be actuated for handover).
        # Return the same structure as action_features for consistency with left/right arms.
        left_arm_features = self.left_arm.feedback_features
        right_arm_features = self.right_arm.feedback_features

        return {
            **{f"left_{k}": v for k, v in left_arm_features.items()},
            **{f"right_{k}": v for k, v in right_arm_features.items()},
        }

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        action_dict = {}

        # Add "left_" prefix
        left_action = self.left_arm.get_action()
        action_dict.update({f"left_{key}": value for key, value in left_action.items()})

        # Add "right_" prefix
        right_action = self.right_arm.get_action()
        action_dict.update({f"right_{key}": value for key, value in right_action.items()})

        return action_dict

    def enable_torque(self) -> None:
        """Enable torque on both leader arms for smooth handover."""
        self.left_arm.enable_torque()
        self.right_arm.enable_torque()

    def disable_torque(self) -> None:
        """Disable torque on both leader arms to allow human control."""
        self.left_arm.disable_torque()
        self.right_arm.disable_torque()

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, float]) -> None:
        """Route bimanual feedback to left and right arms with proper prefix stripping.
        
        Receives feedback dict with keys like: left_shoulder_pan.pos, right_shoulder_pan.pos, ...
        Splits and routes to each arm by removing the prefix.
        
        This enables DAgger smooth handover: when transitioning from policy control to human
        intervention, both leader arms are commanded to the follower's current pose to avoid
        discontinuities.
        """
        # Split feedback by arm prefix
        left_feedback = {}
        right_feedback = {}

        for key, value in feedback.items():
            if key.startswith("left_"):
                # Strip "left_" prefix and pass to left arm
                stripped_key = key[5:]  # len("left_") == 5
                left_feedback[stripped_key] = value
            elif key.startswith("right_"):
                # Strip "right_" prefix and pass to right arm
                stripped_key = key[6:]  # len("right_") == 6
                right_feedback[stripped_key] = value

        # Send to each arm
        if left_feedback:
            self.left_arm.send_feedback(left_feedback)
        if right_feedback:
            self.right_arm.send_feedback(right_feedback)
