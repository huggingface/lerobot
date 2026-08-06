#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from lerobot.lerobot_types import RobotAction
from lerobot.utils.bimanual import BimanualMixin
from lerobot.utils.decorators import check_if_not_connected

from ..openarm_leader import OpenArmLeader, OpenArmLeaderConfig
from ..teleoperator import Teleoperator
from .config_bi_openarm_leader import BiOpenArmLeaderConfig

logger = logging.getLogger(__name__)


class BiOpenArmLeader(BimanualMixin, Teleoperator):
    """A bimanual pair of [`~teleoperators.openarm_leader.OpenArmLeader`] arms."""

    config_class = BiOpenArmLeaderConfig
    name = "bi_openarm_leader"

    def __init__(self, config: BiOpenArmLeaderConfig):
        """Build the teleoperator from its configuration.

        Args:
            config (`BiOpenArmLeaderConfig`):
                The teleoperator's configuration. Its `left_arm_config` and `right_arm_config` determine
                what is connected on each side.
        """
        super().__init__(config)
        self.config = config

        left_arm_config = OpenArmLeaderConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_config.port,
            can_interface=config.left_arm_config.can_interface,
            use_can_fd=config.left_arm_config.use_can_fd,
            can_bitrate=config.left_arm_config.can_bitrate,
            can_data_bitrate=config.left_arm_config.can_data_bitrate,
            motor_config=config.left_arm_config.motor_config,
            manual_control=config.left_arm_config.manual_control,
            use_velocity_and_torque=config.left_arm_config.use_velocity_and_torque,
            position_kd=config.left_arm_config.position_kd,
            position_kp=config.left_arm_config.position_kp,
        )

        right_arm_config = OpenArmLeaderConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_config.port,
            can_interface=config.right_arm_config.can_interface,
            use_can_fd=config.right_arm_config.use_can_fd,
            can_bitrate=config.right_arm_config.can_bitrate,
            can_data_bitrate=config.right_arm_config.can_data_bitrate,
            motor_config=config.right_arm_config.motor_config,
            manual_control=config.right_arm_config.manual_control,
            use_velocity_and_torque=config.right_arm_config.use_velocity_and_torque,
            position_kd=config.right_arm_config.position_kd,
            position_kp=config.right_arm_config.position_kp,
        )

        self.left_arm = OpenArmLeader(left_arm_config)
        self.right_arm = OpenArmLeader(right_arm_config)

    @cached_property
    def action_features(self) -> dict[str, type]:
        """See [`~teleoperators.Teleoperator.action_features`].

        Merges both arms' features, each key prefixed with `left_` or `right_`.
        """
        left_arm_features = self.left_arm.action_features
        right_arm_features = self.right_arm.action_features

        return {
            **{f"left_{k}": v for k, v in left_arm_features.items()},
            **{f"right_{k}": v for k, v in right_arm_features.items()},
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        """See [`~teleoperators.Teleoperator.feedback_features`].

        Always empty: feedback is not implemented for the OpenArm leader.
        """
        return {}

    def setup_motors(self) -> None:
        """Not supported: raises `NotImplementedError`.

        Motor ID configuration for CAN motors is typically done via manufacturer tools rather than through
        LeRobot.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "Motor ID configuration is typically done via manufacturer tools for CAN motors."
        )

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """See [`~teleoperators.Teleoperator.get_action`].

        Merges both arms' actions, each key prefixed with `left_` or `right_`.
        """
        action_dict = {}

        # Add "left_" prefix
        left_action = self.left_arm.get_action()
        action_dict.update({f"left_{key}": value for key, value in left_action.items()})

        # Add "right_" prefix
        right_action = self.right_arm.get_action()
        action_dict.update({f"right_{key}": value for key, value in right_action.items()})

        return action_dict

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """Not supported: raises `NotImplementedError`.

        Args:
            feedback (`dict[str, float]`):
                Unused.

        Raises:
            NotImplementedError: Always.
        """
        # TODO: Implement force feedback
        raise NotImplementedError
