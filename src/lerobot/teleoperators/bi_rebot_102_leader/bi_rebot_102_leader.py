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

from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..rebot_102_leader import RebotArm102Leader, RebotArm102LeaderTeleopConfig
from ..teleoperator import Teleoperator
from .config_bi_rebot_102_leader import BiRebotArm102LeaderConfig

logger = logging.getLogger(__name__)


class BiRebotArm102Leader(Teleoperator):
    """Bimanual Seeed Studio StarArm102 / reBot Arm 102 leader.

    Composes two single-arm :class:`RebotArm102Leader` instances. Action keys of
    each arm are namespaced with a ``left_`` / ``right_`` prefix, so a bimanual
    leader can teleoperate a bimanual reBot B601 follower.
    """

    config_class = BiRebotArm102LeaderConfig
    name = "bi_rebot_102_leader"

    def __init__(self, config: BiRebotArm102LeaderConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = RebotArm102LeaderTeleopConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_config.port,
            baudrate=config.left_arm_config.baudrate,
            joint_ids=config.left_arm_config.joint_ids,
            joint_directions=config.left_arm_config.joint_directions,
            joint_ranges=config.left_arm_config.joint_ranges,
        )

        right_arm_config = RebotArm102LeaderTeleopConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_config.port,
            baudrate=config.right_arm_config.baudrate,
            joint_ids=config.right_arm_config.joint_ids,
            joint_directions=config.right_arm_config.joint_directions,
            joint_ranges=config.right_arm_config.joint_ranges,
        )

        self.left_arm = RebotArm102Leader(left_arm_config)
        self.right_arm = RebotArm102Leader(right_arm_config)

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            **{f"left_{k}": v for k, v in self.left_arm.action_features.items()},
            **{f"right_{k}": v for k, v in self.right_arm.action_features.items()},
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    @check_if_already_connected
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

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        action_dict = {}
        action_dict.update({f"left_{k}": v for k, v in self.left_arm.get_action().items()})
        action_dict.update({f"right_{k}": v for k, v in self.right_arm.get_action().items()})
        return action_dict

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError("Feedback is not implemented for the reBot Arm 102 leader.")

    @check_if_not_connected
    def disconnect(self) -> None:
        self.left_arm.disconnect()
        self.right_arm.disconnect()
