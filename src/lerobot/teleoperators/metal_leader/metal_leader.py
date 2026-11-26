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

"""A Metal arm used as a leader, built on the Metal follower."""

import logging
from copy import deepcopy
from dataclasses import fields
from functools import cached_property

from lerobot.lerobot_types import RobotAction
from lerobot.robots.metal_follower import MetalFollower, MetalFollowerConfig, MetalFollowerRobotConfig
from lerobot.utils import check_if_not_connected

from ..teleoperator import Teleoperator
from .config_metal_leader import MetalLeaderTeleopConfig

logger = logging.getLogger(__name__)


class MetalLeader(Teleoperator):
    """A Metal arm you move by hand, whose joint positions drive a Metal follower.

    It is a `MetalFollower` that never receives an action, so it stays in float mode.

    Args:
        config (`MetalLeaderTeleopConfig`):
            Hardware and control settings.
    """

    config_class = MetalLeaderTeleopConfig
    name = "metal_leader"

    def __init__(self, config: MetalLeaderTeleopConfig) -> None:
        super().__init__(config)
        self.config = config
        values = {field.name: deepcopy(getattr(config, field.name)) for field in fields(MetalFollowerConfig)}
        self.arm = MetalFollower(
            MetalFollowerRobotConfig(id=config.id, calibration_dir=config.calibration_dir, **values)
        )

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self.arm.action_features

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.arm.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self.arm.is_calibrated

    def connect(self, calibrate: bool = True) -> None:
        """Connect the arm and let it float from its initial position.

        Args:
            calibrate (`bool`, *optional*, defaults to `True`):
                Unused: the joint zeros are stored in the motors.
        """
        self.arm.connect(calibrate)

    def calibrate(self) -> None:
        """Log where the zeros are: Metal keeps them in its motors."""
        self.arm.calibrate()

    def configure(self) -> None:
        """Nothing to configure: the motors run in MIT mode."""

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """Return the measured joint positions and gripper opening, in the follower's action units."""
        observation = self.arm.get_observation()
        return {key: observation[key] for key in self.action_features}

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError("Feedback is not implemented for the Metal leader.")

    def disconnect(self) -> None:
        """Return to the start pose, then stop control and release the CAN interface."""
        self.arm.disconnect()
