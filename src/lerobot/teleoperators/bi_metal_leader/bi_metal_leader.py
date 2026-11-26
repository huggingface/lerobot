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

from functools import cached_property

from lerobot.lerobot_types import RobotAction
from lerobot.robots.bi_metal_follower import BiMetalFollower, BiMetalFollowerConfig
from lerobot.utils.decorators import check_if_not_connected

from ..teleoperator import Teleoperator
from .config_bi_metal_leader import BiMetalLeaderConfig


class BiMetalLeader(Teleoperator):
    """Two Metal leaders with `left_` and `right_` prefixed actions, for a bimanual Metal follower.

    It is a `BiMetalFollower` that never receives an action, so both arms stay in float mode. As for the
    follower, neither arm gets torque until both are in their initial position, a fault on either stops both,
    and disconnecting attempts both.

    Args:
        config (`BiMetalLeaderConfig`):
            Settings of both arms.
    """

    config_class = BiMetalLeaderConfig
    name = "bi_metal_leader"

    def __init__(self, config: BiMetalLeaderConfig) -> None:
        super().__init__(config)
        self.config = config
        self.arms = BiMetalFollower(
            BiMetalFollowerConfig(
                id=config.id,
                calibration_dir=config.calibration_dir,
                left_arm_config=config.left_arm_config,
                right_arm_config=config.right_arm_config,
            )
        )

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self.arms.action_features

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.arms.is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        """Connect both arms and let them float from their initial positions.

        Args:
            calibrate (`bool`, *optional*, defaults to `True`):
                Unused: the joint zeros are stored in the motors.
        """
        self.arms.connect(calibrate)

    def calibrate(self) -> None:
        """Log where the zeros are: Metal keeps them in its motors."""
        self.arms.calibrate()

    def configure(self) -> None:
        """Nothing to configure: the motors run in MIT mode."""

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        observation = self.arms.get_observation()
        return {key: observation[key] for key in self.action_features}

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError("Feedback is not implemented for the Metal leader.")

    def disconnect(self) -> None:
        """Return both arms to their start poses and stop them, attempting both even if one fails."""
        self.arms.disconnect()
