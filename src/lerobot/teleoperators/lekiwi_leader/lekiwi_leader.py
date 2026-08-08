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
from functools import cached_property
from typing import Any

from lerobot.lerobot_types import RobotAction
from lerobot.utils.decorators import check_if_not_connected

from ..keyboard import KeyboardTeleop, KeyboardTeleopConfig
from ..so_leader import SOLeader, SOLeaderTeleopConfig
from ..teleoperator import Teleoperator
from .config_lekiwi_leader import LeKiwiLeaderConfig

logger = logging.getLogger(__name__)

BASE_FEATURES = ("x.vel", "y.vel", "theta.vel")


class LeKiwiLeader(Teleoperator):
    """Teleoperator for [LeKiwi](https://github.com/SIGRobotics-UIUC/LeKiwi).

    LeKiwi is a mobile manipulator, so it takes two devices to drive: a leader arm for the
    follower arm, and a keyboard for the holonomic base. This drives both from a single
    teleoperator, the way `bi_so_leader` drives two arms, so the recording and teleoperation
    loops see one device and need no special handling.
    """

    config_class = LeKiwiLeaderConfig
    name = "lekiwi_leader"

    def __init__(self, config: LeKiwiLeaderConfig):
        super().__init__(config)
        self.config = config

        # The arm keeps this teleoperator's own id, unsuffixed. `bi_so_leader` appends
        # `_left`/`_right` because its two arms would otherwise share a calibration file;
        # LeKiwi has one arm, so a suffix would only orphan an existing leader calibration
        # (`<calibration_dir>/<id>.json`) and force a needless recalibration.
        arm_config = SOLeaderTeleopConfig(
            id=config.id,
            calibration_dir=config.calibration_dir,
            port=config.arm_config.port,
            use_degrees=config.arm_config.use_degrees,
            num_read_retries=config.arm_config.num_read_retries,
        )

        self.arm = SOLeader(arm_config)
        # The keyboard needs no calibration, and sharing the arm's directory would make it
        # read the arm's calibration file as its own.
        self.keyboard = KeyboardTeleop(KeyboardTeleopConfig(id=config.id))
        self.speed_index = 0  # Start at slow

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            **{f"arm_{key}": value for key, value in self.arm.action_features.items()},
            **dict.fromkeys(BASE_FEATURES, float),
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        # The keyboard is best-effort: pynput cannot capture keys on Wayland or on a headless
        # machine, and there the arm should still be teleoperable on its own.
        return self.arm.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self.arm.is_calibrated

    def connect(self, calibrate: bool = True) -> None:
        self.arm.connect(calibrate)
        self.keyboard.connect()
        if not self.keyboard.is_connected:
            logger.warning(
                "LeKiwi's base keyboard is unavailable, so the base will not move. The arm "
                "remains teleoperable. See the KeyboardTeleop warning above for the cause."
            )

    def calibrate(self) -> None:
        self.arm.calibrate()

    def configure(self) -> None:
        self.arm.configure()

    def setup_motors(self) -> None:
        self.arm.setup_motors()

    def _base_action(self) -> RobotAction:
        """Turn the currently held keys into base velocities."""
        if not self.keyboard.is_connected:
            # Keep the action space complete so the base is commanded to hold still.
            return dict.fromkeys(BASE_FEATURES, 0.0)

        pressed_keys = self.keyboard.get_action()
        keys = self.config.teleop_keys

        if keys["speed_up"] in pressed_keys:
            self.speed_index = min(self.speed_index + 1, len(self.config.speed_levels) - 1)
        if keys["speed_down"] in pressed_keys:
            self.speed_index = max(self.speed_index - 1, 0)

        speed_setting = self.config.speed_levels[self.speed_index]
        xy_speed = speed_setting["xy"]  # m/s
        theta_speed = speed_setting["theta"]  # deg/s

        x_cmd = 0.0  # m/s forward/backward
        y_cmd = 0.0  # m/s lateral
        theta_cmd = 0.0  # deg/s rotation

        if keys["forward"] in pressed_keys:
            x_cmd += xy_speed
        if keys["backward"] in pressed_keys:
            x_cmd -= xy_speed
        if keys["left"] in pressed_keys:
            y_cmd += xy_speed
        if keys["right"] in pressed_keys:
            y_cmd -= xy_speed
        if keys["rotate_left"] in pressed_keys:
            theta_cmd += theta_speed
        if keys["rotate_right"] in pressed_keys:
            theta_cmd -= theta_speed

        return {"x.vel": x_cmd, "y.vel": y_cmd, "theta.vel": theta_cmd}

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        # The follower arm's joints are prefixed on LeKiwi, the base velocities are not.
        action = {f"arm_{key}": value for key, value in self.arm.get_action().items()}
        action.update(self._base_action())
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        self.arm.disconnect()
        if self.keyboard.is_connected:
            self.keyboard.disconnect()
