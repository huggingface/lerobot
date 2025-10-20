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

from __future__ import annotations

from functools import cached_property
from typing import Any, Dict

from ..teleoperator import Teleoperator
from ..bi_joycon.bi_joycon import BiJoycon
from ..lekiwi_base_gamepad.teleop_lekiwi_base_gamepad import LeKiwiBaseTeleop
from ..xlerobot_mount_idle.teleop import XLeRobotMountIdle
from .config import XLeRobotJoyconGamepadConfig


class XLeRobotJoyconGamepad(Teleoperator):
    """Composite teleoperator: JoyCon arms + gamepad base for XLeRobot."""

    config_class = XLeRobotJoyconGamepadConfig
    name = "xlerobot_joycon_gamepad"

    def __init__(self, config: XLeRobotJoyconGamepadConfig):
        super().__init__(config)
        self.config = config
        self.arm_teleop = BiJoycon(config.arms_config)
        self.base_teleop = LeKiwiBaseTeleop(config.base_config)
        self.mount_teleop = XLeRobotMountIdle(config.mount_config)

    @cached_property
    def action_features(self) -> dict[str, type]:
        features: dict[str, type] = {}
        features.update(self.arm_teleop.action_features)
        features.update(self.base_teleop.action_features)
        features.update(self.mount_teleop.action_features)
        return features

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return self.arm_teleop.feedback_features

    @property
    def is_connected(self) -> bool:
        return (
            self.arm_teleop.is_connected
            and self.base_teleop.is_connected
            and self.mount_teleop.is_connected
        )

    def connect(self, calibrate: bool = True) -> None:
        self.arm_teleop.connect(calibrate=calibrate)
        self.base_teleop.connect(calibrate=calibrate)
        self.mount_teleop.connect(calibrate=calibrate)

    def disconnect(self) -> None:
        self.arm_teleop.disconnect()
        self.base_teleop.disconnect()
        self.mount_teleop.disconnect()

    def calibrate(self) -> None:
        self.arm_teleop.calibrate()
        self.mount_teleop.calibrate()

    def configure(self) -> None:
        self.arm_teleop.configure()
        self.base_teleop.configure()
        self.mount_teleop.configure()

    def get_action(self) -> Dict[str, float]:
        action = dict(self.arm_teleop.get_action())
        action.update(self.base_teleop.get_action())
        action.update(self.mount_teleop.get_action())
        return action

    def send_feedback(self, feedback: Dict[str, float]) -> None:
        self.arm_teleop.send_feedback(feedback)

    @property
    def is_calibrated(self) -> bool:
        return True
