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

# XLeRobot integration based on
#
#   https://github.com/Astera-org/brainbot
#   https://github.com/Vector-Wangel/XLeRobot
#   https://github.com/bingogome/lerobot

from dataclasses import dataclass, field
from typing import Any

import draccus

from lerobot.configs import parser

from ...config import TeleoperatorConfig
from ...keyboard.configuration_keyboard import KeyboardRoverTeleopConfig
from ..sub_teleoperators.panthera_keyboard_ee import PantheraKeyboardEETeleopConfig
from ..sub_teleoperators.xlerobot_mount_gamepad.config import XLeRobotMountGamepadTeleopConfig

DEFAULT_BASE_KEYBOARD_CONFIG = {
    "linear_speed": 0.2,
    "angular_speed": 40.0,
    "speed_increment": 0.05,
    "turn_assist_ratio": 0.0,
    "angular_speed_ratio": 200.0,
    "min_linear_speed": 0.05,
    "min_angular_speed": 10.0,
}

PantheraArmKeyboardTeleopConfig = PantheraKeyboardEETeleopConfig


@TeleoperatorConfig.register_subclass("xlerobot_keyboard_composite")
@dataclass
class XLeRobotKeyboardCompositeConfig(TeleoperatorConfig):
    """Config for XLeRobot teleop with keyboard-controlled base and optional Panthera arm."""

    _comment: str | None = None
    config_file: str | None = None
    arm: dict[str, Any] | None = field(default_factory=dict)
    arm_side: str = "left"
    base: dict[str, Any] | None = field(default_factory=dict)
    mount: dict[str, Any] | None = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.config_file:
            self._load_from_config_file(self.config_file)

        if self.arm_side not in ("left", "right"):
            raise ValueError("xlerobot_keyboard_composite.arm_side must be either 'left' or 'right'.")

        self.arm_config = self._coerce_arm_config(self.arm)
        self.base_config = self._coerce_base_config(self.base)
        self.mount_config = self._coerce_mount_config(self.mount)

    def _load_from_config_file(self, config_file: str) -> None:
        cli_overrides = parser.get_cli_overrides("teleop") or []
        cli_overrides = [arg for arg in cli_overrides if not arg.startswith("--config_file=")]
        loaded = draccus.parse(config_class=TeleoperatorConfig, config_path=config_file, args=cli_overrides)
        self.__dict__.update(loaded.__dict__)
        self.config_file = config_file

    def _coerce_arm_config(
        self, value: PantheraArmKeyboardTeleopConfig | dict[str, Any] | None
    ) -> PantheraArmKeyboardTeleopConfig | None:
        if isinstance(value, PantheraKeyboardEETeleopConfig):
            return value
        if not value:
            return None

        data = dict(value)
        arm_type = str(data.pop("type", "panthera_keyboard_ee"))
        if arm_type == "panthera_keyboard_ee":
            return PantheraKeyboardEETeleopConfig(**data)
        raise ValueError(
            "xlerobot_keyboard_composite.arm.type must be 'panthera_keyboard_ee'."
        )

    def _coerce_base_config(
        self, value: KeyboardRoverTeleopConfig | dict[str, Any] | None
    ) -> KeyboardRoverTeleopConfig | None:
        if isinstance(value, KeyboardRoverTeleopConfig):
            return value
        if value is None:
            return None

        data = dict(DEFAULT_BASE_KEYBOARD_CONFIG)
        data.update(dict(value))
        data.pop("type", None)
        return KeyboardRoverTeleopConfig(**data)

    def _coerce_mount_config(
        self, value: XLeRobotMountGamepadTeleopConfig | dict[str, Any] | None
    ) -> XLeRobotMountGamepadTeleopConfig | None:
        if isinstance(value, XLeRobotMountGamepadTeleopConfig):
            return value
        if not value:
            return None
        data = dict(value)
        data.pop("type", None)
        return XLeRobotMountGamepadTeleopConfig(**data)
