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

from dataclasses import dataclass, field
from typing import Any

from ..config import TeleoperatorConfig
from ..bi_joycon.config_bi_joycon import BiJoyconConfig
from ..lekiwi_base_gamepad.config_lekiwi_base_gamepad import LeKiwiBaseTeleopConfig
from ..xlerobot_mount_gamepad.config import XLeRobotMountGamepadTeleopConfig


@TeleoperatorConfig.register_subclass("xlerobot_joycon_gamepad")
@dataclass
class XLeRobotJoyconGamepadConfig(TeleoperatorConfig):
    arms: dict[str, Any] = field(default_factory=dict)
    base: dict[str, Any] = field(default_factory=dict)
    mount: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arms_cfg = self.arms if isinstance(self.arms, BiJoyconConfig) else BiJoyconConfig(**self.arms)
        base_cfg = self.base if isinstance(self.base, LeKiwiBaseTeleopConfig) else LeKiwiBaseTeleopConfig(**self.base)
        mount_cfg = self.mount if isinstance(self.mount, XLeRobotMountGamepadTeleopConfig) else XLeRobotMountGamepadTeleopConfig(**self.mount)
        self.arms_config = arms_cfg
        self.base_config = base_cfg
        self.mount_config = mount_cfg
