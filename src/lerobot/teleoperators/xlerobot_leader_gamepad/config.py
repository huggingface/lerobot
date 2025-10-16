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
from ..bi_so101_leader.config_bi_so101_leader import BiSO101LeaderConfig
from ..lekiwi_base_gamepad.config_lekiwi_base_gamepad import LeKiwiBaseTeleopConfig
from ..xlerobot_mount_idle.config import XLeRobotMountIdleConfig


@TeleoperatorConfig.register_subclass("xlerobot_leader_gamepad")
@dataclass
class XLeRobotLeaderGamepadConfig(TeleoperatorConfig):
    arms: dict[str, Any] = field(default_factory=dict)
    base: dict[str, Any] = field(default_factory=dict)
    mount: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arms_cfg = self.arms if isinstance(self.arms, BiSO101LeaderConfig) else BiSO101LeaderConfig(**self.arms)
        base_cfg = self.base if isinstance(self.base, LeKiwiBaseTeleopConfig) else LeKiwiBaseTeleopConfig(**self.base)
        mount_cfg = self.mount if isinstance(self.mount, XLeRobotMountIdleConfig) else XLeRobotMountIdleConfig(**self.mount)
        self.arms_config = arms_cfg
        self.base_config = base_cfg
        self.mount_config = mount_cfg
