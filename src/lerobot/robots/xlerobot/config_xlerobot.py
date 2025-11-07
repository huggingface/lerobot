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

from lerobot.cameras import CameraConfig

from ..bi_so101_follower.config_bi_so101_follower import BiSO101FollowerConfig
from ..config import RobotConfig
from ..lekiwi_base.config import LeKiwiBaseConfig
from ..xlerobot_mount.config import XLeRobotMountConfig


@RobotConfig.register_subclass("xlerobot")
@dataclass
class XLeRobotConfig(RobotConfig):
    BASE_TYPE_LEKIWI = "lekiwi_base"
    BASE_TYPE_BIWHEEL = "biwheel_base"  # TODO: add config/dataclass once biwheel base exists.

    arms: dict[str, Any] = field(default_factory=dict)
    base: dict[str, Any] = field(default_factory=dict)
    mount: dict[str, Any] = field(default_factory=dict)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    base_type: str = BASE_TYPE_LEKIWI

    def __post_init__(self) -> None:
        super().__post_init__()

        if isinstance(self.arms, BiSO101FollowerConfig):
            arms_cfg = self.arms
        else:
            arms_cfg = BiSO101FollowerConfig(**self.arms)
        base_type = self.base_type or self.BASE_TYPE_LEKIWI
        if base_type == self.BASE_TYPE_LEKIWI:
            if isinstance(self.base, LeKiwiBaseConfig):
                base_cfg = self.base
            else:
                base_cfg = LeKiwiBaseConfig(**self.base)
        elif base_type == self.BASE_TYPE_BIWHEEL:
            # TODO: Replace with the biwheel base config once it is implemented.
            base_cfg = self.base
        else:
            raise ValueError(f"Unsupported XLeRobot base type: {base_type}")
        if isinstance(self.mount, XLeRobotMountConfig):
            mount_cfg = self.mount
        else:
            mount_cfg = XLeRobotMountConfig(**self.mount)

        self.arms = arms_cfg
        self.base = base_cfg
        self.mount = mount_cfg
        self.arms_config: BiSO101FollowerConfig = arms_cfg
        self.base_config = base_cfg
        self.mount_config: XLeRobotMountConfig = mount_cfg
        self.base_type = base_type

        if self.id:
            if arms_cfg.id is None:
                arms_cfg.id = f"{self.id}_arms"
            if base_cfg.id is None:
                base_cfg.id = f"{self.id}_base"
            if mount_cfg.id is None:
                mount_cfg.id = f"{self.id}_mount"

        if self.calibration_dir:
            if arms_cfg.calibration_dir is None:
                arms_cfg.calibration_dir = self.calibration_dir
            if base_cfg.calibration_dir is None:
                base_cfg.calibration_dir = self.calibration_dir
            if mount_cfg.calibration_dir is None:
                mount_cfg.calibration_dir = self.calibration_dir
