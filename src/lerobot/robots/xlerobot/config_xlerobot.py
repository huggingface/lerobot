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
#   https://www.hackster.io/brainbot/brainbot-big-brain-with-xlerobot-ad1b4c
#   https://github.com/Astera-org/brainbot
#   https://github.com/Vector-Wangel/XLeRobot
#   https://github.com/bingogome/lerobot

from dataclasses import dataclass, field
from typing import Any

from lerobot.cameras import CameraConfig

from ..bi_so101_follower.config_bi_so101_follower import BiSO101FollowerConfig
from ..biwheel_base.config_biwheel_base import BiWheelBaseConfig
from ..config import RobotConfig
from ..lekiwi_base.config import LeKiwiBaseConfig
from ..xlerobot_mount.config import XLeRobotMountConfig


@RobotConfig.register_subclass("xlerobot")
@dataclass
class XLeRobotConfig(RobotConfig):
    BASE_TYPE_LEKIWI = "lekiwi_base"
    BASE_TYPE_BIWHEEL = "biwheel_base"

    arms: dict[str, Any] = field(default_factory=dict)
    base: dict[str, Any] = field(default_factory=dict)
    mount: dict[str, Any] = field(default_factory=dict)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    base_type: str | None = BASE_TYPE_LEKIWI

    def __post_init__(self) -> None:
        super().__post_init__()

        arms_cfg: BiSO101FollowerConfig | None = None
        if isinstance(self.arms, BiSO101FollowerConfig):
            arms_cfg = self.arms
        elif self.arms:
            arms_cfg = BiSO101FollowerConfig(**self.arms)

        base_cfg: LeKiwiBaseConfig | BiWheelBaseConfig | None = None
        base_type = self.base_type or self.BASE_TYPE_LEKIWI
        if self.base:
            if base_type == self.BASE_TYPE_LEKIWI:
                base_cfg = (
                    self.base if isinstance(self.base, LeKiwiBaseConfig) else LeKiwiBaseConfig(**self.base)
                )
            elif base_type == self.BASE_TYPE_BIWHEEL:
                base_cfg = (
                    self.base if isinstance(self.base, BiWheelBaseConfig) else BiWheelBaseConfig(**self.base)
                )
            else:
                raise ValueError(f"Unsupported XLeRobot base type: {base_type}")
        else:
            base_type = None

        mount_cfg: XLeRobotMountConfig | None = None
        if isinstance(self.mount, XLeRobotMountConfig):
            mount_cfg = self.mount
        elif self.mount:
            mount_cfg = XLeRobotMountConfig(**self.mount)

        self.arms = arms_cfg
        self.base = base_cfg
        self.mount = mount_cfg
        self.arms_config: BiSO101FollowerConfig | None = arms_cfg
        self.base_config: LeKiwiBaseConfig | BiWheelBaseConfig | None = base_cfg
        self.mount_config: XLeRobotMountConfig | None = mount_cfg
        self.base_type = base_type

        if self.id:
            if arms_cfg and arms_cfg.id is None:
                arms_cfg.id = f"{self.id}_arms"
            if base_cfg and getattr(base_cfg, "id", None) is None:
                base_cfg.id = f"{self.id}_base"
            if mount_cfg and mount_cfg.id is None:
                mount_cfg.id = f"{self.id}_mount"

        if self.calibration_dir:
            if arms_cfg and arms_cfg.calibration_dir is None:
                arms_cfg.calibration_dir = self.calibration_dir
            if base_cfg and getattr(base_cfg, "calibration_dir", None) is None:
                base_cfg.calibration_dir = self.calibration_dir
            if mount_cfg and mount_cfg.calibration_dir is None:
                mount_cfg.calibration_dir = self.calibration_dir
