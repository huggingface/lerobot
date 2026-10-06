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

"""Bimanual YAM follower configuration."""

import math
from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig
from ..yam_follower.config_yam_follower import BI_YAM_FEATURE_NAMES, YamArmConfig


@RobotConfig.register_subclass("bi_yam_follower")
@dataclass
class BiYamFollowerConfig(RobotConfig):
    left_arm: YamArmConfig = field(default_factory=lambda: YamArmConfig(port="can_left"))
    right_arm: YamArmConfig = field(default_factory=lambda: YamArmConfig(port="can_right"))
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    # Opt in only after verifying CAN side assignment, encoder frame, and gripper calibration.
    read_only: bool = True
    # Leave classic CAN bandwidth for both refresh and MIT command feedback.
    control_frequency: float = 100.0
    feedback_timeout_s: float = 0.2
    command_timeout_s: float = 1.0
    freeze_gc: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.left_arm.port == self.right_arm.port:
            raise ValueError("YAM arms use duplicate motor IDs and require distinct CAN interfaces")
        for name in ("control_frequency", "feedback_timeout_s", "command_timeout_s"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 20 <= self.control_frequency <= 250:
            raise ValueError("control_frequency must be between 20 and 250 Hz")
        if set(self.cameras) & set(BI_YAM_FEATURE_NAMES):
            raise ValueError("Camera names collide with YAM motor features")
