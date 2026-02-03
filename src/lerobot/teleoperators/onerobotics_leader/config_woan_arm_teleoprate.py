#!/usr/bin/env python

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
from typing import List

from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig


@dataclass
class WoanTeleopLeaderConfigBase(RobotConfig):
    """Base configuration for the Woan teleoperation leader arm."""

    # Connection default settings for woanarm_api_py
    device_path: str = "/dev/ttyACM0"
    baud_rate: int = 961200
    robot_model: str = "a1_r"
    version: str = "A1"
    woan_description_path: str = "/home/woan/WoanLerobotAdapter/woan_arm/woan_description"
    slcan_type: str = "damiao"  # "damiao" or "canable"
    is_teleop_leader: bool = True

    use_velocity: bool = True
    use_acceleration: bool = False
    
    # Robot Physical settings
    home_joints_positions: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def __post_init__(self):
        super().__post_init__()

        if not self.id:
            self.id = self.robot_model


@TeleoperatorConfig.register_subclass("woan_teleop_leader")
@dataclass
class WoanTeleopLeaderConfig(TeleoperatorConfig, WoanTeleopLeaderConfigBase):
    """Registered config combining Woan teleop leader defaults with Teleoperator registry."""

    pass