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

from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig


@dataclass
class OneroTeleopLeaderConfigBase(RobotConfig):
    """Base configuration for the Onero teleoperation leader arm."""

    # Connection default settings for oneroarm_api_py
    port: str = "/dev/ttyACM0"
    baud_rate: int = 961200
    robot_model: str = "a1_r"
    version: str = "A1"
    onero_description_path: str = "/home/onero/OneroLerobotAdapter/onero_arm/onero_description"
    slcan_type: str = "damiao"  # "damiao" or "canable"
    is_teleop_leader: bool = True

    use_velocity: bool = True

    enable_gripper: bool = False
    enable_gripper_joystick: bool = True

    # Optional low-level MIT gains passed to oneroarm_api_py (length should match dof)
    mit_kp: list[float] | None = None
    mit_kd: list[float] | None = None

    # Robot Physical settings
    home_joints_positions: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def __post_init__(self):
        super().__post_init__()

        if not self.id:
            self.id = self.robot_model


@TeleoperatorConfig.register_subclass("onero_teleop_leader")
@dataclass
class OneroTeleopLeaderConfig(TeleoperatorConfig, OneroTeleopLeaderConfigBase):
    """Registered config combining Onero teleop leader defaults with Teleoperator registry."""

    pass
