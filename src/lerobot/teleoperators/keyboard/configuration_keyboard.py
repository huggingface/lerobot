#!/usr/bin/env python

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
"""Configuration for keyboard teleoperators."""

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@dataclass
class KeyboardTeleopConfig(TeleoperatorConfig):
    """
    Base configuration for all keyboard teleoperators.
    """

    type: str | None = None


@dataclass
class KeyboardEndEffectorTeleopConfig(KeyboardTeleopConfig):
    """
    Configuration for the end-effector keyboard teleoperator.
    """

    type: str = "end_effector"
    use_gripper: bool = True


@dataclass
class KeyboardRoverTeleopConfig(KeyboardTeleopConfig):
    """
    Configuration for the rover keyboard teleoperator.
    """

    type: str = "rover"
    linear_speed: float = 1.0
    angular_speed: float = 1.0
    speed_increment: float = 0.1
    angular_speed_ratio: float = 1.0
    turn_assist_ratio: float = 0.5
    min_linear_speed: float = 0.1
    min_angular_speed: float = 0.1


@TeleoperatorConfig.register_subclass("keyboard_reachy_mini")
@dataclass
class KeyboardReachyMiniTeleopConfig(KeyboardTeleopConfig):
    """
    Configuration for the Reachy Mini keyboard teleoperator.
    """

    type: str = "reachy_mini"
    head_speed_deg: float = 1.0
    body_speed_deg: float = 1.0
    antenna_speed_deg: float = 2.0

