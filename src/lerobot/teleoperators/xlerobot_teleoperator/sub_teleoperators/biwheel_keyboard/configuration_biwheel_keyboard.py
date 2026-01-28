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

from ....config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("biwheel_keyboard")
@dataclass
class BiwheelKeyboardTeleopConfig(TeleoperatorConfig):
    """
    Configuration for bidirectional wheel (differential drive) keyboard teleop.

    This configuration supports smooth acceleration/deceleration control for
    differential drive mobile bases with configurable speed parameters.
    """

    # Smooth control parameters
    acceleration_rate: float = 3.0  # acceleration slope (speed/second) - reduced from 10.0
    deceleration_rate: float = 5.0  # deceleration slope (speed/second) - reduced from 10.0
    max_speed_multiplier: float = 3.0  # maximum speed multiplier - reduced from 6.0
    min_velocity_threshold: float = 0.01  # minimum velocity to send during deceleration

    # Key mappings for base control
    key_forward: str = "w"
    key_backward: str = "s"
    key_rotate_left: str = "a"
    key_rotate_right: str = "d"
    key_speed_up: str = "="
    key_speed_down: str = "-"
    key_quit: str = "q"

    # Speed level settings (list of dicts with 'linear' and 'angular' keys)
    speed_levels: list = field(
        default_factory=lambda: [
            {"linear": 0.04, "angular": 20},  # Level 1: Slow (half of max)
            {"linear": 0.08, "angular": 40},  # Level 2: Fast (motor limit)
        ]
    )

    # Initial speed level index (0-based)
    initial_speed_index: int = 0  # Default to very slow speed (Level 1)

    # Enable debug output
    debug: bool = True
