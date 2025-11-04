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
from dataclasses import dataclass

from ..config import RobotConfig


@RobotConfig.register_subclass("biwheel_base")
@dataclass
class BiWheelBaseConfig(RobotConfig):
    port: str = "/dev/ttyACM0"

    disable_torque_on_disconnect: bool = True

    # Differential drive parameters
    wheel_radius: float = 0.05
    wheel_base: float = 0.25
    max_wheel_raw: int = 2000  # Reduced from 3000 for safety

    # Motor IDs for the left and right wheels
    base_motor_ids: tuple[int, int] = (9, 10)

    # Motor direction inversion flags (True to invert, False to keep original)
    # Set to True if motor rotates in opposite direction
    invert_left_motor: bool = True
    invert_right_motor: bool = False  # Right motor needs inversion for proper differential drive

    handshake_on_connect: bool = True
