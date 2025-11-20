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

from ....config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("biwheel_gamepad")
@dataclass
class BiwheelGamepadTeleopConfig(TeleoperatorConfig):
    """Configuration for the differential (biwheel) base gamepad teleoperator."""

    joystick_index: int = 0
    max_speed_mps: float = 0.2
    yaw_speed_deg: float = 40.0
    deadzone: float = 0.15
    polling_fps: int = 60
    axis_forward: int = 1
    axis_yaw: int = 0
    invert_forward: bool = True
    invert_yaw: bool = False
