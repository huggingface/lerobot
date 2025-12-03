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


@TeleoperatorConfig.register_subclass("lekiwi_base_gamepad")
@dataclass
class LeKiwiBaseTeleopConfig(TeleoperatorConfig):
    """Configuration for the LeKiwi base gamepad teleoperator."""

    joystick_index: int = 0
    max_speed_mps: float = 1.0
    deadzone: float = 0.15
    yaw_speed_deg: float = 30.0
    polling_fps: int = 60
    normalize_diagonal: bool = True
    axis_x: int = 0
    axis_y: int = 1
    invert_x: bool = False
    invert_y: bool = True
    hat_index: int = 0
    dpad_left_button: int = 14
    dpad_right_button: int = 13
