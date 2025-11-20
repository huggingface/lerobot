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


@TeleoperatorConfig.register_subclass("xlerobot_mount_gamepad")
@dataclass
class XLeRobotMountGamepadTeleopConfig(TeleoperatorConfig):
    joystick_index: int = 0  # Controller index (0 for first controller)
    deadzone: float = 0.15  # Deadzone threshold (0.0-1.0)
    polling_fps: int = 50  # Control loop frequency in Hz

    max_pan_speed_dps: float = 60.0  # Maximum pan angular velocity
    max_tilt_speed_dps: float = 45.0  # Maximum tilt angular velocity
    pan_axis: int = 3  # Right stick horizontal
    tilt_axis: int = 4  # Right stick vertical

    invert_pan: bool = False  # Invert pan direction if needed
    invert_tilt: bool = False  # Invert tilt direction if needed

    pan_range: tuple[float, float] = (-90.0, 90.0)
    tilt_range: tuple[float, float] = (-30.0, 60.0)
