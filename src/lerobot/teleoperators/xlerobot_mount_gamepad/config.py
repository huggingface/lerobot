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

"""Configuration for XLeRobot mount gamepad teleoperation."""

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("xlerobot_mount_gamepad")
@dataclass
class XLeRobotMountGamepadTeleopConfig(TeleoperatorConfig):
    """Configuration for Xbox gamepad control of XLeRobot mount.

    The right stick controls the pan/tilt mount:
    - Right stick horizontal (axis 2): Pan control
    - Right stick vertical (axis 3): Tilt control

    The control uses incremental velocity integration, so holding the stick
    will continuously move the mount, while releasing returns to zero velocity.
    """

    # Joystick settings
    joystick_index: int = 0  # Controller index (0 for first controller)
    deadzone: float = 0.15  # Deadzone threshold (0.0-1.0)
    polling_fps: int = 50  # Control loop frequency in Hz

    # Speed limits (degrees per second)
    max_pan_speed_dps: float = 60.0  # Maximum pan angular velocity
    max_tilt_speed_dps: float = 45.0  # Maximum tilt angular velocity

    # Axis configuration for Xbox controller right stick. Note that pygame exposes
    # right stick horizontal as axis 3 and right stick vertical as axis 4 on
    # Microsoft/XInput-style gamepads (axis 2 corresponds to the left trigger and
    # defaults to -1.0, which would introduce constant motion if used directly).
    pan_axis: int = 3  # Right stick horizontal
    tilt_axis: int = 4  # Right stick vertical

    # Axis inversion flags
    invert_pan: bool = False  # Invert pan direction if needed
    invert_tilt: bool = False  # Invert tilt direction if needed

    # Safety limits (degrees) - should match robot config
    pan_range: tuple[float, float] = (-90.0, 90.0)
    tilt_range: tuple[float, float] = (-30.0, 60.0)
