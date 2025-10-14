#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Dict, Tuple

from ..config import TeleoperatorConfig


def _default_joint_sensitivity() -> Dict[str, float]:
    return {
        "shoulder_pan": 0.6,
        "shoulder_lift": 0.6,
        "elbow_flex": 0.5,
        "wrist_flex": 0.45,
        "wrist_roll": 0.45,
        "gripper": 1.0,
    }


OrientationMatrix = Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]


@TeleoperatorConfig.register_subclass("bi_joycon")
@dataclass
class BiJoyconConfig(TeleoperatorConfig):
    """
    Configuration for the JoyCon-powered bimanual SO-ARM teleoperator.

    The defaults align with a paired left/right JoyCon using the pyjoycon backend.
    Override any of the button mappings or gains if your setup reports different
    attribute names or requires different sensitivity.
    """

    deadzone: float = 0.12
    axis_speed: float = 120.0  # units per second for analog sticks
    button_speed: float = 120.0  # units per second for digital buttons / hats
    gripper_speed: float = 180.0  # units per second for the digital gripper buttons
    gripper_open_value: float = 100.0
    gyro_roll_gain: float = 45.0  # units per radian of JoyCon roll
    gyro_deadzone: float = 0.01  # radians, to filter gyroscope noise
    gyro_roll_axis_index: int = 0  # index into the mapped rotation vector (default: roll about X)
    use_gyro_roll: bool = True
    stick_x_directions: Tuple[float, float] = (1.0, 1.0)
    stick_y_directions: Tuple[float, float] = (1.0, 1.0)
    joint_sensitivity: Dict[str, float] = field(default_factory=_default_joint_sensitivity)
    left_wrist_flex_buttons: Tuple[str, str] = ("left_sl", "left_sr")  # (decrease, increase)
    right_wrist_flex_buttons: Tuple[str, str] = ("right_sr", "right_sl")  # (decrease, increase)
    left_gripper_buttons: Tuple[str, str] = ("l", "zl")  # (open, close)
    right_gripper_buttons: Tuple[str, str] = ("r", "zr")  # (open, close)
    orientation_map: OrientationMatrix | None = None
    right_orientation_map: OrientationMatrix | None = None
    discovery_timeout: float = 6.0
    discovery_poll_interval: float = 0.4
    left_serial_hint: str | None = None
    right_serial_hint: str | None = None
