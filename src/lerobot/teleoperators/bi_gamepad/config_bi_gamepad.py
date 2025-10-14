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


def _default_button_mapping() -> Dict[str, int]:
    """
    Default mapping derived from the Xbox / XInput layout exposed by pygame.

    Users can override any entry by passing a custom dictionary when instantiating
    the teleoperator config.
    """

    return {
        "a": 0,
        "b": 1,
        "x": 2,
        "y": 3,
        "lb": 4,
        "rb": 5,
        "back": 6,
        "start": 7,
        "guide": 8,
        "left_stick": 9,
        "right_stick": 10,
    }


def _default_joint_sensitivity() -> Dict[str, float]:
    return {
        "shoulder_pan": 0.15,
        "shoulder_lift": 0.15,
        "elbow_flex": 0.15,
        "wrist_flex": 0.2,
        "wrist_roll": 0.2,
        "gripper": 1.0,
    }


@TeleoperatorConfig.register_subclass("bi_gamepad")
@dataclass
class BiGamepadConfig(TeleoperatorConfig):
    """
    Configuration for the bimanual SO-ARM teleoperation driven by a single gamepad.

    The defaults match the layout observed on the Brainbot teleop reference setup.
    Override any mapping if your gamepad reports different indices or sensitivities,
    and adjust ``gripper_open_value`` to change the neutral opening.
    """

    deadzone: float = 0.1
    axis_speed: float = 120.0  # units per second for analog sticks
    button_speed: float = 120.0  # units per second for digital buttons / hats
    left_stick_axes: Tuple[int, int] = (0, 1)  # (x, y)
    right_stick_axes: Tuple[int, int] = (3, 4)  # (x, y)
    left_trigger_axis: int = 2
    right_trigger_axis: int = 5
    hat_index: int = 0
    button_mapping: Dict[str, int] = field(default_factory=_default_button_mapping)
    joint_sensitivity: Dict[str, float] = field(default_factory=_default_joint_sensitivity)
    gripper_open_value: float = 100.0
    use_hid_controller: bool | None = None  # Force HID backend if desired (macOS defaults to True)
