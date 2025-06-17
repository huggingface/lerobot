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

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("keyboard")
@dataclass
class KeyboardTeleopConfig(TeleoperatorConfig):
    # TODO(Steven): Consider setting in here the keys that we want to capture/listen
    mock: bool = False
    
    # Motor mappings for SO100 follower
    motor_mappings: dict[str, dict[str, str]] = field(
        default_factory=lambda: {
            "shoulder_pan": {
                "left": "a",
                "right": "d",
            },
            "shoulder_lift": {
                "up": "w",
                "down": "s",
            },
            "elbow_flex": {
                "up": "q",
                "down": "e",
            },
            "wrist_flex": {
                "up": "r",
                "down": "f",
            },
            "wrist_roll": {
                "left": "z",
                "right": "x",
            },
            "gripper": {
                "open": "g",
                "close": "h",
            },
        }
    )
    
    # Step size for each motor movement
    step_size: float = 5.0


@TeleoperatorConfig.register_subclass("keyboard_ee")
@dataclass
class KeyboardEndEffectorTeleopConfig(KeyboardTeleopConfig):
    use_gripper: bool = True
