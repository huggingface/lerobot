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

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("joystick")
@dataclass
class JoystickTeleopConfig(TeleoperatorConfig):
    # Joystick device index (0 for first joystick)
    device_index: int = 0
    
    # Deadzone for joystick axes to prevent drift
    deadzone: float = 0.1
    
    # Step size for relative movement sensitivity
    step_size: float = 0.05
    
    # Mapping of joystick axes to robot joints
    # Format: {"axis_index": "joint_name"}
    axis_mapping: dict[int, str] = None
    
    def __post_init__(self):
        # Set default id if not provided
        if self.id is None:
            self.id = "fsi6x"
            
        if self.axis_mapping is None:
            # Default mapping for SO101 - can be overridden
            self.axis_mapping = {
                0: "shoulder_pan",
                1: "shoulder_lift", 
                2: "elbow_flex",
                3: "wrist_flex",
                4: "wrist_roll",
                5: "gripper"
            } 