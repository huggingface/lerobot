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

from lerobot.common.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("bimanual_parcelot")
@dataclass
class BimanualParcelotConfig(RobotConfig):
    # Ports for the two follower arms
    left_arm_port: str  # Left follower arm port
    right_arm_port: str  # Right follower arm port
    
    # Optional separate ID for each arm
    left_arm_id: str | None = None
    right_arm_id: str | None = None

    # Safety configurations
    disable_torque_on_disconnect: bool = True
    
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Can be set per arm or globally
    max_relative_target: int | None = None
    left_arm_max_relative_target: int | None = None
    right_arm_max_relative_target: int | None = None

    # Cameras configuration - three cameras: top, left_wrist, right_wrist
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False