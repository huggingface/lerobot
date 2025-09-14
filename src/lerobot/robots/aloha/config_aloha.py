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

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("aloha")
@dataclass
class AlohaConfig(RobotConfig):
    left_arm_port: str
    right_arm_port: str

    # Optional parameters for left arm (ViperX)
    left_arm_max_relative_target: float | dict[str, float] = 20.0
    left_arm_use_degrees: bool = True

    # Optional parameters for right arm (ViperX)
    right_arm_max_relative_target: float | dict[str, float] = 20.0
    right_arm_use_degrees: bool = True

    # cameras (shared between both arms)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
