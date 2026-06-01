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

from ..config import RobotConfig
from ..so_follower import SOFollowerConfig

from lerobot.cameras.opencv import OpenCVCameraConfig

@RobotConfig.register_subclass("bi_so_follower")
@dataclass
class BiSOFollowerConfig(RobotConfig):
    """Configuration class for Bi SO Follower robots."""

    left_arm_config: SOFollowerConfig
    right_arm_config: SOFollowerConfig
    top_cameras: dict[str, OpenCVCameraConfig] = field(default_factory=dict)
