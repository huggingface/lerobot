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

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("bi_yam_follower")
@dataclass
class BiYamFollowerConfig(RobotConfig):
    # Server ports for left and right arm followers
    # These should match the ports in the bimanual_lead_follower.py script
    # Default: 1235 for left arm, 1234 for right arm
    left_arm_port: int = 1235
    right_arm_port: int = 1234

    # Server host (usually localhost for local setup)
    server_host: str = "localhost"

    # Optional: Maximum relative target for safety
    left_arm_max_relative_target: float | dict[str, float] | None = None
    right_arm_max_relative_target: float | dict[str, float] | None = None

    # Cameras (shared between both arms)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
