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
from lerobot.cameras.opencv import OpenCVCameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("piper_follower_dual")
@dataclass
class PIPERFollowerDualConfig(RobotConfig):
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "wrist_left": OpenCVCameraConfig(
                index_or_path="/dev/video0",
                fps=30,
                width=480,
                height=640,
                rotation=-90,
            ),
            "wrist_right": OpenCVCameraConfig(
                index_or_path="/dev/video2",
                fps=30,
                width=480,
                height=640,
                rotation=90,
            ),
            "top": OpenCVCameraConfig(
                index_or_path="/dev/video4",
                fps=30,
                width=640,
                height=480,
            ),
        }
    )
