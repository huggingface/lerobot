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


@RobotConfig.register_subclass("piper_dual_teleop")
@dataclass
class PIPERDualTeleopConfig(RobotConfig):
    """Configuration for Piper Dual Arm with Software-level Teleoperation.

    This plugin connects to 4 Piper arms:
    - 2 Leader arms (for reading teleop commands)
    - 2 Follower arms (for executing actions)

    During teleop recording, the software reads leader positions and writes to followers.
    During inference/replay, only the follower arms are controlled.
    """

    # Leader arm CAN ports (for reading teleop commands)
    left_leader_port: str = "can_left_leader"
    right_leader_port: str = "can_right_leader"

    # Follower arm CAN ports (for executing actions)
    left_follower_port: str = "can_left_follower"
    right_follower_port: str = "can_right_follower"

    # Whether to use software teleop (read from leaders, write to followers)
    # If False, only followers are used (for inference/replay)
    use_teleop: bool = True

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "left": OpenCVCameraConfig(
                index_or_path="/dev/video4",
                fps=30,
                width=640,
                height=480,
            ),
            "right": OpenCVCameraConfig(
                index_or_path="/dev/video12",
                fps=30,
                width=640,
                height=480,
            ),
            "middle": OpenCVCameraConfig(
                index_or_path="/dev/video6",
                fps=30,
                width=640,
                height=480,
            ),
        }
    )
