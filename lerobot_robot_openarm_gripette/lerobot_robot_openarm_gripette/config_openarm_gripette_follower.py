#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from lerobot.robots.config import RobotConfig
from lerobot.robots.openarm_follower import OpenArmFollowerConfigBase


@RobotConfig.register_subclass("openarm_gripette_follower")
@dataclass
class OpenArmGripetteFollowerConfig(RobotConfig):
    """OpenArm 7-joint follower with a networked Gripette gripper (gRPC)."""

    arm_config: OpenArmFollowerConfigBase = field(
        default_factory=lambda: OpenArmFollowerConfigBase(port="can0")
    )
    # gRPC target of the Gripette server, e.g. "192.168.1.x:50051".
    gripette_addr: str = "localhost:50051"
    # proximal.pos / distal.pos goals (radians, Gripette robot frame: 0 = open, positive = closing)
    # are clipped to [0, gripper_motor_max] since the server rejects out-of-range goals.
    gripper_motor_max: tuple[float, float] = (1.632, 2.025)
    # Expose the Gripette's JPEG stream as a camera observation. Frames are resized to
    # (height, width) so the feature shape is known before connecting.
    use_gripper_camera: bool = True
    gripper_camera_name: str = "gripette"
    gripper_camera_height: int = 486
    gripper_camera_width: int = 648
