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

from dataclasses import dataclass

from ..config import RobotConfig
from ..openarm_follower import OpenArmFollowerConfig


@RobotConfig.register_subclass("openarm_gripette_follower")
@dataclass
class OpenArmGripetteFollowerConfig(OpenArmFollowerConfig):
    """OpenArm 7-joint follower with a networked Gripette gripper (gRPC)."""

    # gRPC target of the Gripette server, e.g. "192.168.1.x:50051".
    gripette_addr: str = "localhost:50051"
    # proximal.pos / distal.pos goals (radians, Gripette robot frame: 0 = open, positive = closing)
    # are clipped to [0, gripper_motor_max] since the server rejects out-of-range goals.
    gripper_motor_max: tuple[float, float] = (1.632, 2.025)

    def __post_init__(self) -> None:
        # The Gripette replaces the CAN gripper motor.
        self.motor_config.pop("gripper", None)
        super().__post_init__()
