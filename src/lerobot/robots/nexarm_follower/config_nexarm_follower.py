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


@RobotConfig.register_subclass("nexarm_follower")
@dataclass
class NexArmFollowerConfig(RobotConfig):
    """Configuration for the NexArm follower (slave) robot.

    Attributes:
        port: Serial port for the slave ESP32 (e.g. "COM19", "/dev/ttyUSB1").
        baudrate: Serial baud rate. NexArm uses 1 Mbps.
        disable_torque_on_disconnect: If True, disable servo torque on disconnect.
        max_relative_target: Per-step ceiling on |goal - present|. Float applies
            uniformly; dict applies per joint name. ``None`` disables clamping.
        motion_acc: CMD 56 acceleration for every CMD 97 write. 0 = max accel.
        motion_speed: CMD 56 speed limit (raw units/s). 0 = no limit.
        cameras: Camera configs keyed by name (e.g. ``front``, ``wrist``).
    """

    port: str
    baudrate: int = 1_000_000
    disable_torque_on_disconnect: bool = True
    max_relative_target: float | dict[str, float] | None = None
    motion_acc: int = 100
    motion_speed: int = 2000
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
