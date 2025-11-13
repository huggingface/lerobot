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

from ....config import RobotConfig


@RobotConfig.register_subclass("xlerobot_mount")
@dataclass
class XLeRobotMountConfig(RobotConfig):
    # Serial communication
    port: str = "/dev/ttyACM5"  # USB-to-serial port (adjust for your system)

    # Motor IDs
    pan_motor_id: int = 1
    tilt_motor_id: int = 2
    motor_model: str = "sts3215"  # Feetech STS3215 servo

    # Feature keys for observation/action dictionaries
    pan_key: str = "mount_pan.pos"
    tilt_key: str = "mount_tilt.pos"

    # Operational parameters
    max_pan_speed_dps: float = 60.0  # Maximum pan speed in degrees per second
    max_tilt_speed_dps: float = 45.0  # Maximum tilt speed in degrees per second

    # Safety limits (degrees)
    pan_range: tuple[float, float] = (-90.0, 90.0)  # Pan angle limits
    tilt_range: tuple[float, float] = (-30.0, 60.0)  # Tilt angle limits
