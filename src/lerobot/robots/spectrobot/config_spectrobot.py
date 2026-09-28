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

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@dataclass
class TactileSensorConfig:
    channel: int
    observation_key: str | None = None
    sample_rate_hz: int = 20_000
    nfft: int = 512
    min_db: float = -120.0
    max_db: float = -50.0
    measurement: str = "Voltage"  # "Voltage" or "IEPE"
    range_choice: int = 10_000
    hpf_choice: float = 0.1
    excitation_choice: int = 4


@RobotConfig.register_subclass("spectrobot")
@dataclass
class SpectRoFollowerConfig(RobotConfig):
    """Configuration for the tactile SO follower robot."""

    port: str
    disable_torque_on_disconnect: bool = True
    max_relative_target: float | dict[str, float] | None = None
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    use_degrees: bool = True
    tactile_data_stream: str = "teensy_ADS8688"  # "teensy_ADS8688" or "OpenDAQ"
    tactile_serial_port: str = "/dev/ttyACM0" #use only for teensy_ADS8688
    tactile_fps: float = 30.0
    tactile_sensors: dict[str, TactileSensorConfig] = field(default_factory=dict)


SO100FollowerConfig = SpectRoFollowerConfig
SO101FollowerConfig = SpectRoFollowerConfig

