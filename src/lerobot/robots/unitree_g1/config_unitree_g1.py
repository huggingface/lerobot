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


@RobotConfig.register_subclass("unitree_g1")
@dataclass
class UnitreeG1Config(RobotConfig):
   # id: str = "unitree_g1"
    motion_mode: bool = False
    simulation_mode: bool = True
    kp_high = 40.0
    kd_high = 3.0
    kp_low = 80.0
    kd_low = 3.0
    kp_wrist = 40.0
    kd_wrist = 1.5
    all_motor_q = None
    arm_velocity_limit = 100.0
    control_dt = 1.0 / 250.0

    all_motor_q = None
    arm_velocity_limit = 100.0
    control_dt = 1.0 / 250.0

    speed_gradual_max = False
    gradual_start_time = None
    gradual_time = None

    audio_client = True

    freeze_body = False

    gravity_compensation = False

    cameras: dict[str, CameraConfig] = field(default_factory=dict)
