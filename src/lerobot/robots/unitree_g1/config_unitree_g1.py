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


@RobotConfig.register_subclass("unitree_g1")
@dataclass
class UnitreeG1Config(RobotConfig):
    # id: str = "unitree_g1"
    simulation_mode: bool = False

    kp: list = field(
        default_factory=lambda: [
            150,
            150,
            150,
            300,
            40,
            40,  # Left leg pitch, roll, yaw, knee, ankle pitch, ankle roll
            150,
            150,
            150,
            300,
            40,
            40,  # Right leg pitch, roll, yaw, knee, ankle pitch, ankle roll
            250,
            250,
            250,  # Waist yaw, roll, pitch
            80,
            80,
            80,
            80,  # Left shoulder pitch, roll, yaw, elbow (kp_low)
            40,
            40,
            40,  # Left wrist roll, pitch, yaw (kp_wrist)
            80,
            80,
            80,
            80,  # Right shoulder pitch, roll, yaw, elbow (kp_low)
            40,
            40,
            40,  # Right wrist roll, pitch, yaw (kp_wrist)
            80,
            80,
            80,
            80,
            80,
            80,  # Other
        ]
    )

    kd: list = field(
        default_factory=lambda: [
            2,
            2,
            2,
            4,
            2,
            2,  # Left leg pitch, roll, yaw, knee, ankle pitch, ankle roll
            2,
            2,
            2,
            4,
            2,
            2,  # Right leg pitch, roll, yaw, knee, ankle pitch, ankle roll
            5,
            5,
            5,  # Waist yaw, roll, pitch
            3,
            3,
            3,
            3,  # Left shoulder pitch, roll, yaw, elbow (kd_low)
            1.5,
            1.5,
            1.5,  # Left wrist roll, pitch, yaw (kd_wrist)
            3,
            3,
            3,
            3,  # Right shoulder pitch, roll, yaw, elbow (kd_low)
            1.5,
            1.5,
            1.5,  # Right wrist roll, pitch, yaw (kd_wrist)
            3,
            3,
            3,
            3,
            3,
            3,  # Other
        ]
    )

    arm_velocity_limit = 100.0
    control_dt = 1.0 / 250.0

    leg_joint2motor_idx: list = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    default_leg_angles: list = field(
        default_factory=lambda: [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0]
    )
