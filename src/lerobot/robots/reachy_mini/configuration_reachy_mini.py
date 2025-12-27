#!/usr/bin/env python

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
from lerobot.robots import RobotConfig
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig


@RobotConfig.register_subclass("reachy_mini")
@dataclass
class ReachyMiniConfig(RobotConfig):
    """
    Configuration for the Reachy Mini robot.
    """
    # IP address or hostname of the robot
    # IP address or hostname of the robot. For Reachy Mini Lite (USB), use 'localhost'
    # as it connects to a local daemon (Reachy Dashboard) usually at http://localhost:8000.
    # For the wireless version, use its assigned IP address or hostname.
    ip_address: str = "localhost"

    # Camera configuration
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "wrist_camera": OpenCVCameraConfig(
                # Adjust camera index or path as needed
                index_or_path=0,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    # Movement limits
    # Estimated head Z position limits in mm (min, max)
    head_z_pos_limits_mm: tuple[float, float] = (-50.0, 50.0)
    # Body yaw limits in degrees (min, max)
    body_yaw_limits_deg: tuple[float, float] = (-160.0, 160.0)
    # Antennas position limits in degrees (min, max)
    antennas_pos_limits_deg: tuple[float, float] = (-90.0, 90.0)
