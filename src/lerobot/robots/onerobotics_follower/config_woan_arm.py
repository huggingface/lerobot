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

from lerobot.cameras.configs import CameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("woan_arm")
@dataclass
class WoanRobotConfig(RobotConfig):
    """
    Configuration for the Woan Robot Arm adapter.
    Inherits standard fields (id, type, calibration_dir, etc.) from RobotConfig.
    """

    # Connection default settings for woanarm_api_py
    port: str = "/dev/ttyACM0"  # e.g., "/dev/ttyACM0", use 'ls /dev/tty*' to find the correct port
    baud_rate: int = 961200
    robot_model: str = "x1_r"
    version: str = "v3.2"
    woan_description_path: str = (
        "/path/to/woan_description"  # Path to Woan description package, including urdf and meshes
    )
    slcan_type: str = "damiao"  # "damiao" or "canable"
    is_teleop_leader: bool = False

    use_velocity: bool = False

    # Robot Physical settings
    home_joints_positions: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )  # Default home position for the robot's joints

    enable_gripper: bool = True  # Whether to enable gripper control

    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()

        if not self.id:
            self.id = self.robot_model


@RobotConfig.register_subclass("woan_teleop_follower")
@dataclass
class WoanTeleopFollowerConfig(WoanRobotConfig):
    """
    Configuration for the Woan Robot Arm Teleoperation Follower adapter.
    Inherits from WoanRobotConfig and sets is_teleop_leader to False by default.
    """

    use_velocity: bool = True  # Teleop follower typically uses velocity for feedfoward control
    is_teleop_leader: bool = False
    home_joints_positions: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    enable_gripper: bool = True  # Whether to enable gripper control
