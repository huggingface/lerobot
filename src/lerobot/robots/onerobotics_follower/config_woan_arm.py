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
from typing import List

# 引入 LeRobot 的基础配置类
from lerobot.robots.config import RobotConfig

@RobotConfig.register_subclass("woan_arm")  # 注册类型名，对应 json 里的 "type": "woan_arm"
@dataclass
class WoanRobotConfig(RobotConfig):
    """
    Configuration for the Woan Robot Arm adapter.
    Inherits standard fields (id, type, calibration_dir, etc.) from RobotConfig.
    """
    
    # Connection default settings for woanarm_api_py
    device_path: str = "/dev/ttyACM0"
    baud_rate: int = 961200
    robot_model: str = "a1_r"
    version: str = "A1"
    woan_description_path: str = "/home/woan/WoanLerobotAdapter/woan_arm/woan_description"
    slcan_type: str = "damiao"  # "damiao" or "canable"
    is_teleop_leader: bool = False
    
    use_velocity: bool = False
    use_acceleration: bool = False
    
    # Robot Physical settings
    home_joints_positions: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

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
    use_velocity: bool = True 
    is_teleop_leader: bool = False
    # Robot Physical settings
    home_joints_positions: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])