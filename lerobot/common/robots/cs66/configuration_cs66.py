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
from typing import Dict

from lerobot.common.cameras import CameraConfig

from ..config import RobotConfig

@RobotConfig.register_subclass("cs66")
@dataclass
class EliteCS66Config(RobotConfig):
    # A connection is established with the robot controller. 
    ip: str = "192.168.101.11"
    
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # 关节偏移量，按 joint 名称映射偏移值
    joint_offsets: Dict[str, float] = field(default_factory=lambda: {
        "joint_1": 0.0,
        "joint_2": -90.0,
        "joint_3": 0.0,
        "joint_4": -90.0,
        "joint_5": 180.0,
        "joint_6": 0.0,
    })

    dt = 1 / 25  # 25Hz
    lookahead_time = 1/5
    gain = 300