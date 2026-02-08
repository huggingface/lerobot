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

"""
OpenLoong "Qinglong" humanoid robot support for LeRobot.

OpenLoong is a humanoid robot from Shanghai Humanoid Robotics Innovation Center.
It uses MPC (Model Predictive Control) and WBC (Whole-Body Control) for dynamic locomotion.

Reference: https://github.com/loongOpen/OpenLoong-Dyn-Control

Example:
    ```python
    from lerobot.robots.openloong import OpenLoong, OpenLoongConfig
    
    # Create configuration
    config = OpenLoongConfig(
        is_simulation=True,
        control_dt=1/500,  # 500Hz
    )
    
    # Initialize robot
    robot = OpenLoong(config)
    robot.connect()
    
    # Get observation
    obs = robot.get_observation()
    
    # Send action (joint positions)
    action = {
        "kLeftHipPitch.q": 0.1,
        "kRightHipPitch.q": 0.1,
        "kLeftKnee.q": 0.3,
        "kRightKnee.q": 0.3,
    }
    robot.send_action(action)
    
    # Reset to default position
    robot.reset()
    
    # Disconnect
    robot.disconnect()
    ```
"""

from .config_openloong import OpenLoongConfig
from .openloong import OpenLoong
from .openloong_utils import (
    NUM_MOTORS,
    OPENLOONG_DEFAULT_GAINS,
    OPENLOONG_DEFAULT_STANDING_POSITION,
    OPENLOONG_JOINT_LIMITS,
    OPENLOONG_JOINT_NAMES,
    OpenLoongArmJointIndex,
    OpenLoongJointIndex,
    OpenLoongLegJointIndex,
)

__all__ = [
    "OpenLoong",
    "OpenLoongConfig",
    "OpenLoongJointIndex",
    "OpenLoongArmJointIndex",
    "OpenLoongLegJointIndex",
    "NUM_MOTORS",
    "OPENLOONG_DEFAULT_GAINS",
    "OPENLOONG_DEFAULT_STANDING_POSITION",
    "OPENLOONG_JOINT_LIMITS",
    "OPENLOONG_JOINT_NAMES",
]
