#!/usr/bin/env python

# Copyright 2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

"""NVIDIA Isaac Teleop teleoperators for LeRobot.

Each input device is an :class:`IsaacTeleopTeleoperator` subclass: :class:`XRController`
(XR/VR controller) and :class:`SO101LeaderArm` (back-drivable SO-101 leader arm) ship today.
"""

from .base import IsaacTeleopTeleoperator
from .clutch import Clutch
from .config_isaac_teleop import IsaacTeleopConfig, SO101LeaderArmConfig, XRControllerConfig
from .teleop_so101_leader_arm import SO101LeaderArm, leader_joints_to_robot_action
from .teleop_xr_controller import XRController
from .xr_controller_processor import MapXRControllerActionToRobotAction

__all__ = [
    "Clutch",
    "IsaacTeleopConfig",
    "IsaacTeleopTeleoperator",
    "MapXRControllerActionToRobotAction",
    "SO101LeaderArm",
    "SO101LeaderArmConfig",
    "XRController",
    "XRControllerConfig",
    "leader_joints_to_robot_action",
]
