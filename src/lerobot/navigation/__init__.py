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

"""Spatial-memory navigation for LeRobot mobile bases.

Online spatio-semantic mapping (DynaMem-style), A* planning, obstacle
avoidance and open-vocabulary goto/explore, driving any LeRobot ``Robot``
that exposes body-velocity actions and planar odometry. Ported from the
dyna360 research stack; the physical robot layer lives in
``lerobot.robots`` (e.g. ``unitree_go2``).
"""

from .base_controller import (
    BaseController,
    RobotBaseController,
    SafeBaseController,
    StubBaseController,
    odometry_to_world_pose,
    world_velocity_to_body,
)

__all__ = [
    "BaseController",
    "RobotBaseController",
    "SafeBaseController",
    "StubBaseController",
    "odometry_to_world_pose",
    "world_velocity_to_body",
]
