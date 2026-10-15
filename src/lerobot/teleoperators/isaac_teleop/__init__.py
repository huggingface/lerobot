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

Each input device is an :class:`IsaacTeleopTeleoperator` subclass; :class:`XRController`
(``--teleop.type=isaac_teleop``) ships today. Importing this package registers it
on the global :class:`~lerobot.teleoperators.config.TeleoperatorConfig` choice registry.
"""

from .base import IsaacTeleopTeleoperator
from .config import IsaacTeleopConfig, XRControllerConfig
from .xr_controller import XRController
from .xr_controller_processor import MapXRControllerActionToRobotAction

__all__ = [
    "IsaacTeleopConfig",
    "IsaacTeleopTeleoperator",
    "MapXRControllerActionToRobotAction",
    "XRController",
    "XRControllerConfig",
]
