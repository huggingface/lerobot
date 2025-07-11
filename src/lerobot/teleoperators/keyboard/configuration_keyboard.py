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
from enum import Enum

from ..config import TeleoperatorConfig


class CoordinateSystem(Enum):
    XYZ = "xyz"
    RTZ = "rtz"


@TeleoperatorConfig.register_subclass("keyboard")
@dataclass
class KeyboardTeleopConfig(TeleoperatorConfig):
    # TODO(Steven): Consider setting in here the keys that we want to capture/listen
    mock: bool = False
    # Coordinate system to use for movement
    coordinate_system: CoordinateSystem = CoordinateSystem.XYZ
    # Step sizes for movement sensitivity (in meters)
    # For XYZ: x, y, z represent cartesian coordinates
    # For RTZ: r, t, z represent radial, theta (angular), and vertical coordinates
    deltas: dict[str, float] = field(
        default_factory=lambda: {"x": 1.0, "y": 1.0, "z": 1.0}
        # default_factory=lambda: {"r": 1.0, "t": 0.25, "z": 0.5}
    )


@TeleoperatorConfig.register_subclass("keyboard_ee")
@dataclass
class KeyboardEndEffectorTeleopConfig(KeyboardTeleopConfig):
    use_gripper: bool = True
