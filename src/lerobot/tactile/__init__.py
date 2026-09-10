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

"""Tactile sensor module for LeRobot.

This module provides a unified interface for tactile sensors, following the same
design pattern as the cameras module. Tactile data is represented as a point set
where each sensing point is a fixed-dimension vector, selected by
``TactileDataType``:

- ``displacement``: per-point 3D displacement ``[dx, dy, dz]`` (optical sensors)
- ``force``: per-point 3D force ``[Fx, Fy, Fz]`` (taxel arrays, per-fingertip resultants)
- ``full``: per-point 6D ``[dx, dy, dz, Fx, Fy, Fz]`` (optical sensors with force fields)
- ``wrench``: per-point 6D force-torque ``[Fx, Fy, Fz, Mx, My, Mz]`` (F/T-style sensors)

Supported backends:
    - simulated: Simulated tactile sensor for testing and development
    - paxini: PaXini multi-fingertip force sensor arrays (serial)

Example:
    ```python
    from lerobot.tactile.simulated import SimulatedTactile, SimulatedTactileConfig

    config = SimulatedTactileConfig(num_points=400, fps=30)
    with SimulatedTactile(config) as sensor:
        data = sensor.read()  # Returns (400, 6) array
    ```
"""

from .configs import TactileDataType, TactileSensorConfig
from .tactile import TactileSensor
from .utils import make_tactile_sensors_from_configs

__all__ = [
    "TactileDataType",
    "TactileSensor",
    "TactileSensorConfig",
    "make_tactile_sensors_from_configs",
]
