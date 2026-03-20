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

"""Configuration for Tac3D tactile sensors."""

from dataclasses import dataclass, field

from ..configs import TactileDataType, TactileSensorConfig

__all__ = ["Tac3DConfig"]


@TactileSensorConfig.register_subclass("tac3d")
@dataclass
class Tac3DConfig(TactileSensorConfig):
    """Configuration for Tac3D tactile sensors.

    Tac3D sensors provide high-resolution tactile feedback with 20x20 sensing arrays,
    measuring both 3D displacement and 3D force distributions. Each sensing point
    outputs a 6D vector [dx, dy, dz, Fx, Fy, Fz].

    Note:
        Tac3D sensors only support FULL (6D) data type. Attempting to configure
        DISPLACEMENT or FORCE (3D) will raise a ValueError.

    Attributes:
        udp_port: UDP port for receiving sensor data from Tac3D-Desktop. Defaults to 9988.
        sensor_sn: Serial number of the sensor to connect to. Defaults to None (auto-detect).
        tare_on_startup: Whether to perform tare calibration on connect. Defaults to True.
        displacement_range: Range for displacement normalization in mm. Defaults to (-2.0, 3.0).
        force_range: Range for force normalization in N. Defaults to (-0.8, 0.8).
        timeout_ms: Timeout for frame reception in milliseconds. Defaults to 1000.

    Example:
        ```python
        from lerobot.tactile.tac3d import Tac3DTactile, Tac3DConfig

        config = Tac3DConfig(udp_port=9988)
        with Tac3DTactile(config) as sensor:
            sensor.tare()
            data = sensor.read()  # (400, 6) array
        ```
    """

    udp_port: int = 9988
    sensor_sn: str | None = None
    tare_on_startup: bool = True
    displacement_range: tuple[float, float] = field(default_factory=lambda: (-2.0, 3.0))
    force_range: tuple[float, float] = field(default_factory=lambda: (-0.8, 0.8))
    timeout_ms: float = 1000.0

    def __post_init__(self) -> None:
        """Validate Tac3D-specific configuration."""
        super().__post_init__()

        if self.num_points != 400:
            raise ValueError(f"Tac3D sensors have exactly 400 points (20x20 grid), got {self.num_points}")

        # Tac3D only outputs 6D data (displacement + force)
        if self.data_type != TactileDataType.FULL:
            raise ValueError(
                f"Tac3D sensors only support data_type=FULL (6D), got {self.data_type}. "
                "Tac3D hardware always outputs both displacement and force data."
            )
