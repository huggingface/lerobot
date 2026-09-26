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

"""Configuration for PaXini multi-fingertip tactile force sensors."""

from dataclasses import dataclass

from ..configs import TactileDataType, TactileSensorConfig

__all__ = ["PaxiniTactileConfig"]


@TactileSensorConfig.register_subclass("paxini")
@dataclass
class PaxiniTactileConfig(TactileSensorConfig):
    """Configuration for PaXini fingertip force sensor arrays.

    PaXini sensors are taxel-based force-distribution sensors commonly mounted
    on the fingertips of dexterous hands (up to five sensors on one serial hub).
    The hub streams per-fingertip resultant 3D forces ``[Fx, Fy, Fz]``, so the
    natural configuration is ``data_type=FORCE`` with one point per fingertip.

    Note:
        Only ``data_type=FORCE`` is supported: the hub streams 3D force vectors.
        Displacement and wrench outputs are not provided by this hardware.

    Attributes:
        port: Serial port of the sensor hub (e.g., "/dev/ttyACM1").
        baudrate: Serial baudrate. Defaults to 921600.
        timeout_ms: Timeout for frame reception in milliseconds. Defaults to 1000.
        tare_on_connect: Whether to zero readings on connect. Defaults to True.

    Example:
        ```python
        from lerobot.tactile.paxini import PaxiniTactile, PaxiniTactileConfig

        config = PaxiniTactileConfig(port="/dev/ttyACM1", num_points=5)
        with PaxiniTactile(config) as sensor:
            data = sensor.read()  # (5, 3) array, one [Fx, Fy, Fz] per fingertip
        ```
    """

    port: str = "/dev/ttyACM1"
    baudrate: int = 921600
    timeout_ms: float = 1000.0
    tare_on_connect: bool = True

    num_points: int = 5
    data_type: TactileDataType = TactileDataType.FORCE
    fps: int = 90

    def __post_init__(self) -> None:
        """Validate PaXini-specific configuration."""
        super().__post_init__()

        if self.data_type != TactileDataType.FORCE:
            raise ValueError(
                f"PaXini sensors only support data_type=FORCE (3D per point), got {self.data_type}. "
                "The hub streams per-fingertip resultant force vectors."
            )
        if not 1 <= self.num_points <= 5:
            raise ValueError(f"PaXini hubs support 1-5 fingertip sensors, got {self.num_points}")
