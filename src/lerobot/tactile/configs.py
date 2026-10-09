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

"""Configuration classes for tactile sensors."""

import abc
from dataclasses import dataclass
from enum import Enum

import draccus  # type: ignore  # TODO: add type stubs for draccus


class TactileDataType(str, Enum):  # noqa: UP042
    """Supported tactile data types determining the dimensionality per point.

    Covers the three sensing families found in practice:
    optical elastomer sensors (displacement and optionally dense force fields),
    taxel/force-distribution arrays (per-point 3D force), and
    force-torque style outputs (per-point 6D wrench).
    """

    DISPLACEMENT = "displacement"  # 3D displacement field (dx, dy, dz)
    FORCE = "force"  # 3D force distribution (Fx, Fy, Fz)
    FULL = "full"  # Displacement and force (6D: dx, dy, dz, Fx, Fy, Fz)
    WRENCH = "wrench"  # Force-torque (6D: Fx, Fy, Fz, Mx, My, Mz)

    @classmethod
    def _missing_(cls, value: object) -> None:
        raise ValueError(f"`data_type` is expected to be in {list(cls)}, but {value} is provided.")


@dataclass(kw_only=True)
class TactileSensorConfig(draccus.ChoiceRegistry, abc.ABC):
    """Base configuration class for tactile sensors.

    This abstract base class defines the common configuration parameters for all
    tactile sensor implementations, following the same pattern as CameraConfig.

    Attributes:
        fps: Frames per second for data acquisition. Defaults to 30.
        num_points: Number of tactile sensing points (e.g., 400 for a 20x20 array,
            or 5 for one resultant point per fingertip of a five-finger hand).
        data_type: Type of tactile data to capture. Defaults to FULL (6D).

    Example:
        ```python
        # Create a simulated tactile sensor
        from lerobot.tactile.simulated import SimulatedTactile, SimulatedTactileConfig

        config = SimulatedTactileConfig(num_points=400)
        with SimulatedTactile(config) as sensor:
            data = sensor.read()
        ```
    """

    fps: int = 30
    num_points: int = 400
    data_type: TactileDataType = TactileDataType.FULL

    def __post_init__(self) -> None:
        """Validate and coerce configuration values."""
        self.data_type = TactileDataType(self.data_type)

        if self.fps <= 0:
            raise ValueError(f"`fps` must be positive, got {self.fps}")
        if self.num_points <= 0:
            raise ValueError(f"`num_points` must be positive, got {self.num_points}")

    @property
    def type(self) -> str:
        """Return the sensor type identifier."""
        return str(self.get_choice_name(self.__class__))

    @property
    def data_dim(self) -> int:
        """Return the dimensionality of tactile data per point.

        Returns:
            int: 3 for displacement-only or force-only, 6 for full or wrench data.
        """
        dim_map = {
            TactileDataType.DISPLACEMENT: 3,
            TactileDataType.FORCE: 3,
            TactileDataType.FULL: 6,
            TactileDataType.WRENCH: 6,
        }
        return dim_map[self.data_type]

    @property
    def expected_shape(self) -> tuple[int, ...]:
        """Return the expected shape of tactile data.

        Returns:
            tuple: Shape as (num_points, data_dim).
        """
        return (self.num_points, self.data_dim)
