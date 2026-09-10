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

"""Simulated tactile sensor for testing and development."""

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..tactile import TactileSensor
from .configuration_simulated import SimulatedTactileConfig


class SimulatedTactile(TactileSensor):
    """Simulated tactile sensor for testing and development.

    Generates random tactile data with optional noise, useful for testing
    pipelines without physical hardware.

    Example:
        ```python
        config = SimulatedTactileConfig(num_points=400, seed=42)
        with SimulatedTactile(config) as sensor:
            data = sensor.read()  # (400, 6) array
        ```
    """

    def __init__(self, config: SimulatedTactileConfig):
        super().__init__(config)
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @staticmethod
    def find_sensors() -> list[dict[str, Any]]:
        """Return a single simulated sensor.

        Returns:
            list: Single simulated sensor entry.
        """
        return [{"id": "simulated_0", "type": "simulated"}]

    def connect(self, warmup: bool = True) -> None:
        """Connect to the simulated sensor.

        Args:
            warmup: If True, generates warmup frames to simulate real sensor behavior.
        """
        self._is_connected = True

        if warmup:
            for _ in range(self.config.warmup_frames):
                self._generate_frame()
                if self.config.simulate_delay:
                    time.sleep(1.0 / self.fps)

    def disconnect(self) -> None:
        """Disconnect from the simulated sensor."""
        self._is_connected = False

    def read(self) -> NDArray[np.float64]:
        """Generate and return a simulated tactile frame.

        Returns:
            np.ndarray: Simulated tactile data with shape (num_points, data_dim).

        Raises:
            ConnectionError: If sensor is not connected.
        """
        if not self._is_connected:
            raise ConnectionError("Sensor is not connected. Call connect() first.")

        frame = self._generate_frame()
        if self.config.simulate_delay:
            time.sleep(1.0 / self.fps)
        return frame

    def async_read(self, timeout_ms: float = 1000.0) -> NDArray[np.float64]:
        """Return the most recent simulated frame.

        For the simulated sensor, this behaves identically to read().

        Args:
            timeout_ms: Ignored for simulated sensor.

        Returns:
            np.ndarray: Simulated tactile data.
        """
        return self.read()

    def _generate_frame(self) -> NDArray[np.float64]:
        """Generate a single simulated tactile frame.

        Returns:
            np.ndarray: Generated frame with shape (num_points, data_dim).
        """
        # Generate base random data
        frame = self._rng.standard_normal((self.num_points, self.data_dim)) * 0.1

        # Add noise
        if self.config.noise_std > 0:
            frame += self._rng.standard_normal(frame.shape) * self.config.noise_std

        # For full 6D data, scale displacement and force differently
        if self.data_dim == 6:
            frame[:, :3] *= 2.0  # Displacement in mm range
            frame[:, 3:6] *= 0.5  # Force in N range

        return frame.astype(np.float64)
