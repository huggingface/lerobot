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

"""Base tactile sensor interface for LeRobot."""

import abc
import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray  # type: ignore  # TODO: add type stubs for numpy.typing

from .configs import TactileSensorConfig


class TactileSensor(abc.ABC):
    """Base class for tactile sensor implementations.

    Defines a standard interface for tactile sensor operations across different backends,
    following the same design pattern as Camera. Subclasses must implement all abstract methods.

    Tactile sensors provide high-resolution contact information as point clouds,
    where each sensing point is represented as a 6D vector [dx, dy, dz, Fx, Fy, Fz]
    capturing both 3D displacement and 3D force information.

    Attributes:
        fps (int | None): Configured frames per second
        num_points (int | None): Number of sensing points in the sensor array
        data_dim (int | None): Dimensionality of data per point (3 or 6)
    """

    def __init__(self, config: TactileSensorConfig):
        """Initialize the tactile sensor with the given configuration.

        Args:
            config: TactileSensorConfig containing sensor parameters.
        """
        self.fps: int | None = config.fps
        self.num_points: int | None = config.num_points
        self.data_dim: int | None = config.data_dim

    def __enter__(self):
        """Context manager entry. Automatically connects to the sensor."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Context manager exit. Automatically disconnects."""
        self.disconnect()

    def __del__(self) -> None:
        """Destructor safety net."""
        try:
            if self.is_connected:
                self.disconnect()
        except Exception:  # nosec B110
            pass

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """Check if the sensor is currently connected.

        Returns:
            bool: True if sensor is connected and ready, False otherwise.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def find_sensors() -> list[dict[str, Any]]:
        """Detect available tactile sensors connected to the system.

        Returns:
            list[dict[str, Any]]: List of dictionaries containing information
                about each detected sensor.
        """
        pass

    @abc.abstractmethod
    def connect(self, warmup: bool = True) -> None:
        """Establish connection to the tactile sensor.

        Args:
            warmup: If True, captures a warmup frame before returning.
        """
        pass

    @abc.abstractmethod
    def read(self) -> NDArray[np.float64]:
        """Capture and return a single tactile frame synchronously.

        Returns:
            np.ndarray: Tactile data array with shape (num_points, data_dim).
                For full 6D data: (N, 6) where each row is [dx, dy, dz, Fx, Fy, Fz].

        Raises:
            ConnectionError: If the sensor is not connected.
            TimeoutError: If data cannot be read within the timeout period.
        """
        pass

    @abc.abstractmethod
    def async_read(self, timeout_ms: float = 1000.0) -> NDArray[np.float64]:
        """Return the most recent new tactile frame.

        Blocks up to timeout_ms waiting for a new frame if none available.

        Args:
            timeout_ms: Maximum time to wait for a new frame in milliseconds.

        Returns:
            np.ndarray: Tactile data array with shape (num_points, data_dim).

        Raises:
            TimeoutError: If no new frame arrives within timeout_ms.
            ConnectionError: If the sensor is not connected.
        """
        pass

    def read_latest(self, max_age_ms: int = 500) -> NDArray[np.float64]:
        """Return the most recent frame immediately (non-blocking).

        This method returns whatever is currently in the buffer without waiting.
        The frame may be stale (captured some time ago).

        Usage:
            Ideal for scenarios requiring zero latency or decoupled frequencies,
            such as visualization or non-critical monitoring.

        Args:
            max_age_ms: Maximum acceptable age of the frame in milliseconds.
                Raises TimeoutError if the latest frame is older than this.

        Returns:
            np.ndarray: The latest tactile data frame.

        Raises:
            TimeoutError: If the latest frame is older than max_age_ms.
            ConnectionError: If the sensor is not connected.
            RuntimeError: If no frames have been captured yet.
        """
        # Default implementation: delegates to async_read
        # Subclasses may override for true non-blocking behavior
        warnings.warn(
            f"{self.__class__.__name__}.read_latest() default implementation "
            "calls async_read(). Override for true non-blocking behavior.",
            FutureWarning,
            stacklevel=2,
        )
        return self.async_read(timeout_ms=max_age_ms)

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the sensor and release all resources."""
        pass
