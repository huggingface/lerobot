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

"""Tac3D tactile sensor implementation.

Requires the PyTac3D SDK: https://github.com/tactile-sensor/Tac3D-Python
"""

import contextlib
import importlib.util
import logging
import threading
import time
from queue import Empty, Queue
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..tactile import TactileSensor
from .configuration_tac3d import Tac3DConfig

logger = logging.getLogger(__name__)


class Tac3DTactile(TactileSensor):
    """Tac3D high-resolution tactile sensor implementation.

    Interfaces with Tac3D tactile sensors via the PyTac3D SDK, providing
    6D point cloud data (3D displacement + 3D force) for a 20x20 sensing array.

    Args:
        config: Tac3DConfig with sensor parameters.

    Raises:
        ImportError: If PyTac3D SDK is not installed.

    Example:
        ```python
        from lerobot.tactile.tac3d import Tac3DTactile, Tac3DConfig

        config = Tac3DConfig(udp_port=9988)
        with Tac3DTactile(config) as sensor:
            data = sensor.read()  # (400, 6) array
        ```
    """

    def __init__(self, config: Tac3DConfig):
        super().__init__(config)
        self.config = config
        self._sensor = None
        self._is_connected = False
        self._frame_queue: Queue[NDArray[np.float64]] = Queue(maxsize=2)
        self._capture_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Tare calibration
        self._tare_offset: NDArray[np.float64] | None = None
        self._is_tared = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @staticmethod
    def find_sensors() -> list[dict[str, Any]]:
        """Detect available Tac3D sensors.

        Returns:
            list: Detected sensors with id and type information.
        """
        if importlib.util.find_spec("PyTac3D") is None:
            logger.warning("PyTac3D SDK not installed. Cannot detect Tac3D sensors.")
            return []

        return [{"id": "tac3d_default", "type": "tac3d"}]

    def connect(self, warmup: bool = True) -> None:
        """Connect to the Tac3D sensor.

        Args:
            warmup: If True, captures warmup frames after connecting.

        Raises:
            ImportError: If PyTac3D SDK is not installed.
            ConnectionError: If sensor connection fails.
        """
        try:
            from PyTac3D import PyTac3D_Sensor
        except ImportError as e:
            raise ImportError(
                "PyTac3D SDK is required for Tac3D sensors. "
                "Install from: https://github.com/tactile-sensor/Tac3D-Python"
            ) from e

        self._sensor = PyTac3D_Sensor(
            udpPort=self.config.udp_port,
            SN=self.config.sensor_sn,
        )

        self._is_connected = True

        # Start background capture thread
        self._stop_event.clear()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        if warmup:
            time.sleep(0.5)  # Allow sensor to stabilize

        logger.info(f"Tac3D sensor connected on UDP port {self.config.udp_port}")

        if self.config.tare_on_startup:
            self.tare()

    def disconnect(self) -> None:
        """Disconnect from the Tac3D sensor."""
        self._stop_event.set()

        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None

        self._is_connected = False
        logger.info("Tac3D sensor disconnected")

    def read(self) -> NDArray[np.float64]:
        """Capture a single tactile frame synchronously.

        Returns:
            np.ndarray: Tactile data with shape (400, 6) containing
                [dx, dy, dz, Fx, Fy, Fz] for each of the 400 sensing points.

        Raises:
            ConnectionError: If sensor is not connected.
            TimeoutError: If no frame available within timeout.
        """
        if not self._is_connected:
            raise ConnectionError("Sensor is not connected. Call connect() first.")

        try:
            frame = self._frame_queue.get(timeout=self.config.timeout_ms / 1000.0)
            return self._apply_tare(frame)
        except Empty as e:
            raise TimeoutError(f"No frame received within {self.config.timeout_ms}ms") from e

    def async_read(self, timeout_ms: float = 1000.0) -> NDArray[np.float64]:
        """Return the most recent new tactile frame.

        Args:
            timeout_ms: Maximum time to wait for a new frame.

        Returns:
            np.ndarray: Tactile data with shape (400, 6).

        Raises:
            TimeoutError: If no new frame arrives within timeout.
        """
        if not self._is_connected:
            raise ConnectionError("Sensor is not connected.")

        try:
            frame = self._frame_queue.get(timeout=timeout_ms / 1000.0)
            return self._apply_tare(frame)
        except Empty as e:
            raise TimeoutError(f"No frame received within {timeout_ms}ms") from e

    def tare(self, num_samples: int = 10) -> None:
        """Perform tare/zeroing calibration.

        Captures samples and computes offset to subtract from future readings.

        Args:
            num_samples: Number of samples to average for tare.
        """
        if not self._is_connected:
            raise ConnectionError("Cannot tare: sensor is not connected")

        logger.info(f"Performing tare with {num_samples} samples...")
        samples = []

        for i in range(num_samples):
            try:
                frame = self._frame_queue.get(timeout=self.config.timeout_ms / 1000.0)
                samples.append(frame)
            except Empty as e:
                raise RuntimeError(f"Failed to capture tare sample {i}") from e

        self._tare_offset = np.mean(samples, axis=0)
        self._is_tared = True
        logger.info("Tare calibration completed")

    def _apply_tare(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply tare offset if calibrated."""
        if self._is_tared and self._tare_offset is not None:
            return data - self._tare_offset
        return data

    def _capture_loop(self) -> None:
        """Background thread for continuous frame capture.

        Sleeps based on configured FPS to avoid CPU spinning.
        """
        frame_interval = 1.0 / max(self.fps, 1)  # seconds per frame

        while not self._stop_event.is_set():
            loop_start = time.perf_counter()

            try:
                if self._sensor is not None:
                    data = self._sensor.get_TactileData()
                    if data is not None:
                        frame = np.array(data, dtype=np.float64).reshape(self.num_points, self.data_dim)
                        # Non-blocking put, drop oldest if full
                        if self._frame_queue.full():
                            with contextlib.suppress(Empty):
                                self._frame_queue.get_nowait()
                        self._frame_queue.put_nowait(frame)
            except Exception as e:
                logger.warning(f"Frame capture error: {e}")

            # Sleep for remaining frame interval to avoid CPU spinning
            elapsed = time.perf_counter() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
