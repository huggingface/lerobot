#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
ZMQCamera - Captures frames from remote cameras via ZeroMQ using JSON protocol in the
following format:
    {
        "timestamps": {"camera_name": float},
        "images": {"camera_name": "<base64-jpeg>"}
    }
"""

import base64
import json
import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from .configuration_zmq import ZMQCameraConfig

logger = logging.getLogger(__name__)


class ZMQCamera(Camera):
    """
    Manages camera interactions via ZeroMQ for receiving frames from a remote server.

    This class connects to a ZMQ Publisher, subscribes to frame topics, and decodes
    incoming JSON messages containing Base64 encoded images. It supports both
    synchronous and asynchronous frame reading patterns.

    Example usage:
        ```python
        from lerobot.cameras.zmq import ZMQCamera, ZMQCameraConfig

        config = ZMQCameraConfig(server_address="192.168.123.164", port=5555, camera_name="head_camera")
        camera = ZMQCamera(config)
        camera.connect()

        # Read 1 frame synchronously (blocking)
        color_image = camera.read()

        # Read 1 frame asynchronously (waits for new frame with a timeout)
        async_image = camera.async_read()

        # Get the latest frame immediately (no wait, returns timestamp)
        latest_image, timestamp = camera.read_latest()

        camera.disconnect()
        ```
    """

    def __init__(self, config: ZMQCameraConfig):
        super().__init__(config)
        import zmq

        self.config = config
        self.server_address = config.server_address
        self.port = config.port
        self.camera_name = config.camera_name
        self.color_mode = config.color_mode
        self.timeout_ms = config.timeout_ms

        # ZMQ Context and Socket
        self.context: zmq.Context | None = None
        self.socket: zmq.Socket | None = None
        self._connected = False

        # Threading resources
        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self.new_frame_event: Event = Event()

    def __str__(self) -> str:
        return f"ZMQCamera({self.camera_name}@{self.server_address}:{self.port})"

    @property
    def is_connected(self) -> bool:
        """Checks if the ZMQ socket is initialized and connected."""
        return self._connected and self.context is not None and self.socket is not None

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        """Connect to ZMQ camera server.

        Args:
            warmup (bool): If True, waits for the camera to provide at least one
                           valid frame before returning. Defaults to True.
        """

        logger.info(f"Connecting to {self}...")

        try:
            import zmq

            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.SUB)
            self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
            self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            self.socket.setsockopt(zmq.CONFLATE, True)
            self.socket.connect(f"tcp://{self.server_address}:{self.port}")
            self._connected = True

            # Auto-detect resolution if not provided
            if self.width is None or self.height is None:
                # Read directly from hardware because the thread isn't running yet
                temp_frame = self._read_from_hardware()
                h, w = temp_frame.shape[:2]
                self.height = h
                self.width = w
                logger.info(f"{self} resolution detected: {w}x{h}")

            self._start_read_thread()
            logger.info(f"{self} connected.")

            if warmup:
                # Ensure we have captured at least one frame via the thread
                start_time = time.time()
                while time.time() - start_time < (self.config.warmup_s):  # Wait a bit more than timeout
                    self.async_read(timeout_ms=self.config.warmup_s * 1000)
                    time.sleep(0.1)

                with self.frame_lock:
                    if self.latest_frame is None:
                        raise ConnectionError(f"{self} failed to capture frames during warmup.")

        except Exception as e:
            self._cleanup()
            raise RuntimeError(f"Failed to connect to {self}: {e}") from e

    def _cleanup(self):
        """Clean up ZMQ resources."""
        self._connected = False
        if self.socket:
            self.socket.close()
            self.socket = None
        if self.context:
            self.context.term()
            self.context = None

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        Detection not implemented for ZMQ cameras. These cameras require manual configuration (server address/port).
        """
        raise NotImplementedError("Camera detection is not implemented for ZMQ cameras.")

    def _read_from_hardware(self) -> NDArray[Any]:
        """
        Reads a single frame directly from the ZMQ socket.
        """
        if not self.is_connected or self.socket is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        try:
            message = self.socket.recv_string()
        except Exception as e:
            # Check for ZMQ timeout (EAGAIN/Again) without requiring global zmq import
            if type(e).__name__ == "Again":
                raise TimeoutError(f"{self} timeout after {self.timeout_ms}ms") from e
            raise

        # Decode JSON message
        data = json.loads(message)

        if "images" not in data:
            raise RuntimeError(f"{self} invalid message: missing 'images' key")

        images = data["images"]

        # Get image by camera name or first available
        if self.camera_name in images:
            img_b64 = images[self.camera_name]
        elif images:
            img_b64 = next(iter(images.values()))
        else:
            raise RuntimeError(f"{self} no images in message")

        # Decode base64 JPEG
        img_bytes = base64.b64decode(img_b64)
        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        if frame is None:
            raise RuntimeError(f"{self} failed to decode image")

        return frame

    @check_if_not_connected
    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any]:
        """
        Reads a single frame synchronously from the camera.

        This is a blocking call. It waits for the next available frame from the
        camera background thread.

        Returns:
            np.ndarray: Decoded frame (height, width, 3)
        """
        start_time = time.perf_counter()

        if color_mode is not None:
            logger.warning(
                f"{self} read() color_mode parameter is deprecated and will be removed in future versions."
            )

        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        self.new_frame_event.clear()
        frame = self.async_read(timeout_ms=10000)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")

        return frame

    def _read_loop(self) -> None:
        """
        Internal loop run by the background thread for asynchronous reading.
        """
        if self.stop_event is None:
            raise RuntimeError(f"{self}: stop_event is not initialized.")

        failure_count = 0
        while not self.stop_event.is_set():
            try:
                frame = self._read_from_hardware()
                capture_time = time.perf_counter()

                with self.frame_lock:
                    self.latest_frame = frame
                    self.latest_timestamp = capture_time
                self.new_frame_event.set()
                failure_count = 0

            except DeviceNotConnectedError:
                break
            except (TimeoutError, Exception) as e:
                if failure_count <= 10:
                    failure_count += 1
                    logger.warning(f"Read error: {e}")
                else:
                    raise RuntimeError(f"{self} exceeded maximum consecutive read failures.") from e

    def _start_read_thread(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        with self.frame_lock:
            self.latest_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, daemon=True, name=f"{self}_read_loop")
        self.thread.start()
        time.sleep(0.1)

    def _stop_read_thread(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None

        with self.frame_lock:
            self.latest_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """
        Reads the latest available frame asynchronously.

        Args:
            timeout_ms (float): Maximum time in milliseconds to wait for a frame
                to become available. Defaults to 200ms.

        Returns:
            np.ndarray: The latest captured frame.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            TimeoutError: If no frame data becomes available within the specified timeout.
            RuntimeError: If the background thread is not running.
        """

        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"{self} async_read timeout after {timeout_ms}ms")

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"{self} no frame available")

        return frame

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 1000) -> NDArray[Any]:
        """Return the most recent frame captured immediately (Peeking).

        This method is non-blocking and returns whatever is currently in the
        memory buffer. The frame may be stale,
        meaning it could have been captured a while ago (hanging camera scenario e.g.).

        Returns:
            NDArray[Any]: The frame image (numpy array).

        Raises:
            TimeoutError: If the latest frame is older than `max_age_ms`.
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If the camera is connected but has not captured any frames yet.
        """

        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        with self.frame_lock:
            frame = self.latest_frame
            timestamp = self.latest_timestamp

        if frame is None or timestamp is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")

        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"{self} latest frame is too old: {age_ms:.1f} ms (max allowed: {max_age_ms} ms)."
            )

        return frame

    def disconnect(self) -> None:
        """Disconnect from ZMQ camera."""
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(f"{self} not connected.")

        if self.thread is not None:
            self._stop_read_thread()

        self._cleanup()

        with self.frame_lock:
            self.latest_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

        logger.info(f"{self} disconnected.")
