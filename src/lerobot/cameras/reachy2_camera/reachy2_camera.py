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

"""
Provides the Reachy2Camera class for capturing frames from Reachy 2 cameras using Reachy 2's CameraManager.
"""

import logging
import os
import platform
import time
from threading import Event, Lock, Thread
from typing import Any

from numpy.typing import NDArray  # type: ignore  # TODO: add type stubs for numpy.typing

# Fix MSMF hardware transform compatibility for Windows before importing cv2
if platform.system() == "Windows" and "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS" not in os.environ:
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2  # type: ignore  # TODO: add type stubs for OpenCV
import numpy as np  # type: ignore  # TODO: add type stubs for numpy
from reachy2_sdk.media.camera import CameraView  # type: ignore  # TODO: add type stubs for reachy2_sdk
from reachy2_sdk.media.camera_manager import (  # type: ignore  # TODO: add type stubs for reachy2_sdk
    CameraManager,
)

from lerobot.utils.errors import DeviceNotConnectedError

from ..camera import Camera
from .configuration_reachy2_camera import ColorMode, Reachy2CameraConfig

logger = logging.getLogger(__name__)


class Reachy2Camera(Camera):
    """
    Manages Reachy 2 camera using Reachy 2 CameraManager.

    This class provides a high-level interface to connect to, configure, and read
    frames from Reachy 2 cameras. It supports both synchronous and asynchronous
    frame reading.

    An Reachy2Camera instance requires a camera name (e.g., "teleop") and an image
    type (e.g., "left") to be specified in the configuration.

    The camera's default settings (FPS, resolution, color mode) are used unless
    overridden in the configuration.
    """

    def __init__(self, config: Reachy2CameraConfig):
        """
        Initializes the Reachy2Camera instance.

        Args:
            config: The configuration settings for the camera.
        """
        super().__init__(config)

        self.config = config

        self.fps = config.fps
        self.color_mode = config.color_mode

        self.cam_manager: CameraManager | None = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: NDArray[Any] | None = None
        self.new_frame_event: Event = Event()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.config.name}, {self.config.image_type})"

    @property
    def is_connected(self) -> bool:
        """Checks if the camera is currently connected and opened."""
        if self.config.name == "teleop":
            return bool(
                self.cam_manager._grpc_connected and self.cam_manager.teleop if self.cam_manager else False
            )
        elif self.config.name == "depth":
            return bool(
                self.cam_manager._grpc_connected and self.cam_manager.depth if self.cam_manager else False
            )
        else:
            raise ValueError(f"Invalid camera name '{self.config.name}'. Expected 'teleop' or 'depth'.")

    def connect(self, warmup: bool = True) -> None:
        """
        Connects to the Reachy2 CameraManager as specified in the configuration.
        """
        self.cam_manager = CameraManager(host=self.config.ip_address, port=self.config.port)
        self.cam_manager.initialize_cameras()

        logger.info(f"{self} connected.")

    @staticmethod
    def find_cameras(ip_address: str = "localhost", port: int = 50065) -> list[dict[str, Any]]:
        """
        Detects available Reachy 2 cameras.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries,
            where each dictionary contains 'name', 'stereo',
            and the default profile properties (width, height, fps).
        """
        initialized_cameras = []
        camera_manager = CameraManager(host=ip_address, port=port)

        for camera in [camera_manager.teleop, camera_manager.depth]:
            if camera is None:
                continue

            height, width, _, _, _, _, _ = camera.get_parameters()

            camera_info = {
                "name": camera._cam_info.name,
                "stereo": camera._cam_info.stereo,
                "default_profile": {
                    "width": width,
                    "height": height,
                    "fps": 30,
                },
            }
            initialized_cameras.append(camera_info)

        camera_manager.disconnect()
        return initialized_cameras

    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any]:
        """
        Reads a single frame synchronously from the camera.

        This is a blocking call.

        Args:
            color_mode (Optional[ColorMode]): If specified, overrides the default
                color mode (`self.color_mode`) for this read operation (e.g.,
                request RGB even if default is BGR).

        Returns:
            np.ndarray: The captured frame as a NumPy array in the format
                       (height, width, channels), using the specified or default
                       color mode and applying any configured rotation.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start_time = time.perf_counter()

        frame: NDArray[Any] = np.empty((0, 0, 3), dtype=np.uint8)

        if self.cam_manager is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        else:
            if self.config.name == "teleop" and hasattr(self.cam_manager, "teleop"):
                if self.config.image_type == "left":
                    frame = self.cam_manager.teleop.get_frame(CameraView.LEFT, size=(640, 480))[0]
                elif self.config.image_type == "right":
                    frame = self.cam_manager.teleop.get_frame(CameraView.RIGHT, size=(640, 480))[0]
            elif self.config.name == "depth" and hasattr(self.cam_manager, "depth"):
                if self.config.image_type == "depth":
                    frame = self.cam_manager.depth.get_depth_frame()[0]
                elif self.config.image_type == "rgb":
                    frame = self.cam_manager.depth.get_frame(size=(640, 480))[0]

            if frame is None:
                return np.empty((0, 0, 3), dtype=np.uint8)

            if self.config.color_mode == "rgb":
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")

        return frame

    def _read_loop(self) -> None:
        """
        Internal loop run by the background thread for asynchronous reading.

        On each iteration:
        1. Reads a color frame
        2. Stores result in latest_frame (thread-safe)
        3. Sets new_frame_event to notify listeners

        Stops on DeviceNotConnectedError, logs other errors and continues.
        """
        if self.stop_event is None:
            raise RuntimeError(f"{self}: stop_event is not initialized before starting read loop.")

        while not self.stop_event.is_set():
            try:
                color_image = self.read()

                with self.frame_lock:
                    self.latest_frame = color_image
                self.new_frame_event.set()

            except DeviceNotConnectedError:
                break
            except Exception as e:
                logger.warning(f"Error reading frame in background thread for {self}: {e}")

    def _start_read_thread(self) -> None:
        """Starts or restarts the background read thread if it's not running."""
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.1)
        if self.stop_event is not None:
            self.stop_event.set()

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, args=(), name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()

    def _stop_read_thread(self) -> None:
        """Signals the background read thread to stop and waits for it to join."""
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None

    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """
        Reads the latest available frame asynchronously.

        This method retrieves the most recent frame captured by the background
        read thread. It does not block waiting for the camera hardware directly,
        but may wait up to timeout_ms for the background thread to provide a frame.

        Args:
            timeout_ms (float): Maximum time in milliseconds to wait for a frame
                to become available. Defaults to 200ms (0.2 seconds).

        Returns:
            np.ndarray: The latest captured frame as a NumPy array in the format
                       (height, width, channels), processed according to configuration.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            TimeoutError: If no frame becomes available within the specified timeout.
            RuntimeError: If an unexpected error occurs.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            thread_alive = self.thread is not None and self.thread.is_alive()
            raise TimeoutError(
                f"Timed out waiting for frame from camera {self} after {timeout_ms} ms. "
                f"Read thread alive: {thread_alive}."
            )

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")

        return frame

    def disconnect(self) -> None:
        """
        Stops the background read thread (if running).

        Raises:
            DeviceNotConnectedError: If the camera is already disconnected.
        """
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(f"{self} not connected.")

        if self.thread is not None:
            self._stop_read_thread()

        if self.cam_manager is not None:
            self.cam_manager.disconnect()

        logger.info(f"{self} disconnected.")
