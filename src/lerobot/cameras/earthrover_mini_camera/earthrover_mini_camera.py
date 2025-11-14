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
Provides the OpenCVCamera class for capturing frames from cameras using OpenCV.
"""

import logging
import math
import os
import platform
import time
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

# Fix MSMF hardware transform compatibility for Windows before importing cv2
if platform.system() == "Windows" and "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS" not in os.environ:
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..camera import Camera
from ..utils import get_cv2_backend, get_cv2_rotation
from .configuration_earthrover_mini import ColorMode, EarthRoverMiniCameraConfig

logger = logging.getLogger(__name__)

class EarthRoverMiniCamera(Camera):
    """
    Manages camera interactions using OpenCV for efficient frame recording.

    This class provides a high-level interface to connect to, configure, and read
    frames from cameras compatible with OpenCV's VideoCapture. It supports both
    synchronous and asynchronous frame reading.

    There are 4 camera paths available, the front main camera, front sub camera,
    rear camera, and rear main sub camera. Main cameras provide higher resolution
    outputs than sub cameras, but display a similar image. These paths are defined
    as FRONT_CAM_MAIN, FRONT_CAM_SUB, REAR_CAM_MAIN, REAR_CAM_SUB in the 
    EarthRoverMiniCameraConfig class in configuration_earthrover_mini.py.

    The camera's default settings (FPS, resolution, color mode) are used unless
    overridden in the configuration. FPS and resolution are locked at 30 FPS and
    1920 x 1080 resolution, and will throw an error if modified in the config.

    Example:
        ```python
        from lerobot.cameras.earthrover_mini_camera import EarthRoverMiniCamera
        from lerobot.cameras.earthrover_mini_camera.configuration_earthrover_mini import EarthRoverMiniCameraConfig, ColorMode

        # Front main camera config
        config = EarthRoverMiniCameraConfig(
            index_or_path=EarthRoverMiniCameraConfig.FRONT_CAM_MAIN, 
            color_mode=ColorMode.RGB
        )

        # Instantiate the camera
        camera = EarthRoverMiniCamera(config)

        # Connect to the camera
        camera.connect()

        # Display frames in a loop
        try:
            while True:
                frame = camera.read() 
                cv2.imshow("Earth Rover Camera", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            # Disconnect and cleanup
            camera.disconnect()
            cv2.destroyAllWindows()
        ```
    """

    def __init__(self, config: EarthRoverMiniCameraConfig):
        """
        Initializes the EarthRoverMiniCamera instance.

        Args:
            config: The configuration settings for the camera.
        """
        super().__init__(config)

        self.config = config
        self.index_or_path = config.index_or_path

        self.fps = config.fps
        self.color_mode = config.color_mode
        self.warmup_s = config.warmup_s

        self.videocapture: cv2.VideoCapture | None = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: np.ndarray | None = None
        self.new_frame_event: Event = Event()

        self.rotation: int | None = get_cv2_rotation(config.rotation)
        self.backend: int = get_cv2_backend()

        if self.height and self.width:
            self.capture_width, self.capture_height = self.width, self.height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.capture_width, self.capture_height = self.height, self.width

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.index_or_path})"

    @property
    def is_connected(self) -> bool:
        """Checks if the camera is currently connected and opened."""
        return isinstance(self.videocapture, cv2.VideoCapture) and self.videocapture.isOpened()

    def connect(self, warmup: bool = True):
        """
        Connects to the EarthRoverMini camera specified in the configuration.

        Initializes the OpenCV VideoCapture object, sets desired camera properties
        (FPS, width, height), and performs initial checks.

        Raises:
            DeviceAlreadyConnectedError: If the camera is already connected.
            ConnectionError: If the specified camera index/path is not found or the camera is found but fails to open.
            RuntimeError: If the camera opens but fails to apply requested FPS/resolution settings.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        # Use 1 thread for OpenCV operations to avoid potential conflicts or
        # blocking in multi-threaded applications, especially during data collection.
        cv2.setNumThreads(1)

        self.videocapture = cv2.VideoCapture(self.index_or_path, self.backend)

        if not self.videocapture.isOpened():
            self.videocapture.release()
            self.videocapture = None
            raise ConnectionError(
                f"Failed to open {self}.Run `lerobot-find-cameras opencv` to find available cameras."
            )

        if warmup:
            start_time = time.time()
            while time.time() - start_time < self.warmup_s:
                self.read()
                time.sleep(0.1)

        logger.info(f"{self} connected.")

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        Detects available EarthRoverMini cameras connected to the system.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries,
            where each dictionary contains 'type', 'id' (port index or path),
            and the default profile properties (width, height, fps, format).
        """

        found_cameras_info = []

        targets_to_scan = [EarthRoverMiniCameraConfig.FRONT_CAM_MAIN, 
                           EarthRoverMiniCameraConfig.FRONT_CAM_SUB, 
                           EarthRoverMiniCameraConfig.REAR_CAM_MAIN,
                           EarthRoverMiniCameraConfig.REAR_CAM_SUB]
        
        target_names = ["Front Main", "Front Sub", "Rear Main", "Rear Sub"]

        for target, target_name in zip(targets_to_scan, target_names):
            camera = cv2.VideoCapture(target)
            if camera.isOpened():
                default_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                default_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                default_fps = camera.get(cv2.CAP_PROP_FPS)
                default_format = camera.get(cv2.CAP_PROP_FORMAT)
                camera_info = {
                    "name": f"{target_name} Camera @ {target}",
                    "type": "OpenCV",
                    "id": target,
                    "backend_api": camera.getBackendName(),
                    "default_stream_profile": {
                        "format": default_format,
                        "width": default_width,
                        "height": default_height,
                        "fps": default_fps,
                    },
                }

                found_cameras_info.append(camera_info)
                camera.release()

        return found_cameras_info

    def read(self, color_mode: ColorMode | None = None) -> np.ndarray:
        """
        Reads a single frame synchronously from the camera.

        This is a blocking call. It waits for the next available frame from the
        camera hardware via OpenCV.

        Args:
            color_mode (Optional[ColorMode]): If specified, overrides the default
                color mode (`self.color_mode`) for this read operation (e.g.,
                request RGB even if default is BGR).

        Returns:
            np.ndarray: The captured frame as a NumPy array in the format
                       (height, width, channels), using the specified or default
                       color mode and applying any configured rotation.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If reading the frame from the camera fails or if the
                          received frame dimensions don't match expectations before rotation.
            ValueError: If an invalid `color_mode` is requested.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start_time = time.perf_counter()

        ret, frame = self.videocapture.read()

        if not ret or frame is None:
            raise RuntimeError(f"{self} read failed (status={ret}).")

        processed_frame = self._postprocess_image(frame, color_mode)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")

        return processed_frame

    def _postprocess_image(self, image: np.ndarray, color_mode: ColorMode | None = None) -> np.ndarray:
        """
        Applies color conversion, dimension validation, and rotation to a raw frame.

        Args:
            image (np.ndarray): The raw image frame (expected BGR format from OpenCV).
            color_mode (Optional[ColorMode]): The target color mode (RGB or BGR). If None,
                                             uses the instance's default `self.color_mode`.

        Returns:
            np.ndarray: The processed image frame.

        Raises:
            ValueError: If the requested `color_mode` is invalid.
            RuntimeError: If the raw frame dimensions do not match the configured
                          `width` and `height`.
        """
        requested_color_mode = self.color_mode if color_mode is None else color_mode

        if requested_color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"Invalid color mode '{requested_color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )

        h, w, c = image.shape

        if c != 3:
            raise RuntimeError(f"{self} frame channels={c} do not match expected 3 channels (RGB/BGR).")

        processed_image = image
        if requested_color_mode == ColorMode.RGB:
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed_image = cv2.rotate(processed_image, self.rotation)

        return processed_image

    def _read_loop(self):
        """
        Internal loop run by the background thread for asynchronous reading.

        On each iteration:
        1. Reads a color frame
        2. Stores result in latest_frame (thread-safe)
        3. Sets new_frame_event to notify listeners

        Stops on DeviceNotConnectedError, logs other errors and continues.
        """
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

    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
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

    def disconnect(self):
        """
        Disconnects from the camera and cleans up resources.

        Stops the background read thread (if running) and releases the OpenCV
        VideoCapture object.

        Raises:
            DeviceNotConnectedError: If the camera is already disconnected.
        """
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(f"{self} not connected.")

        if self.thread is not None:
            self._stop_read_thread()

        if self.videocapture is not None:
            self.videocapture.release()
            self.videocapture = None

        logger.info(f"{self} disconnected.")
