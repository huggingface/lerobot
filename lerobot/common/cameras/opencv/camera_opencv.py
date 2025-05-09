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

import contextlib
import logging
import math
import platform
import queue
import time
from pathlib import Path
from threading import Event, Thread
from typing import Any, Dict, List

import cv2
import numpy as np

from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.utils.utils import capture_timestamp_utc

from ..camera import Camera
from ..utils import IndexOrPath, get_cv2_backend, get_cv2_rotation
from .configuration_opencv import ColorMode, OpenCVCameraConfig

# The maximum opencv device index depends on your operating system. For instance,
# if you have 3 cameras, they should be associated to index 0, 1, and 2. This is the case
# on MacOS. However, on Ubuntu, the indices are different like 6, 16, 23.
# When you change the USB port or reboot the computer, the operating system might
# treat the same cameras as new devices. Thus we select a higher bound to search indices.
MAX_OPENCV_INDEX = 60

logger = logging.getLogger(__name__)


class OpenCVCamera(Camera):
    """
    Manages camera interactions using OpenCV for efficient frame recording.

    This class provides a high-level interface to connect to, configure, and read
    frames from cameras compatible with OpenCV's VideoCapture. It supports both
    synchronous and asynchronous frame reading.

    An OpenCVCamera instance requires a camera index (e.g., 0) or a device path
    (e.g., '/dev/video0' on Linux). Camera indices can be unstable across reboots
    or port changes, especially on Linux. Use the provided utility script to find
    available camera indices or paths:
    ```bash
    python -m lerobot.find_cameras
    ```

    The camera's default settings (FPS, resolution, color mode) are used unless
    overridden in the configuration.

    Args:
        config (OpenCVCameraConfig): Configuration object containing settings like
            camera index/path, desired FPS, width, height, color mode, and rotation.

    Example:
        ```python
        from lerobot.common.cameras.opencv import OpenCVCamera
        from lerobot.common.cameras.configuration_opencv import OpenCVCameraConfig, ColorMode

        # Basic usage with camera index 0
        config = OpenCVCameraConfig(index_or_path=0)
        camera = OpenCVCamera(config)
        try:
            camera.connect()
            print(f"Connected to {camera}")
            color_image = camera.read() # Synchronous read
            print(f"Read frame shape: {color_image.shape}")
            async_image = camera.async_read() # Asynchronous read
            print(f"Async read frame shape: {async_image.shape}")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            camera.disconnect()
            print(f"Disconnected from {camera}")

        # Example with custom settings
        custom_config = OpenCVCameraConfig(
            index_or_path='/dev/video0', # Or use an index
            fps=30,
            width=1280,
            height=720,
            color_mode=ColorMode.RGB,
            rotation=90
        )
        custom_camera = OpenCVCamera(custom_config)
        # ... connect, read, disconnect ...
        ```
    """

    def __init__(self, config: OpenCVCameraConfig):
        """
        Initializes the OpenCVCamera instance.

        Args:
            config: The configuration settings for the camera.
        """
        self.config = config
        self.index_or_path: IndexOrPath = config.index_or_path

        self.capture_width: int | None = config.width
        self.capture_height: int | None = config.height
        self.width: int | None = None
        self.height: int | None = None

        self.fps: int | None = config.fps
        self.channels: int = config.channels
        self.color_mode: ColorMode = config.color_mode

        self.videocapture_camera: cv2.VideoCapture | None = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_queue: queue.Queue = queue.Queue(maxsize=1)

        self.logs: dict = {}  # NOTE(Steven): Might be removed in the future

        self.rotation: int | None = get_cv2_rotation(config.rotation)
        self.backend: int = get_cv2_backend()  # NOTE(Steven): If I specify backend the opencv open fails

    def __str__(self) -> str:
        """Returns a string representation of the camera instance."""
        return f"{self.__class__.__name__}({self.index_or_path})"

    @property
    def is_connected(self) -> bool:
        """Checks if the camera is currently connected and opened."""
        return isinstance(self.videocapture_camera, cv2.VideoCapture) and self.videocapture_camera.isOpened()

    def _configure_capture_settings(self) -> None:
        """
        Applies the specified FPS, width, and height settings to the connected camera.

        This method attempts to set the camera properties via OpenCV. It checks if
        the camera successfully applied the settings and raises an error if not.

        Args:
            fps: The desired frames per second. If None, the setting is skipped.
            width: The desired capture width. If None, the setting is skipped.
            height: The desired capture height. If None, the setting is skipped.

        Raises:
            RuntimeError: If the camera fails to set any of the specified properties
                          to the requested value.
            DeviceNotConnectedError: If the camera is not connected when attempting
                                     to configure settings.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"Cannot configure settings for {self} as it is not connected.")

        self._validate_fps()
        self._validate_capture_width()
        self._validate_capture_height()

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            self.width, self.height = self.capture_height, self.capture_width
        else:
            self.width, self.height = self.capture_width, self.capture_height
        logger.debug(f"Final image dimensions set to: {self.width}x{self.height} (after rotation if any)")

    def connect(self):
        """
        Connects to the OpenCV camera specified in the configuration.

        Initializes the OpenCV VideoCapture object, sets desired camera properties
        (FPS, width, height), and performs initial checks.

        Raises:
            DeviceAlreadyConnectedError: If the camera is already connected.
            ValueError: If the specified camera index/path is not found or accessible.
            ConnectionError: If the camera is found but fails to open.
            RuntimeError: If the camera opens but fails to apply requested FPS/resolution settings.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        # Use 1 thread for OpenCV operations to avoid potential conflicts or
        # blocking in multi-threaded applications, especially during data collection.
        cv2.setNumThreads(1)

        logger.debug(f"Attempting to connect to camera {self.index_or_path} using backend {self.backend}...")
        self.videocapture_camera = cv2.VideoCapture(self.index_or_path)

        if not self.videocapture_camera.isOpened():
            self.videocapture_camera.release()
            self.videocapture_camera = None
            raise ConnectionError(
                f"Failed to open OpenCV camera {self.index_or_path}."
                f"Run 'python -m find_cameras list-cameras' for details."
            )

        logger.debug(f"Successfully opened camera {self.index_or_path}. Applying configuration...")
        self._configure_capture_settings()
        logger.debug(f"Camera {self.index_or_path} connected and configured successfully.")

    def _validate_fps(self) -> None:
        """Validates and sets the camera's frames per second (FPS)."""

        if self.fps is None:
            self.fps = self.videocapture_camera.get(cv2.CAP_PROP_FPS)
            logger.info(f"FPS set to camera default: {self.fps}.")
            return

        success = self.videocapture_camera.set(cv2.CAP_PROP_FPS, float(self.fps))
        actual_fps = self.videocapture_camera.get(cv2.CAP_PROP_FPS)
        # Use math.isclose for robust float comparison
        if not success or not math.isclose(self.fps, actual_fps, rel_tol=1e-3):
            logger.warning(
                f"Requested FPS {self.fps} for {self}, but camera reported {actual_fps} (set success: {success}). "
                "This might be due to camera limitations."
            )
            raise RuntimeError(
                f"Failed to set requested FPS {self.fps} for {self}. Actual value reported: {actual_fps}."
            )
        logger.debug(f"FPS set to {actual_fps} for {self}.")

    def _validate_capture_width(self) -> None:
        """Validates and sets the camera's frame capture width."""

        actual_width = int(round(self.videocapture_camera.get(cv2.CAP_PROP_FRAME_WIDTH)))

        if self.capture_width is None:
            self.capture_width = actual_width
            logger.info(f"Capture width set to camera default: {self.capture_width}.")
            return

        success = self.videocapture_camera.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.capture_width))
        if not success or self.capture_width != actual_width:
            logger.warning(
                f"Requested capture width {self.capture_width} for {self}, but camera reported {actual_width} (set success: {success})."
            )
            raise RuntimeError(
                f"Failed to set requested capture width {self.capture_width} for {self}. Actual value: {actual_width}."
            )
        logger.debug(f"Capture width set to {actual_width} for {self}.")

    def _validate_capture_height(self) -> None:
        """Validates and sets the camera's frame capture height."""

        actual_height = int(round(self.videocapture_camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        if self.capture_height is None:
            self.capture_height = actual_height
            logger.info(f"Capture height set to camera default: {self.capture_height}.")
            return

        success = self.videocapture_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.capture_height))
        if not success or self.capture_height != actual_height:
            logger.warning(
                f"Requested capture height {self.capture_height} for {self}, but camera reported {actual_height} (set success: {success})."
            )
            raise RuntimeError(
                f"Failed to set requested capture height {self.capture_height} for {self}. Actual value: {actual_height}."
            )
        logger.debug(f"Capture height set to {actual_height} for {self}.")

    @staticmethod
    def find_cameras(
        max_index_search_range=MAX_OPENCV_INDEX, raise_when_empty: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Detects available OpenCV cameras connected to the system.

        On Linux, it scans '/dev/video*' paths. On other systems (like macOS, Windows),
        it checks indices from 0 up to `max_index_search_range`.

        Args:
            max_index_search_range (int): The maximum index to check on non-Linux systems.
            raise_when_empty (bool): If True, raises an OSError if no cameras are found.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries,
            where each dictionary contains 'type', 'id' (port index or path),
            and the default profile properties (width, height, fps, format).
        """
        found_cameras_info = []

        if platform.system() == "Linux":
            logger.info("Linux detected. Scanning '/dev/video*' device paths...")
            possible_paths = sorted(Path("/dev").glob("video*"), key=lambda p: p.name)
            targets_to_scan = [str(p) for p in possible_paths]
            logger.debug(f"Found potential paths: {targets_to_scan}")
        else:
            logger.info(
                f"{platform.system()} system detected. Scanning indices from 0 to {max_index_search_range}..."
            )
            targets_to_scan = list(range(max_index_search_range))

        for target in targets_to_scan:
            camera = cv2.VideoCapture(target)
            if camera.isOpened():
                default_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                default_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                default_fps = camera.get(cv2.CAP_PROP_FPS)
                default_format = camera.get(cv2.CAP_PROP_FORMAT)
                camera_info = {
                    "name": f"OpenCV Camera @ {target}",
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
                logger.debug(f"Found OpenCV camera:: {camera_info}")
                camera.release()

        if not found_cameras_info:
            logger.warning("No OpenCV devices detected.")
            if raise_when_empty:
                raise OSError("No OpenCV devices detected. Ensure cameras are connected.")

        logger.info(f"Detected OpenCV cameras: {[cam['id'] for cam in found_cameras_info]}")
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

        # NOTE(Steven): Are we okay with this blocking an undefined amount of time?
        ret, frame = self.videocapture_camera.read()

        if not ret or frame is None:
            raise RuntimeError(
                f"Failed to capture frame from {self}. '.read()' returned status={ret} and frame is None."
            )

        # Post-process the frame (color conversion, dimension check, rotation)
        processed_frame = self._postprocess_image(frame, color_mode)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} synchronous read took: {read_duration_ms:.1f}ms")

        self.logs["timestamp_utc"] = capture_timestamp_utc()
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
                          `capture_width` and `capture_height`.
        """
        requested_color_mode = self.color_mode if color_mode is None else color_mode

        if requested_color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"Invalid requested color mode '{requested_color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )

        h, w, c = image.shape

        if h != self.capture_height or w != self.capture_width:
            raise RuntimeError(
                f"Captured frame dimensions ({h}x{w}) do not match configured capture dimensions ({self.capture_height}x{self.capture_width}) for {self}."
            )
        if c != self.channels:
            logger.warning(
                f"Captured frame channels ({c}) do not match configured channels ({self.channels}) for {self}."
            )

        processed_image = image
        if requested_color_mode == ColorMode.RGB:
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            logger.debug(f"Converted frame from BGR to RGB for {self}.")

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            processed_image = cv2.rotate(processed_image, self.rotation)
            logger.debug(f"Rotated frame by {self.config.rotation} degrees for {self}.")

        return processed_image

    def _read_loop(self):
        """
        Internal loop run by the background thread for asynchronous reading.

        Continuously reads frames from the camera using the synchronous `read()`
        method and places the latest frame into the `frame_queue`. It overwrites
        any previous frame in the queue.
        """
        logger.debug(f"Starting read loop thread for {self}.")
        while not self.stop_event.is_set():
            try:
                color_image = self.read()

                with contextlib.suppress(queue.Empty):
                    _ = self.frame_queue.get_nowait()
                self.frame_queue.put(color_image)
                logger.debug(f"Frame placed in queue for {self}.")

            except DeviceNotConnectedError:
                logger.error(f"Read loop for {self} stopped: Camera disconnected.")
                break
            except Exception as e:
                logger.warning(f"Error reading frame in background thread for {self}: {e}")

        logger.debug(f"Stopping read loop thread for {self}.")

    def _ensure_read_thread_running(self):
        """Starts or restarts the background read thread if it's not running."""
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.1)
        if self.stop_event is not None:
            self.stop_event.set()

        self.stop_event = Event()
        self.thread = Thread(
            target=self._read_loop, args=(), name=f"OpenCVCameraReadLoop-{self}-{self.index_or_path}"
        )
        self.thread.daemon = True
        self.thread.start()
        logger.debug(f"Read thread started for {self}.")

    def async_read(self, timeout_ms: float = 2000) -> np.ndarray:
        """
        Reads the latest available frame asynchronously.

        This method retrieves the most recent frame captured by the background
        read thread. It does not block waiting for the camera hardware directly,
        only waits for a frame to appear in the internal queue up to the specified
        timeout.

        Args:
            timeout_ms (float): Maximum time in milliseconds to wait for a frame
                to become available in the queue. Defaults to 2000ms (2 seconds).

        Returns:
            np.ndarray: The latest captured frame as a NumPy array in the format
                       (height, width, channels), processed according to configuration.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            TimeoutError: If no frame becomes available within the specified timeout.
            RuntimeError: If an unexpected error occurs while retrieving from the queue.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.thread is None or not self.thread.is_alive():
            self._ensure_read_thread_running()

        try:
            return self.frame_queue.get(timeout=timeout_ms / 1000.0)
        except queue.Empty as e:
            thread_alive = self.thread is not None and self.thread.is_alive()
            logger.error(
                f"Timeout waiting for frame from {self} queue after {timeout_ms}ms. "
                f"(Read thread alive: {thread_alive})"
            )
            raise TimeoutError(
                f"Timed out waiting for frame from camera {self.index_or_path} after {timeout_ms} ms. "
                f"Read thread alive: {thread_alive}."
            ) from e
        except Exception as e:
            logger.exception(f"Unexpected error getting frame from queue for {self}: {e}")
            raise RuntimeError(f"Error getting frame from queue for camera {self.index_or_path}: {e}") from e

    def _shutdown_read_thread(self):
        """Signals the background read thread to stop and waits for it to join."""
        if self.stop_event is not None:
            logger.debug(f"Signaling stop event for read thread of {self}.")
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            logger.debug(f"Waiting for read thread of {self} to join...")
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                logger.warning(f"Read thread for {self} did not terminate gracefully after 2 seconds.")
            else:
                logger.debug(f"Read thread for {self} joined successfully.")

        self.thread = None
        self.stop_event = None

    def disconnect(self):
        """
        Disconnects from the camera and cleans up resources.

        Stops the background read thread (if running) and releases the OpenCV
        VideoCapture object.

        Raises:
            DeviceNotConnectedError: If the camera is already disconnected.
        """
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(
                f"Attempted to disconnect {self}, but it appears already disconnected."
            )

        logger.debug(f"Disconnecting from camera {self.index_or_path}...")

        if self.thread is not None:
            self._shutdown_read_thread()

        if self.videocapture_camera is not None:
            logger.debug(f"Releasing OpenCV VideoCapture object for {self}.")
            self.videocapture_camera.release()
            self.videocapture_camera = None

        logger.info(f"Camera {self.index_or_path} disconnected successfully.")
