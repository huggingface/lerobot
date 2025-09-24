# Copyright 2025 Nathan Lewis. All rights reserved.
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
Provides the RPiCamera class for capturing frames from cameras using Picamera2.
"""

import logging
import time
from typing import Any

from libcamera import Transform
from picamera2 import Picamera2
import numpy as np

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..camera import Camera
from .configuration_rpi import ColorMode, RPiCameraConfig, Cv2Rotation

logger = logging.getLogger(__name__)


class RPiCamera(Camera):
    """
    Manages camera interactions using Picamera2 for efficient frame recording.

    This class provides a high-level interface to connect to, configure, and read
    frames from cameras compatible with Raspberry Pi's Picamera2. It supports both
    synchronous and asynchronous frame reading.

    A RPiCamera instance requires a camera index (e.g., 0). Camera indices for
    the CSI ports are stable across reboots, but indicies for USB devices can
    be unstable across reboots or port changes. Use the provided utility script
    to find available camera indices:
    ```bash
    lerobot-find-cameras rpi
    ```

    The camera's default settings (FPS, resolution, color mode) are used unless
    overridden in the configuration.

    Example:
        ```python
        from lerobot.cameras.rpi import RPiCamera
        from lerobot.cameras.configuration_rpi import RPiCameraConfig, ColorMode, Cv2Rotation

        # Basic usage with camera index 0
        config = RPiCameraConfig(index=0)
        camera = RPiCamera(config)
        camera.connect()

        # Read 1 frame synchronously
        color_image = camera.read()
        print(color_image.shape)

        # Read 1 frame asynchronously
        async_image = camera.async_read()

        # When done, properly disconnect the camera using
        camera.disconnect()

        # Example with custom settings
        custom_config = RPiCameraConfig(
            index=0,
            fps=30,
            width=1280,
            height=720,
            color_mode=ColorMode.RGB,
            rotation=Cv2Rotation.ROTATE_90
        )
        custom_camera = RPiCamera(custom_config)
        # ... connect, read, disconnect ...
        ```
    """

    def __init__(self, config: RPiCameraConfig):
        """
        Initializes the RPiCamera instance.

        Args:
            config: The configuration settings for the camera.
        """
        super().__init__(config)

        self.config = config
        self.index = config.index

        self.fps = config.fps
        self.color_mode = config.color_mode
        self.warmup_s = config.warmup_s
        self.rotation = config.rotation

        self.camera: Picamera2 | None = None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.index})"

    @property
    def is_connected(self) -> bool:
        """Checks if the camera is currently connected and opened."""
        return isinstance(self.camera, Picamera2) and self.camera.started

    def connect(self, warmup: bool = True):
        """
        Connects to the Raspberry Pi camera specified in the configuration.

        Initializes the Picamera2 object, sets desired camera properties
        (FPS, width, height), and performs initial checks.

        Raises:
            DeviceAlreadyConnectedError: If the camera is already connected.
            ConnectionError: If the specified camera index is not found or the camera is found but fails to open.
            RuntimeError: If the camera opens but fails to apply requested FPS/resolution settings.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        try:
            self.camera = Picamera2(self.index)
        except IndexError:
            self.camera = None
            raise ConnectionError(
                f"Failed to open {self}.Run `lerobot-find-cameras rpi` to find available cameras."
            )

        color_mode = "BGR888"
        if self.color_mode is ColorMode.RGB:
            color_mode = "RGB888"

        main = { 'format' : color_mode }
        if self.config.width is not None and self.config.height is not None:
            main['size'] = (self.config.width, self.config.height)

        controls = {}
        if "FrameDurationLimits" in self.camera.camera_controls.keys() and self.fps is not None:
            controls['FrameRate'] = self.fps

        transform = Transform()
        if self.rotation is Cv2Rotation.ROTATE_180:
            transform = Transform(vflip=1, hflip=1)

        config = self.camera.create_video_configuration(
                main=main,
                controls=controls,
                transform=transform
        )
        self.camera.configure(config)
        self.camera.start()

        if warmup:
            start_time = time.time()
            while time.time() - start_time < self.warmup_s:
                self.read()
                time.sleep(0.1)

        logger.info(f"{self} connected.")

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        Detects available Raspberry Pi cameras connected to the system.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries,
            where each dictionary contains 'type', 'id' (port index or path),
            and the default profile properties (width, height, fps, format).
        """
        found_cameras_info = []

        for index, camera_info in enumerate(Picamera2.global_camera_info()):
            with Picamera2(index) as camera:
                default_configuration = camera.create_video_configuration()
                sensor_modes = camera.sensor_modes

            default_format = "BGR888"
            (default_width, default_height) = default_configuration['main']['size']

            # identify the sensor mode that would apply to this resolution (modes are sorted)
            default_fps : float | None = None
            for mode in sensor_modes:
                if "size" not in mode.keys():
                    continue

                (mode_width, mode_height) = mode['size']
                if default_width <= mode_width and default_height <= mode_height:
                    default_fps = mode['fps']
                    break

            found_cameras_info.append({
                "name": f"'{camera_info['Model']}' @ {camera_info['Id']}",
                "type": "RPi",
                "id": index,
                "default_stream_profile": {
                    "format": default_format,
                    "width": default_width,
                    "height": default_height,
                    "fps": default_fps,
                },
            })

        return found_cameras_info

    def read(self, color_mode: ColorMode | None = None) -> np.ndarray:
        """
        Reads a single frame synchronously from the camera.

        This is a blocking call. It waits for the next available frame from the
        camera hardware via Picamera2.

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

        frame = self.camera.capture_array(wait=True)
        if frame is None:
            raise RuntimeError(f"{self} read failed.")

        processed_frame = self._postprocess_image(frame, color_mode)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")

        return processed_frame

    def _postprocess_image(self, image: np.ndarray, color_mode: ColorMode | None = None) -> np.ndarray:
        """
        Applies color conversion, dimension validation, and rotation to a raw frame.

        Args:
            image (np.ndarray): The raw image frame (expected format specified at initialization).
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

        processed_image = image
        if requested_color_mode != self.color_mode:
            processed_image = processed_image[:, : [2, 1, 0]]

        if self.rotation is Cv2Rotation.ROTATE_90:
            processed_image = np.rot90(processed_image, k=-1)
        elif self.rotation is Cv2Rotation.ROTATE_270:
            processed_image = np.rot90(processed_image, k=1)

        return processed_image


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

        job = self.camera.capture_array(wait=False)
        frame = job.get_result(timeout=timeout_ms / 1000.0)

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")

        return self._postprocess_image(frame)


    def disconnect(self):
        """
        Disconnects from the camera and cleans up resources.

        Stops and releases the Picamera2 object.

        Raises:
            DeviceNotConnectedError: If the camera is already disconnected.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} not connected.")

        if self.camera is not None:
            self.camera.stop()
            self.camera = None

        logger.info(f"{self} disconnected.")
