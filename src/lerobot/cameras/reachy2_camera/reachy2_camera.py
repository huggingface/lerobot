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

from __future__ import annotations

import logging
import os
import platform
import time
from typing import TYPE_CHECKING, Any

from numpy.typing import NDArray  # type: ignore  # TODO: add type stubs for numpy.typing

# Fix MSMF hardware transform compatibility for Windows before importing cv2
if platform.system() == "Windows" and "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS" not in os.environ:
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2  # type: ignore  # TODO: add type stubs for OpenCV
import numpy as np  # type: ignore  # TODO: add type stubs for numpy

from lerobot.utils.import_utils import _reachy2_sdk_available

if TYPE_CHECKING or _reachy2_sdk_available:
    from reachy2_sdk.media.camera import CameraView
    from reachy2_sdk.media.camera_manager import CameraManager
else:
    CameraManager = None

    class CameraView:
        LEFT = 0
        RIGHT = 1


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

        self.color_mode = config.color_mode

        self.cam_manager: CameraManager | None = None

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

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
        """
        self.cam_manager = CameraManager(host=self.config.ip_address, port=self.config.port)
        if self.cam_manager is None:
            raise DeviceNotConnectedError(f"Could not connect to {self}.")
        self.cam_manager.initialize_cameras()

        logger.info(f"{self} connected.")

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        Detection not implemented for Reachy2 cameras.
        """
        raise NotImplementedError("Camera detection is not implemented for Reachy2 cameras.")

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
        start_time = time.perf_counter()

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.cam_manager is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        frame: NDArray[Any] = np.empty((0, 0, 3), dtype=np.uint8)

        if self.config.name == "teleop" and hasattr(self.cam_manager, "teleop"):
            if self.config.image_type == "left":
                frame = self.cam_manager.teleop.get_frame(
                    CameraView.LEFT, size=(self.config.width, self.config.height)
                )[0]
            elif self.config.image_type == "right":
                frame = self.cam_manager.teleop.get_frame(
                    CameraView.RIGHT, size=(self.config.width, self.config.height)
                )[0]
        elif self.config.name == "depth" and hasattr(self.cam_manager, "depth"):
            if self.config.image_type == "depth":
                frame = self.cam_manager.depth.get_depth_frame()[0]
            elif self.config.image_type == "rgb":
                frame = self.cam_manager.depth.get_frame(size=(self.config.width, self.config.height))[0]
        else:
            raise ValueError(f"Invalid camera name '{self.config.name}'. Expected 'teleop' or 'depth'.")

        if frame is None:
            return np.empty((0, 0, 3), dtype=np.uint8)

        if self.config.color_mode == "rgb":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")

        return frame

    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """
        Reads the latest available frame.

        This method retrieves the most recent frame available in Reachy 2's low-level software.

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

        frame = self.read()

        if frame is None:
            raise RuntimeError(f"Internal error: No frame available for {self}.")

        return frame

    def disconnect(self) -> None:
        """
        Stops the background read thread (if running).

        Raises:
            DeviceNotConnectedError: If the camera is already disconnected.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} not connected.")

        if self.cam_manager is not None:
            self.cam_manager.disconnect()

        logger.info(f"{self} disconnected.")
