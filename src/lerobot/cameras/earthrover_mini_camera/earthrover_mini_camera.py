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
"""Virtual camera adapter for SDK-based robots like EarthRover Mini Plus."""

import numpy as np

from lerobot.cameras.camera import Camera

from .configuration_virtual_camera import VirtualCameraConfig


class VirtualCamera(Camera):
    """
    Virtual camera adapter that reads frames from SDK via robot's observation cache.

    This adapter allows SDK-based robots (like EarthRoverMiniPlus) to provide camera
    frames through the standard Camera interface without direct hardware access.
    The frames are retrieved from the robot's _last_observation cache, which is
    populated by SDK HTTP API calls.

    The EarthRover Mini uses cloud-based camera streaming via WebRTC.
    Cameras stream to WebRTC/Agora cloud, and the SDK connects via browser
    to extract frames served through HTTP API endpoints.

    Example:
        ```python
        from lerobot.cameras.earthrover_mini_camera import VirtualCamera, VirtualCameraConfig

        # Using VirtualCameraConfig
        config = VirtualCameraConfig(name="front", fps=30, width=640, height=480)
        camera = VirtualCamera(config, robot)

        # Or using convenience parameters (creates config internally)
        camera = VirtualCamera("front", robot, fps=30, width=640, height=480)

        # Later, after robot.get_observation() has been called:
        frame = camera.read()  # Gets frame from cache
        ```
    """

    def __init__(
        self,
        config_or_name: VirtualCameraConfig | str,
        robot,
        fps: int = 30,
        width: int = 640,
        height: int = 480,
    ):
        """Initialize the virtual camera adapter.

        Args:
            config_or_name: Either a VirtualCameraConfig object or a camera name string
            robot: Reference to the parent robot instance that provides observations
            fps: Frames per second (used if config_or_name is a string)
            width: Frame width in pixels (used if config_or_name is a string)
            height: Frame height in pixels (used if config_or_name is a string)
        """
        # Create config if a name string was provided
        if isinstance(config_or_name, str):
            config = VirtualCameraConfig(
                name=config_or_name,
                fps=fps,
                width=width,
                height=height,
            )
        else:
            config = config_or_name

        super().__init__(config)

        self.name = config.name
        self.robot = robot

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"

    @property
    def is_connected(self) -> bool:
        """Check if the camera is connected (delegates to robot connection status)."""
        return self.robot.is_connected

    def connect(self) -> None:
        """Connect to camera (no-op, connection handled by robot).

        The actual connection to SDK cameras is managed by the robot's connect() method.
        """
        pass

    def read(self) -> np.ndarray:
        """Read the latest frame from robot's observation cache.

        Returns:
            np.ndarray: Camera frame as (height, width, 3) array in the configured color format,
                       or black frame if no cached observation is available.

        Note:
            This method reads from the robot's _last_observation dictionary which
            is populated by the robot's get_observation() method. Frames are stored
            in BGR format and converted to RGB if color_mode is RGB.
        """
        import cv2

        from .configuration_virtual_camera import ColorMode

        frame = None
        if hasattr(self.robot, "_last_observation") and self.name in self.robot._last_observation:
            frame = self.robot._last_observation[self.name]

        # Return black frame if no observation available
        if frame is None:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Apply color conversion based on config
        if hasattr(self.config, "color_mode") and self.config.color_mode == ColorMode.RGB:
            # Convert BGR (OpenCV default) to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame

    def async_read(self) -> np.ndarray:
        """Read frame asynchronously (delegates to synchronous read for SDK cameras).

        Returns:
            np.ndarray: Camera frame, same as read()
        """
        return self.read()

    def disconnect(self) -> None:
        """Disconnect from camera (no-op, disconnection handled by robot).

        The actual disconnection is managed by the robot's disconnect() method.
        """
        pass

    @staticmethod
    def find_cameras() -> list:
        """Find available cameras (not applicable for virtual cameras).

        Virtual cameras don't represent physical hardware that can be discovered.
        They are created programmatically by robots that use SDK/API for camera access.

        Returns:
            list: Empty list, as virtual cameras cannot be discovered
        """
        return []
