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

import abc
from typing import Any

from numpy.typing import NDArray  # type: ignore  # TODO: add type stubs for numpy.typing

from .configs import CameraConfig, ColorMode


class Camera(abc.ABC):
    """Base class for camera implementations.

    Defines a standard interface for camera operations across different backends.
    Subclasses must implement all abstract methods.

    Manages basic camera properties (FPS, resolution) and core operations:
    - Connection/disconnection
    - Frame capture (sync/async/latest)

    Attributes:
        fps (int | None): Configured frames per second
        width (int | None): Frame width in pixels
        height (int | None): Frame height in pixels
    """

    def __init__(self, config: CameraConfig):
        """Initialize the camera with the given configuration.

        Args:
            config: Camera configuration containing FPS and resolution.
        """
        self.fps: int | None = config.fps
        self.width: int | None = config.width
        self.height: int | None = config.height

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """Check if the camera is currently connected.

        Returns:
            bool: True if the camera is connected and ready to capture frames,
                  False otherwise.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def find_cameras() -> list[dict[str, Any]]:
        """Detects available cameras connected to the system.
        Returns:
            List[Dict[str, Any]]: A list of dictionaries,
            where each dictionary contains information about a detected camera.
        """
        pass

    @abc.abstractmethod
    def connect(self, warmup: bool = True) -> None:
        """Establish connection to the camera.

        Args:
            warmup: If True (default), captures a warmup frame before returning. Useful
                   for cameras that require time to adjust capture settings.
                   If False, skips the warmup frame.
        """
        pass

    @abc.abstractmethod
    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any]:
        """Capture and return a single frame from the camera synchronously.

        This is a blocking call that will wait for the hardware and its SDK.

        Args:
            color_mode: Desired color mode for the output frame. If None,
                        uses the camera's default color mode.

        Returns:
            np.ndarray: Captured frame as a numpy array.
        """
        pass

    @abc.abstractmethod
    def async_read(self, timeout_ms: float = ...) -> NDArray[Any]:
        """Return the most recent new frame.

        This method retrieves the latest frame captured by the background thread.
        If a new frame is already available in the buffer (captured since the last call),
        it returns it immediately.

        It blocks up to `timeout_ms` only if the buffer is empty or if the latest frame
        was already consumed by a previous `async_read` call.

        Usage:
            - Ideal for control loops where you want to ensure every processed frame
            is fresh, effectively synchronizing your loop to the camera's FPS.
            - Causes of a timeout usually include: very low camera FPS, heavy processing load,
            or if the camera is disconnected.

        Args:
            timeout_ms: Maximum time to wait for a new frame in milliseconds.
                        Defaults to 200ms (0.2s).

        Returns:
            np.ndarray: Captured frame as a numpy array.

        Raises:
            TimeoutError: If no new frame arrives within `timeout_ms`.
        """
        pass

    @abc.abstractmethod
    def read_latest(self) -> tuple[NDArray[Any], float]:
        """Return the most recent frame captured immediately (Peeking).

        This method is non-blocking and returns whatever is currently in the
        memory buffer, along with its capture timestamp. The frame may be stale,
        meaning it could have been captured a while ago (hanging camera scenario e.g.).

        Usage:
            The user is responsible for checking the timestamp to handle stale data.
            Ideal for scenarios requiring zero latency or decoupled frequencies & when
            we want a guaranteed frame, such as UI visualization, logging, or
            non-critical monitoring.

        Returns:
            tuple[NDArray[Any], float]:
                - The frame image (numpy array).
                - The timestamp (time.perf_counter) when this frame was captured.

        Raises:
            NotConnectedError: If the camera is not connected.
            RuntimeError: If the camera is connected but has not captured any frames yet.
        """
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the camera and release resources."""
        pass
