# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from lerobot.cameras import Camera, CameraConfig
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

if TYPE_CHECKING:
    import menlo.asimov as menlo


@CameraConfig.register_subclass("asimov")
@dataclass
class AsimovCameraConfig(CameraConfig):
    """Onboard RGB stream, resized to the dataset's fixed output dimensions."""

    fps: int = 30
    width: int = 640
    height: int = 480
    connect_timeout_s: float = 5.0

    def __post_init__(self) -> None:
        if any(value <= 0 for value in (self.fps, self.width, self.height)):
            raise ValueError("Asimov camera fps, width and height must be positive.")
        if not isfinite(self.connect_timeout_s) or self.connect_timeout_s <= 0:
            raise ValueError("Asimov camera connect_timeout_s must be finite and positive.")


class AsimovCamera(Camera):
    """Camera view of the Asimov robot's shared SDK connection; never sends commands."""

    def __init__(self, config: AsimovCameraConfig) -> None:
        super().__init__(config)
        self.config = config
        self._stream: menlo.Camera | None = None

    @property
    def is_connected(self) -> bool:
        return self._stream is not None

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        return []  # The onboard stream is discovered when Asimov joins its room.

    @check_if_already_connected
    def connect(self, warmup: bool = True, *, stream: menlo.Camera | None = None) -> None:
        if stream is None:
            raise ValueError("The onboard camera must be connected through an Asimov robot.")
        self._stream = stream
        try:
            if warmup:
                self._pixels(stream.photo(timeout=self.config.connect_timeout_s))
        except BaseException:
            self.disconnect()
            raise

    def _pixels(self, frame: menlo.Frame) -> NDArray[np.uint8]:
        if frame.encoding != "rgb8":
            raise ValueError(f"Expected the SDK's decoded RGB camera frame, got {frame.encoding!r}.")
        pixels = np.asarray(frame.to_numpy())
        if pixels.dtype != np.uint8 or pixels.shape != (frame.height, frame.width, 3):
            raise ValueError("Invalid Asimov RGB frame dimensions or dtype.")
        if pixels.shape[:2] != (self.config.height, self.config.width):
            pixels = np.asarray(
                Image.fromarray(pixels).resize(
                    (self.config.width, self.config.height), Image.Resampling.BILINEAR
                )
            )
        return pixels

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[np.uint8]:
        assert self._stream is not None
        frame = self._stream.latest()
        if frame is None or not isfinite(frame.age_s) or frame.age_s > max_age_ms / 1000:
            raise TimeoutError("Asimov camera has no recent frame.")
        return self._pixels(frame)

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[np.uint8]:
        assert self._stream is not None
        return self._pixels(self._stream.photo(timeout=timeout_ms / 1000))

    def read(self) -> NDArray[np.uint8]:
        return self.async_read(timeout_ms=self.config.connect_timeout_s * 1000)

    def disconnect(self) -> None:
        self._stream = None  # The robot owns and closes the shared SDK connection.
