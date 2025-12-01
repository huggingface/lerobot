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
"""Configuration for VirtualCamera used in SDK-based robots."""

from dataclasses import dataclass

from lerobot.cameras.configs import CameraConfig, ColorMode, Cv2Rotation

__all__ = ["VirtualCameraConfig", "ColorMode", "Cv2Rotation"]


@CameraConfig.register_subclass("virtual")
@dataclass
class VirtualCameraConfig(CameraConfig):
    """Configuration class for VirtualCamera used in SDK-based robots.

    VirtualCamera is used by robots that access cameras through an SDK or API
    rather than direct hardware access. The camera frames are provided by the
    robot's observation cache, which is populated by SDK HTTP API calls.

    This is used by robots like EarthRover Mini Plus, where cameras stream
    via WebRTC through cloud services.

    Example configurations:
    ```python
    # Basic front camera
    VirtualCameraConfig("front", 30, 640, 480)

    # Rear camera with rotation
    VirtualCameraConfig("rear", 30, 640, 480, rotation=Cv2Rotation.ROTATE_180)

    # Custom resolution
    VirtualCameraConfig("front", 10, 1280, 720)
    ```

    Attributes:
        name: Camera identifier (e.g., "front", "rear"). This must match
              the key used in the robot's observation dictionary.
        fps: Target frames per second (informational, not enforced)
        width: Frame width in pixels
        height: Frame height in pixels
        color_mode: Color mode for image output (RGB or BGR). Defaults to RGB.
        rotation: Image rotation setting (0°, 90°, 180°, or 270°). Defaults to no rotation.

    Note:
        - The actual frame rate depends on the SDK and network conditions
        - Frames are read from robot's _last_observation cache
        - No direct hardware connection is made
    """

    name: str
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("`name` must be a non-empty string (e.g., 'front', 'rear')")

        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"`color_mode` must be {ColorMode.RGB.value} or {ColorMode.BGR.value}, "
                f"but {self.color_mode} is provided."
            )

        if self.rotation not in (
            Cv2Rotation.NO_ROTATION,
            Cv2Rotation.ROTATE_90,
            Cv2Rotation.ROTATE_180,
            Cv2Rotation.ROTATE_270,
        ):
            raise ValueError(
                f"`rotation` must be one of {(Cv2Rotation.NO_ROTATION, Cv2Rotation.ROTATE_90, Cv2Rotation.ROTATE_180, Cv2Rotation.ROTATE_270)}, "
                f"but {self.rotation} is provided."
            )
