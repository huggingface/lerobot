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

from dataclasses import dataclass

from ..configs import CameraConfig, ColorMode, Cv2Rotation


@CameraConfig.register_subclass("intelrealsense")
@dataclass
class RealSenseCameraConfig(CameraConfig):
    """Configuration class for Intel RealSense cameras.

    This class provides specialized configuration options for Intel RealSense cameras,
    including support for depth sensing and device identification via serial number or name.

    Example configurations for Intel RealSense D405:
    ```python
    # Basic configurations
    RealSenseCameraConfig("0123456789", 30, 1280, 720)  # 1280x720 @ 30FPS
    RealSenseCameraConfig("0123456789", 60, 640, 480)  # 640x480 @ 60FPS

    # Advanced configurations
    RealSenseCameraConfig("0123456789", 30, 640, 480, use_depth=True)  # With depth sensing
    RealSenseCameraConfig("0123456789", 30, 640, 480, rotation=Cv2Rotation.ROTATE_90)  # With 90° rotation
    ```

    Attributes:
        fps: Requested frames per second for the color stream.
        width: Requested frame width in pixels for the color stream.
        height: Requested frame height in pixels for the color stream.
        serial_number_or_name: Unique serial number or human-readable name to identify the camera.
        color_mode: Color mode for image output (RGB or BGR). Defaults to RGB.
        use_depth: Whether to enable depth stream. Defaults to False.
        rotation: Image rotation setting (0°, 90°, 180°, or 270°). Defaults to no rotation.
        warmup_s: Time reading frames before returning from connect (in seconds)

    Note:
        - Either name or serial_number must be specified.
        - Depth stream configuration (if enabled) will use the same FPS as the color stream.
        - The actual resolution and FPS may be adjusted by the camera to the nearest supported mode.
        - For `fps`, `width` and `height`, either all of them need to be set, or none of them.
    """

    serial_number_or_name: str
    color_mode: ColorMode = ColorMode.RGB
    use_depth: bool = False
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1

    def __post_init__(self) -> None:
        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"`color_mode` is expected to be {ColorMode.RGB.value} or {ColorMode.BGR.value}, but {self.color_mode} is provided."
            )

        if self.rotation not in (
            Cv2Rotation.NO_ROTATION,
            Cv2Rotation.ROTATE_90,
            Cv2Rotation.ROTATE_180,
            Cv2Rotation.ROTATE_270,
        ):
            raise ValueError(
                f"`rotation` is expected to be in {(Cv2Rotation.NO_ROTATION, Cv2Rotation.ROTATE_90, Cv2Rotation.ROTATE_180, Cv2Rotation.ROTATE_270)}, but {self.rotation} is provided."
            )

        values = (self.fps, self.width, self.height)
        if any(v is not None for v in values) and any(v is None for v in values):
            raise ValueError(
                "For `fps`, `width` and `height`, either all of them need to be set, or none of them."
            )
