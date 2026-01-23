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

from ..configs import CameraConfig, ColorMode

__all__ = ["CameraConfig", "ColorMode", "Reachy2CameraConfig"]


@CameraConfig.register_subclass("reachy2_camera")
@dataclass
class Reachy2CameraConfig(CameraConfig):
    """Configuration class for Reachy 2 camera devices.

    This class provides configuration options for Reachy 2 cameras,
    supporting both the teleop and depth cameras. It includes settings
    for resolution, frame rate, color mode, and the selection of the cameras.

    Example configurations:
    ```python
    # Basic configurations
    Reachy2CameraConfig(
        name="teleop",
        image_type="left",
        ip_address="192.168.0.200",  # IP address of the robot
        port=50065,  # Port of the camera server
        width=640,
        height=480,
        fps=30,  # Not configurable for Reachy 2 cameras
        color_mode=ColorMode.RGB,
    )  # Left teleop camera, 640x480 @ 30FPS
    ```

    Attributes:
        name: Name of the camera device. Can be "teleop" or "depth".
        image_type: Type of image stream. For "teleop" camera, can be "left" or "right".
                    For "depth" camera, can be "rgb" or "depth". (depth is not supported yet)
        fps: Requested frames per second for the color stream. Not configurable for Reachy 2 cameras.
        width: Requested frame width in pixels for the color stream.
        height: Requested frame height in pixels for the color stream.
        color_mode: Color mode for image output (RGB or BGR). Defaults to RGB.
        ip_address: IP address of the robot. Defaults to "localhost".
        port: Port number for the camera server. Defaults to 50065.

    Note:
        - Only 3-channel color output (RGB/BGR) is currently supported.
    """

    name: str
    image_type: str
    color_mode: ColorMode = ColorMode.RGB
    ip_address: str | None = "localhost"
    port: int = 50065

    def __post_init__(self) -> None:
        if self.name not in ["teleop", "depth"]:
            raise ValueError(f"`name` is expected to be 'teleop' or 'depth', but {self.name} is provided.")
        if (self.name == "teleop" and self.image_type not in ["left", "right"]) or (
            self.name == "depth" and self.image_type not in ["rgb", "depth"]
        ):
            raise ValueError(
                f"`image_type` is expected to be 'left' or 'right' for teleop camera, and 'rgb' or 'depth' for depth camera, but {self.image_type} is provided."
            )

        if self.color_mode not in ["rgb", "bgr"]:
            raise ValueError(
                f"`color_mode` is expected to be 'rgb' or 'bgr', but {self.color_mode} is provided."
            )
