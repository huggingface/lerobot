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
from dataclasses import dataclass

import draccus


@dataclass
class CameraConfig(draccus.ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@CameraConfig.register_subclass("opencv")
@dataclass
class OpenCVCameraConfig(CameraConfig):
    """
    Example of tested options for Intel Real Sense D405:

    ```python
    OpenCVCameraConfig(0, 30, 640, 480)
    OpenCVCameraConfig(0, 60, 640, 480)
    OpenCVCameraConfig(0, 90, 640, 480)
    OpenCVCameraConfig(0, 30, 1280, 720)
    ```
    """

    camera_index: int
    fps: int | None = None
    width: int | None = None
    height: int | None = None
    color_mode: str = "rgb"
    channels: int | None = None
    rotation: int | None = None
    mock: bool = False

    def __post_init__(self):
        if self.color_mode not in ["rgb", "bgr"]:
            raise ValueError(
                f"`color_mode` is expected to be 'rgb' or 'bgr', but {self.color_mode} is provided."
            )

        self.channels = 3

        if self.rotation not in [-90, None, 90, 180]:
            raise ValueError(f"`rotation` must be in [-90, None, 90, 180] (got {self.rotation})")


@CameraConfig.register_subclass("intelrealsense")
@dataclass
class IntelRealSenseCameraConfig(CameraConfig):
    """
    Example of tested options for Intel Real Sense D405:

    ```python
    IntelRealSenseCameraConfig(128422271347, 30, 640, 480)
    IntelRealSenseCameraConfig(128422271347, 60, 640, 480)
    IntelRealSenseCameraConfig(128422271347, 90, 640, 480)
    IntelRealSenseCameraConfig(128422271347, 30, 1280, 720)
    IntelRealSenseCameraConfig(128422271347, 30, 640, 480, use_depth=True)
    IntelRealSenseCameraConfig(128422271347, 30, 640, 480, rotation=90)
    ```
    """

    name: str | None = None
    serial_number: int | None = None
    fps: int | None = None
    width: int | None = None
    height: int | None = None
    color_mode: str = "rgb"
    channels: int | None = None
    use_depth: bool = False
    force_hardware_reset: bool = True
    rotation: int | None = None
    mock: bool = False

    def __post_init__(self):
        # bool is stronger than is None, since it works with empty strings
        if bool(self.name) and bool(self.serial_number):
            raise ValueError(
                f"One of them must be set: name or serial_number, but {self.name=} and {self.serial_number=} provided."
            )

        if self.color_mode not in ["rgb", "bgr"]:
            raise ValueError(
                f"`color_mode` is expected to be 'rgb' or 'bgr', but {self.color_mode} is provided."
            )

        self.channels = 3

        at_least_one_is_not_none = self.fps is not None or self.width is not None or self.height is not None
        at_least_one_is_none = self.fps is None or self.width is None or self.height is None
        if at_least_one_is_not_none and at_least_one_is_none:
            raise ValueError(
                "For `fps`, `width` and `height`, either all of them need to be set, or none of them, "
                f"but {self.fps=}, {self.width=}, {self.height=} were provided."
            )

        if self.rotation not in [-90, None, 90, 180]:
            raise ValueError(f"`rotation` must be in [-90, None, 90, 180] (got {self.rotation})")


@CameraConfig.register_subclass("ros2")
@dataclass
class ROS2CameraConfig(CameraConfig):
    """Configuration for cameras connected through ROS 2 topics.

    This class allows configuring cameras that publish images to ROS 2 topics, making
    them available for use in LeRobot. The class supports RGB/BGR color formats, image
    resizing, and rotation.

    Examples:
        Basic usage with a standard color camera:

        ```python
        ROS2CameraConfig(
            topic="/camera/color/image_raw",
            fps=30,
            width=640,
            height=480
        )
        ```

        Using a depth camera:

        ```python
        ROS2CameraConfig(
            topic="/camera/depth/image_rect_raw",
            fps=30,
            width=640,
            height=480,
            channels=1  # Depth cameras typically use 1 channel
        )
        ```

        With image rotation and BGR format:

        ```python
        ROS2CameraConfig(
            topic="/camera/image_raw",
            fps=30,
            width=1280,
            height=720,
            encoding="bgr8",
            rotation=90
        )
        ```

        Mock camera for testing:

        ```python
        ROS2CameraConfig(
            topic="/test/image",
            mock=True
        )
        ```

        Example of a dictionary with multiple camera configurations:
        ```python
        cameras: dict[str, CameraConfig] = field(
            default_factory=lambda: {
                "rs_camera_color": ROS2CameraConfig(
                    topic="/rs/color/image_raw",
                    fps=30,
                    width=640,
                    height=480,
                    channels=3,
                ),
                "rs_camera_depth": ROS2CameraConfig(
                    topic="/rs/depth/image_rect_raw",
                    fps=30,
                    width=640,
                    height=480,
                    channels=1,
                ),
                "oak_color_camera": ROS2CameraConfig(
                    topic="/oak/rgb/image_raw",
                    fps=30,
                    width=640,
                    height=480,
                    channels=3,
                ),
                "oak_camera_depth": ROS2CameraConfig(
                    topic="/oak/stereo/image_raw",
                    fps=30,
                    width=640,
                    height=480,
                    channels=1,
                ),
            }
        )
        ```

    Args:
        topic: ROS 2 topic to subscribe to for images
        fps: Target frames per second (optional)
        width: Target image width in pixels (optional)
        height: Target image height in pixels (optional)
        encoding: ROS image coding (default is "passthrough"). See https://github.com/ros2/common_interfaces/blob/rolling/sensor_msgs/include/sensor_msgs/image_encodings.hpp
        channels: Number of color channels (auto-detected if None)
        rotation: Optional rotation in degrees, must be one of [-90, None, 90, 180]
        mock: Whether to use a mock camera instead of connecting to ROS 2
    """

    topic: str
    fps: int | None = None
    width: int | None = None
    height: int | None = None
    encoding: str | None = "passthrough"
    channels: int | None = None
    rotation: int | None = None
    mock: bool = False

    def __post_init__(self):
        if self.rotation not in [-90, None, 90, 180]:
            raise ValueError(f"`rotation` must be in [-90, None, 90, 180] (got {self.rotation})")
