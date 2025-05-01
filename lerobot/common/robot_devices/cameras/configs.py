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
from typing import Optional

import draccus


@dataclass
class CameraConfig(draccus.ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    def _validate_common(self, color_mode: str, rotation: Optional[int]) -> None:
        if color_mode not in {"rgb", "bgr"}:
            raise ValueError(f"`color_mode` must be 'rgb' or 'bgr', got '{color_mode}'.")

        if rotation not in {None, -90, 90, 180}:
            raise ValueError(f"`rotation` must be one of [None, -90, 90, 180], got {rotation}.")


@CameraConfig.register_subclass("opencv")
@dataclass
class OpenCVCameraConfig(CameraConfig):
    """
    Example tested configurations for Intel Real Sense D405:

    ```python
    OpenCVCameraConfig(0, 30, 640, 480)
    OpenCVCameraConfig(0, 60, 640, 480)
    OpenCVCameraConfig(0, 90, 640, 480)
    OpenCVCameraConfig(0, 30, 1280, 720)
    ```
    """

    camera_index: int
    fps: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    color_mode: str = "rgb"
    channels: Optional[int] = None
    rotation: Optional[int] = None
    mock: bool = False

    def __post_init__(self):
        self._validate_common(self.color_mode, self.rotation)
        self.channels = 3


@CameraConfig.register_subclass("intelrealsense")
@dataclass
class IntelRealSenseCameraConfig(CameraConfig):
    """
    Example tested configurations for Intel Real Sense D405:

    ```python
    IntelRealSenseCameraConfig(128422271347, 30, 640, 480)
    IntelRealSenseCameraConfig(128422271347, 60, 640, 480)
    IntelRealSenseCameraConfig(128422271347, 90, 640, 480)
    IntelRealSenseCameraConfig(128422271347, 30, 1280, 720)
    IntelRealSenseCameraConfig(128422271347, 30, 640, 480, use_depth=True)
    IntelRealSenseCameraConfig(128422271347, 30, 640, 480, rotation=90)
    ```
    """

    name: Optional[str] = None
    serial_number: Optional[int] = None
    fps: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    color_mode: str = "rgb"
    channels: Optional[int] = None
    use_depth: bool = False
    force_hardware_reset: bool = True
    rotation: Optional[int] = None
    mock: bool = False

    def __post_init__(self):
        self._validate_common(self.color_mode, self.rotation)
        self.channels = 3

        if self.name and self.serial_number:
            raise ValueError(
                f"Only one of `name` or `serial_number` should be set. Got name={self.name!r}, serial_number={self.serial_number!r}."
            )

        fps_set = self.fps is not None
        dims_set = self.width is not None and self.height is not None

        if fps_set != dims_set:
            raise ValueError(
                f"`fps`, `width`, and `height` must either all be set or all be None. "
                f"Got fps={self.fps}, width={self.width}, height={self.height}."
            )
