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
from enum import Enum
from pathlib import Path

from ..configs import CameraConfig, ColorMode, Cv2Rotation


class EyeSelection(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"


__all__ = ["PS4EyeCameraConfig", "EyeSelection", "ColorMode", "Cv2Rotation"]


@CameraConfig.register_subclass("ps4eye")
@dataclass
class PS4EyeCameraConfig(CameraConfig):
    """Configuration class for Sony PS4 Eye camera devices.

    The PS4 Eye is a stereo USB camera that enumerates as a standard V4L2/UVC
    device (on Linux). It outputs a wide panoramic
    frame with both eyes side-by-side, which this driver slices into individual
    eye images.

    Default stereo geometry (at 3448×808 resolution):
        - Left eye:  frame[0:800, 64:1328]  → 1264×800
        - Right eye: frame[0:800, 1328:2592] → 1264×800

    Use `lerobot-find-cameras opencv` to discover the camera's device index or
    path, then pass that value as `index_or_path`.

    Example configurations:
    ```python
    # Left eye at default resolution
    PS4EyeCameraConfig(index_or_path=0, eye="left")

    # Right eye with explicit resolution
    PS4EyeCameraConfig(index_or_path="/dev/video0", fps=60, width=3448, height=808, eye="right")

    # Full panoramic (both eyes, unsplit)
    PS4EyeCameraConfig(index_or_path=0, eye="both")
    ```

    Attributes:
        index_or_path: Integer camera index or Path/str to the device node (e.g. '/dev/video0').
        color_mode: Color mode for image output (RGB or BGR). Defaults to RGB.
        rotation: Image rotation applied after stereo crop. Defaults to no rotation.
        warmup_s: Seconds spent reading warm-up frames after connect. Defaults to 1.
        eye: Which stereo eye to return on `read()`.
            - ``"left"``  – left image slice only (default)
            - ``"right"`` – right image slice only
            - ``"both"``  – full unsplit panoramic frame
    """

    index_or_path: int | str | Path
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1
    eye: EyeSelection = EyeSelection.LEFT

    def __post_init__(self) -> None:
        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"`color_mode` is expected to be {ColorMode.RGB.value!r} or {ColorMode.BGR.value!r}, "
                f"but {self.color_mode!r} is provided."
            )

        if self.rotation not in (
            Cv2Rotation.NO_ROTATION,
            Cv2Rotation.ROTATE_90,
            Cv2Rotation.ROTATE_180,
            Cv2Rotation.ROTATE_270,
        ):
            raise ValueError(
                f"`rotation` is expected to be one of "
                f"{(Cv2Rotation.NO_ROTATION, Cv2Rotation.ROTATE_90, Cv2Rotation.ROTATE_180, Cv2Rotation.ROTATE_270)}, "
                f"but {self.rotation!r} is provided."
            )

        if self.eye not in (EyeSelection.LEFT, EyeSelection.RIGHT, EyeSelection.BOTH):
            raise ValueError(
                f"`eye` must be 'left', 'right', or 'both', but {self.eye!r} is provided."
            )
