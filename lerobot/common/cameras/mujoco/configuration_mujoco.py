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
from typing import Any

try:
    import mujoco
except ImportError:
    mujoco = None

from ..configs import CameraConfig, ColorMode, Cv2Rotation


@CameraConfig.register_subclass("mujoco")
@dataclass
class MuJoCoCameraConfig(CameraConfig):
    """Configuration class for MuJoCo simulation cameras.

    This class provides configuration options for cameras in MuJoCo simulations,
    supporting both named cameras and camera indices within the simulation environment.

    Example configurations:
    ```python
    # Basic configurations
    MuJoCoCameraConfig(model, data, 30, 640, 480)   # 640x480 @ 30FPS
    MuJoCoCameraConfig(model, data, 60, 1280, 720)  # 1280x720 @ 60FPS

    # Advanced configurations
    MuJoCoCameraConfig(model, data, 30, 640, 480, cam="top_view")          # Named camera
    MuJoCoCameraConfig(model, data, 30, 640, 480, cam=0)                   # Camera index
    MuJoCoCameraConfig(model, data, 30, 640, 480, rotation=Cv2Rotation.ROTATE_90)  # With 90° rotation
    ```

    Attributes:
        model: MuJoCo model object (mjModel).
        data: MuJoCo data object (mjData).
        fps: Requested frames per second for the simulation rendering.
        width: Requested frame width in pixels.
        height: Requested frame height in pixels.
        cam: Camera name (string) or index (int) in the MuJoCo model. If None, uses default camera.
        color_mode: Color mode for image output (RGB or BGR). Defaults to RGB.
        rotation: Image rotation setting (0°, 90°, 180°, or 270°). Defaults to no rotation.

    Note:
        - The MuJoCo model and data objects must be provided and valid.
        - Camera name/index must exist in the MuJoCo model if specified.
        - Only RGB color output is supported natively; BGR conversion is handled if requested.
    """

    model: Any  # mujoco.MjModel, but using Any to avoid import issues
    data: Any   # mujoco.MjData, but using Any to avoid import issues
    cam: str | int | None = None
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION

    def __post_init__(self):
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

        if mujoco is not None:
            # Validate that model and data are proper MuJoCo objects
            if not hasattr(self.model, 'nq') or not hasattr(self.data, 'qpos'):
                raise ValueError("Invalid MuJoCo model or data objects provided.")
