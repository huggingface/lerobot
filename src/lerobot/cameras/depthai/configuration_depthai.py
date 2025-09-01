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


@CameraConfig.register_subclass("depthai")
@dataclass
class DepthAICameraConfig(CameraConfig):
    """Configuration for Luxonis DepthAI cameras (OAK-D series).

    Supports device identification via MxID or device name, RGB stream from CAM_A,
    and depth stream from stereo cameras CAM_B and CAM_C.

    Examples:
        ```python
        # RGB only using MxID
        DepthAICameraConfig("14442C10D13EABF200")

        # Depth only using device name
        DepthAICameraConfig("OAK-D-LITE", enable_rgb=False, enable_depth=True)

        # Both RGB and depth with custom settings
        DepthAICameraConfig(
            "14442C10D13EABF200", fps=30, width=1280, height=720, enable_rgb=True, enable_depth=True
        )
        ```

    Attributes:
        mxid_or_name: Unique MxID (18-char hex string) or device name to identify the camera.
        color_mode: Output color format (RGB or BGR). Defaults to RGB.
        enable_rgb: Enable RGB stream from CAM_A. Defaults to True.
        enable_depth: Enable depth stream from CAM_B and CAM_C stereo cameras. Defaults to False.
        rotation: Image rotation (0°, 90°, 180°, 270°). Defaults to no rotation.
        warmup_s: Warmup time in seconds after connection. Defaults to 1.

    Note:
        For fps, width, and height: either set all three or none (uses defaults: 30fps, 640x480).
        At least one of enable_rgb or enable_depth must be True.
    """

    mxid_or_name: str
    color_mode: ColorMode = ColorMode.RGB
    enable_rgb: bool = True
    enable_depth: bool = False
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1

    def __post_init__(self):
        # Validate that at least one stream is enabled
        if not self.enable_rgb and not self.enable_depth:
            raise ValueError("At least one of enable_rgb or enable_depth must be True")

        # Validate color mode
        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(f"color_mode must be RGB or BGR, got {self.color_mode}")

        # Validate rotation
        valid_rotations = (
            Cv2Rotation.NO_ROTATION,
            Cv2Rotation.ROTATE_90,
            Cv2Rotation.ROTATE_180,
            Cv2Rotation.ROTATE_270,
        )
        if self.rotation not in valid_rotations:
            raise ValueError(f"rotation must be one of {valid_rotations}, got {self.rotation}")

        # Validate fps/width/height consistency
        resolution_values = (self.fps, self.width, self.height)
        if any(v is not None for v in resolution_values) and any(v is None for v in resolution_values):
            raise ValueError("fps, width, and height must all be set or all be None")

    @property
    def use_depth(self) -> bool:
        """Backwards compatibility property."""
        return self.enable_depth
