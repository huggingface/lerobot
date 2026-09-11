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

# NOTE: this file imports NOTHING from pyzed, so it is safe to import at startup
# (in cameras/__init__.py) to register the "zed" draccus choice without pulling
# the ZED SDK dependency.


@CameraConfig.register_subclass("zed")
@dataclass
class ZedCameraConfig(CameraConfig):
    """Configuration for Stereolabs ZED cameras (via the ZED SDK Python API `pyzed`).

    The camera is identified by serial number (recommended); omit it to open the
    first available ZED. The `resolution` string drives width/height/fps, so leave
    `fps`/`width`/`height` as None unless you want to override the reported values.

    Attributes:
        serial_number: ZED serial number. None → first available device.
        resolution: sl.RESOLUTION name; default "AUTO" (picks the camera's native mode).
            USB ZED/ZED2: "HD2K"|"HD1080"|"HD720"|"VGA". GMSL ZED X / X Mini: "HD1200"|"HD1080"|"SVGA"
            (HD720 is INVALID). The ZED X HDR models open only in "AUTO" (native 1920x1536); every
            explicit mode is INVALID. When in doubt, use "AUTO".
        rectified: return the rectified color image (default) or the raw, distorted sensor image.
            Rectification removes lens distortion and is what `intrinsics` describes, so unrectified
            output cannot be used with those intrinsics and is incompatible with `use_depth`.
        depth_mode: sl.DEPTH_MODE name — "NEURAL" | "NEURAL_PLUS" | "NEURAL_LIGHT" | "NONE". The SDK
            still accepts "PERFORMANCE", "QUALITY" and "ULTRA", but deprecates them.
        color_mode: RGB (default) or BGR.
        use_rgb: enable the left color stream. Default True.
        use_depth: enable the metric depth stream (uint16 millimetres). Default False.
        rotation: 0 / 90 / 180 / -90 (`Cv2Rotation`; 270 is not accepted).
        warmup_s: seconds of warmup grabbing before connect() returns.
    """

    serial_number: int | None = None
    resolution: str = "AUTO"  # AUTO picks each model's native mode (ZED X=HD1080, ZED X HDR=1920x1536, ...)
    rectified: bool = True
    depth_mode: str = "NEURAL"
    color_mode: ColorMode = ColorMode.RGB
    use_rgb: bool = True
    use_depth: bool = False
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        self.rotation = Cv2Rotation(self.rotation)

        if not self.use_rgb and not self.use_depth:
            raise ValueError("At least one of `use_rgb` or `use_depth` must be enabled.")

        if not self.rectified and self.use_depth:
            raise ValueError(
                "`rectified=False` cannot be combined with `use_depth=True`: the ZED SDK computes depth "
                "from the rectified stereo pair, so the depth map would not align with the raw image."
            )

        values = (self.fps, self.width, self.height)
        if any(v is not None for v in values) and any(v is None for v in values):
            raise ValueError(
                "For `fps`, `width` and `height`, either all of them need to be set, or none of them."
            )
