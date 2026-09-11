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
# (in cameras/__init__.py) to register the "zed_one" draccus choice without pulling
# the ZED SDK dependency.


@CameraConfig.register_subclass("zed_one")
@dataclass
class ZedOneCameraConfig(CameraConfig):
    """Configuration for the monocular Stereolabs ZED X One (via the ZED SDK Python API `pyzed`).

    The ZED X One is a single-sensor camera, so it has no depth stream and no `use_depth` /
    `depth_mode` settings — use :class:`ZedCameraConfig` for stereo models.

    Attributes:
        serial_number: ZED X One serial number. None → first available single-sensor camera.
        resolution: sl.RESOLUTION name; default "AUTO" (1920x1200 @ 30 on the ZED X One GS and UHD).
            "HD1200" | "HD1080" work on both; "HD4K" (3840x2160 @ 15) on the UHD only. "SVGA",
            "HD720" and "VGA" are INVALID on the X One.
        rectified: return the rectified image (default) or the raw, distorted sensor image.
            `intrinsics` follows this choice.
        color_mode: RGB (default) or BGR.
        rotation: 0 / 90 / 180 / -90 (`Cv2Rotation`; 270 is not accepted).
        warmup_s: seconds of warmup grabbing before connect() returns.
    """

    serial_number: int | None = None
    resolution: str = "AUTO"
    rectified: bool = True
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        self.rotation = Cv2Rotation(self.rotation)

        values = (self.fps, self.width, self.height)
        if any(v is not None for v in values) and any(v is None for v in values):
            raise ValueError(
                "For `fps`, `width` and `height`, either all of them need to be set, or none of them."
            )
