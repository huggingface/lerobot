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


@CameraConfig.register_subclass("oakd")
@dataclass
class OAKDCameraConfig(CameraConfig):
    """Configuration class for Luxonis OAK-D cameras (DepthAI).

    This class provides configuration options for OAK-D series cameras,
    including support for on-device stereo depth computation aligned to the
    RGB camera.

    Example configurations:
    ```python
    # RGB only, auto-detect device
    OAKDCameraConfig(fps=30, width=640, height=480)

    # RGB + depth, specific device
    OAKDCameraConfig(device_id="14442C1091E0B5D700", fps=30, width=640, height=480, use_depth=True)

    # With rotation
    OAKDCameraConfig(fps=30, width=640, height=480, rotation=Cv2Rotation.ROTATE_90)
    ```

    Attributes:
        device_id: MX ID of the OAK device, or empty string to auto-detect.
            Run ``lerobot-find-cameras oakd`` to list connected devices.
        color_mode: Color mode for image output (RGB or BGR). Defaults to RGB.
        use_depth: Whether to enable the stereo depth stream aligned to RGB.
            Defaults to False.
        rotation: Image rotation setting (0, 90, 180, or 270 degrees).
            Defaults to no rotation.
        warmup_s: Seconds to read frames before returning from connect().
        stereo_preset: Stereo profile preset (e.g. ``FAST_ACCURACY``, ``HIGH_DETAIL``,
            ``DEFAULT``). Unknown names fall back to ``FAST_ACCURACY``.
        stereo_confidence_threshold: Confidence 0--255 (higher = stricter). Use ``-1``
            to skip calling ``setConfidenceThreshold`` (SDK default).
        stereo_extended_disparity: Enable extended disparity for **closer** minimum range
            (needed for many eye-in-hand / near-table setups). Slightly more computation.

    Note:
        - Depth alignment to RGB requires left-right check to be enabled
          (handled automatically).
        - For ``fps``, ``width`` and ``height``, either all must be set or none.
    """

    device_id: str = ""
    color_mode: ColorMode = ColorMode.RGB
    use_depth: bool = False
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 2
    stereo_preset: str = "FAST_ACCURACY"
    stereo_confidence_threshold: int = 200
    stereo_extended_disparity: bool = False

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        self.rotation = Cv2Rotation(self.rotation)

        values = (self.fps, self.width, self.height)
        if any(v is not None for v in values) and any(v is None for v in values):
            raise ValueError(
                "For `fps`, `width` and `height`, either all of them need to be set, or none of them."
            )
