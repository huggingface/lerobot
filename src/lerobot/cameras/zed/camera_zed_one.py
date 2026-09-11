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
"""ZED X One (monocular Stereolabs camera) backend for LeRobot.

The ZED SDK drives single-sensor cameras through ``sl.CameraOne``, a separate API from the
``sl.Camera`` used for stereo models: no depth, its own device enumeration, and a flat
calibration structure. Everything else — the background capture thread and the read families —
is shared with :class:`ZedCamera` via :class:`ZedCameraBase`.
"""

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base_zed import ZedCameraBase, _require_sl, logger, sl
from .configuration_zed_one import ZedOneCameraConfig


class ZedOneCamera(ZedCameraBase):
    """Manages a monocular Stereolabs ZED X One via the ZED SDK (`pyzed.sl.CameraOne`).

    Same API as :class:`ZedCamera` minus depth: RGB via ``read()`` / ``async_read()`` /
    ``read_latest()`` and the ZED hardware image timestamp via ``latest_hw_timestamp_ns``.
    Depth methods raise, since a single sensor cannot triangulate.

    Example:
        ```python
        from lerobot.cameras.zed import ZedOneCamera, ZedOneCameraConfig

        cam = ZedOneCamera(ZedOneCameraConfig(serial_number=12345678))  # replace with your camera's serial
        cam.connect()
        rgb = cam.read()  # (H, W, 3) uint8
        cam.disconnect()
        ```
    """

    def __init__(self, config: ZedOneCameraConfig):
        super().__init__(config)
        self.use_rgb = True
        self.use_depth = False  # monocular: no depth stream exists

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """List monocular ZED cameras, which the SDK enumerates through its own device list."""
        _require_sl()
        return [
            {
                "name": str(dev.camera_model),
                "type": "zed_one",
                "id": int(dev.serial_number),
                "serial_number": int(dev.serial_number),
            }
            for dev in sl.CameraOne.get_device_list()
        ]

    # ---- SDK-specific hooks ----
    def _open_device(self) -> None:
        init = sl.InitParametersOne()
        init.camera_resolution = self._resolve_resolution()
        init.sdk_verbose = 1 if logger.isEnabledFor(logging.DEBUG) else 0  # pyzed defaults to 1
        if self.fps:
            init.camera_fps = int(self.fps)
        if self.serial_number is not None:
            init.set_from_serial_number(int(self.serial_number))

        self.cam = sl.CameraOne()
        status = self.cam.open(init)
        if status > sl.ERROR_CODE.SUCCESS:
            raise ConnectionError(
                f"Failed to open {self}: {status}. Run `lerobot-find-cameras zed` to find available cameras."
            )
        if status != sl.ERROR_CODE.SUCCESS:  # negative codes are warnings, the camera is usable
            logger.warning(f"{self} opened with a warning: {status}")

    def _grab(self) -> Any:
        # CameraOne.grab() takes no RuntimeParameters
        return self.cam.grab()

    def _retrieve_color(self) -> NDArray[Any]:
        self.cam.retrieve_image(self._img_mat, self._color_view)
        return self._img_mat.get_data()

    def _read_calibration(self, camera_configuration: Any) -> Any:
        # CameraOne exposes a flat `CameraParameters` (no `.left_cam`, since there is one sensor).
        return (
            camera_configuration.calibration_parameters
            if self.config.rectified
            else camera_configuration.calibration_parameters_raw
        )

    # ---- depth is not available on a single sensor ----
    def _no_depth(self) -> None:
        raise NotImplementedError(
            f"{self} is a monocular camera and has no depth stream. Use a stereo ZED with "
            "`ZedCameraConfig(use_depth=True)` if you need depth."
        )

    def read_depth(self, timeout_ms: int = 0) -> NDArray[np.uint16]:
        """Always raises ``NotImplementedError``: a monocular camera has no depth stream."""
        self._no_depth()

    def async_read_depth(self, timeout_ms: float = 200) -> NDArray[np.uint16]:
        """Always raises ``NotImplementedError``: a monocular camera has no depth stream."""
        self._no_depth()

    def read_latest_depth(self, max_age_ms: int = 500) -> NDArray[np.uint16]:
        """Always raises ``NotImplementedError``: a monocular camera has no depth stream."""
        self._no_depth()
