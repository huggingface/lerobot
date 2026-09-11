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
"""ZED (Stereolabs) stereo camera backend for LeRobot.

Provides RGB + optional metric depth (uint16 mm) with a background read thread,
plus the ZED hardware image timestamp (for downstream multi-stream sync).
"""

import logging
from typing import Any

from numpy.typing import NDArray

from .base_zed import ZedCameraBase, _require_sl, logger, sl
from .configuration_zed import ZedCameraConfig


class ZedCamera(ZedCameraBase):
    """Manages a Stereolabs stereo ZED camera via the ZED SDK (`pyzed.sl`).

    Follows the standard LeRobot camera interface: identify by serial number,
    RGB via ``read()``/``async_read()``, metric depth (uint16 mm) via
    ``read_depth()``/``async_read_depth()``. Depth is returned in millimetres, the
    unit LeRobot's depth path expects, so no conversion is needed downstream.

    Note on RGB-D pairs: ``read()``/``read_depth()`` (and their ``async_`` variants) each wait
    for the *next* frame, so calling them in sequence returns images from two different grabs.
    For a color/depth pair captured at the same instant, use ``read_latest()`` and
    ``read_latest_depth()``, which both serve the most recent grab.

    For the monocular ZED X One, use :class:`ZedOneCamera` — the SDK exposes single-sensor
    cameras through a different API.

    Example:
        ```python
        from lerobot.cameras.zed import ZedCamera, ZedCameraConfig

        cfg = ZedCameraConfig(serial_number=12345678, resolution="HD720", use_depth=True)
        cam = ZedCamera(cfg)
        cam.connect()
        rgb = cam.read_latest()  # (H, W, 3) uint8
        depth = cam.read_latest_depth()  # (H, W, 1) uint16 millimetres, same grab as `rgb`
        cam.disconnect()
        ```
    """

    def __init__(self, config: ZedCameraConfig):
        super().__init__(config)

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """List the stereo ZED cameras the SDK can see (``id`` and ``serial_number`` are the serial).

        Raises:
            ImportError: If the ZED SDK Python API is not installed or fails to import.
        """
        _require_sl()
        cameras = []
        for dev in sl.Camera.get_device_list():
            cameras.append(
                {
                    "name": str(dev.camera_model),
                    "type": "zed",
                    "id": int(dev.serial_number),
                    "serial_number": int(dev.serial_number),
                }
            )
        return cameras

    # ---- SDK-specific hooks ----
    def _open_device(self) -> None:
        init = sl.InitParameters()
        init.camera_resolution = self._resolve_resolution()
        init.sdk_verbose = 1 if logger.isEnabledFor(logging.DEBUG) else 0  # pyzed defaults to 1
        if self.fps:
            init.camera_fps = int(self.fps)
        init.enable_image_validity_check = 1  # pyzed defaults to 0; without it CORRUPTED_FRAME never fires
        try:
            init.depth_mode = (
                getattr(sl.DEPTH_MODE, self.depth_mode_name) if self.use_depth else sl.DEPTH_MODE.NONE
            )
        except AttributeError as e:
            valid = [m.name for m in sl.DEPTH_MODE if m.name != "LAST"]
            raise ValueError(f"{self}: unknown depth_mode {self.depth_mode_name!r}; valid: {valid}") from e
        if self.serial_number is not None:
            init.set_from_serial_number(int(self.serial_number))

        self.cam = sl.Camera()
        status = self.cam.open(init)
        if status > sl.ERROR_CODE.SUCCESS:
            raise ConnectionError(
                f"Failed to open {self}: {status}. Run `lerobot-find-cameras zed` to find available cameras."
            )
        if status != sl.ERROR_CODE.SUCCESS:  # negative codes are warnings, the camera is usable
            logger.warning(f"{self} opened with a warning: {status}")
        self.runtime = sl.RuntimeParameters()

    def _grab(self) -> Any:
        return self.cam.grab(self.runtime)

    def _retrieve_color(self) -> NDArray[Any]:
        self.cam.retrieve_image(self._img_mat, self._color_view)
        return self._img_mat.get_data()

    def _retrieve_depth(self) -> NDArray[Any]:
        # uint16 mm regardless of coordinate_units, invalid pixels already 0
        self.cam.retrieve_measure(self._depth_mat, sl.MEASURE.DEPTH_U16_MM)
        return self._depth_mat.get_data()

    def _read_calibration(self, camera_configuration: Any) -> Any:
        # `calibration_parameters` describes the rectified image, `calibration_parameters_raw` the raw one.
        calib = (
            camera_configuration.calibration_parameters
            if self.config.rectified
            else camera_configuration.calibration_parameters_raw
        )
        return calib.left_cam
