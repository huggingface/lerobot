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

from dataclasses import dataclass, field
from enum import Enum

from ..configs import CameraConfig, ColorMode, Cv2Rotation


class D2CMode(str, Enum):
    HARDWARE = "hardware"
    SOFTWARE = "software"


__all__ = ["OrbbecCameraConfig", "ColorMode", "Cv2Rotation", "D2CMode"]


@CameraConfig.register_subclass("orbbec")
@dataclass
class OrbbecCameraConfig(CameraConfig):
    """Configuration class for Orbbec cameras (pyorbbecsdk2).

    Supports device selection by integer index (as in ``enumerate.py`` / ``multi_device.py``
    from the Orbbec SDK examples) or by serial number string (``device_info.get_serial_number()``).

    Example configurations:

    .. code-block:: python

        # Select by device index (0 = first connected device)
        OrbbecCameraConfig(0, 30, 640, 480)

        # Select by serial number
        OrbbecCameraConfig("CL8FC3100EY", 30, 1280, 720)

        # Enable depth stream with hardware depth-to-color alignment
        OrbbecCameraConfig(0, 30, 640, 480, use_depth=True, align_depth=True)

        # BGR output with 90-degree rotation
        OrbbecCameraConfig(0, 30, 640, 480, color_mode=ColorMode.BGR, rotation=Cv2Rotation.ROTATE_90)

    Use ``OrbbecCamera.find_cameras()`` (or ``lerobot-find-cameras orbbec``) to list
    the index and serial number of every connected device.

    Attributes:
        index_or_serial_number: Device index (``int``) matching the enumeration order
            from ``Context().query_devices()``, **or** a serial number string obtained
            from ``device_info.get_serial_number()``.
        fps: Desired capture frame rate. ``None`` uses the camera default.
        width: Desired output frame width in pixels (after rotation). ``None`` uses
            the camera default.
        height: Desired output frame height in pixels (after rotation). ``None`` uses
            the camera default.
        color_mode: Output color channel ordering — ``RGB`` (default) or ``BGR``.
        use_depth: Enable the depth sensor stream. Defaults to ``False``.
        align_depth: Align depth frames to the color frame coordinate system.
            Requires ``use_depth=True``. Defaults to ``False``.
        d2c_mode: Depth-to-color alignment mode when ``align_depth=True``.
            - ``D2CMode.HARDWARE`` (default): prefers hardware D2C.
            - ``D2CMode.SOFTWARE``: uses software alignment filter.
        rotation: Post-capture rotation applied to both color and depth output.
            Defaults to no rotation.
        warmup_s: Seconds spent reading and discarding frames inside ``connect()``
            before returning. Allows the auto-exposure / white-balance to stabilise.
            Defaults to ``1``.

    Note:
        ``fps``, ``width``, and ``height`` must either all be ``None`` (use camera
        defaults) or all be specified together.
    """

    index_or_serial_number: int | str = field(default=0)
    color_mode: ColorMode = ColorMode.RGB
    use_depth: bool = False
    align_depth: bool = False
    d2c_mode: D2CMode = D2CMode.HARDWARE
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.index_or_serial_number, (int, str)):
            raise ValueError(
                f"`index_or_serial_number` must be an int (device index) or str (serial number), "
                f"but got {type(self.index_or_serial_number).__name__}."
            )
        if isinstance(self.index_or_serial_number, str) and not self.index_or_serial_number.strip():
            raise ValueError("`index_or_serial_number` must not be an empty string.")

        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"`color_mode` is expected to be {ColorMode.RGB.value} or {ColorMode.BGR.value}, "
                f"but {self.color_mode} is provided."
            )

        if self.rotation not in (
            Cv2Rotation.NO_ROTATION,
            Cv2Rotation.ROTATE_90,
            Cv2Rotation.ROTATE_180,
            Cv2Rotation.ROTATE_270,
        ):
            raise ValueError(
                f"`rotation` is expected to be in "
                f"{(Cv2Rotation.NO_ROTATION, Cv2Rotation.ROTATE_90, Cv2Rotation.ROTATE_180, Cv2Rotation.ROTATE_270)}, "
                f"but {self.rotation} is provided."
            )

        if self.align_depth and not self.use_depth:
            raise ValueError("align_depth=True requires use_depth=True.")

        if isinstance(self.d2c_mode, str):
            try:
                self.d2c_mode = D2CMode(self.d2c_mode.lower())
            except ValueError as e:
                raise ValueError(
                    f"`d2c_mode` must be one of {[m.value for m in D2CMode]}, but got {self.d2c_mode!r}."
                ) from e
        elif not isinstance(self.d2c_mode, D2CMode):
            raise ValueError(f"`d2c_mode` must be D2CMode, but got {type(self.d2c_mode).__name__}.")

        fps_width_height = (self.fps, self.width, self.height)
        if any(v is not None for v in fps_width_height) and any(v is None for v in fps_width_height):
            raise ValueError(
                "Either all of `fps`, `width`, and `height` must be set, or all must be None. "
                f"Got fps={self.fps}, width={self.width}, height={self.height}."
            )
