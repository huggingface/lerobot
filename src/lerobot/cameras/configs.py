#!/usr/bin/env python

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
from enum import Enum

import draccus  # type: ignore  # TODO: add type stubs for draccus


class ColorMode(str, Enum):
    """Color channel order for frames returned by a camera.

    **Attributes**:
        - **RGB** -- Red-green-blue channel order.
        - **BGR** -- Blue-green-red channel order, OpenCV's native order.
    """

    RGB = "rgb"
    BGR = "bgr"

    @classmethod
    def _missing_(cls, value: object) -> None:
        """Reject a value that is not a valid `ColorMode`.

        Raises:
            ValueError: Always, naming the invalid value and the valid choices.
        """
        raise ValueError(f"`color_mode` is expected to be in {list(cls)}, but {value} is provided.")


class Cv2Rotation(int, Enum):
    """Clockwise rotation to apply to a frame after capture, in degrees.

    **Attributes**:
        - **NO_ROTATION** -- No rotation.
        - **ROTATE_90** -- Rotate 90° clockwise.
        - **ROTATE_180** -- Rotate 180°.
        - **ROTATE_270** -- Rotate 270° clockwise (90° counter-clockwise).
    """

    NO_ROTATION = 0
    ROTATE_90 = 90
    ROTATE_180 = 180
    ROTATE_270 = -90

    @classmethod
    def _missing_(cls, value: object) -> None:
        """Reject a value that is not a valid `Cv2Rotation`.

        Raises:
            ValueError: Always, naming the invalid value and the valid choices.
        """
        raise ValueError(f"`rotation` is expected to be in {list(cls)}, but {value} is provided.")


# Subset from https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
class Cv2Backends(int, Enum):
    """OpenCV capture backend to request when opening a device.

    See the [OpenCV `VideoCaptureAPIs` reference](https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html)
    for the full list this is a subset of.

    **Attributes**:
        - **ANY** -- Let OpenCV auto-detect the backend.
        - **V4L2** -- Video4Linux2, the usual choice on Linux.
        - **DSHOW** -- DirectShow, a Windows backend.
        - **PVAPI** -- PvAPI, for Prosilica GigE cameras.
        - **ANDROID** -- Android's native camera API.
        - **AVFOUNDATION** -- AVFoundation, the usual choice on macOS.
        - **MSMF** -- Microsoft Media Foundation, a Windows backend.
    """

    ANY = 0
    V4L2 = 200
    DSHOW = 700
    PVAPI = 800
    ANDROID = 1000
    AVFOUNDATION = 1200
    MSMF = 1400

    @classmethod
    def _missing_(cls, value: object) -> None:
        """Reject a value that is not a valid `Cv2Backends`.

        Raises:
            ValueError: Always, naming the invalid value and the valid choices.
        """
        raise ValueError(f"`backend` is expected to be in {list(cls)}, but {value} is provided.")


@dataclass(kw_only=True)
class CameraConfig(draccus.ChoiceRegistry, abc.ABC):  # type: ignore  # TODO: add type stubs for draccus
    """Base configuration shared by every camera backend.

    Concrete backends subclass this and register themselves with
    `@CameraConfig.register_subclass("name")`, which is what makes `--camera.type=name` work on the
    command line. Subclasses inherit the three fields below and must document them alongside their own.

    Args:
        fps (`int`, *optional*):
            Requested frames per second for the color stream. `None` leaves it at the backend's default.
        width (`int`, *optional*):
            Requested frame width in pixels. `None` leaves it at the backend's default.
        height (`int`, *optional*):
            Requested frame height in pixels. `None` leaves it at the backend's default.
    """

    fps: int | None = None
    width: int | None = None
    height: int | None = None

    @property
    def type(self) -> str:
        """Return the registered name this config was registered under.

        Returns:
            `str`: The name passed to `@CameraConfig.register_subclass`, e.g. `"opencv"`.
        """
        return str(self.get_choice_name(self.__class__))
