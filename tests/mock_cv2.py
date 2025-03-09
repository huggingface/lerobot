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
from functools import cache

import numpy as np

CAP_V4L2 = 200
CAP_DSHOW = 700
CAP_AVFOUNDATION = 1200
CAP_ANY = -1

CAP_PROP_FPS = 5
CAP_PROP_FRAME_WIDTH = 3
CAP_PROP_FRAME_HEIGHT = 4
COLOR_RGB2BGR = 4
COLOR_BGR2RGB = 4

ROTATE_90_COUNTERCLOCKWISE = 2
ROTATE_90_CLOCKWISE = 0
ROTATE_180 = 1


@cache
def _generate_image(width: int, height: int):
    return np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)


def cvtColor(color_image, color_conversion):  # noqa: N802
    if color_conversion in [COLOR_RGB2BGR, COLOR_BGR2RGB]:
        return color_image[:, :, [2, 1, 0]]
    else:
        raise NotImplementedError(color_conversion)


def rotate(color_image, rotation):
    if rotation is None:
        return color_image
    elif rotation == ROTATE_90_CLOCKWISE:
        return np.rot90(color_image, k=1)
    elif rotation == ROTATE_180:
        return np.rot90(color_image, k=2)
    elif rotation == ROTATE_90_COUNTERCLOCKWISE:
        return np.rot90(color_image, k=3)
    else:
        raise NotImplementedError(rotation)


class VideoCapture:
    def __init__(self, *args, **kwargs):
        self._mock_dict = {
            CAP_PROP_FPS: 30,
            CAP_PROP_FRAME_WIDTH: 640,
            CAP_PROP_FRAME_HEIGHT: 480,
        }
        self._is_opened = True

    def isOpened(self):  # noqa: N802
        return self._is_opened

    def set(self, propId: int, value: float) -> bool:  # noqa: N803
        if not self._is_opened:
            raise RuntimeError("Camera is not opened")
        self._mock_dict[propId] = value
        return True

    def get(self, propId: int) -> float:  # noqa: N803
        if not self._is_opened:
            raise RuntimeError("Camera is not opened")
        value = self._mock_dict[propId]
        if value == 0:
            if propId == CAP_PROP_FRAME_HEIGHT:
                value = 480
            elif propId == CAP_PROP_FRAME_WIDTH:
                value = 640
        return value

    def read(self):
        if not self._is_opened:
            raise RuntimeError("Camera is not opened")
        h = self.get(CAP_PROP_FRAME_HEIGHT)
        w = self.get(CAP_PROP_FRAME_WIDTH)
        ret = True
        return ret, _generate_image(width=w, height=h)

    def release(self):
        self._is_opened = False

    def __del__(self):
        if self._is_opened:
            self.release()
