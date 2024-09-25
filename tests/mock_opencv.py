from functools import cache

import cv2
import numpy as np


@cache
def _generate_image(width: int, height: int):
    return np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)


class MockVideoCapture:
    def __init__(self, *args, **kwargs):
        self._mock_dict = {
            cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
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
            if propId == cv2.CAP_PROP_FRAME_HEIGHT:
                value = 480
            elif propId == cv2.CAP_PROP_FRAME_WIDTH:
                value = 640
        return value

    def read(self):
        if not self._is_opened:
            raise RuntimeError("Camera is not opened")
        h = self.get(cv2.CAP_PROP_FRAME_HEIGHT)
        w = self.get(cv2.CAP_PROP_FRAME_WIDTH)
        ret = True
        return ret, _generate_image(width=w, height=h)

    def release(self):
        self._is_opened = False

    def __del__(self):
        if self._is_opened:
            self.release()
