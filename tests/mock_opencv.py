import cv2
import numpy as np


class MockVideoCapture(cv2.VideoCapture):
    image = {
        "480x640": np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8),
        "720x1280": np.random.randint(0, 256, size=(720, 1280, 3), dtype=np.uint8),
    }

    def __init__(self, *args, **kwargs):
        self._mock_dict = {
            cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
        }

    def isOpened(self):  # noqa: N802
        return True

    def set(self, propId: int, value: float) -> bool:  # noqa: N803
        self._mock_dict[propId] = value
        return True

    def get(self, propId: int) -> float:  # noqa: N803
        value = self._mock_dict[propId]
        if value == 0:
            if propId == cv2.CAP_PROP_FRAME_HEIGHT:
                value = 480
            elif propId == cv2.CAP_PROP_FRAME_WIDTH:
                value = 640
        return value

    def read(self):
        h = self.get(cv2.CAP_PROP_FRAME_HEIGHT)
        w = self.get(cv2.CAP_PROP_FRAME_WIDTH)
        ret = True
        return ret, self.image[f"{h}x{w}"]

    def release(self):
        pass
