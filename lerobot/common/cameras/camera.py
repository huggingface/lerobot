import abc

import numpy as np


class Camera(abc.ABC):
    @abc.abstractmethod
    def connect(self):
        pass

    @abc.abstractmethod
    def read(self, temporary_color: str | None = None) -> np.ndarray:
        pass

    @abc.abstractmethod
    def async_read(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def disconnect(self):
        pass

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
