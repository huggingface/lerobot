import abc

import numpy as np


class Camera(abc.ABC):
    @abc.abstractproperty
    def is_connected(self) -> bool:
        pass

    @abc.abstractmethod
    def connect(self) -> None:
        pass

    @abc.abstractmethod
    def read(self, temporary_color: str | None = None) -> np.ndarray:
        pass

    @abc.abstractmethod
    def async_read(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        pass
