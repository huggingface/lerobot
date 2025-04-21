import abc
from dataclasses import dataclass
from enum import Enum

import draccus


class ColorMode(Enum):
    RGB = 0
    BGR = 1


@dataclass
class CameraConfig(draccus.ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)
