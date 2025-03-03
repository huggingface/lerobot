import abc
from dataclasses import dataclass
from pathlib import Path

import draccus


@dataclass(kw_only=True)
class TeleoperatorConfig(draccus.ChoiceRegistry, abc.ABC):
    # Allows to distinguish between different teleoperators of the same type
    id: str | None = None
    # Directory to store calibration file
    calibration_dir: Path | None = None

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)
