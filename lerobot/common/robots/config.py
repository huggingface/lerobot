import abc
import enum
from dataclasses import dataclass
from pathlib import Path

import draccus


class RobotMode(enum.Enum):
    TELEOP = 0
    AUTO = 1


@dataclass(kw_only=True)
class RobotConfig(draccus.ChoiceRegistry, abc.ABC):
    # Allows to distinguish between different robots of the same type
    id: str | None = None
    # Directory to store calibration file
    calibration_dir: Path | None = None
    robot_mode: RobotMode | None = None

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)
