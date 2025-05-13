import abc
from dataclasses import dataclass
from pathlib import Path

import draccus


@dataclass(kw_only=True)
class RobotConfig(draccus.ChoiceRegistry, abc.ABC):
    # Allows to distinguish between different robots of the same type
    id: str | None = None
    # Directory to store calibration file
    calibration_dir: Path | None = None

    def __post_init__(self):
        if hasattr(self, "cameras"):
            cameras = self.cameras
            if cameras:
                for cam_name, cam_config in cameras.items():
                    for attr in ["width", "height", "fps"]:
                        if getattr(cam_config, attr) is None:
                            raise ValueError(
                                f"Camera config for '{cam_name}' has None value for required attribute '{attr}'"
                            )

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)
