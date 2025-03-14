import abc
from typing import Any

from lerobot.common.constants import HF_LEROBOT_CALIBRATION, TELEOPERATORS

from .config import TeleoperatorConfig


class Teleoperator(abc.ABC):
    """The main LeRobot class for implementing teleoperation devices."""

    # Set these in ALL subclasses
    config_class: TeleoperatorConfig
    name: str

    def __init__(self, config: TeleoperatorConfig):
        self.id = config.id
        self.calibration_dir = (
            config.calibration_dir
            if config.calibration_dir
            else HF_LEROBOT_CALIBRATION / TELEOPERATORS / self.name
        )
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_fpath = self.calibration_dir / f"{self.id}.json"

    @abc.abstractproperty
    def action_feature(self) -> dict:
        pass

    @abc.abstractproperty
    def feedback_feature(self) -> dict:
        pass

    @abc.abstractmethod
    def connect(self) -> None:
        """Connects to the teleoperator."""
        pass

    @abc.abstractmethod
    def calibrate(self) -> None:
        """Calibrates the teleoperator."""
        pass

    @abc.abstractmethod
    def get_action(self) -> dict[str, Any]:
        """Gets the action to send to a teleoperator."""
        pass

    @abc.abstractmethod
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """Sends feedback captured from a robot to the teleoperator."""
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Disconnects from the teleoperator."""
        pass

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
