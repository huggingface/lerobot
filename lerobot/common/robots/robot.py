import abc
from typing import Any

from lerobot.common.constants import HF_LEROBOT_CALIBRATION, ROBOTS

from .config import RobotConfig


# TODO(aliberts): action/obs typing such as Generic[ObsType, ActType] similar to gym.Env ?
# https://github.com/Farama-Foundation/Gymnasium/blob/3287c869f9a48d99454306b0d4b4ec537f0f35e3/gymnasium/core.py#L23
class Robot(abc.ABC):
    """The main LeRobot class for implementing robots."""

    # Set these in ALL subclasses
    config_class: RobotConfig
    name: str

    def __init__(self, config: RobotConfig):
        self.robot_type = self.name
        self.id = config.id
        self.calibration_dir = (
            config.calibration_dir if config.calibration_dir else HF_LEROBOT_CALIBRATION / ROBOTS / self.name
        )
        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        self.calibration_fpath = self.calibration_dir / f"{self.id}.json"

    # TODO(aliberts): create a proper Feature class for this that links with datasets
    @abc.abstractproperty
    def state_feature(self) -> dict:
        pass

    @abc.abstractproperty
    def action_feature(self) -> dict:
        pass

    @abc.abstractproperty
    def camera_features(self) -> dict[str, dict]:
        pass

    @abc.abstractmethod
    def connect(self) -> None:
        """Connects to the robot."""
        pass

    @abc.abstractmethod
    def calibrate(self) -> None:
        """Calibrates the robot."""
        pass

    @abc.abstractmethod
    def get_observation(self) -> dict[str, Any]:
        """Gets observation from the robot."""
        pass

    @abc.abstractmethod
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Sends actions to the robot."""
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Disconnects from the robot."""
        pass

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
