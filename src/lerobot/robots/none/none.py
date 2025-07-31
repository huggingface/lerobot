from ..robot import Robot
from .config_none import NoneRobotConfig
from typing import Any


class NoneRobot(Robot):
    cfg: NoneRobotConfig
    name = "none"
    config_class = NoneRobotConfig

    # all abstract methods become no-ops
    @property
    def observation_features(self) -> dict:
        return {"dummy_obs": float}

    @property
    def action_features(self) -> dict:
        return {"dummy_act": float}

    @property
    def feedback_features(self) -> dict:
        return {"dummy_fb": float}

    @property
    def is_connected(self) -> bool:
        return True

    @property
    def is_calibrated(self) -> bool:
        return True

    # connectivity --------------------------------------------------------- #
    def connect(self, calibrate: bool = True) -> None:
        pass

    def disconnect(self) -> None:
        pass

    # calibration / config ------------------------------------------------- #
    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    # data paths ----------------------------------------------------------- #
    def get_observation(self) -> dict[str, Any]:
        return {}

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        return action  # echo back unchanged for completeness

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass
