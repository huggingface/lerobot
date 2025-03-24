from dataclasses import dataclass, field
from pathlib import Path

from lerobot.common.cameras import CameraConfig

from ..config import RobotConfig

# TODO(pepijn): Remove these two configs this after test and remove BASE_CALIBRATION_DIR ...
BASE_CALIBRATION_DIR = Path(".cache/calibration/so100")
BASE_CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)


@RobotConfig.register_subclass("so100leader")
@dataclass
class SO100RobotLeaderConfig(RobotConfig):
    # Port to connect to the robot
    port: str

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    calibration_fpath: Path = field(default=BASE_CALIBRATION_DIR / "leader_calibration.json")


@RobotConfig.register_subclass("so100follower")
@dataclass
class SO100RobotFollowerConfig(RobotConfig):
    # Port to connect to the robot
    port: str

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    calibration_fpath: Path = field(default=BASE_CALIBRATION_DIR / "follower_calibration.json")
