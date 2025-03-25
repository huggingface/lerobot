from dataclasses import dataclass, field
from pathlib import Path

from lerobot.common.cameras import CameraConfig

from ..config import RobotConfig

# TODO(pepijn): Do this differently
BASE_CALIBRATION_DIR = Path(".cache/calibration/so100")
BASE_CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)


@RobotConfig.register_subclass("so100")
@dataclass
class SO100RobotConfig(RobotConfig):
    # Port to connect to the robot
    port: str = "/dev/tty.usbmodem58760431201"

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    calibration_fpath: Path = field(default=BASE_CALIBRATION_DIR / "leader_calibration.json")
