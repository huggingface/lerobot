from dataclasses import dataclass, field

from lerobot.common.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("moss")
@dataclass
class MossRobotConfig(RobotConfig):
    # Port to connect to the robot
    port: str

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    mock: bool = False

    # motors
    shoulder_pan: tuple = (1, "sts3215")
    shoulder_lift: tuple = (2, "sts3215")
    elbow_flex: tuple = (3, "sts3215")
    wrist_flex: tuple = (4, "sts3215")
    wrist_roll: tuple = (5, "sts3215")
    gripper: tuple = (6, "sts3215")

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
