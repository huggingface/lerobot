from dataclasses import dataclass, field

from lerobot.common.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("koch")
@dataclass
class KochRobotConfig(RobotConfig):
    # Port to connect to the robot
    port: str

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    # Sets the leader arm in torque mode with the gripper motor set to this angle. This makes it possible
    # to squeeze the gripper and have it spring back to an open position on its own.
    gripper_open_degree: float = 35.156

    mock: bool = False

    # motors
    shoulder_pan: tuple = (1, "xl430-w250")
    shoulder_lift: tuple = (2, "xl430-w250")
    elbow_flex: tuple = (3, "xl330-m288")
    wrist_flex: tuple = (4, "xl330-m288")
    wrist_roll: tuple = (5, "xl330-m288")
    gripper: tuple = (6, "xl330-m288")

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
