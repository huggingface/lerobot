from dataclasses import dataclass, field

from lerobot.common.cameras.configs import CameraConfig
from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.common.motors.configs import FeetechMotorsBusConfig, MotorsBusConfig
from lerobot.common.robots.config import RobotConfig


@RobotConfig.register_subclass("daemon_lekiwi")
@dataclass
class DaemonLeKiwiRobotConfig(RobotConfig):
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    # Network Configuration
    remote_ip: str = "192.168.0.193"
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556

    teleop_keys: dict[str, str] = field(
        default_factory=lambda: {
            # Movement
            "forward": "w",
            "backward": "s",
            "left": "a",
            "right": "d",
            "rotate_left": "z",
            "rotate_right": "x",
            # Speed control
            "speed_up": "r",
            "speed_down": "f",
            # quit teleop
            "quit": "q",
        }
    )