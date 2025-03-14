from dataclasses import dataclass, field

from lerobot.common.robots.config import RobotConfig


@RobotConfig.register_subclass("daemon_lekiwi")
@dataclass
class DaemonLeKiwiRobotConfig(RobotConfig):
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
