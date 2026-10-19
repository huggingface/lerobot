import math
from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("jaka_zu7_follower")
@dataclass
class JAKAZu7FollowerConfig(RobotConfig):
    """Configuration for a JAKA Zu 7 follower arm, driven via the bridge server.

    The real ``jkrc`` SDK runs in a separate Python 3.10 process (see
    ``jaka_bridge_server.py``). This follower is a socket client that talks to that
    server over localhost; it does not import ``jkrc`` directly.
    """

    # Bridge server address. The server holds the controller IP / SDK, not this config.
    bridge_host: str = "127.0.0.1"
    bridge_port: int = 10086

    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Software velocity/acceleration limits applied on the lerobot side (deg / deg/s^2).
    # The bridge server's NLF filter acts as a separate hardware-side safety ceiling.
    max_joint_velocity_deg_s: float = 15.0
    max_joint_acceleration_deg_s2: float = 30.0

    # Direct 1:1 mapping between the SO leader's six joints and JAKA J1..J6:
    # controller_deg = sign * leader_deg + offset.
    joint_signs: tuple[int, ...] = (1, 1, 1, 1, 1, 1)
    joint_offsets_deg: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not math.isfinite(self.max_joint_velocity_deg_s) or self.max_joint_velocity_deg_s <= 0:
            raise ValueError("max_joint_velocity_deg_s must be positive")
        if not math.isfinite(self.max_joint_acceleration_deg_s2) or self.max_joint_acceleration_deg_s2 <= 0:
            raise ValueError("max_joint_acceleration_deg_s2 must be positive")
        if len(self.joint_signs) != 6 or any(sign not in (-1, 1) for sign in self.joint_signs):
            raise ValueError("joint_signs must contain six values, each either -1 or 1")
        if len(self.joint_offsets_deg) != 6 or not all(
            math.isfinite(offset) for offset in self.joint_offsets_deg
        ):
            raise ValueError("joint_offsets_deg must contain six finite values")
