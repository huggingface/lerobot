import math
from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("jaka_zu7_follower")
@dataclass
class JAKAZu7FollowerConfig(RobotConfig):
    """Configuration for a JAKA Zu 7 follower arm."""

    ip_address: str
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    auto_power_on: bool = False
    auto_enable_robot: bool = False
    power_on_delay_s: float = 8.0
    servo_step_num: int = 2
    max_joint_velocity_deg_s: float = 15.0
    max_joint_acceleration_deg_s2: float = 30.0
    joint_signs: tuple[int, ...] = (1, 1, 1, 1, 1, 1)
    joint_offsets_deg: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.servo_step_num < 1:
            raise ValueError("servo_step_num must be at least 1")
        if not math.isfinite(self.max_joint_velocity_deg_s) or self.max_joint_velocity_deg_s <= 0:
            raise ValueError("max_joint_velocity_deg_s must be positive")
        if not math.isfinite(self.max_joint_acceleration_deg_s2) or self.max_joint_acceleration_deg_s2 <= 0:
            raise ValueError("max_joint_acceleration_deg_s2 must be positive")
        if not math.isfinite(self.power_on_delay_s) or self.power_on_delay_s < 0:
            raise ValueError("power_on_delay_s must be non-negative")
        if len(self.joint_signs) != 6 or any(sign not in (-1, 1) for sign in self.joint_signs):
            raise ValueError("joint_signs must contain six values, each either -1 or 1")
        if len(self.joint_offsets_deg) != 6 or not all(
            math.isfinite(offset) for offset in self.joint_offsets_deg
        ):
            raise ValueError("joint_offsets_deg must contain six finite values")
