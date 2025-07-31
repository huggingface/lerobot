from dataclasses import dataclass
from ..config import RobotConfig


@RobotConfig.register_subclass("none")
@dataclass
class NoneRobotConfig(RobotConfig):
    """Placeholder robot for teleop setups that don’t need local hardware."""

    pass
