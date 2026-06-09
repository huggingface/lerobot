from dataclasses import dataclass

from lerobot.robots import RobotConfig
from lerobot.robots.so_follower import SOFollowerConfig


@RobotConfig.register_subclass("bi_so_follower_mobile")
@dataclass
class BiSOFollowerMobileConfig(RobotConfig):
    """
    Configuration for bimanual SO-101 follower arms with a dual-wheel mobile base.
    The wheel motors (STS3250, IDs 9 & 10) share the right arm's motor bus.
    """
    left_arm_config: SOFollowerConfig
    right_arm_config: SOFollowerConfig
