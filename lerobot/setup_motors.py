from dataclasses import dataclass

import draccus

from .common.robots import RobotConfig, koch_follower, make_robot_from_config, so100_follower  # noqa: F401
from .common.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
)

COMPATIBLE_DEVICES = [
    "koch_follower",
    "koch_leader",
    "so100_follower",
    "so100_leader",
]


@dataclass
class SetupConfig:
    device: RobotConfig | TeleoperatorConfig


@draccus.wrap()
def setup_motors(cfg: SetupConfig):
    if cfg.device.type not in COMPATIBLE_DEVICES:
        raise NotImplementedError

    if isinstance(cfg.device, RobotConfig):
        device = make_robot_from_config(cfg.device)
    else:
        device = make_teleoperator_from_config(cfg.device)

    device.setup_motors()


if __name__ == "__main__":
    setup_motors()
