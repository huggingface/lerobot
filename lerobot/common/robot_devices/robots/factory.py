import hydra
from omegaconf import DictConfig

from lerobot.common.robot_devices.robots.utils import Robot


def make_robot(cfg: DictConfig) -> Robot:
    robot = hydra.utils.instantiate(cfg)
    return robot
