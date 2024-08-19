import hydra
from omegaconf import DictConfig


def make_robot(cfg: DictConfig):
    robot = hydra.utils.instantiate(cfg)
    return robot
