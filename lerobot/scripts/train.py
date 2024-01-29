import hydra
import torch
from termcolor import colored

from ..lib.utils import set_seed


@hydra.main(version_base=None, config_name="default", config_path="../configs")
def train(cfg: dict):
    assert torch.cuda.is_available()
    set_seed(cfg.seed)
    print(colored("Work dir:", "yellow", attrs=["bold"]), cfg.log_dir)


if __name__ == "__main__":
    train()
