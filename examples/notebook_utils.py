from pprint import pprint

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

CONFIG_DIR = "../lerobot/configs"
DEFAULT_CONFIG = "default"


def config_notebook(
    policy: str = "diffusion",
    env: str = "pusht",
    device: str = "cpu",
    config_name=DEFAULT_CONFIG,
    config_path=CONFIG_DIR,
    print_config: bool = False,
) -> DictConfig:
    GlobalHydra.instance().clear()
    initialize(config_path=config_path)
    overrides = [
        f"env={env}",
        f"policy={policy}",
        f"device={device}",
    ]
    cfg = compose(config_name=config_name, overrides=overrides)
    if print_config:
        pprint(OmegaConf.to_container(cfg))

    return cfg
