import os
import hydra
from hydra import compose, initialize

CONFIG_PATH = "../lerobot/configs"

DEVICE = os.environ.get('LEROBOT_TESTS_DEVICE', "cuda")

def init_config(config_name="default", overrides=None):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(config_path=CONFIG_PATH)
    cfg = compose(config_name=config_name, overrides=overrides)
    return cfg
