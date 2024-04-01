import os

# Pass this as the first argument to init_hydra_config.
DEFAULT_CONFIG_PATH = "lerobot/configs/default.yaml"

DEVICE = os.environ.get('LEROBOT_TESTS_DEVICE', "cuda")
