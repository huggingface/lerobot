import torch

# Pass this as the first argument to init_hydra_config.
DEFAULT_CONFIG_PATH = "lerobot/configs/default.yaml"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
