import importlib
import pytest
import torch
from lerobot.common.datasets.factory import make_dataset
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from lerobot.common.envs.factory import make_env
from lerobot.common.utils.import_utils import is_package_available
from lerobot.common.utils.utils import init_hydra_config

import lerobot
from lerobot.common.envs.utils import preprocess_observation

from .utils import DEVICE, DEFAULT_CONFIG_PATH

OBS_TYPES = ["state", "pixels", "pixels_agent_pos"]


@pytest.mark.parametrize("obs_type", OBS_TYPES)
@pytest.mark.parametrize("env_name, env_task", lerobot.env_task_pairs)
def test_env(env_name, env_task, obs_type):
    if env_name == "aloha" and obs_type == "state":
        pytest.skip("`state` observations not available for aloha")
    
    package_name = f"gym_{env_name}"
    if not is_package_available(package_name):
        pytest.skip(f"gym-{env_name} not installed")

    importlib.import_module(package_name)
    env = gym.make(f"{package_name}/{env_task}", obs_type=obs_type)
    check_env(env.unwrapped, skip_render_check=True)
    env.close()


@pytest.mark.parametrize("env_name", lerobot.available_envs)
def test_factory(env_name):
    cfg = init_hydra_config(
        DEFAULT_CONFIG_PATH,
        overrides=[f"env={env_name}", f"device={DEVICE}"],
    )

    dataset = make_dataset(cfg)

    env = make_env(cfg, num_parallel_envs=1)
    obs, _ = env.reset()
    obs = preprocess_observation(obs, transform=dataset.transform)
    for key in dataset.image_keys:
        img = obs[key]
        assert img.dtype == torch.float32
        # TODO(rcadene): we assume for now that image normalization takes place in the model
        assert img.max() <= 1.0
        assert img.min() >= 0.0

    env.close()
