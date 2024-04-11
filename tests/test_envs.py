import importlib
import pytest
import torch
from lerobot.common.datasets.factory import make_dataset
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from lerobot.common.envs.factory import make_env
from lerobot.common.utils import init_hydra_config

from lerobot.common.envs.utils import preprocess_observation

from .utils import DEVICE, DEFAULT_CONFIG_PATH


@pytest.mark.parametrize(
    "env_name, task, obs_type",
    [
        # ("AlohaInsertion-v0", "state"),
        ("aloha", "AlohaInsertion-v0", "pixels"),
        ("aloha", "AlohaInsertion-v0", "pixels_agent_pos"),
        ("aloha", "AlohaTransferCube-v0", "pixels"),
        ("aloha", "AlohaTransferCube-v0", "pixels_agent_pos"),
        ("xarm", "XarmLift-v0", "state"),
        ("xarm", "XarmLift-v0", "pixels"),
        ("xarm", "XarmLift-v0", "pixels_agent_pos"),
        ("pusht", "PushT-v0", "state"),
        ("pusht", "PushT-v0", "pixels"),
        ("pusht", "PushT-v0", "pixels_agent_pos"),
    ],
)
def test_env(env_name, task, obs_type):
    package_name = f"gym_{env_name}"
    importlib.import_module(package_name)
    env = gym.make(f"{package_name}/{task}", obs_type=obs_type)
    check_env(env.unwrapped, skip_render_check=True)
    env.close()

@pytest.mark.parametrize(
    "env_name",
    [
        "pusht",
        "xarm",
        "aloha",
    ],
)
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
