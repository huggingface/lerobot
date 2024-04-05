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
    "env_task, obs_type",
    [
        # ("AlohaInsertion-v0", "state"),
        ("AlohaInsertion-v0", "pixels"),
        ("AlohaInsertion-v0", "pixels_agent_pos"),
        ("AlohaTransferCube-v0", "pixels"),
        ("AlohaTransferCube-v0", "pixels_agent_pos"),
    ],
)
def test_aloha(env_task, obs_type):
    from lerobot.common.envs import aloha as gym_aloha  # noqa: F401
    env = gym.make(f"gym_aloha/{env_task}", obs_type=obs_type)
    check_env(env.unwrapped)



@pytest.mark.parametrize(
    "env_task, obs_type",
    [
        ("XarmLift-v0", "state"),
        ("XarmLift-v0", "pixels"),
        ("XarmLift-v0", "pixels_agent_pos"),
        # TODO(aliberts): Add gym_xarm other tasks
    ],
)
def test_xarm(env_task, obs_type):
    import gym_xarm  # noqa: F401
    env = gym.make(f"gym_xarm/{env_task}", obs_type=obs_type)
    check_env(env.unwrapped)



@pytest.mark.parametrize(
    "env_task, obs_type",
    [
        ("PushTPixels-v0", "state"),
        ("PushTPixels-v0", "pixels"),
        ("PushTPixels-v0", "pixels_agent_pos"),
    ],
)
def test_pusht(env_task, obs_type):
    import gym_pusht  # noqa: F401
    env = gym.make(f"gym_pusht/{env_task}", obs_type=obs_type)
    check_env(env.unwrapped)


@pytest.mark.parametrize(
    "env_name",
    [
        "pusht",
        "simxarm",
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
    obs, info = env.reset()
    obs = preprocess_observation(obs, transform=dataset.transform)
    for key in dataset.image_keys:
        img = obs[key]
        assert img.dtype == torch.float32
        # TODO(rcadene): we assume for now that image normalization takes place in the model
        assert img.max() <= 1.0
        assert img.min() >= 0.0
