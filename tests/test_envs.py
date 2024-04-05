import pytest
import torch
from lerobot.common.datasets.factory import make_dataset
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from lerobot.common.envs.factory import make_env
from lerobot.common.utils import init_hydra_config

from lerobot.common.envs.utils import preprocess_observation

# import dmc_aloha  # noqa: F401

from .utils import DEVICE, DEFAULT_CONFIG_PATH


# def print_spec_rollout(env):
#     print("observation_spec:", env.observation_spec)
#     print("action_spec:", env.action_spec)
#     print("reward_spec:", env.reward_spec)
#     print("done_spec:", env.done_spec)

#     td = env.reset()
#     print("reset tensordict", td)

#     td = env.rand_step(td)
#     print("random step tensordict", td)

#     def simple_rollout(steps=100):
#         # preallocate:
#         data = TensorDict({}, [steps])
#         # reset
#         _data = env.reset()
#         for i in range(steps):
#             _data["action"] = env.action_spec.rand()
#             _data = env.step(_data)
#             data[i] = _data
#             _data = step_mdp(_data, keep_other=True)
#         return data

#     print("data from rollout:", simple_rollout(100))


@pytest.mark.skip("TODO")
@pytest.mark.parametrize(
    "task,from_pixels,pixels_only",
    [
        ("sim_insertion", True, False),
        ("sim_insertion", True, True),
        ("sim_transfer_cube", True, False),
        ("sim_transfer_cube", True, True),
    ],
)
def test_aloha(task, from_pixels, pixels_only):
    env = AlohaEnv(
        task,
        from_pixels=from_pixels,
        pixels_only=pixels_only,
        image_size=[3, 480, 640] if from_pixels else None,
    )
    # print_spec_rollout(env)
    check_env_specs(env)


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
    check_env(env)



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
    check_env(env)


@pytest.mark.parametrize(
    "env_name",
    [
        "pusht",
        "simxarm",
        # "aloha",
    ],
)
def test_factory(env_name):
    cfg = init_hydra_config(
        DEFAULT_CONFIG_PATH,
        overrides=[f"env={env_name}", f"device={DEVICE}"],
    )

    dataset = make_dataset(cfg)

    env = make_env(cfg)
    obs, info = env.reset()
    obs = {key: obs[key][None, ...] for key in obs}
    obs = preprocess_observation(obs, transform=dataset.transform)
    for key in dataset.image_keys:
        img = obs[key]
        assert img.dtype == torch.float32
        # TODO(rcadene): we assume for now that image normalization takes place in the model
        assert img.max() <= 1.0
        assert img.min() >= 0.0
