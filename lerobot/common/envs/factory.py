import importlib

import gymnasium as gym


def make_env(cfg, num_parallel_envs=0) -> gym.Env | gym.vector.SyncVectorEnv:
    """
    Note: When `num_parallel_envs > 0`, this function returns a `SyncVectorEnv` which takes batched action as input and
    returns batched observation, reward, terminated, truncated of `num_parallel_envs` items.
    """
    kwargs = {
        "obs_type": "pixels_agent_pos",
        "render_mode": "rgb_array",
        "max_episode_steps": cfg.env.episode_length,
        "visualization_width": 384,
        "visualization_height": 384,
    }

    package_name = f"gym_{cfg.env.name}"

    try:
        importlib.import_module(package_name)
    except ModuleNotFoundError as e:
        print(
            f"{package_name} is not installed. Please install it with `pip install 'lerobot[{cfg.env.name}]'`"
        )
        raise e

    gym_handle = f"{package_name}/{cfg.env.task}"

    if num_parallel_envs == 0:
        # non-batched version of the env that returns an observation of shape (c)
        env = gym.make(gym_handle, disable_env_checker=True, **kwargs)
    else:
        # batched version of the env that returns an observation of shape (b, c)
        env = gym.vector.SyncVectorEnv(
            [
                lambda: gym.make(gym_handle, disable_env_checker=True, **kwargs)
                for _ in range(num_parallel_envs)
            ]
        )

    return env
