#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib

import gymnasium as gym

from lerobot.envs.configs import AlohaEnv, EnvConfig, LiberoEnv, PushtEnv


def make_env_config(env_type: str, **kwargs) -> EnvConfig:
    if env_type == "aloha":
        return AlohaEnv(**kwargs)
    elif env_type == "pusht":
        return PushtEnv(**kwargs)
    elif env_type == "libero":
        return LiberoEnv(**kwargs)
    else:
        raise ValueError(f"Policy type '{env_type}' is not available.")


def make_env(
    cfg: EnvConfig, n_envs: int = 1, use_async_envs: bool = False
) -> dict[str, dict[int, gym.vector.VectorEnv]]:
    """Makes a gym vector environment according to the config.

    Args:
        cfg (EnvConfig): the config of the environment to instantiate.
        n_envs (int, optional): The number of parallelized env to return. Defaults to 1.
        use_async_envs (bool, optional): Whether to return an AsyncVectorEnv or a SyncVectorEnv. Defaults to
            False.

    Raises:
        ValueError: if n_envs < 1
        ModuleNotFoundError: If the requested env package is not installed

    Returns:
        dict[str, dict[int, gym.vector.VectorEnv]]:
            A mapping from suite name to indexed vectorized environments.
            - For multi-task benchmarks (e.g., LIBERO): one entry per suite, and one vec env per task_id.
            - For single-task environments: a single suite entry (cfg.type) with task_id=0.

    """
    if n_envs < 1:
        raise ValueError("`n_envs` must be at least 1")

    env_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv

    if "libero" in cfg.type:
        from lerobot.envs.libero import create_libero_envs

        if cfg.task is None:
            raise ValueError("LiberoEnv requires a task to be specified")

        return create_libero_envs(
            task=cfg.task,
            n_envs=n_envs,
            camera_name=cfg.camera_name,
            init_states=cfg.init_states,
            gym_kwargs=cfg.gym_kwargs,
            env_cls=env_cls,
        )
    elif "metaworld" in cfg.type:
        from lerobot.envs.metaworld import create_metaworld_envs

        if cfg.task is None:
            raise ValueError("MetaWorld requires a task to be specified")

        return create_metaworld_envs(
            task=cfg.task,
            n_envs=n_envs,
            gym_kwargs=cfg.gym_kwargs,
            env_cls=env_cls,
        )
    package_name = f"gym_{cfg.type}"
    try:
        importlib.import_module(package_name)
    except ModuleNotFoundError as e:
        print(f"{package_name} is not installed. Please install it with `pip install 'lerobot[{cfg.type}]'`")
        raise e

    gym_handle = f"{package_name}/{cfg.task}"

    def _make_one():
        return gym.make(gym_handle, disable_env_checker=cfg.disable_env_checker, **(cfg.gym_kwargs or {}))

    vec = env_cls([_make_one for _ in range(n_envs)], autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)

    # normalize to {suite: {task_id: vec_env}} for consistency
    suite_name = cfg.type  # e.g., "pusht", "aloha"
    return {suite_name: {0: vec}}
