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
from typing import Union

import gymnasium as gym

from lerobot.common.envs.configs import AlohaEnv, EnvConfig, GenesisEnv, PushtEnv, XarmEnv
from lerobot.common.envs.utils import BatchedEnv


def make_env_config(env_type: str, **kwargs) -> EnvConfig:
    if env_type == "aloha":
        return AlohaEnv(**kwargs)
    elif env_type == "pusht":
        return PushtEnv(**kwargs)
    elif env_type == "xarm":
        return XarmEnv(**kwargs)
    elif env_type == "genesis":
        return GenesisEnv(**kwargs)
    else:
        raise ValueError(f"Policy type '{env_type}' is not available.")


def make_env(
    cfg: EnvConfig, n_envs: int = 1, use_async_envs: bool = False
) -> Union[gym.vector.VectorEnv, BatchedEnv] | None:
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
        gym.vector.VectorEnv: The parallelized gym.env instance.
    """
    if n_envs < 1:
        raise ValueError("`n_envs must be at least 1")

    package_name = f"gym_{cfg.type}"

    try:
        importlib.import_module(package_name)
    except ModuleNotFoundError as e:
        print(f"{package_name} is not installed. Please install it with `pip install 'lerobot[{cfg.type}]'`")
        raise e

    gym_handle = f"{package_name}/{cfg.task}"

    if cfg.type in ["genesis"]:
        # genesis env is natively batched—just create it with num_envs=n_envs
        env = gym.make(
            gym_handle,
            num_envs=n_envs,
            disable_env_checker=True,
            **cfg.gym_kwargs,
        )
        return env

    # otherwise, use vectorized Gym environments
    # batched version of the env that returns an observation of shape (b, c)
    env_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv
    env = env_cls(
        [lambda: gym.make(gym_handle, disable_env_checker=True, **cfg.gym_kwargs) for _ in range(n_envs)]
    )

    return env
