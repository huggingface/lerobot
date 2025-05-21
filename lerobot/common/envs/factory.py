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
from gymnasium.envs.registration import registry as gym_registry

from lerobot.common.envs.configs import AlohaEnv, EnvConfig, PushtEnv, XarmEnv


def make_env_config(env_type: str, **kwargs) -> EnvConfig:
    if env_type == "aloha":
        return AlohaEnv(**kwargs)
    elif env_type == "pusht":
        return PushtEnv(**kwargs)
    elif env_type == "xarm":
        return XarmEnv(**kwargs)
    else:
        raise ValueError(f"Policy type '{env_type}' is not available.")


def make_env(cfg: EnvConfig, n_envs: int = 1, use_async_envs: bool = False) -> gym.vector.VectorEnv | None:
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

    if cfg.gym_id not in gym_registry:
        print(
            f"Gym id {cfg.gym_id} not found in gym registry. Attempting to import package {cfg.package_name}..."
        )
        try:
            importlib.import_module(cfg.package_name)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"Required package '{cfg.package_name}' not found for environment '{cfg.type}'. "
                f"Is it installed and discoverable?"
            ) from e

        if cfg.gym_id not in gym_registry:
            raise gym.error.NameNotFound(
                f"Environment '{cfg.gym_id}' not found. "
                f"Package '{cfg.package_name}' was imported, but the environment ID "
                f"was still not registered. Check the `gym.register()` call within the package."
            )
        else:
            print(f"Successfully imported package {cfg.package_name} and registered gym id {cfg.gym_id}.")

    # batched version of the env that returns an observation of shape (b, c)
    env_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv
    env = env_cls(
        [lambda: gym.make(cfg.gym_id, disable_env_checker=True, **cfg.gym_kwargs) for _ in range(n_envs)]
    )

    return env
