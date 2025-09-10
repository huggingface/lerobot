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

from lerobot.envs.configs import AlohaEnv, EnvConfig, HILEnvConfig, LiberoEnv, PushtEnv, XarmEnv


def make_env_config(env_type: str, **kwargs) -> EnvConfig:
    if env_type == "aloha":
        return AlohaEnv(**kwargs)
    elif env_type == "pusht":
        return PushtEnv(**kwargs)
    elif env_type == "xarm":
        return XarmEnv(**kwargs)
    elif env_type == "hil":
        return HILEnvConfig(**kwargs)
    elif env_type == "libero":
        return LiberoEnv(**kwargs)
    else:
        raise ValueError(f"Policy type '{env_type}' is not available.")


def make_env(
    cfg: EnvConfig, n_envs: int = 1, use_async_envs: bool = False
) -> gym.vector.VectorEnv | dict[str, dict[int, gym.vector.VectorEnv]]:
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
        dict[str, dict[int, gym.vector.VectorEnv]]: A mapping from task suite
            names to indexed vectorized environments (when multitask eval is used).

    """
        if n_envs < 1:
            raise ValueError("`n_envs` must be at least 1")

        env_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv

    
        if "libero" in cfg.type:
            from lerobot.envs.libero import create_libero_envs
            return create_libero_envs(
                task=cfg.task,
                n_envs=n_envs,
                camera_name=cfg.camera_name,
                init_states=cfg.init_states,
                gym_kwargs=cfg.gym_kwargs,
                env_cls=env_cls,
                multitask_eval=cfg.multitask_eval,
            )

        
        package_name = f"gym_{cfg.type}"
        try:
            importlib.import_module(package_name)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"{package_name} is not installed. Install with: pip install \"lerobot[{cfg.type}]\""
            ) from e

        gym_handle = f"{package_name}/{cfg.task}"
        
        def _make_one():
            return gym.make(gym_handle, disable_env_checker=True, **(cfg.gym_kwargs or {}))

        return env_cls([_make_one for _ in range(n_envs)])
