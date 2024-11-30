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
from typing import Any, Dict
import importlib

import gymnasium as gym
from omegaconf import DictConfig

def make_env(cfg: DictConfig, n_envs: int | None = None, out_dir: str = "") -> gym.vector.VectorEnv | None:
    """Makes a gym vector environment according to the evaluation config.

    n_envs can be used to override eval.batch_size in the configuration. Must be at least 1.
    """
    if n_envs is not None and n_envs < 1:
        raise ValueError("`n_envs must be at least 1")

    if cfg.env.name == "real_world":
        return

    package_name = f"gym_{cfg.env.name}"

    try:
        importlib.import_module(package_name)
    except ModuleNotFoundError as e:
        print(
            f"{package_name} is not installed. Please install it with `pip install 'lerobot[{cfg.env.name}]'`"
        )
        raise e

    gym_handle = f"{package_name}/{cfg.env.task}"
    gym_kwgs = dict(cfg.env.get("gym", {}))

    if cfg.env.get("episode_length"):
        gym_kwgs["max_episode_steps"] = cfg.env.episode_length

    # e.g. in case the evaluation on a remote server
    disable_rendering = cfg.env.get("disable_rendering", False)
    # if disable_rendering:
    #     print(f"Disabeling video rendering.")
    #     disable_view_window()
        
    # batched version of the env that returns an observation of shape (b, c)
    env_cls = gym.vector.AsyncVectorEnv if cfg.eval.use_async_envs else gym.vector.SyncVectorEnv
    env = env_cls(
        [
            lambda: gym.make(gym_handle, disable_env_checker=True, **gym_kwgs)
            for _ in range(n_envs if n_envs is not None else cfg.eval.batch_size)
        ]
    )

    return env

# get_gym_env_func(gym_handle, disable_env_checker=True, gym_kwgs=gym_kwgs, disable_rendering=disable_rendering, gym_video_path=f"{out_dir}/videos")

# def disable_view_window():
#     org_constructor = gym.envs.classic_control.rendering.Viewer.__init__

#     def constructor(self, *args, **kwargs):
#         org_constructor(self, *args, **kwargs)
#         self.window.set_visible(visible=False)

#     gym.envs.classic_control.rendering.Viewer.__init__ = constructor

# def get_gym_env_func(gym_handle: str, disable_env_checker: bool, gym_kwgs: Dict[str, Any], disable_rendering: bool = False, gym_video_path: str = ""):
#     if disable_rendering:
#         return lambda: gym.wrappers.RecordVideo(gym.make(gym_handle, disable_env_checker=disable_env_checker, **gym_kwgs), gym_video_path, episode_trigger=lambda t: False)
#     else:
#     return lambda: gym.make(gym_handle, disable_env_checker=True, **gym_kwgs)
