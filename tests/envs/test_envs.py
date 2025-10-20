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
from dataclasses import dataclass, field

import gymnasium as gym
import pytest
import torch
from gymnasium.envs.registration import register, registry as gym_registry
from gymnasium.utils.env_checker import check_env

import lerobot
from lerobot.configs.types import PolicyFeature
from lerobot.envs.configs import EnvConfig
from lerobot.envs.factory import make_env, make_env_config
from lerobot.envs.utils import preprocess_observation
from tests.utils import require_env

OBS_TYPES = ["state", "pixels", "pixels_agent_pos"]


@pytest.mark.parametrize("obs_type", OBS_TYPES)
@pytest.mark.parametrize("env_name, env_task", lerobot.env_task_pairs)
@require_env
def test_env(env_name, env_task, obs_type):
    if env_name == "aloha" and obs_type == "state":
        pytest.skip("`state` observations not available for aloha")

    package_name = f"gym_{env_name}"
    importlib.import_module(package_name)
    env = gym.make(f"{package_name}/{env_task}", obs_type=obs_type)
    check_env(env.unwrapped, skip_render_check=True)
    env.close()


@pytest.mark.parametrize("env_name", lerobot.available_envs)
@require_env
def test_factory(env_name):
    cfg = make_env_config(env_name)
    envs = make_env(cfg, n_envs=1)
    suite_name = next(iter(envs))
    task_id = next(iter(envs[suite_name]))
    env = envs[suite_name][task_id]
    obs, _ = env.reset()
    obs = preprocess_observation(obs)

    # test image keys are float32 in range [0,1]
    for key in obs:
        if "image" not in key:
            continue
        img = obs[key]
        assert img.dtype == torch.float32
        # TODO(rcadene): we assume for now that image normalization takes place in the model
        assert img.max() <= 1.0
        assert img.min() >= 0.0

    env.close()


def test_factory_custom_gym_id():
    gym_id = "dummy_gym_pkg/DummyTask-v0"
    if gym_id in gym_registry:
        pytest.skip(f"Environment ID {gym_id} is already registered")

    @EnvConfig.register_subclass("dummy")
    @dataclass
    class DummyEnv(EnvConfig):
        task: str = "DummyTask-v0"
        fps: int = 10
        features: dict[str, PolicyFeature] = field(default_factory=dict)

        @property
        def package_name(self) -> str:
            return "dummy_gym_pkg"

        @property
        def gym_id(self) -> str:
            return gym_id

        @property
        def gym_kwargs(self) -> dict:
            return {}

    try:
        register(id=gym_id, entry_point="gymnasium.envs.classic_control:CartPoleEnv")

        cfg = DummyEnv()
        envs_dict = make_env(cfg, n_envs=1)
        dummy_envs = envs_dict["dummy"]
        assert len(dummy_envs) == 1
        env = next(iter(dummy_envs.values()))
        assert env is not None and isinstance(env, gym.vector.VectorEnv)
        env.close()

    finally:
        if gym_id in gym_registry:
            del gym_registry[gym_id]
