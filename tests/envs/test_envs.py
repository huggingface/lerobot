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
import numpy as np
import pytest
import torch
from gymnasium.envs.registration import register, registry as gym_registry
from gymnasium.utils.env_checker import check_env

import lerobot
from lerobot.configs.types import PolicyFeature
from lerobot.envs.configs import EnvConfig
from lerobot.envs.factory import make_env, make_env_config
from lerobot.envs.utils import (
    _normalize_hub_result,
    _parse_hub_url,
    preprocess_observation,
)
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


# Hub environment loading tests


def test_make_env_hub_url_parsing():
    """Test URL parsing for hub environment references."""
    # simple repo_id
    repo_id, revision, file_path = _parse_hub_url("user/repo")
    assert repo_id == "user/repo"
    assert revision is None
    assert file_path == "env.py"

    # repo with revision
    repo_id, revision, file_path = _parse_hub_url("user/repo@main")
    assert repo_id == "user/repo"
    assert revision == "main"
    assert file_path == "env.py"

    # repo with custom file path
    repo_id, revision, file_path = _parse_hub_url("user/repo:custom_env.py")
    assert repo_id == "user/repo"
    assert revision is None
    assert file_path == "custom_env.py"

    # repo with revision and custom file path
    repo_id, revision, file_path = _parse_hub_url("user/repo@v1.0:envs/my_env.py")
    assert repo_id == "user/repo"
    assert revision == "v1.0"
    assert file_path == "envs/my_env.py"

    # repo with commit hash
    repo_id, revision, file_path = _parse_hub_url("org/repo@abc123def456")
    assert repo_id == "org/repo"
    assert revision == "abc123def456"
    assert file_path == "env.py"


def test_normalize_hub_result():
    """Test normalization of different return types from hub make_env."""
    # test with VectorEnv (most common case)
    mock_vec_env = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1")])
    result = _normalize_hub_result(mock_vec_env)
    assert isinstance(result, dict)
    assert len(result) == 1
    suite_name = next(iter(result))
    assert 0 in result[suite_name]
    assert isinstance(result[suite_name][0], gym.vector.VectorEnv)
    mock_vec_env.close()

    # test with single Env
    mock_env = gym.make("CartPole-v1")
    result = _normalize_hub_result(mock_env)
    assert isinstance(result, dict)
    suite_name = next(iter(result))
    assert 0 in result[suite_name]
    assert isinstance(result[suite_name][0], gym.vector.VectorEnv)
    result[suite_name][0].close()

    # test with dict (already normalized)
    mock_vec_env = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1")])
    input_dict = {"my_suite": {0: mock_vec_env}}
    result = _normalize_hub_result(input_dict)
    assert result == input_dict
    assert "my_suite" in result
    assert 0 in result["my_suite"]
    mock_vec_env.close()

    # test with invalid type
    with pytest.raises(ValueError, match="Hub `make_env` must return"):
        _normalize_hub_result("invalid_type")


def test_make_env_from_hub_requires_trust_remote_code():
    """Test that loading from hub requires explicit trust_remote_code=True."""
    hub_id = "lerobot/cartpole-env"

    # Should raise RuntimeError when trust_remote_code=False (default)
    with pytest.raises(RuntimeError, match="Refusing to execute remote code"):
        make_env(hub_id, trust_remote_code=False)

    # Should also raise when not specified (defaults to False)
    with pytest.raises(RuntimeError, match="Refusing to execute remote code"):
        make_env(hub_id)


@pytest.mark.parametrize(
    "hub_id",
    [
        "lerobot/cartpole-env",
        "lerobot/cartpole-env@main",
        "lerobot/cartpole-env:env.py",
    ],
)
def test_make_env_from_hub_with_trust(hub_id):
    """Test loading environment from Hugging Face Hub with trust_remote_code=True."""
    # load environment from hub
    envs_dict = make_env(hub_id, n_envs=2, trust_remote_code=True)

    # verify structure
    assert isinstance(envs_dict, dict)
    assert len(envs_dict) >= 1

    # get the first suite and task
    suite_name = next(iter(envs_dict))
    task_id = next(iter(envs_dict[suite_name]))
    env = envs_dict[suite_name][task_id]

    # verify it's a vector environment
    assert isinstance(env, gym.vector.VectorEnv)
    assert env.num_envs == 2

    # test basic environment interaction
    obs, info = env.reset()
    assert obs is not None
    assert isinstance(obs, (dict, np.ndarray))

    # take a random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs is not None
    assert isinstance(reward, np.ndarray)
    assert len(reward) == 2

    # clean up
    env.close()


def test_make_env_from_hub_async():
    """Test loading hub environment with async vector environments."""
    hub_id = "lerobot/cartpole-env"

    # load with async envs
    envs_dict = make_env(hub_id, n_envs=2, use_async_envs=True, trust_remote_code=True)

    suite_name = next(iter(envs_dict))
    task_id = next(iter(envs_dict[suite_name]))
    env = envs_dict[suite_name][task_id]

    # verify it's an async vector environment
    assert isinstance(env, gym.vector.AsyncVectorEnv)
    assert env.num_envs == 2

    # test basic interaction
    obs, info = env.reset()
    assert obs is not None

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert len(reward) == 2

    # clean up
    env.close()
