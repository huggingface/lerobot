#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import os

import numpy as np
import pytest

from lerobot.envs.factory import make_env, make_env_config

# Set MuJoCo rendering backend before importing environment
os.environ["MUJOCO_GL"] = "egl"


def assert_observations_equal(obs1, obs2, path="", atol=1e-8):
    """
    Recursively compare two observations and assert they are equal.

    Args:
        obs1: First observation (dict or numpy array)
        obs2: Second observation (dict or numpy array)
        path: Current path in nested structure (for error messages)
        atol: Absolute tolerance for numpy array comparisons
    """
    if isinstance(obs1, dict) and isinstance(obs2, dict):
        assert obs1.keys() == obs2.keys(), f"Keys differ at {path}: {obs1.keys()} != {obs2.keys()}"
        for key in obs1:
            assert_observations_equal(obs1[key], obs2[key], path=f"{path}.{key}" if path else key, atol=atol)
    elif isinstance(obs1, np.ndarray) and isinstance(obs2, np.ndarray):
        assert obs1.shape == obs2.shape, f"Shape mismatch at {path}: {obs1.shape} != {obs2.shape}"
        assert obs1.dtype == obs2.dtype, f"Dtype mismatch at {path}: {obs1.dtype} != {obs2.dtype}"
        assert np.allclose(obs1, obs2, atol=atol), (
            f"Array values differ at {path}: max abs diff = {np.abs(obs1 - obs2).max()}"
        )
    else:
        assert type(obs1) is type(obs2), f"Type mismatch at {path}: {type(obs1)} != {type(obs2)}"
        assert obs1 == obs2, f"Values differ at {path}: {obs1} != {obs2}"


def test_libero_env_creation():
    """Test that the libero environment can be created successfully."""
    config = make_env_config("libero", task="libero_spatial")
    envs_dict = make_env(config)

    assert "libero_spatial" in envs_dict
    assert 0 in envs_dict["libero_spatial"]

    env = envs_dict["libero_spatial"][0]
    assert env is not None

    # Test basic reset
    observation, info = env.reset(seed=42)
    assert observation is not None
    assert info is not None

    env.close()


def test_libero_reset_determinism():
    """Test that resetting with the same seed produces identical observations."""
    config = make_env_config("libero", task="libero_spatial")
    envs_dict = make_env(config)
    env = envs_dict["libero_spatial"][0]

    # Reset multiple times with the same seed
    obs1, info1 = env.reset(seed=42)
    obs2, info2 = env.reset(seed=42)
    obs3, info3 = env.reset(seed=42)

    # All observations should be identical
    assert_observations_equal(obs1, obs2)
    assert_observations_equal(obs1, obs3)
    assert_observations_equal(obs2, obs3)

    env.close()


def test_libero_step_determinism():
    """Test that step() is deterministic when resetting with the same seed."""
    config = make_env_config("libero", task="libero_spatial")
    envs_dict = make_env(config)
    env = envs_dict["libero_spatial"][0]

    seed = 42

    # First rollout
    obs1, info1 = env.reset(seed=seed)
    action = env.action_space.sample()
    obs_after_step1, reward1, terminated1, truncated1, info_step1 = env.step(action)

    # Second rollout with identical seed and action
    obs2, info2 = env.reset(seed=seed)
    obs_after_step2, reward2, terminated2, truncated2, info_step2 = env.step(action)

    # Initial observations should be identical
    assert_observations_equal(obs1, obs2)

    # Post-step observations should be identical
    assert_observations_equal(obs_after_step1, obs_after_step2)

    # Rewards and termination flags should be identical
    assert np.allclose(reward1, reward2), f"Rewards differ: {reward1} != {reward2}"
    assert np.array_equal(terminated1, terminated2), (
        f"Terminated flags differ: {terminated1} != {terminated2}"
    )
    assert np.array_equal(truncated1, truncated2), f"Truncated flags differ: {truncated1} != {truncated2}"

    env.close()


@pytest.mark.parametrize("task", ["libero_spatial", "libero_object", "libero_goal", "libero_10"])
def test_libero_tasks(task):
    """Test that different libero tasks can be created and used."""
    config = make_env_config("libero", task=task)
    envs_dict = make_env(config)

    assert task in envs_dict
    assert 0 in envs_dict[task]

    env = envs_dict[task][0]
    observation, info = env.reset(seed=42)

    assert observation is not None
    assert info is not None

    # Take a step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs is not None
    assert reward is not None
    assert isinstance(terminated, (bool, np.ndarray))
    assert isinstance(truncated, (bool, np.ndarray))

    env.close()
