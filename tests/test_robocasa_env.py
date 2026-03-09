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
"""Tests for RoboCasa LeRobot integration.

Requires: robocasa installed + kitchen assets downloaded.
Tests are skipped automatically if robocasa is not available.
"""
from __future__ import annotations

import numpy as np
import pytest

# Skip entire module if robocasa is not installed or assets are missing
robocasa = pytest.importorskip("robocasa", reason="robocasa not installed")

from lerobot.envs.robocasa import ACTION_DIM, STATE_DIM, CAM_KEY_TO_NAME, RoboCasaEnv, create_robocasa_envs

# The 5 benchmark tasks (3 short + 2 long)
BENCHMARK_TASKS = [
    "PickPlaceCounterToCabinet",  # short
    "PrepareToast",               # short
    "CoffeeSetupMug",             # short
    "PrepareCoffee",              # long
    "RestockPantry",              # long
]
SHORT_TASKS = BENCHMARK_TASKS[:3]
LONG_TASKS = BENCHMARK_TASKS[3:]

IMAGE_SIZE = 64  # small for fast tests


@pytest.fixture(scope="module")
def single_env():
    """Shared env instance for lightweight tests."""
    env = RoboCasaEnv(task="PickPlaceCounterToCabinet", image_size=IMAGE_SIZE)
    yield env
    env.close()


class TestRoboCasaEnvSpaces:
    def test_action_space_is_flat_box(self, single_env):
        import gymnasium as gym

        assert isinstance(single_env.action_space, gym.spaces.Box)
        assert single_env.action_space.shape == (ACTION_DIM,)
        assert single_env.action_space.dtype == np.float32

    def test_action_bounds(self, single_env):
        assert np.all(single_env.action_space.low == -1.0)
        assert np.all(single_env.action_space.high == 1.0)

    def test_observation_space_has_pixels_and_state(self, single_env):
        import gymnasium as gym

        assert isinstance(single_env.observation_space, gym.spaces.Dict)
        assert "pixels" in single_env.observation_space.spaces
        assert "robot_state" in single_env.observation_space.spaces

    def test_observation_space_cameras(self, single_env):
        pixels_space = single_env.observation_space["pixels"]
        expected_cams = set(CAM_KEY_TO_NAME.values())
        assert set(pixels_space.spaces.keys()) == expected_cams

    def test_state_dim(self, single_env):
        state_space = single_env.observation_space["robot_state"]
        assert state_space.shape == (STATE_DIM,)


class TestRoboCasaEnvReset:
    def test_reset_returns_obs_and_info(self, single_env):
        obs, info = single_env.reset()
        assert isinstance(obs, dict)
        assert isinstance(info, dict)

    def test_reset_obs_has_pixels(self, single_env):
        obs, _ = single_env.reset()
        assert "pixels" in obs
        for cam_name in CAM_KEY_TO_NAME.values():
            assert cam_name in obs["pixels"], f"Missing camera: {cam_name}"

    def test_reset_obs_image_shape(self, single_env):
        obs, _ = single_env.reset()
        for cam_name, img in obs["pixels"].items():
            assert img.shape == (IMAGE_SIZE, IMAGE_SIZE, 3), f"Bad shape for {cam_name}: {img.shape}"
            assert img.dtype == np.uint8

    def test_reset_obs_state_shape(self, single_env):
        obs, _ = single_env.reset()
        assert obs["robot_state"].shape == (STATE_DIM,)
        assert obs["robot_state"].dtype == np.float32

    def test_reset_info_has_task(self, single_env):
        _, info = single_env.reset()
        assert "task" in info
        assert info["task"] == "PickPlaceCounterToCabinet"


class TestRoboCasaEnvStep:
    def test_step_10_random_actions(self, single_env):
        single_env.reset()
        for _ in range(10):
            action = single_env.action_space.sample()
            obs, reward, terminated, truncated, info = single_env.step(action)
        assert obs["robot_state"].shape == (STATE_DIM,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_step_bad_action_raises(self, single_env):
        single_env.reset()
        with pytest.raises(ValueError, match="Expected 1-D action"):
            single_env.step(np.zeros((2, ACTION_DIM)))

    def test_step_info_has_is_success(self, single_env):
        single_env.reset()
        _, _, _, _, info = single_env.step(single_env.action_space.sample())
        assert "is_success" in info


class TestRoboCasaConfig:
    def test_robocasa_env_config(self):
        from lerobot.envs.configs import RoboCasaEnv as RoboCasaEnvConfig
        from lerobot.configs.types import FeatureType

        cfg = RoboCasaEnvConfig(task="PickPlaceCounterToCabinet", image_size=IMAGE_SIZE)
        assert cfg.type == "robocasa"
        # action feature
        assert "action" in cfg.features
        assert cfg.features["action"].shape == (ACTION_DIM,)
        # camera features
        for cam in ("agentview_left", "agentview_right", "eye_in_hand"):
            assert cam in cfg.features
            assert cfg.features[cam].type == FeatureType.VISUAL
            assert cfg.features[cam].shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
        # state feature
        assert "robot_state" in cfg.features
        assert cfg.features["robot_state"].shape == (STATE_DIM,)

    def test_make_env_config_robocasa(self):
        from lerobot.envs.factory import make_env_config
        cfg = make_env_config("robocasa", task="PickPlaceCounterToCabinet")
        assert cfg.type == "robocasa"


class TestRoboCasaProcessorStep:
    def test_processor_remaps_keys(self):
        import torch
        from lerobot.processor.env_processor import RoboCasaProcessorStep
        from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

        step = RoboCasaProcessorStep()
        B = 2
        obs = {
            f"{OBS_IMAGES}.agentview_left":  torch.zeros(B, 3, IMAGE_SIZE, IMAGE_SIZE),
            f"{OBS_IMAGES}.agentview_right": torch.zeros(B, 3, IMAGE_SIZE, IMAGE_SIZE),
            f"{OBS_IMAGES}.eye_in_hand":     torch.zeros(B, 3, IMAGE_SIZE, IMAGE_SIZE),
            f"observation.robot_state":      torch.zeros(B, STATE_DIM),
        }
        out = step._process_observation(obs)
        assert OBS_STATE in out
        assert out[OBS_STATE].dtype == torch.float32
        for cam in ("agentview_left", "agentview_right", "eye_in_hand"):
            assert f"{OBS_IMAGES}.{cam}" in out
