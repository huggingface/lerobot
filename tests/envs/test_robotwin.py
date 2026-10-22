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
"""Unit tests for the RoboTwin 2.0 Gymnasium wrapper.

These tests mock out the SAPIEN-based RoboTwin runtime (task modules +
YAML config loader) so they run without the full RoboTwin installation
(SAPIEN, CuRobo, mplib, asset downloads, etc.).
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import gymnasium as gym
import numpy as np
import pytest

from lerobot.envs.robotwin import (
    ACTION_DIM,
    ROBOTWIN_CAMERA_NAMES,
    ROBOTWIN_TASKS,
    RoboTwinEnv,
    create_robotwin_envs,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_mock_task_env(
    height: int = 240,
    width: int = 320,
    cameras: tuple[str, ...] = ROBOTWIN_CAMERA_NAMES,
) -> MagicMock:
    """Return a mock that mimics the RoboTwin task class API.

    RoboTwin's real get_obs returns
        {"observation": {cam: {"rgb": img}}, "joint_action": {"vector": np.ndarray}, ...}
    so the mock follows the same nested shape.
    """
    obs_dict = {
        "observation": {cam: {"rgb": np.zeros((height, width, 3), dtype=np.uint8)} for cam in cameras},
        "joint_action": {"vector": np.zeros(ACTION_DIM, dtype=np.float32)},
        "endpose": {},
    }

    mock = MagicMock()
    mock.get_obs.return_value = obs_dict
    mock.setup_demo.return_value = None
    mock.take_action.return_value = None
    mock.eval_success = False
    mock.check_success.return_value = False
    mock.close_env.return_value = None
    return mock


@contextmanager
def _patch_runtime(mock_task_instance: MagicMock):
    """Patch both the task-class loader and the YAML config loader so the
    env can construct + reset without a real RoboTwin install."""
    task_cls = MagicMock(return_value=mock_task_instance)
    fake_setup = {
        "head_camera_h": 240,
        "head_camera_w": 320,
        "left_embodiment_config": {},
        "right_embodiment_config": {},
        "left_robot_file": "",
        "right_robot_file": "",
        "dual_arm_embodied": True,
        "render_freq": 0,
        "task_name": "beat_block_hammer",
        "task_config": "demo_clean",
    }
    with (
        patch("lerobot.envs.robotwin._load_robotwin_task", return_value=task_cls),
        patch("lerobot.envs.robotwin._load_robotwin_setup_kwargs", return_value=fake_setup),
    ):
        yield


# ---------------------------------------------------------------------------
# RoboTwinEnv unit tests
# ---------------------------------------------------------------------------


class TestRoboTwinEnv:
    def test_observation_space_shape(self):
        """observation_space should have the configured h×w×3 for every camera."""
        h, w = 240, 320
        env = RoboTwinEnv(
            task_name="beat_block_hammer",
            observation_height=h,
            observation_width=w,
            camera_names=["head_camera", "left_camera"],
        )
        pixels_space = env.observation_space["pixels"]
        assert pixels_space["head_camera"].shape == (h, w, 3)
        assert pixels_space["left_camera"].shape == (h, w, 3)
        assert "right_camera" not in pixels_space

    def test_action_space(self):
        env = RoboTwinEnv(task_name="beat_block_hammer")
        assert env.action_space.shape == (ACTION_DIM,)
        assert env.action_space.dtype == np.float32

    def test_reset_returns_correct_obs_keys(self):
        mock_task = _make_mock_task_env()
        env = RoboTwinEnv(task_name="beat_block_hammer")
        with _patch_runtime(mock_task):
            obs, info = env.reset()

        assert "pixels" in obs
        for cam in ROBOTWIN_CAMERA_NAMES:
            assert cam in obs["pixels"], f"Missing camera '{cam}' in obs"
        assert "agent_pos" in obs
        assert obs["agent_pos"].shape == (ACTION_DIM,)
        assert info["is_success"] is False

    def test_reset_calls_setup_demo(self):
        mock_task = _make_mock_task_env()
        env = RoboTwinEnv(task_name="beat_block_hammer")
        with _patch_runtime(mock_task):
            env.reset(seed=42)
        # setup_demo receives the full YAML-derived kwargs plus seed + is_test;
        # we only assert the caller-provided bits.
        assert mock_task.setup_demo.call_count == 1
        call_kwargs = mock_task.setup_demo.call_args.kwargs
        assert call_kwargs["seed"] == 42
        assert call_kwargs["is_test"] is True

    def test_step_returns_correct_types(self):
        mock_task = _make_mock_task_env()
        env = RoboTwinEnv(task_name="beat_block_hammer")
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        with _patch_runtime(mock_task):
            env.reset()
            obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_wrong_action_shape_raises(self):
        mock_task = _make_mock_task_env()
        env = RoboTwinEnv(task_name="beat_block_hammer")
        bad_action = np.zeros(7, dtype=np.float32)  # wrong dim
        with _patch_runtime(mock_task):
            env.reset()
            with pytest.raises(ValueError, match="Expected 1-D action"):
                env.step(bad_action)

    def test_success_terminates_episode(self):
        mock_task = _make_mock_task_env()
        mock_task.check_success.return_value = True
        env = RoboTwinEnv(task_name="beat_block_hammer")
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        with _patch_runtime(mock_task):
            env.reset()
            _, _, terminated, _, info = env.step(action)
        assert terminated is True
        assert info["is_success"] is True

    def test_truncation_after_episode_length(self):
        mock_task = _make_mock_task_env()
        env = RoboTwinEnv(task_name="beat_block_hammer", episode_length=2)
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        with _patch_runtime(mock_task):
            env.reset()
            env.step(action)  # step 1
            _, _, _, truncated, _ = env.step(action)  # step 2 → truncated
        assert truncated is True

    def test_close_calls_close_env(self):
        mock_task = _make_mock_task_env()
        env = RoboTwinEnv(task_name="beat_block_hammer")
        with _patch_runtime(mock_task):
            env.reset()
            env.close()
        mock_task.close_env.assert_called_once()

    def test_black_frame_for_missing_camera(self):
        """If a camera key is absent from get_obs(), a black frame is returned."""
        # Mock exposes only head_camera; we ask for both head_camera + left_camera.
        mock_task = _make_mock_task_env(height=10, width=10, cameras=("head_camera",))
        env = RoboTwinEnv(
            task_name="beat_block_hammer",
            camera_names=["head_camera", "left_camera"],
            observation_height=10,
            observation_width=10,
        )
        with _patch_runtime(mock_task):
            obs, _ = env.reset()
        assert obs["pixels"]["left_camera"].shape == (10, 10, 3)
        assert obs["pixels"]["left_camera"].sum() == 0

    def test_task_and_task_description_attributes(self):
        env = RoboTwinEnv(task_name="beat_block_hammer")
        assert env.task == "beat_block_hammer"
        assert isinstance(env.task_description, str)

    def test_deferred_init_env_is_none_before_reset(self):
        env = RoboTwinEnv(task_name="beat_block_hammer")
        assert env._env is None  # noqa: SLF001  (testing internal state)


# ---------------------------------------------------------------------------
# create_robotwin_envs tests
# ---------------------------------------------------------------------------


class TestCreateRoboTwinEnvs:
    def test_returns_correct_structure(self):
        mock_task = _make_mock_task_env()
        with _patch_runtime(mock_task):
            envs = create_robotwin_envs(
                task="beat_block_hammer",
                n_envs=1,
                env_cls=gym.vector.SyncVectorEnv,
            )
        assert "beat_block_hammer" in envs
        assert 0 in envs["beat_block_hammer"]
        assert isinstance(envs["beat_block_hammer"][0], gym.vector.SyncVectorEnv)

    def test_multi_task(self):
        mock_task = _make_mock_task_env()
        with _patch_runtime(mock_task):
            envs = create_robotwin_envs(
                task="beat_block_hammer,click_bell",
                n_envs=1,
                env_cls=gym.vector.SyncVectorEnv,
            )
        assert set(envs.keys()) == {"beat_block_hammer", "click_bell"}

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown RoboTwin tasks"):
            create_robotwin_envs(
                task="not_a_real_task",
                n_envs=1,
                env_cls=gym.vector.SyncVectorEnv,
            )

    def test_invalid_n_envs_raises(self):
        with pytest.raises(ValueError, match="n_envs must be a positive int"):
            create_robotwin_envs(
                task="beat_block_hammer",
                n_envs=0,
                env_cls=gym.vector.SyncVectorEnv,
            )


# ---------------------------------------------------------------------------
# ROBOTWIN_TASKS list
# ---------------------------------------------------------------------------


def test_task_list_not_empty():
    assert len(ROBOTWIN_TASKS) >= 50


def test_all_tasks_are_strings():
    assert all(isinstance(t, str) and t for t in ROBOTWIN_TASKS)


def test_no_duplicate_tasks():
    assert len(ROBOTWIN_TASKS) == len(set(ROBOTWIN_TASKS))
