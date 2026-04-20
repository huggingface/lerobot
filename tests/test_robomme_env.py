# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Unit tests for the RoboMME env wrapper and config.

RoboMME requires Linux + ManiSkill (Vulkan/SAPIEN), so tests that touch the
env wrapper mock the ``robomme`` package. Tests that only exercise the
dataclass config run without any mocking.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np


def _install_robomme_stub():
    """Register a minimal stub for the ``robomme`` package on sys.modules."""
    stub = ModuleType("robomme")
    wrapper_stub = ModuleType("robomme.env_record_wrapper")

    class FakeBuilder:
        def __init__(self, **kwargs):
            pass

        def make_env_for_episode(self, episode_idx: int, max_steps: int):
            env = MagicMock()
            obs = {
                "front_rgb_list": [np.zeros((256, 256, 3), dtype=np.uint8)],
                "wrist_rgb_list": [np.zeros((256, 256, 3), dtype=np.uint8)],
                "joint_state_list": [np.zeros(7, dtype=np.float32)],
                "gripper_state_list": [np.zeros(2, dtype=np.float32)],
            }
            env.reset.return_value = (obs, {"status": "ongoing", "task_goal": "pick the cube"})
            env.step.return_value = (obs, 0.0, False, False, {"status": "ongoing", "task_goal": ""})
            return env

    wrapper_stub.BenchmarkEnvBuilder = FakeBuilder
    stub.env_record_wrapper = wrapper_stub
    sys.modules["robomme"] = stub
    sys.modules["robomme.env_record_wrapper"] = wrapper_stub


def _uninstall_robomme_stub():
    sys.modules.pop("robomme", None)
    sys.modules.pop("robomme.env_record_wrapper", None)


# ---------------------------------------------------------------------------
# Config tests (no sim required)
# ---------------------------------------------------------------------------


def test_robomme_env_config_defaults():
    from lerobot.envs.configs import RoboMMEEnv

    cfg = RoboMMEEnv()
    assert cfg.task == "PickXtimes"
    assert cfg.fps == 10
    assert cfg.episode_length == 300
    assert cfg.action_space == "joint_angle"
    assert cfg.dataset_split == "test"
    assert cfg.task_ids is None


def test_robomme_env_config_type():
    from lerobot.envs.configs import RoboMMEEnv

    cfg = RoboMMEEnv()
    assert cfg.type == "robomme"


def test_robomme_features_map():
    from lerobot.envs.configs import RoboMMEEnv
    from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

    cfg = RoboMMEEnv()
    assert cfg.features_map[ACTION] == ACTION
    assert cfg.features_map["pixels/image"] == f"{OBS_IMAGES}.image"
    assert cfg.features_map["pixels/wrist_image"] == f"{OBS_IMAGES}.wrist_image"
    assert cfg.features_map["agent_pos"] == OBS_STATE


def test_robomme_features_action_dim_joint_angle():
    from lerobot.envs.configs import RoboMMEEnv
    from lerobot.utils.constants import ACTION

    cfg = RoboMMEEnv(action_space="joint_angle")
    assert cfg.features[ACTION].shape == (8,)


def test_robomme_features_action_dim_ee_pose():
    """`ee_pose` uses a 7-D action; __post_init__ sets the correct shape."""
    from lerobot.envs.configs import RoboMMEEnv
    from lerobot.utils.constants import ACTION

    cfg = RoboMMEEnv(action_space="ee_pose")
    assert cfg.features[ACTION].shape == (7,)


# ---------------------------------------------------------------------------
# Obs conversion (pure Python, no sim)
# ---------------------------------------------------------------------------


def test_convert_obs_list_format():
    """_convert_obs takes the last element from list-format obs fields and
    emits a nested ``pixels`` dict (image, wrist_image) plus ``agent_pos``.

    The nested layout is required so ``preprocess_observation()`` in
    ``envs/utils.py`` maps each camera to ``observation.images.<cam>``.
    """
    _install_robomme_stub()
    try:
        from lerobot.envs.robomme import RoboMMEGymEnv

        env = RoboMMEGymEnv.__new__(RoboMMEGymEnv)

        front = np.full((256, 256, 3), 42, dtype=np.uint8)
        wrist = np.full((256, 256, 3), 7, dtype=np.uint8)
        joints = np.arange(7, dtype=np.float32)
        gripper = np.array([0.5, 0.5], dtype=np.float32)

        obs_raw = {
            "front_rgb_list": [np.zeros_like(front), front],
            "wrist_rgb_list": [np.zeros_like(wrist), wrist],
            "joint_state_list": [np.zeros(7, dtype=np.float32), joints],
            "gripper_state_list": [np.zeros(2, dtype=np.float32), gripper],
        }

        result = env._convert_obs(obs_raw)
        np.testing.assert_array_equal(result["pixels"]["image"], front)
        np.testing.assert_array_equal(result["pixels"]["wrist_image"], wrist)
        assert result["agent_pos"].shape == (8,)
        np.testing.assert_array_almost_equal(result["agent_pos"][:7], joints)
        assert result["agent_pos"][7] == gripper[0]
    finally:
        _uninstall_robomme_stub()


def test_convert_obs_array_format():
    """_convert_obs also handles non-list (direct array) obs."""
    _install_robomme_stub()
    try:
        from lerobot.envs.robomme import RoboMMEGymEnv

        env = RoboMMEGymEnv.__new__(RoboMMEGymEnv)

        front = np.zeros((256, 256, 3), dtype=np.uint8)
        obs_raw = {
            "front_rgb_list": front,
            "wrist_rgb_list": front,
            "joint_state_list": np.zeros(7, dtype=np.float32),
            "gripper_state_list": np.zeros(2, dtype=np.float32),
        }
        result = env._convert_obs(obs_raw)
        assert result["pixels"]["image"].shape == (256, 256, 3)
        assert result["pixels"]["wrist_image"].shape == (256, 256, 3)
        assert result["agent_pos"].shape == (8,)
    finally:
        _uninstall_robomme_stub()


# ---------------------------------------------------------------------------
# create_robomme_envs (mocked sim)
# ---------------------------------------------------------------------------


def test_create_robomme_envs_returns_correct_structure():
    """Single task -> {task_name: {task_id: VectorEnv}} with one entry per task_id."""
    _install_robomme_stub()
    try:
        from lerobot.envs.robomme import create_robomme_envs

        env_cls = MagicMock(return_value=MagicMock())
        result = create_robomme_envs(
            task="PickXtimes",
            n_envs=1,
            task_ids=[0, 1],
            env_cls=env_cls,
        )

        assert "PickXtimes" in result
        assert 0 in result["PickXtimes"]
        assert 1 in result["PickXtimes"]
        assert env_cls.call_count == 2
    finally:
        _uninstall_robomme_stub()


def test_create_robomme_envs_multi_task():
    """Comma-separated task list produces one suite per task."""
    _install_robomme_stub()
    try:
        from lerobot.envs.robomme import create_robomme_envs

        env_cls = MagicMock(return_value=MagicMock())
        result = create_robomme_envs(
            task="PickXtimes,BinFill,StopCube",
            n_envs=1,
            env_cls=env_cls,
        )

        assert set(result.keys()) == {"PickXtimes", "BinFill", "StopCube"}
    finally:
        _uninstall_robomme_stub()


def test_create_robomme_envs_raises_on_invalid_env_cls():
    _install_robomme_stub()
    try:
        import pytest

        from lerobot.envs.robomme import create_robomme_envs

        with pytest.raises(ValueError, match="env_cls must be a callable"):
            create_robomme_envs(task="PickXtimes", n_envs=1, env_cls=None)
    finally:
        _uninstall_robomme_stub()
