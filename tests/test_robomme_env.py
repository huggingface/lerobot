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

RoboMME requires Linux + ManiSkill (Vulkan/SAPIEN), so all tests that
instantiate the real env mock the ``robomme`` package.  Tests that only
exercise pure-Python logic (config defaults, obs conversion, lazy env)
run without any mocking.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_robomme_stub():
    """Return a minimal stub for the ``robomme`` package."""
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
    return stub, wrapper_stub


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
    assert cfg.features_map["front_rgb"] == f"{OBS_IMAGES}.front"
    assert cfg.features_map["wrist_rgb"] == f"{OBS_IMAGES}.wrist"
    assert cfg.features_map[OBS_STATE] == OBS_STATE


def test_robomme_features_action_dim_joint_angle():
    from lerobot.envs.configs import RoboMMEEnv
    from lerobot.utils.constants import ACTION

    cfg = RoboMMEEnv(action_space="joint_angle")
    assert cfg.features[ACTION].shape == (8,)


def test_robomme_features_action_dim_ee_pose():
    """ee_pose action space uses 7-D; config declares 8-D default (joint_angle).

    Users switching to ee_pose must override the features dict manually or
    the env wrapper will return 7-D actions while the config claims 8-D.
    This test documents the current behaviour so it is explicit.
    """
    from lerobot.envs.configs import RoboMMEEnv
    from lerobot.utils.constants import ACTION

    cfg = RoboMMEEnv(action_space="ee_pose")
    # Default features still say 8-D — ee_pose override is a user responsibility.
    assert cfg.features[ACTION].shape == (8,)


# ---------------------------------------------------------------------------
# Obs conversion (pure Python, no sim)
# ---------------------------------------------------------------------------


def test_convert_obs_list_format():
    """_convert_obs must take the last element from list-format obs fields."""
    stub, wrapper_stub = _make_robomme_stub()
    sys.modules["robomme"] = stub
    sys.modules["robomme.env_record_wrapper"] = wrapper_stub

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

    np.testing.assert_array_equal(result["front_rgb"], front)
    np.testing.assert_array_equal(result["wrist_rgb"], wrist)
    assert result["state"].shape == (8,)
    np.testing.assert_array_almost_equal(result["state"][:7], joints)
    assert result["state"][7] == pytest.approx(gripper[0])

    # cleanup
    del sys.modules["robomme"]
    del sys.modules["robomme.env_record_wrapper"]


def test_convert_obs_array_format():
    """_convert_obs must also handle non-list (direct array) obs."""
    stub, wrapper_stub = _make_robomme_stub()
    sys.modules["robomme"] = stub
    sys.modules["robomme.env_record_wrapper"] = wrapper_stub

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
    assert result["front_rgb"].shape == (256, 256, 3)

    del sys.modules["robomme"]
    del sys.modules["robomme.env_record_wrapper"]


# ---------------------------------------------------------------------------
# create_robomme_envs (mocked sim)
# ---------------------------------------------------------------------------


def test_create_robomme_envs_returns_correct_structure():
    stub, wrapper_stub = _make_robomme_stub()
    sys.modules["robomme"] = stub
    sys.modules["robomme.env_record_wrapper"] = wrapper_stub

    from lerobot.envs.robomme import create_robomme_envs

    env_cls = MagicMock(return_value=MagicMock())
    result = create_robomme_envs(
        task="PickXtimes",
        n_envs=1,
        task_ids=[0, 1],
        env_cls=env_cls,
    )

    assert "robomme" in result
    assert 0 in result["robomme"]
    assert 1 in result["robomme"]
    assert env_cls.call_count == 2

    del sys.modules["robomme"]
    del sys.modules["robomme.env_record_wrapper"]


def test_create_robomme_envs_raises_on_invalid_env_cls():
    stub, wrapper_stub = _make_robomme_stub()
    sys.modules["robomme"] = stub
    sys.modules["robomme.env_record_wrapper"] = wrapper_stub

    from lerobot.envs.robomme import create_robomme_envs

    with pytest.raises(ValueError, match="env_cls must be a callable"):
        create_robomme_envs(task="PickXtimes", n_envs=1, env_cls=None)

    del sys.modules["robomme"]
    del sys.modules["robomme.env_record_wrapper"]


# ---------------------------------------------------------------------------
# LazyVectorEnv
# ---------------------------------------------------------------------------


def test_lazy_vec_env_used_when_task_ids_gt_50():
    """create_robomme_envs must use LazyVectorEnv when len(task_ids) > 50."""
    stub, wrapper_stub = _make_robomme_stub()
    sys.modules["robomme"] = stub
    sys.modules["robomme.env_record_wrapper"] = wrapper_stub

    from lerobot.envs.lazy_vec_env import LazyVectorEnv
    from lerobot.envs.robomme import create_robomme_envs

    env_cls = MagicMock(return_value=MagicMock())
    task_ids = list(range(51))
    result = create_robomme_envs(task="PickXtimes", n_envs=1, task_ids=task_ids, env_cls=env_cls)

    for tid in task_ids:
        assert isinstance(result["robomme"][tid], LazyVectorEnv)
    # env_cls must NOT have been called yet (lazy)
    env_cls.assert_not_called()

    del sys.modules["robomme"]
    del sys.modules["robomme.env_record_wrapper"]


def test_lazy_vec_env_materializes_on_access():
    from lerobot.envs.lazy_vec_env import LazyVectorEnv

    inner = MagicMock()
    inner.reset.return_value = ({"obs": 1}, {})
    env_cls = MagicMock(return_value=inner)
    factory_fns = [lambda: MagicMock()]

    lazy = LazyVectorEnv(env_cls, factory_fns)
    env_cls.assert_not_called()

    # accessing reset triggers materialization
    lazy.reset()
    env_cls.assert_called_once_with(factory_fns)
    inner.reset.assert_called_once()


def test_lazy_vec_env_close_clears_env():
    from lerobot.envs.lazy_vec_env import LazyVectorEnv

    inner = MagicMock()
    env_cls = MagicMock(return_value=inner)
    lazy = LazyVectorEnv(env_cls, [lambda: MagicMock()])

    lazy.reset()  # materialize
    lazy.close()
    inner.close.assert_called_once()

    # env_cls should be called again on next access after close
    lazy.reset()
    assert env_cls.call_count == 2
