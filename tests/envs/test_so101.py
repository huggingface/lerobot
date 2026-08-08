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
"""Unit tests for the SO-101 MuJoCo sim env + config.

Config-only tests run unconditionally (no `mujoco` needed). Tests that touch the actual
sim (`SO101MujocoEnv` construction, `reset`/`step`, rendering) are skipped when `mujoco`
isn't installed (`pip install lerobot[so101]`) — same pattern as the rest of this repo's
optional-dependency env tests (see `tests/envs/test_envs.py`'s `@require_env`).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from tests.utils import skip_if_package_missing

# ---------------------------------------------------------------------------
# Config tests (no mujoco required)
# ---------------------------------------------------------------------------


def test_so101_env_config_defaults():
    from lerobot.envs.configs import SO101Env

    cfg = SO101Env()
    assert cfg.fps == 30
    assert cfg.episode_length == 300
    assert cfg.obs_type == "state"
    assert cfg.reset_qpos_deg is None


def test_so101_env_config_type():
    from lerobot.envs.configs import SO101Env

    assert SO101Env().type == "so101"


def test_so101_features_map():
    from lerobot.envs.configs import SO101Env
    from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE

    cfg = SO101Env()
    assert cfg.features_map[ACTION] == ACTION
    assert cfg.features_map["agent_pos"] == OBS_STATE
    assert cfg.features_map["pixels"] == OBS_IMAGE


def test_so101_features_action_and_state_dims():
    from lerobot.envs.configs import SO101Env
    from lerobot.utils.constants import ACTION

    cfg = SO101Env()
    assert cfg.features[ACTION].shape == (6,)
    assert cfg.features["agent_pos"].shape == (6,)
    # "state" obs_type (default) doesn't add a visual feature.
    assert "pixels" not in cfg.features


def test_so101_pixels_agent_pos_adds_visual_feature():
    from lerobot.envs.configs import SO101Env

    cfg = SO101Env(obs_type="pixels_agent_pos", observation_height=64, observation_width=96)
    assert cfg.features["pixels"].shape == (64, 96, 3)


def test_so101_invalid_obs_type_raises():
    from lerobot.envs.configs import SO101Env

    with pytest.raises(ValueError, match="Unsupported obs_type"):
        SO101Env(obs_type="bogus")


def test_so101_gym_kwargs_threads_reset_qpos_deg():
    from lerobot.envs.configs import SO101Env

    cfg = SO101Env(reset_qpos_deg=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert cfg.gym_kwargs["reset_qpos_deg"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


# ---------------------------------------------------------------------------
# create_so101_envs — env_cls is mocked, so the real SO101MujocoEnv factories are never
# invoked and no mujoco import happens. Same pattern as
# `tests/test_robomme_env.py::test_create_robomme_envs_returns_correct_structure`.
# ---------------------------------------------------------------------------


def test_create_so101_envs_returns_correct_structure():
    from lerobot.envs.so101 import create_so101_envs

    env_cls = MagicMock(return_value=MagicMock())
    result = create_so101_envs(n_envs=3, env_cls=env_cls)

    assert set(result.keys()) == {"so101"}
    assert set(result["so101"].keys()) == {0}
    (fns,), _ = env_cls.call_args
    assert len(fns) == 3
    env_cls.assert_called_once()


def test_create_so101_envs_raises_on_invalid_env_cls():
    from lerobot.envs.so101 import create_so101_envs

    with pytest.raises(ValueError, match="env_cls must be a callable"):
        create_so101_envs(n_envs=1, env_cls=None)


def test_create_so101_envs_raises_on_invalid_n_envs():
    from lerobot.envs.so101 import create_so101_envs

    with pytest.raises(ValueError, match="n_envs must be a positive int"):
        create_so101_envs(n_envs=0, env_cls=MagicMock())


# ---------------------------------------------------------------------------
# Real-sim tests (mujoco required)
# ---------------------------------------------------------------------------


@skip_if_package_missing("mujoco")
def test_env_passes_gymnasium_check_env_state():
    from gymnasium.utils.env_checker import check_env

    from lerobot.envs.so101 import SO101MujocoEnv

    env = SO101MujocoEnv(obs_type="state", episode_length=5)
    check_env(env.unwrapped, skip_render_check=True)
    env.close()


@skip_if_package_missing("mujoco")
def test_env_passes_gymnasium_check_env_pixels():
    from gymnasium.utils.env_checker import check_env

    from lerobot.envs.so101 import SO101MujocoEnv

    env = SO101MujocoEnv(
        obs_type="pixels_agent_pos", observation_height=64, observation_width=64, episode_length=5
    )
    check_env(env.unwrapped, skip_render_check=False)
    env.close()


@skip_if_package_missing("mujoco")
def test_action_space_matches_mjcf_joint_limits():
    from lerobot.envs.so101 import MOTOR_NAMES, SO101MujocoEnv, joint_limits_deg

    env = SO101MujocoEnv()
    limits = joint_limits_deg(env.model)
    for i, name in enumerate(MOTOR_NAMES):
        lo, hi = limits[name]
        assert env.action_space.low[i] == pytest.approx(lo, abs=1e-3)
        assert env.action_space.high[i] == pytest.approx(hi, abs=1e-3)
    env.close()


@skip_if_package_missing("mujoco")
def test_reset_default_qpos_is_zero():
    from lerobot.envs.so101 import SO101MujocoEnv

    env = SO101MujocoEnv(obs_type="state")
    obs, info = env.reset(seed=0)
    np.testing.assert_allclose(obs["agent_pos"], np.zeros(6), atol=1e-6)
    assert info == {}
    env.close()


@skip_if_package_missing("mujoco")
def test_reset_qpos_deg_override_via_options():
    from lerobot.envs.so101 import SO101MujocoEnv

    env = SO101MujocoEnv(obs_type="state")
    target = [10.0, -5.0, 3.0, 0.0, 0.0, 0.0]
    obs, _ = env.reset(seed=0, options={"reset_qpos_deg": target})
    np.testing.assert_allclose(obs["agent_pos"], target, atol=1e-3)
    env.close()


@skip_if_package_missing("mujoco")
def test_reset_qpos_deg_constructor_default():
    from lerobot.envs.so101 import SO101MujocoEnv

    default_pose = [5.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    env = SO101MujocoEnv(obs_type="state", reset_qpos_deg=default_pose)
    obs, _ = env.reset(seed=0)
    np.testing.assert_allclose(obs["agent_pos"], default_pose, atol=1e-3)
    env.close()


@skip_if_package_missing("mujoco")
def test_step_action_is_clipped_to_action_space():
    from lerobot.envs.so101 import SO101MujocoEnv

    env = SO101MujocoEnv(obs_type="state", episode_length=10)
    env.reset(seed=0)
    huge_action = env.action_space.high + 1000.0
    obs, reward, terminated, truncated, info = env.step(huge_action)
    assert reward == 0.0
    assert terminated is False
    assert np.all(obs["agent_pos"] <= env.action_space.high + 1e-3)
    env.close()


@skip_if_package_missing("mujoco")
def test_episode_truncates_at_episode_length():
    from lerobot.envs.so101 import SO101MujocoEnv

    env = SO101MujocoEnv(obs_type="state", episode_length=3)
    env.reset(seed=0)
    zero_action = np.zeros(6, dtype=np.float32)
    truncated = False
    steps = 0
    for _ in range(3):
        _, _, terminated, truncated, _ = env.step(zero_action)
        steps += 1
        assert terminated is False
    assert truncated is True
    assert steps == 3
    env.close()


@skip_if_package_missing("mujoco")
def test_factory_via_env_config():
    import gymnasium as gym

    from lerobot.envs.configs import SO101Env
    from lerobot.envs.utils import preprocess_observation

    cfg = SO101Env(episode_length=5)
    envs = cfg.create_envs(n_envs=2)
    vec = envs["so101"][0]
    assert isinstance(vec, gym.vector.VectorEnv)
    assert vec.num_envs == 2

    obs, _ = vec.reset()
    obs = preprocess_observation(obs)
    assert obs["observation.state"].shape == (2, 6)

    action = vec.action_space.sample()
    obs, reward, terminated, truncated, info = vec.step(action)
    assert len(reward) == 2
    vec.close()
