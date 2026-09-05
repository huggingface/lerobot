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
"""Unit tests for the MolmoSpaces env wrapper and config.

MolmoSpaces requires Linux + MuJoCo, so tests that touch the env wrapper
mock the ``molmospaces`` package. Tests that only exercise the dataclass
config run without any mocking.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np


def _install_molmospaces_stub():
    """Register a minimal stub for the ``molmospaces`` package on sys.modules."""
    stub = ModuleType("molmospaces")

    episode_stub = ModuleType("molmo_spaces.envs.episode")
    episode_stub.load_benchmark = MagicMock()

    sim_stub = ModuleType("molmo_spaces.simulators.mujoco")
    sim_stub.MuJoCoSimulator = MagicMock

    stub.envs = ModuleType("molmospaces.envs")
    stub.envs.episode = episode_stub
    stub.simulators = ModuleType("molmospaces.simulators")
    stub.simulators.mujoco = sim_stub

    sys.modules["molmospaces"] = stub
    sys.modules["molmospaces.envs"] = stub.envs
    sys.modules["molmospaces.envs.episode"] = episode_stub
    sys.modules["molmospaces.simulators"] = stub.simulators
    sys.modules["molmospaces.simulators.mujoco"] = sim_stub


def _uninstall_molmospaces_stub():
    for mod in [
        "molmospaces",
        "molmospaces.envs",
        "molmospaces.envs.episode",
        "molmospaces.simulators",
        "molmospaces.simulators.mujoco",
    ]:
        sys.modules.pop(mod, None)


# ---------------------------------------------------------------------------
# Config tests (no sim required)
# ---------------------------------------------------------------------------


def test_molmospaces_env_config_defaults():
    from lerobot.envs.configs import MolmoSpacesEnv

    cfg = MolmoSpacesEnv()
    assert cfg.task == "pick"
    assert cfg.fps == 30
    assert cfg.episode_length == 500
    assert cfg.obs_type == "pixels_agent_pos"
    assert cfg.benchmark_name == "molmospaces_bench_v1"


def test_molmospaces_env_config_type():
    from lerobot.envs.configs import MolmoSpacesEnv

    cfg = MolmoSpacesEnv()
    assert cfg.type == "molmospaces"


def test_molmospaces_features_map():
    from lerobot.envs.configs import MolmoSpacesEnv
    from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

    cfg = MolmoSpacesEnv()
    assert cfg.features_map[ACTION] == ACTION
    assert cfg.features_map["pixels/front"] == f"{OBS_IMAGES}.front"


def test_molmospaces_features_action_dim():
    from lerobot.envs.configs import MolmoSpacesEnv
    from lerobot.utils.constants import ACTION

    cfg = MolmoSpacesEnv()
    assert cfg.features[ACTION].shape == (7,)


def test_molmospaces_features_with_pixels():
    from lerobot.envs.configs import MolmoSpacesEnv
    from lerobot.configs import FeatureType

    cfg = MolmoSpacesEnv(obs_type="pixels")
    assert "pixels/front" in cfg.features
    assert cfg.features["pixels/front"].type == FeatureType.VISUAL


def test_molmospaces_features_with_pixels_agent_pos():
    from lerobot.envs.configs import MolmoSpacesEnv
    from lerobot.configs import FeatureType
    from lerobot.utils.constants import MOLMO_SPACES_KEY_JOINT_POS

    cfg = MolmoSpacesEnv(obs_type="pixels_agent_pos")
    assert "pixels/front" in cfg.features
    assert cfg.features["pixels/front"].type == FeatureType.VISUAL
    assert MOLMO_SPACES_KEY_JOINT_POS in cfg.features
    assert cfg.features[MOLMO_SPACES_KEY_JOINT_POS].type == FeatureType.STATE


def test_molmospaces_raises_on_invalid_obs_type():
    from lerobot.envs.configs import MolmoSpacesEnv

    with np.testing.assert_raises(ValueError, match="Unsupported obs_type"):
        MolmoSpacesEnv(obs_type="invalid")


def test_molmospaces_gym_kwargs():
    from lerobot.envs.configs import MolmoSpacesEnv

    cfg = MolmoSpacesEnv(
        task="pick_place",
        obs_type="pixels",
        episode_length=300,
        benchmark_name="molmospaces_bench_v2",
    )
    kwargs = cfg.gym_kwargs
    assert kwargs["obs_type"] == "pixels"
    assert kwargs["max_episode_steps"] == 300
    assert kwargs["benchmark_name"] == "molmospaces_bench_v2"


# ---------------------------------------------------------------------------
# Env wrapper tests (mocked sim)
# ---------------------------------------------------------------------------


def test_molmospaces_env_observation_space_pixels():
    """Observation space for pixels_only uses nested dict with camera name."""
    _install_molmospaces_stub()
    try:
        from lerobot.envs.molmospaces import MolmoSpacesEnv

        env = MolmoSpacesEnv(obs_type="pixels")
        space = dict(env.observation_space)

        assert "pixels" in space
        pixels_space = space["pixels"]
        assert list(pixels_space.keys())[0] == "front"
        assert pixels_space["front"].shape == (256, 256, 3)
    finally:
        _uninstall_molmospaces_stub()


def test_molmospaces_env_observation_space_pixels_agent_pos():
    """Observation space for pixels_agent_pos includes agent_pos."""
    _install_molmospaces_stub()
    try:
        from lerobot.envs.molmospaces import MolmoSpacesEnv

        env = MolmoSpacesEnv(obs_type="pixels_agent_pos")
        space = dict(env.observation_space)

        assert "pixels" in space
        pixels_space = space["pixels"]
        assert list(pixels_space.keys())[0] == "front"
        assert "agent_pos" in space
        assert space["agent_pos"].shape == (14,)
    finally:
        _uninstall_molmospaces_stub()


def test_molmospaces_env_action_space():
    """Action space is Box(-1, 1, shape=(7,))."""
    _install_molmospaces_stub()
    try:
        from lerobot.envs.molmospaces import MolmoSpacesEnv

        env = MolmoSpacesEnv()
        space = env.action_space

        assert space.shape == (7,)
        assert space.low.min() == -1.0
        assert space.high.max() == 1.0
    finally:
        _uninstall_molmospaces_stub()


def test_molmospaces_get_dummy_obs_pixels():
    """_get_dummy_obs returns correct format for pixels_only."""
    _install_molmospaces_stub()
    try:
        from lerobot.envs.molmospaces import MolmoSpacesEnv

        env = MolmoSpacesEnv(obs_type="pixels")
        obs = env._get_dummy_obs()

        assert "pixels" in obs
        assert "front" in obs["pixels"]
        assert obs["pixels"]["front"].shape == (256, 256, 3)
    finally:
        _uninstall_molmospaces_stub()


def test_molmospaces_get_dummy_obs_pixels_agent_pos():
    """_get_dummy_obs returns correct format for pixels_agent_pos."""
    _install_molmospaces_stub()
    try:
        from lerobot.envs.molmospaces import MolmoSpacesEnv

        env = MolmoSpacesEnv(obs_type="pixels_agent_pos")
        obs = env._get_dummy_obs()

        assert "pixels" in obs
        assert "front" in obs["pixels"]
        assert "agent_pos" in obs
        assert obs["agent_pos"].shape == (14,)
    finally:
        _uninstall_molmospaces_stub()


def test_molmospaces_create_envs_returns_correct_structure():
    """create_molmospaces_envs returns {benchmark_name: {task_id: VectorEnv}}."""
    _install_molmospaces_stub()
    try:
        from lerobot.envs.molmospaces import create_molmospaces_envs

        env_cls = MagicMock(return_value=MagicMock())
        result = create_molmospaces_envs(
            task="pick",
            n_envs=2,
            env_cls=env_cls,
            benchmark_name="molmospaces_bench_v1",
        )

        assert "molmospaces_bench_v1" in result
        assert 0 in result["molmospaces_bench_v1"]
    finally:
        _uninstall_molmospaces_stub()


def test_molmospaces_create_envs_raises_on_invalid_env_cls():
    """Raises ValueError when env_cls is not callable."""
    _install_molmospaces_stub()
    try:
        import pytest

        from lerobot.envs.molmospaces import create_molmospaces_envs

        with pytest.raises(ValueError, match="env_cls must be a callable"):
            create_molmospaces_envs(task="pick", n_envs=1, env_cls=None)
    finally:
        _uninstall_molmospaces_stub()


def test_molmospaces_create_envs_raises_on_invalid_n_envs():
    """Raises ValueError when n_envs is not a positive int."""
    _install_molmospaces_stub()
    try:
        import pytest

        from lerobot.envs.molmospaces import create_molmospaces_envs

        with pytest.raises(ValueError, match="n_envs must be a positive int"):
            create_molmospaces_envs(task="pick", n_envs=0, env_cls=MagicMock())
    finally:
        _uninstall_molmospaces_stub()
