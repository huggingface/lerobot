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
"""Unit tests for RoboCerebraEnv — no LIBERO install required (all env creation is mocked)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from lerobot.configs.types import FeatureType
from lerobot.envs.configs import RoboCerebraEnv
from lerobot.envs.factory import make_env_config
from lerobot.utils.constants import ACTION, OBS_IMAGES

# ---------------------------------------------------------------------------
# Test 1: default config values
# ---------------------------------------------------------------------------


def test_default_config():
    cfg = RoboCerebraEnv()
    assert cfg.task == "libero_10"
    assert cfg.fps == 20
    assert cfg.observation_height == 256
    assert cfg.observation_width == 256
    assert cfg.obs_type == "pixels_agent_pos"
    assert cfg.control_mode == "relative"
    assert cfg.init_states is True
    assert cfg.task_ids is None


# ---------------------------------------------------------------------------
# Test 2: camera_name_mapping defaults match dataset keys
# ---------------------------------------------------------------------------


def test_camera_name_mapping_defaults():
    cfg = RoboCerebraEnv()
    assert cfg.camera_name_mapping == {
        "agentview_image": "image",
        "robot0_eye_in_hand_image": "wrist_image",
    }


# ---------------------------------------------------------------------------
# Test 3: features include both camera keys after __post_init__
# ---------------------------------------------------------------------------


def test_features_have_camera_keys():
    cfg = RoboCerebraEnv()
    assert "pixels/image" in cfg.features
    assert "pixels/wrist_image" in cfg.features
    assert cfg.features["pixels/image"].type == FeatureType.VISUAL
    assert cfg.features["pixels/wrist_image"].type == FeatureType.VISUAL
    assert cfg.features["pixels/image"].shape == (256, 256, 3)
    assert cfg.features["pixels/wrist_image"].shape == (256, 256, 3)


# ---------------------------------------------------------------------------
# Test 4: features_map maps camera keys to observation.images.*
# ---------------------------------------------------------------------------


def test_features_map_camera_keys():
    cfg = RoboCerebraEnv()
    assert cfg.features_map["pixels/image"] == f"{OBS_IMAGES}.image"
    assert cfg.features_map["pixels/wrist_image"] == f"{OBS_IMAGES}.wrist_image"
    assert cfg.features_map[ACTION] == ACTION


# ---------------------------------------------------------------------------
# Test 5: task_ids filtering propagates through gym_kwargs
# ---------------------------------------------------------------------------


def test_task_ids_in_gym_kwargs():
    cfg = RoboCerebraEnv(task_ids=[0, 2])
    assert cfg.gym_kwargs["task_ids"] == [0, 2]

    cfg_no_ids = RoboCerebraEnv()
    assert "task_ids" not in cfg_no_ids.gym_kwargs


# ---------------------------------------------------------------------------
# Test 6: env type is registered in EnvConfig registry
# ---------------------------------------------------------------------------


def test_env_type_registered():
    cfg = make_env_config("robocerebra")
    assert isinstance(cfg, RoboCerebraEnv)
    assert cfg.type == "robocerebra"


# ---------------------------------------------------------------------------
# Helpers for tests that call create_envs (libero not installed on macOS)
# ---------------------------------------------------------------------------


def _mock_libero_module():
    """Return a MagicMock that stands in for lerobot.envs.libero."""
    mock_mod = MagicMock()
    mock_mod.create_libero_envs.return_value = {"libero_10": {0: MagicMock()}}
    return mock_mod


def _libero_sys_patches(mock_mod: MagicMock) -> dict:
    """sys.modules patches needed to satisfy the lazy import inside create_envs."""
    return {
        "lerobot.envs.libero": mock_mod,
        "libero": MagicMock(),
        "libero.libero": MagicMock(),
        "libero.libero.benchmark": MagicMock(),
        "libero.libero.envs": MagicMock(),
    }


# ---------------------------------------------------------------------------
# Test 7: create_envs forwards camera_name_mapping to create_libero_envs
# ---------------------------------------------------------------------------


def test_create_envs_forwards_camera_name_mapping():
    cfg = RoboCerebraEnv()
    mock_mod = _mock_libero_module()

    with patch.dict(sys.modules, _libero_sys_patches(mock_mod)):
        cfg.create_envs(n_envs=1, use_async_envs=False)

    mock_mod.create_libero_envs.assert_called_once()
    call_kwargs = mock_mod.create_libero_envs.call_args.kwargs
    assert call_kwargs["camera_name_mapping"] == {
        "agentview_image": "image",
        "robot0_eye_in_hand_image": "wrist_image",
    }
    assert call_kwargs["task"] == "libero_10"


# ---------------------------------------------------------------------------
# Test 8: custom camera_name_mapping is forwarded correctly
# ---------------------------------------------------------------------------


def test_custom_camera_name_mapping_forwarded():
    custom_mapping = {"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}
    cfg = RoboCerebraEnv(camera_name_mapping=custom_mapping)
    mock_mod = _mock_libero_module()

    with patch.dict(sys.modules, _libero_sys_patches(mock_mod)):
        cfg.create_envs(n_envs=1, use_async_envs=False)

    call_kwargs = mock_mod.create_libero_envs.call_args.kwargs
    assert call_kwargs["camera_name_mapping"] == custom_mapping


# ---------------------------------------------------------------------------
# Test 9: unsupported obs_type raises ValueError
# ---------------------------------------------------------------------------


def test_unsupported_obs_type_raises():
    with pytest.raises(ValueError, match="unsupported obs_type"):
        RoboCerebraEnv(obs_type="state")


# ---------------------------------------------------------------------------
# Test 10: pixels-only obs_type omits state features
# ---------------------------------------------------------------------------


def test_pixels_only_obs_type_has_no_state_features():
    cfg = RoboCerebraEnv(obs_type="pixels")
    state_keys = [k for k, v in cfg.features.items() if v.type == FeatureType.STATE]
    assert len(state_keys) == 0, f"Expected no state features for obs_type='pixels', got: {state_keys}"
