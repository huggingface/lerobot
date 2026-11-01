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

"""Unit tests for the env/policy visual resolution mismatch warning in lerobot_eval."""

import logging
from types import SimpleNamespace

import pytest

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.envs.configs import LiberoEnv
from lerobot.scripts.lerobot_eval import _warn_on_visual_resolution_mismatch


def _policy_config(input_features):
    return SimpleNamespace(input_features=input_features)


def test_warns_when_env_resolution_differs_from_policy(caplog):
    # LiberoEnv defaults to 360x360 rendering; a typical SmolVLA LIBERO checkpoint expects 3x256x256.
    env_config = LiberoEnv()
    policy_config = _policy_config(
        {
            "observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        }
    )
    with caplog.at_level(logging.WARNING):
        _warn_on_visual_resolution_mismatch(policy_config, env_config)
    assert any("do not match" in r.message for r in caplog.records)
    assert any("observation_height" in r.message for r in caplog.records)


def test_no_warning_when_resolutions_match(caplog):
    env_config = LiberoEnv(observation_height=256, observation_width=256)
    policy_config = _policy_config(
        {"observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))}
    )
    with caplog.at_level(logging.WARNING):
        _warn_on_visual_resolution_mismatch(policy_config, env_config)
    assert not any("do not match" in r.message for r in caplog.records)


def test_no_warning_when_features_missing(caplog):
    with caplog.at_level(logging.WARNING):
        _warn_on_visual_resolution_mismatch(
            SimpleNamespace(input_features=None), SimpleNamespace(features=None)
        )
    assert caplog.records == []


def test_matching_resolution_via_env_hwc_vs_policy_chw(caplog):
    # Env features are (H, W, C) while policy input features are (C, H, W): same resolution must not warn.
    env_config = SimpleNamespace(
        features={"pixels": PolicyFeature(type=FeatureType.VISUAL, shape=(128, 192, 3))}
    )
    policy_config = _policy_config(
        {"observation.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 192))}
    )
    with caplog.at_level(logging.WARNING):
        _warn_on_visual_resolution_mismatch(policy_config, env_config)
    assert not any("do not match" in r.message for r in caplog.records)


if __name__ == "__main__":
    pytest.main([__file__])
