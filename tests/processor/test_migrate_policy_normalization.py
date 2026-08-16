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

"""
Tests for normalization statistic extraction during policy migration.
"""

import torch

from lerobot.processor.migrate_policy_normalization import extract_normalization_stats


def _buffer(feature_name: str, prefix: str = "normalize_inputs.buffer_") -> dict[str, torch.Tensor]:
    """A checkpoint's stat buffers for one feature, keyed the way the old modules stored them."""
    flattened = feature_name.replace(".", "_")
    return {
        f"{prefix}{flattened}.mean": torch.tensor([1.0]),
        f"{prefix}{flattened}.std": torch.tensor([2.0]),
    }


def test_declared_name_with_an_underscore_survives():
    """Guards #4451: `observation.environment_state` must not migrate to
    `observation.environment.state`, which the runtime never looks up."""
    state_dict = _buffer("observation.environment_state")

    stats = extract_normalization_stats(state_dict, feature_names=["observation.environment_state"])

    assert set(stats) == {"observation.environment_state"}
    assert set(stats["observation.environment_state"]) == {"mean", "std"}


def test_declared_names_disambiguate_a_shared_flattened_form():
    """`observation.environment_state` and `observation.environment.state` flatten identically,
    so only the declared name decides which one a buffer meant."""
    state_dict = _buffer("observation.environment.state")

    stats = extract_normalization_stats(state_dict, feature_names=["observation.environment.state"])

    assert set(stats) == {"observation.environment.state"}


def test_undeclared_name_keeps_the_historical_reading():
    """A config too old to declare features leaves the buffer key as the only source."""
    state_dict = _buffer("observation.environment_state")

    stats = extract_normalization_stats(state_dict)

    assert set(stats) == {"observation.environment.state"}


def test_declared_names_do_not_disturb_names_without_underscores():
    state_dict = {**_buffer("observation.state"), **_buffer("action")}

    stats = extract_normalization_stats(state_dict, feature_names=["observation.state", "action"])

    assert set(stats) == {"observation.state", "action"}


def test_dot_preserving_prefix_keeps_a_declared_underscore():
    """`normalize.` and friends store the name with its dots intact, so the lookup has to match
    the stored form as well as the flattened one."""
    state_dict = _buffer("observation.environment_state", prefix="normalize.")
    state_dict = {
        key.replace("observation_environment_state", "observation.environment_state"): value
        for key, value in state_dict.items()
    }

    stats = extract_normalization_stats(state_dict, feature_names=["observation.environment_state"])

    assert set(stats) == {"observation.environment_state"}


def test_a_feature_the_config_does_not_declare_still_migrates():
    """An undeclared buffer is not dropped; it falls back rather than disappearing."""
    state_dict = {**_buffer("observation.environment_state"), **_buffer("observation.state")}

    stats = extract_normalization_stats(state_dict, feature_names=["observation.environment_state"])

    assert set(stats) == {"observation.environment_state", "observation.state"}
