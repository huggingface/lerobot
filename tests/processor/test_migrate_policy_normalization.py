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
Tests for the extraction of normalization statistics out of legacy checkpoints.
"""

import pytest
import torch

from lerobot.processor.migrate_policy_normalization import (
    extract_normalization_stats,
    split_dataset_prefix,
)
from lerobot.utils.constants import ACTION, OBS_STATE


def test_split_dataset_prefix_plain_buffer():
    assert split_dataset_prefix("buffer_observation_state") == (None, "observation_state")


def test_split_dataset_prefix_dataset_scoped_buffer():
    assert split_dataset_prefix("so100_buffer_action") == ("so100", "action")
    assert split_dataset_prefix("so100-blue_buffer_observation_state") == (
        "so100-blue",
        "observation_state",
    )


def test_split_dataset_prefix_without_buffer_marker():
    assert split_dataset_prefix("observation_state") == (None, "observation_state")


def test_extract_normalization_stats_ignores_unrelated_keys():
    state_dict = {
        "normalize_inputs.buffer_observation_state.mean": torch.zeros(6),
        "normalize_inputs.buffer_observation_state.std": torch.ones(6),
        "unnormalize_outputs.buffer_action.mean": torch.zeros(6),
        "unnormalize_outputs.buffer_action.std": torch.ones(6),
        "model.layers.0.weight": torch.ones(2, 2),
    }

    stats = extract_normalization_stats(state_dict)

    assert set(stats) == {OBS_STATE, ACTION}
    assert set(stats[ACTION]) == {"mean", "std"}


def test_extract_normalization_stats_matches_plain_input_prefix():
    """`normalize_inputs.` used to be misspelled, which dropped every observation stat."""
    state_dict = {
        "normalize_inputs.so100_buffer_observation_state.mean": torch.zeros(6),
        "normalize_inputs.so100_buffer_observation_state.std": torch.ones(6),
    }

    stats = extract_normalization_stats(state_dict)

    assert OBS_STATE in stats


def test_extract_normalization_stats_strips_dataset_prefix():
    """A dataset-scoped buffer yields `action`, not `so100.buffer.action`."""
    state_dict = {
        "normalize_inputs.so100_buffer_observation_state.mean": torch.zeros(6),
        "normalize_inputs.so100_buffer_observation_state.std": torch.ones(6),
        "normalize_targets.so100_buffer_action.mean": torch.zeros(6),
        "unnormalize_outputs.so100_buffer_action.std": torch.ones(6),
    }

    stats = extract_normalization_stats(state_dict)

    assert set(stats) == {OBS_STATE, ACTION}


def test_extract_normalization_stats_rejects_several_datasets():
    """Keeping one dataset at random would silently normalize with the wrong values."""
    state_dict = {}
    for dataset in ("so100", "so100-red", "so100-blue"):
        state_dict[f"normalize_targets.{dataset}_buffer_action.mean"] = torch.zeros(6)
        state_dict[f"normalize_targets.{dataset}_buffer_action.std"] = torch.ones(6)

    with pytest.raises(ValueError, match="3 datasets"):
        extract_normalization_stats(state_dict)
