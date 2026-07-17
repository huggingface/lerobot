#!/usr/bin/env python

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

import numpy as np

from lerobot.policies.pi052.fit_fast_tokenizer import (
    _apply_relative_actions,
    _dataset_signature,
    _is_global_leader,
    _normalize_actions,
    _select_episode_indices,
)


def test_fast_tokenizer_fit_uses_training_mean_std_normalization():
    actions = np.array([[[1.0, 7.0], [3.0, 3.0]]], dtype=np.float32)
    stats = {"mean": [2.0, 5.0], "std": [0.5, 2.0]}

    normalized = _normalize_actions(actions, "MEAN_STD", stats)

    np.testing.assert_allclose(normalized, [[[-2.0, 1.0], [2.0, -1.0]]])


def test_fast_tokenizer_fit_quantiles_match_training_without_clipping():
    actions = np.array([[[-1.0], [3.0]]], dtype=np.float32)
    stats = {"q01": [0.0], "q99": [2.0]}

    normalized = _normalize_actions(actions, "QUANTILES", stats)

    np.testing.assert_allclose(normalized, [[[-2.0], [2.0]]])


def test_fast_tokenizer_cache_signature_tracks_stats_and_episode_selection():
    kwargs = {
        "dataset_repo_id": "org/dataset",
        "base_tokenizer_name": "physical-intelligence/fast",
        "n_samples": 100,
        "chunk_size": 20,
        "normalization_mode": "QUANTILES",
        "dataset_revision": "main",
        "episodes": [1, 2, 3],
        "exclude_episodes": [2],
        "use_relative_actions": False,
        "relative_action_mask": None,
    }

    first = _dataset_signature(**kwargs, action_stats={"q01": [0.0], "q99": [1.0]})
    changed_stats = _dataset_signature(**kwargs, action_stats={"q01": [0.0], "q99": [2.0]})
    changed_selection = _dataset_signature(
        **{**kwargs, "exclude_episodes": [2, 3]},
        action_stats={"q01": [0.0], "q99": [1.0]},
    )

    assert first != changed_stats
    assert first != changed_selection


def test_fast_tokenizer_uses_only_global_rank_zero(monkeypatch):
    monkeypatch.setenv("RANK", "8")
    monkeypatch.setenv("LOCAL_RANK", "0")
    assert not _is_global_leader()

    monkeypatch.setenv("RANK", "0")
    assert _is_global_leader()


def test_fast_tokenizer_episode_selection_applies_allowlist_and_exclusions():
    selected = _select_episode_indices([0, 1, 2, 3], episodes=[1, 2, 3], exclude_episodes=[2])

    assert selected == [1, 3]


def test_fast_tokenizer_relative_actions_match_training_transform():
    actions = np.array([[[2.0, 10.0], [3.0, 11.0]]], dtype=np.float32)
    states = np.array([[1.0, 4.0]], dtype=np.float32)

    relative = _apply_relative_actions(actions, states, [True, False])

    np.testing.assert_allclose(relative, [[[1.0, 10.0], [2.0, 11.0]]])
