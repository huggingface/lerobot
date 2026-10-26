# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import pickle

import numpy as np
import pytest

from lerobot.datasets.episode_subset_index import _EpisodeSubsetIndexMapping


@pytest.mark.parametrize("selected", [[0, 2], [1], [0, 1, 2], []])
def test_episode_subset_index_matches_dense_mapping(selected: list[int]) -> None:
    episode_starts = np.array([0, 5, 10])
    episode_ends = np.array([5, 10, 14])
    mapping = _EpisodeSubsetIndexMapping(episode_starts[selected], episode_ends[selected])
    absolute_indices = [
        absolute_idx
        for episode in selected
        for absolute_idx in range(episode_starts[episode], episode_ends[episode])
    ]
    dense = dict(zip(absolute_indices, range(len(absolute_indices)), strict=True))

    assert len(mapping) == len(dense)
    assert [mapping.rel_to_abs(relative_idx) for relative_idx in range(len(mapping))] == absolute_indices
    assert {absolute_idx: mapping[absolute_idx] for absolute_idx in mapping} == dense
    assert mapping == dense and dense == mapping
    assert mapping != {0: -1}
    for absolute_idx in range(-1, 16):
        assert (absolute_idx in mapping) == (absolute_idx in dense)
        if absolute_idx not in dense:
            with pytest.raises(KeyError):
                mapping[absolute_idx]
    with pytest.raises(TypeError):
        hash(mapping)


def test_episode_subset_index_survives_pickle() -> None:
    mapping = _EpisodeSubsetIndexMapping(np.array([0, 10]), np.array([5, 14]))
    restored = pickle.loads(pickle.dumps(mapping))
    assert len(restored) == len(mapping)
    assert dict(restored) == dict(mapping)
    assert [restored.rel_to_abs(relative_idx) for relative_idx in range(len(restored))] == [
        mapping.rel_to_abs(relative_idx) for relative_idx in range(len(mapping))
    ]
