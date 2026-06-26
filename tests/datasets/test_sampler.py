#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import logging

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from datasets import Dataset  # noqa: E402

from lerobot.datasets.io_utils import (
    hf_transform_to_torch,
)
from lerobot.datasets.sampler import EpisodeAwareSampler


def calculate_episode_data_index(hf_dataset: Dataset) -> dict[str, torch.Tensor]:
    """Calculate episode data index for testing. Returns {"from": Tensor, "to": Tensor}."""
    episode_data_index: dict[str, list[int]] = {"from": [], "to": []}
    current_episode = None
    if len(hf_dataset) == 0:
        return {"from": torch.tensor([]), "to": torch.tensor([])}
    for idx, episode_idx in enumerate(hf_dataset["episode_index"]):
        if episode_idx != current_episode:
            episode_data_index["from"].append(idx)
            if current_episode is not None:
                episode_data_index["to"].append(idx)
            current_episode = episode_idx
    episode_data_index["to"].append(idx + 1)
    return {k: torch.tensor(v) for k, v in episode_data_index.items()}


def test_drop_n_first_frames():
    dataset = Dataset.from_dict(
        {
            "timestamp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "index": [0, 1, 2, 3, 4, 5],
            "episode_index": [0, 0, 1, 2, 2, 2],
        },
    )
    dataset.set_transform(hf_transform_to_torch)
    episode_data_index = calculate_episode_data_index(dataset)
    sampler = EpisodeAwareSampler(episode_data_index["from"], episode_data_index["to"], drop_n_first_frames=1)
    assert sampler.indices == [1, 4, 5]
    assert len(sampler) == 3
    assert list(sampler) == [1, 4, 5]


def test_drop_n_last_frames():
    dataset = Dataset.from_dict(
        {
            "timestamp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "index": [0, 1, 2, 3, 4, 5],
            "episode_index": [0, 0, 1, 2, 2, 2],
        },
    )
    dataset.set_transform(hf_transform_to_torch)
    episode_data_index = calculate_episode_data_index(dataset)
    sampler = EpisodeAwareSampler(episode_data_index["from"], episode_data_index["to"], drop_n_last_frames=1)
    assert sampler.indices == [0, 3, 4]
    assert len(sampler) == 3
    assert list(sampler) == [0, 3, 4]


def test_episode_indices_to_use():
    dataset = Dataset.from_dict(
        {
            "timestamp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "index": [0, 1, 2, 3, 4, 5],
            "episode_index": [0, 0, 1, 2, 2, 2],
        },
    )
    dataset.set_transform(hf_transform_to_torch)
    episode_data_index = calculate_episode_data_index(dataset)
    sampler = EpisodeAwareSampler(
        episode_data_index["from"], episode_data_index["to"], episode_indices_to_use=[0, 2]
    )
    assert sampler.indices == [0, 1, 3, 4, 5]
    assert len(sampler) == 5
    assert list(sampler) == [0, 1, 3, 4, 5]


def test_shuffle():
    dataset = Dataset.from_dict(
        {
            "timestamp": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "index": [0, 1, 2, 3, 4, 5],
            "episode_index": [0, 0, 1, 2, 2, 2],
        },
    )
    dataset.set_transform(hf_transform_to_torch)
    episode_data_index = calculate_episode_data_index(dataset)
    sampler = EpisodeAwareSampler(episode_data_index["from"], episode_data_index["to"], shuffle=False)
    assert sampler.indices == [0, 1, 2, 3, 4, 5]
    assert len(sampler) == 6
    assert list(sampler) == [0, 1, 2, 3, 4, 5]
    sampler = EpisodeAwareSampler(episode_data_index["from"], episode_data_index["to"], shuffle=True)
    assert sampler.indices == [0, 1, 2, 3, 4, 5]
    assert len(sampler) == 6
    assert set(sampler) == {0, 1, 2, 3, 4, 5}


def test_shuffle_is_reproducible_across_instances():
    # The order is a pure function of (seed, epoch), so two fresh samplers (e.g. two ranks)
    # produce the same permutation without any generator synchronization.
    sampler_a = EpisodeAwareSampler([0], [6], shuffle=True, seed=42)
    sampler_b = EpisodeAwareSampler([0], [6], shuffle=True, seed=42)
    epoch_0 = list(sampler_a)
    assert list(sampler_b) == epoch_0
    # Desyncing the global RNG must not affect the permutation.
    sampler_c = EpisodeAwareSampler([0], [6], shuffle=True, seed=42)
    torch.randperm(1000)  # consume global RNG, as rank-asymmetric code (e.g. eval) would
    assert list(sampler_c) == epoch_0


def test_negative_drop_first_frames_raises():
    with pytest.raises(ValueError, match="drop_n_first_frames must be >= 0"):
        EpisodeAwareSampler([0], [10], drop_n_first_frames=-1)


def test_negative_drop_last_frames_raises():
    with pytest.raises(ValueError, match="drop_n_last_frames must be >= 0"):
        EpisodeAwareSampler([0], [10], drop_n_last_frames=-1)


def test_all_episodes_dropped_raises():
    # All episodes have 1 frame, drop_n_first_frames=1 removes all
    with pytest.raises(ValueError, match="No valid frames remain"):
        EpisodeAwareSampler([0, 1, 2], [1, 2, 3], drop_n_first_frames=1)


def test_partial_episode_drop_warns(caplog):
    # Episode 0: 1 frame (dropped), Episode 1: 5 frames (kept)
    with caplog.at_level(logging.WARNING, logger="lerobot.datasets.sampler"):
        sampler = EpisodeAwareSampler([0, 1], [1, 6], drop_n_first_frames=1)
    # Episode 0 is skipped (1 frame, drop 1), Episode 1 keeps frames 2-5
    assert sampler.indices == [2, 3, 4, 5]
    assert "Episode 0" in caplog.text


# --- seeded (seed, epoch) shuffling, resume, and state ---

from lerobot.datasets.sampler import compute_sampler_state  # noqa: E402

EPISODE_BOUNDS = ([0, 2, 3], [2, 3, 6])  # episodes of 2, 1 and 3 frames


@pytest.mark.parametrize("num_frames", [1, 2, 3, 37, 64, 100])
def test_deterministic_sampler_shuffle_is_permutation(num_frames):
    for seed in (0, 1, 1234):
        sampler = EpisodeAwareSampler([0], [num_frames], shuffle=True, seed=seed)
        assert sorted(sampler) == list(range(num_frames))


def test_deterministic_sampler_epochs_reproduce_and_differ():
    sampler_a = EpisodeAwareSampler([0], [100], shuffle=True, seed=42)
    sampler_b = EpisodeAwareSampler([0], [100], shuffle=True, seed=42)
    epoch_0 = list(sampler_a)
    assert list(sampler_b) == epoch_0  # same (seed, epoch) -> same order on any process
    epoch_1 = list(sampler_a)  # __iter__ auto-advances the epoch
    assert epoch_1 != epoch_0
    assert sorted(epoch_1) == sorted(epoch_0)
    sampler_a.set_epoch(0)
    assert list(sampler_a) == epoch_0
    assert list(EpisodeAwareSampler([0], [100], shuffle=True, seed=7)) != epoch_0


def test_deterministic_sampler_resume_mid_epoch():
    reference = EpisodeAwareSampler(*EPISODE_BOUNDS, shuffle=True, seed=42)
    epoch_0 = list(reference)
    epoch_1 = list(reference)
    for start in (0, 1, 4, len(epoch_0)):
        resumed = EpisodeAwareSampler(*EPISODE_BOUNDS, shuffle=True, seed=42)
        resumed.load_state_dict({"epoch": 0, "start_index": start})
        assert list(resumed) == epoch_0[start:]
        # the resumed sampler continues into the same epoch 1 as the uninterrupted one
        assert list(resumed) == epoch_1


def test_deterministic_sampler_construction_stores_only_boundaries():
    # Construction is O(num_episodes), not O(num_frames): a million-frame single episode
    # instantiates from just its boundaries without materializing a per-frame index list.
    num_frames = 1_000_000
    sampler = EpisodeAwareSampler([0], [num_frames], shuffle=True, seed=0)
    assert len(sampler) == num_frames
    assert sampler._starts.shape == (1,) and sampler._cum_lengths.shape == (1,)


def test_deterministic_sampler_resume_is_exact_at_scale():
    # Seeded randperm makes resume sample-exact at non-trivial sizes: regenerating the epoch's
    # permutation and slicing from the saved offset reproduces the remaining order exactly.
    num_frames = 100_000
    reference = EpisodeAwareSampler([0], [num_frames], shuffle=True, seed=0)
    epoch_0 = list(reference)
    assert sorted(epoch_0) == list(range(num_frames))
    start = num_frames - 5
    resumed = EpisodeAwareSampler([0], [num_frames], shuffle=True, seed=0)
    resumed.load_state_dict({"epoch": 0, "start_index": start})
    assert list(resumed) == epoch_0[start:]


def test_compute_sampler_state():
    # 100 frames, batch 10, 2 ranks -> 10 underlying batches, 5 per rank per epoch.
    assert compute_sampler_state(step=0, num_frames=100, batch_size=10, num_processes=2) == {
        "epoch": 0,
        "start_index": 0,
    }
    # step 7 -> epoch 1, 2 per-rank batches in = 2 * 10 * 2 = 40 samples in
    assert compute_sampler_state(step=7, num_frames=100, batch_size=10, num_processes=2) == {
        "epoch": 1,
        "start_index": 40,
    }
    # uneven epoch: 95 frames -> 10 underlying batches (last short), still 5 per rank
    assert compute_sampler_state(step=12, num_frames=95, batch_size=10, num_processes=2) == {
        "epoch": 2,
        "start_index": 40,
    }
    # uneven sharding: 105 frames -> 11 underlying batches, 6 per rank (even_batches pads)
    assert compute_sampler_state(step=11, num_frames=105, batch_size=10, num_processes=2) == {
        "epoch": 1,
        "start_index": 100,
    }
