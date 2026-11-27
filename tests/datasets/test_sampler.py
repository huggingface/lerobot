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
import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.configs.sampling import SampleKey  # noqa: E402
from lerobot.datasets.sampler import EpisodeAwareSampler, compute_sampler_state  # noqa: E402

pytestmark = pytest.mark.dataset

COUNTS = {0: 2, 2: 1, 3: 0, 5: 3}  # samples per episode; episode 3 gives none


def test_keys_cover_every_sample_of_every_episode_once():
    ordered = list(EpisodeAwareSampler(COUNTS))
    assert ordered == [
        SampleKey(0, 0, 0),
        SampleKey(0, 1, 0),
        SampleKey(2, 0, 0),
        *(SampleKey(5, k, 0) for k in range(3)),
    ]
    sampler = EpisodeAwareSampler(COUNTS, shuffle=True, seed=42)
    assert len(sampler) == 6
    epoch_0 = list(sampler)
    assert sorted(epoch_0) == ordered
    # The order is a pure function of (seed, epoch): a fresh sampler on another rank gives the same
    # one, whatever the global RNG did meanwhile; the next epoch differs and carries its number.
    torch.randperm(1000)
    assert list(EpisodeAwareSampler(COUNTS, shuffle=True, seed=42)) == epoch_0
    epoch_1 = list(sampler)
    assert [key[:2] for key in epoch_1] != [key[:2] for key in epoch_0] and {
        key.epoch for key in epoch_1
    } == {1}
    sampler.set_epoch(0)
    assert list(sampler) == epoch_0
    with pytest.raises(ValueError, match="No episode gives any sample"):
        EpisodeAwareSampler({0: 0})


def test_resume_mid_epoch_continues_the_same_order():
    reference = EpisodeAwareSampler(COUNTS, shuffle=True, seed=42)
    epoch_0, epoch_1 = list(reference), list(reference)
    for start in (0, 1, 4, len(epoch_0)):
        resumed = EpisodeAwareSampler(COUNTS, shuffle=True, seed=42)
        resumed.load_state_dict({"epoch": 0, "start_index": start})
        assert list(resumed) == epoch_0[start:]
        assert list(resumed) == epoch_1


def test_resume_is_exact_at_scale_and_stores_only_counts():
    num_samples = 100_000
    reference = EpisodeAwareSampler({0: num_samples}, shuffle=True, seed=0)
    assert reference._ends.shape == (1,)  # O(num_episodes), not O(num_samples)
    epoch_0 = list(reference)
    assert sorted(key.sample for key in epoch_0) == list(range(num_samples))
    resumed = EpisodeAwareSampler({0: num_samples}, shuffle=True, seed=0)
    resumed.load_state_dict({"epoch": 0, "start_index": num_samples - 5})
    assert list(resumed) == epoch_0[-5:]


def test_compute_sampler_state():
    # 100 samples, batch 10, 2 ranks -> 10 underlying batches, 5 per rank per epoch.
    assert compute_sampler_state(step=0, num_samples=100, batch_size=10, num_processes=2) == {
        "epoch": 0,
        "start_index": 0,
    }
    # step 7 -> epoch 1, 2 per-rank batches in = 2 * 10 * 2 = 40 samples in
    assert compute_sampler_state(step=7, num_samples=100, batch_size=10, num_processes=2) == {
        "epoch": 1,
        "start_index": 40,
    }
    # uneven epoch: 95 samples -> 10 underlying batches (last short), still 5 per rank
    assert compute_sampler_state(step=12, num_samples=95, batch_size=10, num_processes=2) == {
        "epoch": 2,
        "start_index": 40,
    }
    # uneven sharding: 105 samples -> 11 underlying batches, 6 per rank (even_batches pads)
    assert compute_sampler_state(step=11, num_samples=105, batch_size=10, num_processes=2) == {
        "epoch": 1,
        "start_index": 100,
    }
