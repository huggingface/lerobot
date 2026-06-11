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

from lerobot.datasets.partition import partition_episodes


def test_partition_covers_every_episode_exactly_once():
    lengths = [50, 30, 20, 40, 10, 60, 25, 35]
    bins = partition_episodes(lengths, num_partitions=3)
    assert len(bins) == 3
    assigned = sorted(idx for bin_episodes in bins for idx in bin_episodes)
    assert assigned == list(range(len(lengths)))
    assert all(len(bin_episodes) > 0 for bin_episodes in bins)


def test_partition_is_load_balanced():
    lengths = [50, 30, 20, 40, 10, 60, 25, 35]
    bins = partition_episodes(lengths, num_partitions=3)
    loads = [sum(lengths[i] for i in bin_episodes) for bin_episodes in bins]
    # LPT guarantees max load <= (4/3 - 1/(3m)) * optimum; for this input the bound means
    # the spread can never exceed the largest episode.
    assert max(loads) - min(loads) <= max(lengths)


def test_partition_is_deterministic():
    lengths = [7, 7, 7, 3, 3, 9, 1, 5]
    assert partition_episodes(lengths, 3) == partition_episodes(lengths, 3)


def test_partition_single_bin_returns_all():
    assert partition_episodes([5, 1, 3], 1) == [[0, 1, 2]]


def test_partition_validation():
    with pytest.raises(ValueError, match="num_partitions must be >= 1"):
        partition_episodes([1, 2, 3], 0)
    with pytest.raises(ValueError, match="every bin needs at least one episode"):
        partition_episodes([1, 2], 3)
    with pytest.raises(ValueError, match="must be > 0"):
        partition_episodes([1, 0, 2], 2)


def test_partition_skewed_lengths():
    # One giant episode should sit alone in its bin while the small ones share the other.
    lengths = [1000, 1, 1, 1, 1]
    bins = partition_episodes(lengths, 2)
    assert [0] in bins
    other = bins[0] if bins[1] == [0] else bins[1]
    assert sorted(other) == [1, 2, 3, 4]
