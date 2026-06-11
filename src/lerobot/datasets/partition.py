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
import heapq


def partition_episodes(episode_lengths: list[int], num_partitions: int) -> list[list[int]]:
    """Statically partition episodes into near-uniform bins by total frame count.

    Uses the greedy Longest-Processing-Time rule: episodes are processed in descending
    length and each is assigned to the currently least-loaded bin. The result is a pure
    function of the inputs (ties broken by episode index and bin index), so every process
    in a distributed run derives the identical assignment from metadata alone, with no
    coordination.

    Args:
        episode_lengths: Number of frames of each episode. Position i is returned as
            episode index i in the bins.
        num_partitions: Number of bins to partition into.

    Returns:
        One sorted list of episode indices per bin. Every episode appears in exactly
        one bin and every bin receives at least one episode.

    Raises:
        ValueError: If `num_partitions < 1`, fewer episodes than partitions, or any
            episode has a non-positive length.
    """
    if num_partitions < 1:
        raise ValueError(f"num_partitions must be >= 1, got {num_partitions}")
    if len(episode_lengths) < num_partitions:
        raise ValueError(
            f"Cannot partition {len(episode_lengths)} episodes into {num_partitions} bins: "
            "every bin needs at least one episode."
        )
    if any(length <= 0 for length in episode_lengths):
        raise ValueError("All episode lengths must be > 0.")

    # (load, bin_index) heap: ties on load resolve to the lowest bin index.
    heap = [(0, bin_idx) for bin_idx in range(num_partitions)]
    bins: list[list[int]] = [[] for _ in range(num_partitions)]
    by_length_desc = sorted(range(len(episode_lengths)), key=lambda i: (-episode_lengths[i], i))
    for episode_idx in by_length_desc:
        load, bin_idx = heapq.heappop(heap)
        bins[bin_idx].append(episode_idx)
        heapq.heappush(heap, (load + episode_lengths[episode_idx], bin_idx))

    return [sorted(bin_episodes) for bin_episodes in bins]
