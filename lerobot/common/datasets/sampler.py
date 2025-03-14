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
import random
from typing import Iterator, List, Optional, Union

import torch
from torch.utils.data import Sampler


class EpisodeAwareSampler:
    def __init__(
        self,
        episode_data_index: dict,
        episode_indices_to_use: Union[list, None] = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
    ):
        """Sampler that optionally incorporates episode boundary information.

        Args:
            episode_data_index: Dictionary with keys 'from' and 'to' containing the start and end indices of each episode.
            episode_indices_to_use: List of episode indices to use. If None, all episodes are used.
                                    Assumes that episodes are indexed from 0 to N-1.
            drop_n_first_frames: Number of frames to drop from the start of each episode.
            drop_n_last_frames: Number of frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
        """
        indices = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(episode_data_index["from"], episode_data_index["to"], strict=True)
        ):
            if episode_indices_to_use is None or episode_idx in episode_indices_to_use:
                indices.extend(
                    range(start_index.item() + drop_n_first_frames, end_index.item() - drop_n_last_frames)
                )

        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            for i in torch.randperm(len(self.indices)):
                yield self.indices[i]
        else:
            for i in self.indices:
                yield i

    def __len__(self) -> int:
        return len(self.indices)


class SumTree:
    """
    A classic sum-tree data structure for storing priorities.
    Each leaf stores a sample's priority, and internal nodes store sums of children.
    """

    def __init__(self, capacity: int):
        """
        Args:
            capacity: Maximum number of elements.
        """
        self.capacity = capacity
        self.size = capacity
        self.tree = [0.0] * (2 * self.size)

    def initialize_tree(self, priorities: List[float]):
        """
        Initializes the sum tree
        """
        # Set leaf values
        for i, priority in enumerate(priorities):
            self.tree[i + self.size] = priority

        # Compute internal node values
        for i in range(self.size - 1, 0, -1):
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]

    def update(self, idx: int, priority: float):
        """
        Update the priority at leaf index `idx` and propagate changes upwards.
        """
        tree_idx = idx + self.size
        self.tree[tree_idx] = priority  # Set new priority

        # Propagate up, explicitly summing children
        tree_idx //= 2
        while tree_idx >= 1:
            self.tree[tree_idx] = self.tree[2 * tree_idx] + self.tree[2 * tree_idx + 1]
            tree_idx //= 2

    def total_priority(self) -> float:
        """Returns the sum of all priorities (stored at root)."""
        return self.tree[1]

    def sample(self, value: float) -> int:
        """
        Samples an index where the prefix sum up to that leaf is >= `value`.
        """
        value = min(max(value, 0), self.total_priority())  # Clamp value
        idx = 1
        while idx < self.size:
            left = 2 * idx
            if self.tree[left] >= value:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1
        return idx - self.size  # Convert tree index to data index


class PrioritizedSampler(Sampler[int]):
    """
    PyTorch Sampler that draws samples in proportion to their priority using a SumTree.
    """

    def __init__(
        self,
        data_len: int,
        alpha: float = 0.6,
        eps: float = 1e-6,
        num_samples_per_epoch: Optional[int] = None,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        total_steps: int = 1,
    ):
        """
        Args:
            data_len: Total number of samples in the dataset.
            alpha: Exponent for priority scaling. Default is 0.6.
            eps: Small constant to avoid zero priorities.
            replacement: Whether to sample with replacement.
            num_samples_per_epoch: Number of samples per epoch (default is data_len).
        """
        self.data_len = data_len
        self.alpha = alpha
        self.eps = eps
        self.num_samples_per_epoch = num_samples_per_epoch or data_len
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.total_steps = total_steps
        self._beta = self.beta_start

        # Initialize difficulties and sum-tree
        self.difficulties = [1.0] * data_len
        self.priorities = [0.0] * data_len
        initial_priorities = [(1.0 + eps) ** alpha] * data_len

        self.sumtree = SumTree(data_len)
        self.sumtree.initialize_tree(initial_priorities)
        for i, p in enumerate(initial_priorities):
            self.priorities[i] = p

    def update_beta(self, current_step: int):
        frac = min(1.0, current_step / self.total_steps)
        self._beta = self.beta_start + (self.beta_end - self.beta_start) * frac

    def update_priorities(self, indices: List[int], difficulties: List[float]):
        """
        Updates the priorities in the sum-tree.
        """
        for idx, diff in zip(indices, difficulties, strict=False):
            self.difficulties[idx] = diff
            new_priority = (diff + self.eps) ** self.alpha
            self.priorities[idx] = new_priority
            self.sumtree.update(idx, new_priority)

    def __iter__(self) -> Iterator[int]:
        """
        Samples indices based on their priority weights.
        """
        total_p = self.sumtree.total_priority()

        for _ in range(self.num_samples_per_epoch):
            r = random.random() * total_p
            idx = self.sumtree.sample(r)

            yield idx

    def __len__(self) -> int:
        return self.num_samples_per_epoch

    def compute_is_weights(self, indices: List[int]) -> torch.Tensor:
        w = []
        total_p = self.sumtree.total_priority()
        for idx in indices:
            p = self.priorities[idx] / total_p
            w.append((p * self.data_len) ** (-self._beta))
        w = torch.tensor(w, dtype=torch.float32)
        return w / w.max()
