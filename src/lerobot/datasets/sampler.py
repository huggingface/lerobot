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
from collections.abc import Iterator

import torch


class EpisodeAwareSampler:
    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
    ):
        """Sampler that optionally incorporates episode boundary information.

        Args:
            dataset_from_indices: List of indices containing the start of each episode in the dataset.
            dataset_to_indices: List of indices containing the end of each episode in the dataset.
            episode_indices_to_use: List of episode indices to use. If None, all episodes are used.
                                    Assumes that episodes are indexed from 0 to N-1.
            drop_n_first_frames: Number of frames to drop from the start of each episode.
            drop_n_last_frames: Number of frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
        """
        indices = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(dataset_from_indices, dataset_to_indices, strict=True)
        ):
            if episode_indices_to_use is None or episode_idx in episode_indices_to_use:
                indices.extend(range(start_index + drop_n_first_frames, end_index - drop_n_last_frames))

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


class WeightedEpisodeAwareSampler:
    """Sampler that draws frames from multiple datasets according to per-dataset weights.

    Each iteration first selects a sub-dataset proportionally to its weight, then
    uniformly samples a frame from that sub-dataset's valid index set.  Episode
    boundary information is respected so that dropped frames are excluded.

    Args:
        dataset_from_indices: Start index for each episode (global, flat).
        dataset_to_indices: End index (exclusive) for each episode (global, flat).
        dataset_membership: Which sub-dataset each episode belongs to (integer id).
        dataset_weights: Relative sampling weight per sub-dataset.
        episode_indices_to_use: If given, only episodes in this set are used.
        drop_n_first_frames: Frames to skip at the start of each episode.
        drop_n_last_frames: Frames to skip at the end of each episode.
        shuffle: Whether to shuffle within each epoch.
        num_samples: How many samples per epoch. Defaults to total valid frames.
        generator: Optional torch.Generator for reproducibility.
    """

    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        dataset_membership: list[int],
        dataset_weights: list[float],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
        num_samples: int | None = None,
        generator: torch.Generator | None = None,
    ):
        n_datasets = max(dataset_membership) + 1 if dataset_membership else 0
        self._per_dataset_indices: list[list[int]] = [[] for _ in range(n_datasets)]

        episodes_to_use = set(episode_indices_to_use) if episode_indices_to_use is not None else None

        for ep_idx, (start, end, ds_id) in enumerate(
            zip(dataset_from_indices, dataset_to_indices, dataset_membership, strict=True)
        ):
            if episodes_to_use is not None and ep_idx not in episodes_to_use:
                continue
            frame_range = range(start + drop_n_first_frames, end - drop_n_last_frames)
            self._per_dataset_indices[ds_id].extend(frame_range)

        # Normalise weights (only over datasets that actually have frames).
        raw_weights = list(dataset_weights[:n_datasets])
        self._weights = torch.zeros(n_datasets)
        for i, w in enumerate(raw_weights):
            if len(self._per_dataset_indices[i]) > 0:
                self._weights[i] = w
        total_w = self._weights.sum()
        if total_w > 0:
            self._weights /= total_w

        self._total_frames = sum(len(idx) for idx in self._per_dataset_indices)
        self._num_samples = num_samples if num_samples is not None else self._total_frames
        self.shuffle = shuffle
        self._generator = generator

    def __iter__(self) -> Iterator[int]:
        if not self.shuffle:
            for ds_indices in self._per_dataset_indices:
                yield from ds_indices
            return

        for _ in range(self._num_samples):
            ds_id = int(torch.multinomial(self._weights, 1, generator=self._generator).item())
            indices = self._per_dataset_indices[ds_id]
            local_idx = int(torch.randint(len(indices), (1,), generator=self._generator).item())
            yield indices[local_idx]

    def __len__(self) -> int:
        return self._num_samples
