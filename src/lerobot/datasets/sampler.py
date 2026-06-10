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
from collections.abc import Iterator

import torch

logger = logging.getLogger(__name__)


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
        if drop_n_first_frames < 0:
            raise ValueError(f"drop_n_first_frames must be >= 0, got {drop_n_first_frames}")
        if drop_n_last_frames < 0:
            raise ValueError(f"drop_n_last_frames must be >= 0, got {drop_n_last_frames}")

        indices = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(dataset_from_indices, dataset_to_indices, strict=True)
        ):
            if episode_indices_to_use is None or episode_idx in episode_indices_to_use:
                ep_length = end_index - start_index
                if drop_n_first_frames + drop_n_last_frames >= ep_length:
                    logger.warning(
                        "Episode %d has %d frames but drop_n_first_frames=%d and "
                        "drop_n_last_frames=%d removes all frames. Skipping.",
                        episode_idx,
                        ep_length,
                        drop_n_first_frames,
                        drop_n_last_frames,
                    )
                    continue
                indices.extend(range(start_index + drop_n_first_frames, end_index - drop_n_last_frames))

        if not indices:
            raise ValueError(
                "No valid frames remain after applying drop_n_first_frames and drop_n_last_frames. "
                "All episodes were either filtered out or had too few frames."
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


class WeightedEpisodeAwareSampler(EpisodeAwareSampler):
    """``EpisodeAwareSampler`` that draws frames *with replacement* in
    proportion to per-frame weights.

    Used to oversample frames carrying a sparse annotation (e.g. a VQA
    question) so the policy sees them more often than their natural
    dataset density. One epoch still yields ``len(self.indices)``
    samples — the weights only change the *composition* of the stream,
    not its length. Each epoch re-draws, so the oversampled subset
    varies run to run.
    """

    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        frame_weights,
        *,
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
    ):
        """
        Args:
            dataset_from_indices: Episode start indices (see ``EpisodeAwareSampler``).
            dataset_to_indices: Episode end indices.
            frame_weights: 1-D sequence/tensor of non-negative weights, one per
                dataset frame (length == total dataset frames). Higher weight ⇒
                that frame is sampled more often.
            episode_indices_to_use / drop_n_first_frames / drop_n_last_frames:
                Same meaning as ``EpisodeAwareSampler`` — the episode-boundary
                frame filtering is applied first, then weighting is restricted
                to the surviving frames.
        """
        super().__init__(
            dataset_from_indices,
            dataset_to_indices,
            episode_indices_to_use=episode_indices_to_use,
            drop_n_first_frames=drop_n_first_frames,
            drop_n_last_frames=drop_n_last_frames,
            shuffle=False,
        )
        weights = torch.as_tensor(frame_weights, dtype=torch.double).flatten()
        idx = torch.tensor(self.indices, dtype=torch.long)
        if weights.numel() <= int(idx.max()):
            raise ValueError(
                f"frame_weights has {weights.numel()} entries but the sampler "
                f"references frame index {int(idx.max())}."
            )
        selected = weights[idx]
        if not torch.isfinite(selected).all() or bool((selected < 0).any()):
            raise ValueError("frame_weights must be finite and non-negative.")
        if float(selected.sum()) <= 0.0:
            # All surviving frames have zero weight — fall back to uniform.
            selected = torch.ones_like(selected)
        self._weights = selected

    def __iter__(self) -> Iterator[int]:
        picks = torch.multinomial(self._weights, num_samples=len(self.indices), replacement=True)
        for i in picks.tolist():
            yield self.indices[i]
