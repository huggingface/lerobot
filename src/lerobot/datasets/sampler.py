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
from collections.abc import Iterator, Sequence

import torch

logger = logging.getLogger(__name__)

# Torch dtypes that are safe to convert to Python int without silent truncation.
_INTEGER_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)


def _to_int_set(values: Sequence[int] | torch.Tensor | set[int], name: str) -> set[int]:
    """Convert an integer sequence, set, or 1-D/N-D tensor to a ``set[int]``.

    Raises ``TypeError`` if a tensor with a non-integer dtype is passed so that
    float inputs (e.g. ``torch.tensor([0.9, 2.1])``) fail loudly instead of
    being silently truncated by ``int()``.
    """
    if isinstance(values, torch.Tensor):
        if values.dtype not in _INTEGER_DTYPES:
            raise TypeError(
                f"{name} tensor must have an integer dtype, got {values.dtype}. "
                "Float values would be silently truncated."
            )
        return {int(x) for x in values.flatten().cpu().tolist()}
    return {int(x) for x in values}


class EpisodeAwareSampler:
    def __init__(
        self,
        dataset_from_indices: Sequence[int] | torch.Tensor,
        dataset_to_indices: Sequence[int] | torch.Tensor,
        episode_indices_to_use: Sequence[int] | torch.Tensor | set[int] | None = None,
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

        # Validate tensor dtypes up front so float inputs fail loudly rather than
        # being silently truncated by int() inside the loop.
        for arr, name in (
            (dataset_from_indices, "dataset_from_indices"),
            (dataset_to_indices, "dataset_to_indices"),
        ):
            if isinstance(arr, torch.Tensor) and arr.dtype not in _INTEGER_DTYPES:
                raise TypeError(
                    f"{name} tensor must have an integer dtype, got {arr.dtype}. "
                    "Float values would be silently truncated."
                )

        episode_indices_set: set[int] | None = None
        if episode_indices_to_use is not None:
            episode_indices_set = _to_int_set(episode_indices_to_use, "episode_indices_to_use")

        indices = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(dataset_from_indices, dataset_to_indices, strict=True)
        ):
            # Cast per-element to plain Python int: scalar .item() for tensors,
            # int() for plain sequences. Avoids eagerly materialising the full
            # index arrays into intermediate lists.
            start_index = int(start_index)
            end_index = int(end_index)
            if episode_indices_set is None or episode_idx in episode_indices_set:
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
