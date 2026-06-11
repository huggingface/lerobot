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
import math
from collections.abc import Iterator

import numpy as np
import torch

logger = logging.getLogger(__name__)

_MASK_64 = (1 << 64) - 1
_FEISTEL_ROUNDS = 4
# Cycle-walking converges in <4 expected steps on the chosen domain; this bound is a generous
# safety net that should never be hit in practice.
_MAX_CYCLE_WALK_STEPS = 100


def _mix64(x: int) -> int:
    """SplitMix64 finalizer (64-bit integer hash)."""
    x = (x + 0x9E3779B97F4A7C15) & _MASK_64
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & _MASK_64
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & _MASK_64
    x ^= x >> 31
    return x


class EpisodeAwareSampler:
    """Sampler over episode frames with O(num_episodes) memory.

    Only episode boundaries are stored; logical positions map to frame indices on the fly, so
    memory does not grow with the number of frames.

    By default (`deterministic=True`) shuffling uses a seeded Feistel permutation over
    `[0, num_frames)`: the data order is a pure function of `(seed, epoch)`, needs no RNG
    synchronization across distributed ranks, and any position can be sought in O(1), enabling
    sample-exact resume via `state_dict` / `load_state_dict`. Each completed `__iter__`
    advances the epoch. The shuffle is pseudo-random rather than truly uniform — the standard
    large-scale trade-off. During a resumed epoch, `__len__` still reports the full length.

    With `deterministic=False`, shuffling falls back to `torch.randperm` driven by `generator`
    (accelerate synchronizes the generator across ranks when preparing the dataloader).
    """

    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
        generator: torch.Generator | None = None,
        deterministic: bool = True,
        seed: int = 0,
    ):
        """
        Args:
            dataset_from_indices: Start index of each episode in the dataset.
            dataset_to_indices: End index of each episode in the dataset.
            episode_indices_to_use: Episode indices to use; None means all.
            drop_n_first_frames: Frames to drop from the start of each episode.
            drop_n_last_frames: Frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
            generator: Generator for non-deterministic shuffling (global torch RNG when None).
            deterministic: Use the seeded Feistel permutation instead of `torch.randperm`.
            seed: Seed the deterministic permutation is derived from (together with the epoch).
        """
        if drop_n_first_frames < 0:
            raise ValueError(f"drop_n_first_frames must be >= 0, got {drop_n_first_frames}")
        if drop_n_last_frames < 0:
            raise ValueError(f"drop_n_last_frames must be >= 0, got {drop_n_last_frames}")
        if deterministic and generator is not None:
            raise ValueError("generator is unused in deterministic mode; pass seed instead.")

        from_indices = np.asarray(dataset_from_indices, dtype=np.int64)
        to_indices = np.asarray(dataset_to_indices, dtype=np.int64)
        if from_indices.shape != to_indices.shape:
            raise ValueError(
                f"dataset_from_indices and dataset_to_indices must have the same length, "
                f"got {len(from_indices)} and {len(to_indices)}"
            )

        used = np.ones(len(from_indices), dtype=bool)
        if episode_indices_to_use is not None:
            used = np.zeros(len(from_indices), dtype=bool)
            used[np.asarray(episode_indices_to_use, dtype=np.int64)] = True

        starts = from_indices + drop_n_first_frames
        lengths = to_indices - drop_n_last_frames - starts
        for episode_idx in np.flatnonzero(used & (lengths <= 0)):
            logger.warning(
                "Episode %d has %d frames but drop_n_first_frames=%d and "
                "drop_n_last_frames=%d removes all frames. Skipping.",
                episode_idx,
                to_indices[episode_idx] - from_indices[episode_idx],
                drop_n_first_frames,
                drop_n_last_frames,
            )
        used &= lengths > 0
        if not used.any():
            raise ValueError(
                "No valid frames remain after applying drop_n_first_frames and drop_n_last_frames. "
                "All episodes were either filtered out or had too few frames."
            )

        self._starts = starts[used]
        self._cum_lengths = np.cumsum(lengths[used])
        self._num_frames = int(self._cum_lengths[-1])
        self.shuffle = shuffle
        self.generator = generator
        self.deterministic = deterministic
        self.seed = seed
        self._epoch = 0
        self._start_index = 0

        # Smallest even-bit-width power-of-two domain >= num_frames: equal Feistel halves,
        # cycle-walking converges in <4 expected steps.
        bits = max((self._num_frames - 1).bit_length(), 2)
        self._half_bits = (bits + 1) // 2
        self._half_mask = (1 << self._half_bits) - 1

    @property
    def indices(self) -> list[int]:
        """Materialized frame indices in unshuffled order; O(num_frames), introspection only."""
        return [self._frame_index(k) for k in range(self._num_frames)]

    def set_epoch(self, epoch: int) -> None:
        self._require_deterministic("set_epoch")
        self._epoch = epoch

    def state_dict(self) -> dict:
        self._require_deterministic("state_dict")
        return {"epoch": self._epoch, "start_index": self._start_index}

    def load_state_dict(self, state: dict) -> None:
        self._require_deterministic("load_state_dict")
        self._epoch = state["epoch"]
        self._start_index = state["start_index"]

    def _require_deterministic(self, method: str) -> None:
        if not self.deterministic:
            raise RuntimeError(f"{method} requires deterministic=True: an RNG order cannot be sought.")

    def _round_keys(self, epoch: int) -> list[int]:
        state = _mix64(_mix64(self.seed) ^ _mix64(epoch))
        keys = []
        for _ in range(_FEISTEL_ROUNDS):
            state = _mix64(state)
            keys.append(state)
        return keys

    def _permute(self, index: int, keys: list[int]) -> int:
        # Feistel network with cycle-walking: a bijection on [0, num_frames).
        half_bits, half_mask = self._half_bits, self._half_mask
        for _ in range(_MAX_CYCLE_WALK_STEPS):
            left, right = index >> half_bits, index & half_mask
            for key in keys:
                left, right = right, left ^ (_mix64(right ^ key) & half_mask)
            index = (left << half_bits) | right
            if index < self._num_frames:
                return index
        raise RuntimeError(
            f"Feistel cycle-walking did not converge within {_MAX_CYCLE_WALK_STEPS} steps; "
            "this should never happen for a valid domain."
        )

    def _frame_index(self, position: int) -> int:
        episode = int(np.searchsorted(self._cum_lengths, position, side="right"))
        position_in_episode = position - (int(self._cum_lengths[episode - 1]) if episode > 0 else 0)
        return int(self._starts[episode]) + position_in_episode

    def __iter__(self) -> Iterator[int]:
        if not self.deterministic:
            return self._iter_default()
        # Advance epoch state eagerly, not on first consumption of the generator.
        epoch, start = self._epoch, self._start_index
        self._epoch += 1
        self._start_index = 0
        return self._iter_deterministic_epoch(epoch, start)

    def _iter_default(self) -> Iterator[int]:
        if self.shuffle:
            for i in torch.randperm(self._num_frames, generator=self.generator):
                yield self._frame_index(int(i))
        else:
            for k in range(self._num_frames):
                yield self._frame_index(k)

    def _iter_deterministic_epoch(self, epoch: int, start: int) -> Iterator[int]:
        keys = self._round_keys(epoch) if self.shuffle else None
        for k in range(start, self._num_frames):
            yield self._frame_index(self._permute(k, keys) if self.shuffle else k)

    def __len__(self) -> int:
        return self._num_frames


def compute_sampler_state(step: int, num_frames: int, batch_size: int, num_processes: int) -> dict:
    """Map an optimization step to an `EpisodeAwareSampler` state for sample-exact resume.

    Under accelerate's batch sharding, one step consumes `batch_size * num_processes` sampler
    positions and each rank sees `ceil(ceil(num_frames / batch_size) / num_processes)` batches
    per epoch (`even_batches` padding included). The start index provably stays below
    `num_frames`; the `min` is defensive.
    """
    batches_per_epoch = math.ceil(math.ceil(num_frames / batch_size) / num_processes)
    epoch, batches_into_epoch = divmod(step, batches_per_epoch)
    start_index = min(batches_into_epoch * batch_size * num_processes, num_frames)
    return {"epoch": epoch, "start_index": start_index}
