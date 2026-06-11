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


def _mix64(x: int) -> int:
    """SplitMix64 finalizer: a high-quality, cheap 64-bit integer hash."""
    x = (x + 0x9E3779B97F4A7C15) & _MASK_64
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & _MASK_64
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & _MASK_64
    x ^= x >> 31
    return x


class EpisodeAwareSampler:
    """Sampler that incorporates episode boundary information.

    Frame indices are never materialized: only per-episode boundaries are stored (a few numpy
    int64 per episode) and the mapping from a logical position to a frame index is computed on
    the fly via `searchsorted`, so memory does not grow with the number of frames.

    Two shuffling modes are supported:

    - Default (`deterministic=False`): `torch.randperm` over positions, optionally driven by a
      dedicated `generator`. Exposing the `generator` attribute (even when None) lets
      `accelerate` register it as the synchronized RNG in distributed training, so every rank
      draws the same permutation and batch shards stay disjoint.
    - `deterministic=True`: a seeded Feistel permutation over `[0, num_frames)` (cycle-walking
      to the exact domain size). The data order becomes a pure function of `(seed, epoch)`:
      nothing to synchronize across ranks, O(1) seek to any position (enabling sample-exact
      resume via `state_dict` / `load_state_dict`), and zero epoch-boundary cost at any dataset
      size. The shuffle is pseudo-random rather than a true uniform permutation — the standard
      trade-off in large-scale training loaders. Each completed `__iter__` advances the internal
      epoch, so consecutive dataloader passes yield different permutations without `set_epoch`
      calls. During an epoch resumed via `load_state_dict`, `__len__` still reports the full
      epoch length; the first pass simply yields fewer samples.

    In deterministic mode, `shard_rank`/`shard_world_size` make the sampler itself shard the
    (shared) permutation: shard r yields positions r, r + world, r + 2 * world, ... — disjoint
    across shards by construction, with no accelerate-level batch sharding (and therefore no
    `even_batches` padding) needed. `__len__`, `start_index` and `compute_sampler_state` (with
    `num_processes=1`) then all live in shard-local positions.
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
        deterministic: bool = False,
        seed: int = 0,
        shard_rank: int = 0,
        shard_world_size: int = 1,
    ):
        """
        Args:
            dataset_from_indices: List of indices containing the start of each episode in the dataset.
            dataset_to_indices: List of indices containing the end of each episode in the dataset.
            episode_indices_to_use: List of episode indices to use. If None, all episodes are used.
                                    Assumes that episodes are indexed from 0 to N-1.
            drop_n_first_frames: Number of frames to drop from the start of each episode.
            drop_n_last_frames: Number of frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
            generator: Generator used for default-mode shuffling. When None, shuffling falls
                       back to the global torch RNG. Incompatible with `deterministic=True`.
            deterministic: Use the seeded Feistel permutation instead of `torch.randperm`.
            seed: Seed the deterministic permutation is derived from (together with the epoch).
                  Must be identical on all shards of the same data for shards to stay disjoint.
            shard_rank: This sampler's shard of the permutation (e.g. the local process index).
                        Requires `deterministic=True` when sharding.
            shard_world_size: Total number of shards (e.g. processes sharing this dataset).
        """
        if drop_n_first_frames < 0:
            raise ValueError(f"drop_n_first_frames must be >= 0, got {drop_n_first_frames}")
        if drop_n_last_frames < 0:
            raise ValueError(f"drop_n_last_frames must be >= 0, got {drop_n_last_frames}")
        if deterministic and generator is not None:
            raise ValueError("generator is unused in deterministic mode; pass seed instead.")
        if not 0 <= shard_rank < shard_world_size:
            raise ValueError(
                f"shard_rank must be in [0, shard_world_size), got {shard_rank=} {shard_world_size=}"
            )
        if shard_world_size > 1 and not deterministic:
            raise ValueError(
                "Sharding requires deterministic=True: without a shared deterministic "
                "permutation, shards cannot be guaranteed disjoint."
            )

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
        if self._num_frames < shard_world_size:
            raise ValueError(
                f"Cannot shard {self._num_frames} frames across {shard_world_size} processes: "
                "every shard needs at least one frame."
            )
        self._shard_rank = shard_rank
        self._shard_world_size = shard_world_size
        # Positions shard_rank, shard_rank + world, ... that are < num_frames.
        self._shard_len = (self._num_frames - shard_rank + shard_world_size - 1) // shard_world_size
        self.shuffle = shuffle
        self.generator = generator
        self.deterministic = deterministic
        self.seed = seed
        self._epoch = 0
        self._start_index = 0

        # Feistel cipher domain: the smallest even-bit-width power of two >= num_frames,
        # so both halves have equal width and cycle-walking converges in <4 expected steps.
        bits = max((self._num_frames - 1).bit_length(), 2)
        self._half_bits = (bits + 1) // 2
        self._half_mask = (1 << self._half_bits) - 1

    @property
    def indices(self) -> list[int]:
        """Materialized frame indices, in unshuffled order (back-compat / introspection only).

        This builds an O(num_frames) list — avoid on very large datasets; iteration never uses it.
        """
        return [self._frame_index(k) for k in range(self._num_frames)]

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch the next `__iter__` will use (DistributedSampler convention)."""
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
            raise RuntimeError(
                f"{method} is only meaningful with deterministic=True: in default mode the "
                "order is drawn from the (generator) RNG and cannot be sought."
            )

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
        while True:
            left, right = index >> half_bits, index & half_mask
            for key in keys:
                left, right = right, left ^ (_mix64(right ^ key) & half_mask)
            index = (left << half_bits) | right
            if index < self._num_frames:
                return index

    def _frame_index(self, position: int) -> int:
        episode = int(np.searchsorted(self._cum_lengths, position, side="right"))
        position_in_episode = position - (int(self._cum_lengths[episode - 1]) if episode > 0 else 0)
        return int(self._starts[episode]) + position_in_episode

    def __iter__(self) -> Iterator[int]:
        if not self.deterministic:
            return self._iter_default()
        # Capture and advance state eagerly so epoch bookkeeping is not deferred until the
        # returned generator is first consumed.
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
        for local_k in range(start, self._shard_len):
            k = self._shard_rank + local_k * self._shard_world_size
            yield self._frame_index(self._permute(k, keys) if self.shuffle else k)

    def __len__(self) -> int:
        return self._shard_len


def compute_sampler_state(step: int, num_frames: int, batch_size: int, num_processes: int) -> dict:
    """Map a global optimization step to an `EpisodeAwareSampler` state for resume.

    Under accelerate's batch-level sharding, every rank iterates the same underlying sampler and
    keeps every `num_processes`-th batch, so one optimization step consumes
    `batch_size * num_processes` consecutive sampler positions, and (with `even_batches` padding)
    each rank sees `ceil(ceil(num_frames / batch_size) / num_processes)` batches per epoch.

    `batches_into_epoch * num_processes <= ceil(num_frames / batch_size) - 1` always holds, so the
    start index stays strictly below `num_frames`; the `min` is purely defensive. Resume is
    sample-exact up to the `even_batches` padding accelerate appends at epoch boundaries (at most
    `num_processes - 1` duplicated batches per epoch, the same duplication non-resumed runs get).
    """
    batches_per_epoch = math.ceil(math.ceil(num_frames / batch_size) / num_processes)
    epoch, batches_into_epoch = divmod(step, batches_per_epoch)
    start_index = min(batches_into_epoch * batch_size * num_processes, num_frames)
    return {"epoch": epoch, "start_index": start_index}
