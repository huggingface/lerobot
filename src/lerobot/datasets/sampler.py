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
import math
from collections.abc import Iterator, Mapping

import numpy as np
import torch

from lerobot.configs.sampling import SampleKey


class EpisodeAwareSampler:
    """Sampler over the samples of every episode that stores only per-episode counts.

    It yields :class:`~lerobot.configs.sampling.SampleKey` s: sample ``k`` of an episode in the
    current epoch. What a sample reads is up to the dataset's training sampler (one per frame by
    default); this class only orders them. Logical positions map to keys on the fly
    (O(num_episodes) construction memory) instead of materializing every key.

    Each epoch is shuffled with a `torch.randperm` seeded from `(seed, epoch)`, so the data order
    is a pure function of `(seed, epoch)`: it reproduces on every rank without synchronizing the
    global RNG (no `generator` to sync across distributed ranks), and `state_dict` /
    `load_state_dict` resume a run sample-exactly by regenerating the epoch's permutation and
    continuing from the saved offset. Each call to `__iter__` advances the epoch. During a
    resumed epoch, `__len__` still reports the full length.

    Epoch advancement: `__iter__` eagerly advances the epoch, and `set_epoch` / `load_state_dict`
    set it explicitly. Within a single run callers should rely on exactly one of these mechanisms,
    not both: advancing the epoch by hand *and* letting `__iter__` auto-advance over the same
    iterations would skip or repeat epochs. The training loop drives it purely through `__iter__`
    (via `cycle`); `set_epoch` / `load_state_dict` are used only to (re)position before iteration
    starts (e.g. on resume or in tests).
    """

    def __init__(self, sample_counts: Mapping[int, int], shuffle: bool = False, seed: int = 0):
        """
        Args:
            sample_counts: Samples per dataset episode, e.g. ``LeRobotDataset.sample_counts``.
            shuffle: Whether to shuffle the samples.
            seed: Seed the permutation is derived from (together with the epoch).
        """
        episodes = np.asarray([ep for ep, count in sample_counts.items() if count > 0], dtype=np.int64)
        counts = np.asarray([count for count in sample_counts.values() if count > 0], dtype=np.int64)
        if not len(counts):
            raise ValueError("No episode gives any sample: every episode was filtered out or is too short.")
        self._episodes = episodes
        self._ends = np.cumsum(counts)
        self._num_samples = int(self._ends[-1])
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0
        self._start_index = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def state_dict(self) -> dict:
        return {"epoch": self._epoch, "start_index": self._start_index}

    def load_state_dict(self, state: dict) -> None:
        self._epoch = state["epoch"]
        self._start_index = state["start_index"]

    def _epoch_generator(self, epoch: int) -> torch.Generator:
        # Derive a per-epoch seed from (seed, epoch) so the permutation is a pure function of both
        # and reproduces identically on every rank without touching the global RNG.
        epoch_seed = int(np.random.SeedSequence([self.seed, epoch]).generate_state(1, dtype=np.uint64)[0])
        return torch.Generator().manual_seed(epoch_seed)

    def _key(self, position: int, epoch: int) -> SampleKey:
        slot = int(np.searchsorted(self._ends, position, side="right"))
        first = int(self._ends[slot - 1]) if slot else 0
        return SampleKey(int(self._episodes[slot]), position - first, epoch)

    def __iter__(self) -> Iterator[SampleKey]:
        # Advance epoch state eagerly, not on first consumption of the generator.
        epoch, start = self._epoch, self._start_index
        self._epoch += 1
        self._start_index = 0
        return self._iter_epoch(epoch, start)

    def _iter_epoch(self, epoch: int, start: int) -> Iterator[SampleKey]:
        if self.shuffle:
            order = torch.randperm(self._num_samples, generator=self._epoch_generator(epoch))
            for k in range(start, self._num_samples):
                yield self._key(int(order[k]), epoch)
        else:
            for k in range(start, self._num_samples):
                yield self._key(k, epoch)

    def __len__(self) -> int:
        return self._num_samples


def compute_sampler_state(step: int, num_samples: int, batch_size: int, num_processes: int) -> dict:
    """Map an optimization step to an `EpisodeAwareSampler` state for sample-exact resume.

    Under accelerate's batch sharding, one step consumes `batch_size * num_processes` sampler
    positions and each rank sees `ceil(ceil(num_samples / batch_size) / num_processes)` batches
    per epoch (`even_batches` padding included). The start index provably stays below
    `num_samples`; the `min` is defensive.

    Assumptions (resume is only sample-exact when they hold):
        - `num_processes` and `batch_size` match the run that wrote the checkpoint. Both scale how
          many positions a step consumes, so the epoch/offset are wrong if either changed. The
          caller passes the checkpoint's `num_processes` and `batch_size` and warns on a mismatch.
        - accelerate uses `even_batches=True` (its default). The `ceil(... / num_processes)` term
          mirrors that padding; with `even_batches=False` the per-epoch batch count differs and
          the boundary is off.
    """
    batches_per_epoch = math.ceil(math.ceil(num_samples / batch_size) / num_processes)
    epoch, batches_into_epoch = divmod(step, batches_per_epoch)
    start_index = min(batches_into_epoch * batch_size * num_processes, num_samples)
    return {"epoch": epoch, "start_index": start_index}
