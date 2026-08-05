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


class EpisodeAwareSampler:
    """Sampler over episode frames that stores only per-episode boundaries.

    Logical positions map to frame indices on the fly (O(num_episodes) construction memory)
    instead of materializing a Python list of every frame index.

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

    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
        seed: int = 0,
        absolute_to_relative_idx: dict[int, int] | None = None,
    ):
        """
        Args:
            dataset_from_indices: Start index of each episode in the dataset.
            dataset_to_indices: End index of each episode in the dataset.
            episode_indices_to_use: Episode indices to use; None means all.
            drop_n_first_frames: Frames to drop from the start of each episode.
            drop_n_last_frames: Frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
            seed: Seed the permutation is derived from (together with the epoch).
        """
        if drop_n_first_frames < 0:
            raise ValueError(f"drop_n_first_frames must be >= 0, got {drop_n_first_frames}")
        if drop_n_last_frames < 0:
            raise ValueError(f"drop_n_last_frames must be >= 0, got {drop_n_last_frames}")

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
        self.seed = seed
        self._epoch = 0
        self._start_index = 0
        self._absolute_to_relative = absolute_to_relative_idx

    @property
    def indices(self) -> list[int]:
        """Materialized frame indices in unshuffled order; O(num_frames), introspection only."""
        return [self._frame_index(k) for k in range(self._num_frames)]

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

    def _frame_index(self, position: int) -> int:
        episode = int(np.searchsorted(self._cum_lengths, position, side="right"))
        position_in_episode = position - (int(self._cum_lengths[episode - 1]) if episode > 0 else 0)
        absolute_idx = int(self._starts[episode]) + position_in_episode
        if self._absolute_to_relative is not None:
            return self._absolute_to_relative[absolute_idx]
        return absolute_idx

    def __iter__(self) -> Iterator[int]:
        # Advance epoch state eagerly, not on first consumption of the generator.
        epoch, start = self._epoch, self._start_index
        self._epoch += 1
        self._start_index = 0
        return self._iter_epoch(epoch, start)

    def _iter_epoch(self, epoch: int, start: int) -> Iterator[int]:
        if self.shuffle:
            order = torch.randperm(self._num_frames, generator=self._epoch_generator(epoch))
            for k in range(start, self._num_frames):
                yield self._frame_index(int(order[k]))
        else:
            for k in range(start, self._num_frames):
                yield self._frame_index(k)

    def __len__(self) -> int:
        return self._num_frames


class DomainBalancedSampler:
    """Deterministic equal-domain sampler with exact per-batch quotas.

    Each epoch independently shuffles the valid frames inside every domain,
    uses each selected frame at most once, and stops at the largest number of
    complete balanced batches supported by the smallest domain. This avoids
    both implicit oversampling and partial, imbalanced batches.
    """

    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices: list[int],
        domain_episode_groups: dict[str, list[int]],
        batch_size: int,
        episode_indices_to_use: list[int] | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        seed: int = 0,
        absolute_to_relative_idx: dict[int, int] | None = None,
    ):
        if len(domain_episode_groups) < 2:
            raise ValueError("At least two domains are required")
        if batch_size <= 0 or batch_size % len(domain_episode_groups) != 0:
            raise ValueError(
                f"batch_size={batch_size} must be divisible by the "
                f"{len(domain_episode_groups)} configured domains"
            )
        if drop_n_first_frames < 0 or drop_n_last_frames < 0:
            raise ValueError("Frame-drop counts must be non-negative")

        from_indices = np.asarray(dataset_from_indices, dtype=np.int64)
        to_indices = np.asarray(dataset_to_indices, dtype=np.int64)
        episode_indices_array = np.asarray(episode_indices, dtype=np.int64)
        if not (from_indices.shape == to_indices.shape == episode_indices_array.shape):
            raise ValueError("Episode indices and frame-bound arrays must have the same length")
        if len({int(value) for value in episode_indices_array}) != len(episode_indices_array):
            raise ValueError("episode_indices must be unique")

        available = {int(value) for value in episode_indices_array}
        if episode_indices_to_use is not None:
            requested = {int(value) for value in episode_indices_to_use}
            unknown = requested - available
            if unknown:
                raise ValueError(f"Requested episodes are absent from the dataset: {sorted(unknown)}")
            available = requested

        flattened = [int(episode) for episodes in domain_episode_groups.values() for episode in episodes]
        if len(flattened) != len(set(flattened)):
            raise ValueError("An episode may belong to only one domain")
        configured = set(flattened)
        if configured != available:
            raise ValueError(
                "Balanced domain episode groups must cover exactly the active dataset episodes; "
                f"missing={sorted(available - configured)}, extra={sorted(configured - available)}"
            )

        bounds = {
            int(episode): (int(start), int(stop))
            for episode, start, stop in zip(episode_indices_array, from_indices, to_indices, strict=True)
        }
        self.domain_names = list(domain_episode_groups)
        self._domain_frame_indices: list[np.ndarray] = []
        for domain, episodes in domain_episode_groups.items():
            frame_indices: list[int] = []
            for episode in episodes:
                start, stop = bounds[int(episode)]
                start += drop_n_first_frames
                stop -= drop_n_last_frames
                if stop <= start:
                    raise ValueError(
                        f"Episode {episode} in domain {domain!r} has no frames after frame drops"
                    )
                for absolute_idx in range(start, stop):
                    relative_idx = (
                        absolute_to_relative_idx[absolute_idx]
                        if absolute_to_relative_idx is not None
                        else absolute_idx
                    )
                    frame_indices.append(relative_idx)
            self._domain_frame_indices.append(np.asarray(frame_indices, dtype=np.int64))

        self.batch_size = batch_size
        self.samples_per_domain_per_batch = batch_size // len(self.domain_names)
        self.num_batches = min(
            len(indices) // self.samples_per_domain_per_batch for indices in self._domain_frame_indices
        )
        if self.num_batches <= 0:
            raise ValueError("No complete balanced batch can be formed")
        self._num_samples = self.num_batches * batch_size
        self.seed = seed
        self._epoch = 0
        self._start_index = 0

    @property
    def domain_frame_counts(self) -> dict[str, int]:
        return {
            domain: len(indices)
            for domain, indices in zip(self.domain_names, self._domain_frame_indices, strict=True)
        }

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def state_dict(self) -> dict:
        return {"epoch": self._epoch, "start_index": self._start_index}

    def load_state_dict(self, state: dict) -> None:
        self._epoch = state["epoch"]
        self._start_index = state["start_index"]

    def _epoch_generator(self, epoch: int, stream: int) -> torch.Generator:
        epoch_seed = int(
            np.random.SeedSequence([self.seed, epoch, stream]).generate_state(1, dtype=np.uint64)[0]
        )
        return torch.Generator().manual_seed(epoch_seed)

    def __iter__(self) -> Iterator[int]:
        epoch, start = self._epoch, self._start_index
        self._epoch += 1
        self._start_index = 0
        return self._iter_epoch(epoch, start)

    def _iter_epoch(self, epoch: int, start: int) -> Iterator[int]:
        selected_by_domain = []
        samples_per_domain = self.num_batches * self.samples_per_domain_per_batch
        for domain_index, indices in enumerate(self._domain_frame_indices):
            order = torch.randperm(len(indices), generator=self._epoch_generator(epoch, domain_index))[
                :samples_per_domain
            ]
            selected_by_domain.append(indices[order.numpy()])

        sequence: list[int] = []
        for batch_index in range(self.num_batches):
            batch: list[int] = []
            offset = batch_index * self.samples_per_domain_per_batch
            stop = offset + self.samples_per_domain_per_batch
            for selected in selected_by_domain:
                batch.extend(int(value) for value in selected[offset:stop])
            within_batch = torch.randperm(
                self.batch_size,
                generator=self._epoch_generator(epoch, len(self.domain_names) + batch_index),
            )
            sequence.extend(batch[int(index)] for index in within_batch)
        yield from sequence[start:]

    def __len__(self) -> int:
        return self._num_samples


class MatchedTwoStreamSampler:
    """Deterministic fixed-length 4+4-style sampler for matched experiments.

    The first half of every batch comes from stream 0 and the second half from
    stream 1. Streams are independently shuffled and cycled as needed.  A
    stream's sequence depends only on its own frame membership, seed, epoch and
    stream index, so an identical stream 0 is byte-for-byte matched when stream
    1 changes size or membership. Overlap between streams is intentional and
    supports the Real24 repeat-compute control.
    """

    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices: list[int],
        stream_episode_groups: dict[str, list[int]],
        batch_size: int,
        batches_per_epoch: int,
        episode_indices_to_use: list[int] | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        seed: int = 0,
        absolute_to_relative_idx: dict[int, int] | None = None,
    ):
        if len(stream_episode_groups) != 2:
            raise ValueError("matched two-stream sampling requires exactly two streams")
        if batch_size <= 0 or batch_size % 2:
            raise ValueError("batch_size must be a positive even number")
        if batches_per_epoch <= 0:
            raise ValueError("batches_per_epoch must be positive")
        if drop_n_first_frames < 0 or drop_n_last_frames < 0:
            raise ValueError("Frame-drop counts must be non-negative")

        from_indices = np.asarray(dataset_from_indices, dtype=np.int64)
        to_indices = np.asarray(dataset_to_indices, dtype=np.int64)
        episode_indices_array = np.asarray(episode_indices, dtype=np.int64)
        if not (from_indices.shape == to_indices.shape == episode_indices_array.shape):
            raise ValueError("Episode indices and frame-bound arrays must have the same length")
        if len(set(map(int, episode_indices_array))) != len(episode_indices_array):
            raise ValueError("episode_indices must be unique")
        available = set(map(int, episode_indices_array))
        if episode_indices_to_use is not None:
            requested = set(map(int, episode_indices_to_use))
            unknown = requested - available
            if unknown:
                raise ValueError(f"Requested episodes are absent from the dataset: {sorted(unknown)}")
            available = requested

        bounds = {
            int(episode): (int(start), int(stop))
            for episode, start, stop in zip(episode_indices_array, from_indices, to_indices, strict=True)
        }
        self.stream_names = list(stream_episode_groups)
        self._stream_frame_indices: list[np.ndarray] = []
        configured_union: set[int] = set()
        for stream, episodes in stream_episode_groups.items():
            if not episodes or len(episodes) != len(set(episodes)):
                raise ValueError(f"Stream {stream!r} must contain unique episodes")
            stream_set = set(map(int, episodes))
            unknown = stream_set - available
            if unknown:
                raise ValueError(f"Stream {stream!r} contains inactive episodes: {sorted(unknown)}")
            configured_union |= stream_set
            frame_indices: list[int] = []
            for episode in episodes:
                start, stop = bounds[int(episode)]
                start += drop_n_first_frames
                stop -= drop_n_last_frames
                if stop <= start:
                    raise ValueError(f"Episode {episode} in stream {stream!r} has no usable frames")
                for absolute_idx in range(start, stop):
                    frame_indices.append(
                        absolute_to_relative_idx[absolute_idx]
                        if absolute_to_relative_idx is not None
                        else absolute_idx
                    )
            self._stream_frame_indices.append(np.asarray(frame_indices, dtype=np.int64))
        if configured_union != available:
            raise ValueError(
                "Matched stream groups must cover all active episodes; "
                f"missing={sorted(available - configured_union)}"
            )

        self.batch_size = batch_size
        self.samples_per_stream_per_batch = batch_size // 2
        self.num_batches = batches_per_epoch
        self._num_samples = batches_per_epoch * batch_size
        self.seed = seed
        self._epoch = 0
        self._start_index = 0

    @property
    def stream_frame_counts(self) -> dict[str, int]:
        return dict(zip(self.stream_names, map(len, self._stream_frame_indices), strict=True))

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def state_dict(self) -> dict:
        return {"epoch": self._epoch, "start_index": self._start_index}

    def load_state_dict(self, state: dict) -> None:
        self._epoch = state["epoch"]
        self._start_index = state["start_index"]

    def _generator(self, epoch: int, stream: int, cycle: int) -> torch.Generator:
        value = int(
            np.random.SeedSequence([self.seed, epoch, stream, cycle]).generate_state(1, dtype=np.uint64)[0]
        )
        return torch.Generator().manual_seed(value)

    def _stream_sequence(self, epoch: int, stream: int, count: int) -> list[int]:
        indices = self._stream_frame_indices[stream]
        result: list[int] = []
        cycle = 0
        while len(result) < count:
            order = torch.randperm(len(indices), generator=self._generator(epoch, stream, cycle)).numpy()
            take = min(len(indices), count - len(result))
            result.extend(int(value) for value in indices[order[:take]])
            cycle += 1
        return result

    def __iter__(self) -> Iterator[int]:
        epoch, start = self._epoch, self._start_index
        self._epoch += 1
        self._start_index = 0
        per_stream = self.num_batches * self.samples_per_stream_per_batch
        streams = [self._stream_sequence(epoch, i, per_stream) for i in range(2)]
        sequence: list[int] = []
        quota = self.samples_per_stream_per_batch
        for batch in range(self.num_batches):
            offset = batch * quota
            sequence.extend(streams[0][offset : offset + quota])
            sequence.extend(streams[1][offset : offset + quota])
        return iter(sequence[start:])

    def __len__(self) -> int:
        return self._num_samples


def compute_sampler_state(step: int, num_frames: int, batch_size: int, num_processes: int) -> dict:
    """Map an optimization step to an `EpisodeAwareSampler` state for sample-exact resume.

    Under accelerate's batch sharding, one step consumes `batch_size * num_processes` sampler
    positions and each rank sees `ceil(ceil(num_frames / batch_size) / num_processes)` batches
    per epoch (`even_batches` padding included). The start index provably stays below
    `num_frames`; the `min` is defensive.

    Assumptions (resume is only sample-exact when they hold):
        - `num_processes` and `batch_size` match the run that wrote the checkpoint. Both scale how
          many positions a step consumes, so the epoch/offset are wrong if either changed. The
          caller passes the checkpoint's `num_processes` and `batch_size` and warns on a mismatch.
        - accelerate uses `even_batches=True` (its default). The `ceil(... / num_processes)` term
          mirrors that padding; with `even_batches=False` the per-epoch batch count differs and
          the boundary is off.
    """
    batches_per_epoch = math.ceil(math.ceil(num_frames / batch_size) / num_processes)
    epoch, batches_into_epoch = divmod(step, batches_per_epoch)
    start_index = min(batches_into_epoch * batch_size * num_processes, num_frames)
    return {"epoch": epoch, "start_index": start_index}
