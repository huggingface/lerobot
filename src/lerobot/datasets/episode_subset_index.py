# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Compact frame mappings for episode subsets, shared by dataset readers."""

from collections.abc import Iterator, Mapping

import numpy as np


class _EpisodeSubsetIndexMapping(Mapping[int, int]):
    """Map absolute frame indices to relative rows using selected episode intervals.

    Stores O(num_episodes) boundaries; both lookup directions use binary search.
    Keys follow storage order, and unselected frames raise KeyError.
    """

    def __init__(self, episode_starts: np.ndarray, episode_ends: np.ndarray) -> None:
        """Build from ascending, non-overlapping absolute [start, end) intervals."""
        self._episode_starts = np.ascontiguousarray(episode_starts, dtype=np.int64)
        self._episode_ends = np.ascontiguousarray(episode_ends, dtype=np.int64)
        self._relative_episode_ends = np.cumsum(self._episode_ends - self._episode_starts)
        self._num_frames = int(self._relative_episode_ends[-1]) if len(self._relative_episode_ends) else 0

    def rel_to_abs(self, rel_idx: int) -> int:
        """Map a relative row in [0, len(self)) to its absolute frame index."""
        block = int(np.searchsorted(self._relative_episode_ends, rel_idx, side="right"))
        relative_base = int(self._relative_episode_ends[block - 1]) if block > 0 else 0
        return int(self._episode_starts[block]) + (rel_idx - relative_base)

    def __getitem__(self, abs_idx: int) -> int:
        block = int(np.searchsorted(self._episode_starts, abs_idx, side="right")) - 1
        if block < 0 or abs_idx >= int(self._episode_ends[block]):
            raise KeyError(abs_idx)
        relative_base = int(self._relative_episode_ends[block - 1]) if block > 0 else 0
        return relative_base + (abs_idx - int(self._episode_starts[block]))

    def __len__(self) -> int:
        return self._num_frames

    def __iter__(self) -> Iterator[int]:
        for start, end in zip(self._episode_starts.tolist(), self._episode_ends.tolist(), strict=True):
            yield from range(start, end)

    def __eq__(self, other: object) -> bool:
        # Mapping.__eq__ materializes per-frame dicts; compare items incrementally instead.
        if isinstance(other, Mapping):
            if len(self) != len(other):
                return False
            try:
                return all(other[key] == value for key, value in self.items())
            except KeyError:
                return False
        return NotImplemented
