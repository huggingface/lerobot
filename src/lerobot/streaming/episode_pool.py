# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0


from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


class ExactCoveragePool:
    """Deterministic, exactly-once frame coverage over a byte-cache episode pool.

    A with-replacement pool never guarantees a full
    epoch: frames are drawn randomly and episodes rotate on a fixed cadence. This planner instead
    enumerates *every frame of every episode exactly once per epoch* while keeping at most
    ``pool_size`` episodes resident, so batch mixing stays high but coverage is complete and
    reproducible.

    Mechanics (this is the "evict only when all frames sampled" model):
      - Episodes are admitted in a seeded global permutation until either ``pool_size`` or the
        optional indexed-byte budget is reached.
      - Each resident episode carries a seeded shuffle of its own frame indices.
      - Each draw picks a resident episode with probability proportional to its *remaining* frames
        (i.e. a uniform draw over all remaining frames in the pool, the map-style ideal) and pops
        one frame.
      - An episode is evicted only when its last frame is emitted; a new episode is then admitted.
      - The epoch ends when the admission order is exhausted and every resident episode is drained.

    Newly admitted episodes are surfaced via :attr:`newly_admitted` (drain it to drive prefetch)
    and evictions via :attr:`evicted` (drain to release cache bytes). The planner does no I/O and
    is fully unit-testable. It yields ``(episode_index, frame_index)``; map to a decode timestamp
    with ``frame_index / max(frame_count - 1, 1)``.

    Determinism: the order is a pure function of ``(seed, epoch)``, the episode frame counts, and
    optional byte sizes/budget. Resume is a deterministic fast-forward: re-instantiate with the
    same inputs and skip ``n`` samples (tabular only, no decode).
    """

    def __init__(
        self,
        episode_frame_counts: Sequence[tuple[int, int]],
        pool_size: int,
        *,
        seed: int,
        epoch: int = 0,
        episode_byte_sizes: Mapping[int, int] | None = None,
        byte_budget: int | None = None,
    ):
        self._counts = {int(ep): int(n) for ep, n in episode_frame_counts if int(n) > 0}
        self._rng = np.random.default_rng([seed, epoch])
        order = np.array(sorted(self._counts), dtype=np.int64)
        self._rng.shuffle(order)
        self.pool_size = max(1, pool_size)
        self._byte_budget = byte_budget
        if byte_budget is not None and byte_budget <= 0:
            raise ValueError("byte_budget must be positive")
        if byte_budget is not None and episode_byte_sizes is None:
            raise ValueError("episode_byte_sizes are required when byte_budget is set")
        self._byte_sizes = {
            episode: int(episode_byte_sizes[episode]) if episode_byte_sizes is not None else 0
            for episode in self._counts
        }
        if any(size < 0 for size in self._byte_sizes.values()):
            raise ValueError("episode byte sizes must be non-negative")
        if byte_budget is not None:
            oversized = next(
                ((episode, size) for episode, size in self._byte_sizes.items() if size > byte_budget),
                None,
            )
            if oversized is not None:
                episode, size = oversized
                raise ValueError(
                    f"Episode {episode} requires {size} bytes, exceeding the byte budget {byte_budget}"
                )

        # Preserve the full seeded order for benchmark/tooling compatibility. Byte-aware admission
        # may temporarily skip an entry, but every episode remains in this deterministic frontier.
        self.admission_order: list[int] = order.tolist()
        self._pending: list[int] = list(self.admission_order)
        self._admitted_count = 0
        self._remaining: dict[int, tuple[np.ndarray, int]] = {}
        self._remaining_total = 0
        self._resident_bytes = 0
        self.newly_admitted: list[int] = []
        self.evicted: list[int] = []
        self._admit_available()

    def _admit_available(self) -> None:
        while len(self._remaining) < self.pool_size and self._pending:
            available_bytes = None if self._byte_budget is None else self._byte_budget - self._resident_bytes
            pending_index = next(
                (
                    index
                    for index, episode in enumerate(self._pending)
                    if available_bytes is None or self._byte_sizes[episode] <= available_bytes
                ),
                None,
            )
            if pending_index is None:
                return

            episode = self._pending.pop(pending_index)
            frame_count = self._counts[episode]
            frames = np.arange(frame_count, dtype=np.int64)
            self._rng.shuffle(frames)
            self._remaining[episode] = (frames, frame_count)
            self._remaining_total += frame_count
            self._resident_bytes += self._byte_sizes[episode]
            self._admitted_count += 1
            self.newly_admitted.append(episode)

    @property
    def remaining_total(self) -> int:
        return self._remaining_total

    @property
    def admitted_count(self) -> int:
        """Number of episodes pulled from the admission order so far (pool fills + rotations)."""
        return self._admitted_count

    @property
    def resident(self) -> list[int]:
        return list(self._remaining)

    @property
    def resident_bytes(self) -> int:
        return self._resident_bytes

    def prefetch_candidates(self, count: int) -> list[int]:
        """Return the next deterministic pending frontier without admitting it."""
        if count <= 0:
            return []
        return self._pending[:count]

    def __iter__(self) -> ExactCoveragePool:
        return self

    def __next__(self) -> tuple[int, int]:
        if self._remaining_total == 0:
            raise StopIteration
        # Uniform draw over all remaining frames in the pool: walk the residents by cumulative
        # remaining count. O(pool_size) per draw (~1024) -> negligible next to decode.
        target = int(self._rng.integers(self._remaining_total))
        chosen = None
        for ep, (_frames, remaining) in self._remaining.items():
            if target < remaining:
                chosen = ep
                break
            target -= remaining
        frames, remaining = self._remaining[chosen]
        remaining -= 1
        frame_index = int(frames[remaining])
        self._remaining_total -= 1
        if remaining == 0:
            del self._remaining[chosen]
            self.evicted.append(chosen)
            self._resident_bytes -= self._byte_sizes[chosen]
            self._admit_available()
        else:
            self._remaining[chosen] = (frames, remaining)
        return chosen, frame_index
