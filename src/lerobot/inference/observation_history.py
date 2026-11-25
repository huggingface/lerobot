# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Capture-time observation history for policies that condition on past frames."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class HistoryFrame:
    """One owned control-thread capture and its original capture time."""

    capture_time: float
    observation: Mapping[str, Any]


def _nearest(frames: Sequence[HistoryFrame], target: float) -> int:
    return min(range(len(frames)), key=lambda index: abs(frames[index].capture_time - target))


class ObservationHistory:
    """Owned captures covering the past policy ticks a policy declares, sampled by capture time.

    Training offsets count dataset frames recorded at the policy rate, so each offset
    selects by capture time rather than by counting notifications: a starved control
    loop republishes on every motor tick. A capture less than half a policy period after
    the newest slot opened replaces it, so the history keeps about one capture per half
    period and each offset takes the kept capture nearest ``now + offset / fps``, which
    can differ from the nearest raw capture by up to half a period. Offsets that reach
    before the first capture of a run repeat that capture and are marked as padding, as
    datasets pad the start of an episode. Callers serialize ``record``, ``clear`` and
    ``frames``; captures are never mutated once recorded.
    """

    def __init__(self, offsets: Mapping[str, Sequence[int]], fps: float) -> None:
        """Track enough captures for the deepest declared offset at ``fps`` policy ticks."""
        if not offsets:
            raise ValueError("Observation history needs at least one feature.")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Observation history needs a finite, positive policy rate.")
        self.offsets = {name: tuple(values) for name, values in offsets.items()}
        if any(not values or max(values) > 0 for values in self.offsets.values()):
            raise ValueError("Observation history offsets must be nonempty and <= 0.")
        self.period = 1.0 / fps
        self.depth = max(-min(values) for values in self.offsets.values())
        # Two slots per policy period cover the deepest offset; the bound only guards
        # against a clock that stops advancing.
        self._frames: deque[HistoryFrame] = deque(maxlen=2 * self.depth + 4)
        self._slot_start = -math.inf

    def record(self, observation: Mapping[str, Any], capture_time: float) -> None:
        """Keep an owned capture as the newest frame."""
        frame = HistoryFrame(capture_time, observation)
        if self._frames and capture_time - self._slot_start < self.period / 2:
            self._frames[-1] = frame
        else:
            self._slot_start = capture_time
            self._frames.append(frame)
        # A capture at or before the deepest target makes every older one unreachable.
        deepest = capture_time - self.depth * self.period
        while len(self._frames) > 1 and self._frames[1].capture_time <= deepest:
            self._frames.popleft()

    def clear(self) -> None:
        """Start a new run: the next capture becomes the first frame."""
        self._frames.clear()
        self._slot_start = -math.inf

    def frames(self) -> tuple[HistoryFrame, ...]:
        """Snapshot the recorded captures, oldest first."""
        return tuple(self._frames)

    def select(self, frames: Sequence[HistoryFrame]) -> dict[str, tuple[tuple[int, ...], tuple[bool, ...]]]:
        """Return, per feature, the frame index and padding flag of each declared offset."""
        if not frames:
            raise ValueError("Observation history has no capture to sample.")
        now = frames[-1].capture_time
        first = frames[0].capture_time
        selection: dict[str, tuple[tuple[int, ...], tuple[bool, ...]]] = {}
        for name, offsets in self.offsets.items():
            indices: list[int] = []
            padding: list[bool] = []
            for offset in offsets:
                target = now + offset * self.period
                if target < first - self.period / 2:
                    indices.append(0)
                    padding.append(True)
                    continue
                indices.append(_nearest(frames, target))
                padding.append(False)
            selection[name] = (tuple(indices), tuple(padding))
        return selection

    def assemble(
        self,
        frames: Sequence[HistoryFrame],
        prepare: Callable[[Mapping[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        """Prepare each selected capture once, the newest last, and stack declared features.

        ``prepare`` turns one capture into a policy batch and must not mutate it. The
        newest capture is always prepared last so stateful processor steps (relative
        actions, language) end on the current observation; it also supplies every
        undeclared feature. Declared features gain a time dimension at index 1 and an
        ``<name>_is_pad`` mask, matching the training batches of delta-indexed features.
        """
        selection = self.select(frames)
        latest = len(frames) - 1
        needed = sorted({latest} | {index for indices, _ in selection.values() for index in indices})
        prepared = {index: prepare(frames[index].observation) for index in needed}
        batch = dict(prepared[latest])
        for name, (indices, padding) in selection.items():
            if name not in batch:
                raise ValueError(
                    f"Declared observation history feature {name!r} is missing after preprocessing."
                )
            stacked = torch.stack([prepared[index][name] for index in indices], dim=1)
            batch[name] = stacked
            batch[f"{name}_is_pad"] = torch.tensor(padding, dtype=torch.bool, device=stacked.device).expand(
                stacked.shape[0], -1
            )
        return batch
