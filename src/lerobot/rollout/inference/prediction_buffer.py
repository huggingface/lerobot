# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Paced playback of a policy's temporal predictions for visualization.

Intermediate predictions (e.g. a world model's imagined clip) arrive as a ``{key: sequence}`` stack
with time on axis 0, one fresh stack per predicted chunk. Advancement is kept separate from reading
so the loop's two clocks don't get conflated: :meth:`advance` steps the playhead at policy cadence
(one new prediction per chunk), while :meth:`current` is a side-effect-free read for the display,
which may run faster (e.g. action-interpolated). Between policy ticks the viewer just holds the last
frame. Modality-agnostic: a step may be an image, vector, or scalar — only axis 0 is assumed to be
time.
"""

from __future__ import annotations


class PacedPredictionBuffer:
    """Maps a temporal prediction stack onto a chunk's policy ticks for paced visualization.

    Drive from policy inference (:meth:`load` a fresh chunk, :meth:`advance` on its later ticks);
    read from the display loop (:meth:`current`).
    """

    def __init__(self, ticks_per_chunk: int | None = None) -> None:
        # Policy ticks per predicted chunk; the T steps of each stack are spread across this span.
        # ``None`` falls back to one step per policy tick (clamped to the stack length).
        self._ticks_per_chunk = ticks_per_chunk
        self._stacks: dict = {}  # key -> sequence with time on axis 0, for the current chunk
        self._cursor = 0  # policy ticks elapsed since the current chunk's stacks were loaded

    def reset(self) -> None:
        """Drop the current chunk's stacks and rewind the playhead."""
        self._stacks = {}
        self._cursor = 0

    def load(self, stacks: dict) -> None:
        """Store a freshly predicted chunk's stacks and rewind the playhead to its first step."""
        self._stacks = stacks
        self._cursor = 0

    def advance(self) -> None:
        """Move the playhead forward one policy tick (no-op until a chunk is loaded)."""
        if self._stacks:
            self._cursor += 1

    def current(self) -> dict | None:
        """Current playhead step per key (``None`` until a chunk is loaded); no side effects.

        Maps each stack's ``T`` steps onto the ``ticks_per_chunk`` span (one step/tick, clamped, when
        the span is unknown).
        """
        if not self._stacks:
            return None
        tick = self._cursor
        span = self._ticks_per_chunk
        out: dict = {}
        for key, stack in self._stacks.items():
            n = len(stack)
            if n == 0:
                continue
            idx = round(tick / (span - 1) * (n - 1)) if span and span > 1 else tick
            idx = min(max(idx, 0), n - 1)
            step = stack[idx]
            if hasattr(step, "detach"):  # torch tensor -> numpy for the visualization backend
                step = step.detach().cpu().numpy()
            out[key] = step
        return out or None
