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

"""Capture-seq alignment buffer (handoff challenge A).

The cloud receives proprioceptive state (state DataChannel) and camera frames
(media track) on *separate* channels with independent jitter/loss. Each carries the
robot-side capture **seq** of the cycle that produced it (joints + image come from one
``robot.get_observation()`` on the robot, so they share a seq; the frame's seq rides
its ``pts``). Pairing by seq is therefore *exact* — a state and a frame with the same
seq are from the same capture instant — and robust to loss: a dropped frame or state
just means that seq is incomplete; we use the freshest seq present on both sides.

Deliberately tiny and thread-safe via a single lock: producers (asyncio receive
callbacks) push from one thread, the consumer (``get_observation``) pulls from
another.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AlignedObs:
    """A state and a camera frame from the same capture seq."""

    seq: int
    t: float  # robot capture time.monotonic() of this cycle (from the state)
    joints: dict[str, float]
    frame: np.ndarray  # HxWx3 uint8 RGB


class AlignmentBuffer:
    """Bounded, thread-safe pairing of state and frames by capture seq."""

    def __init__(self, maxlen: int = 128) -> None:
        self._states: dict[int, tuple[float, dict[str, float]]] = {}  # seq -> (t, joints)
        self._frames: dict[int, np.ndarray] = {}  # seq -> frame
        self._maxlen = maxlen
        self._lock = threading.Lock()

    def add_state(self, seq: int, t: float, joints: dict[str, float]) -> None:
        with self._lock:
            self._states[seq] = (t, dict(joints))
            self._evict(self._states)

    def add_frame(self, seq: int, frame: np.ndarray) -> None:
        with self._lock:
            self._frames[seq] = frame
            self._evict(self._frames)

    def _evict(self, d: dict) -> None:
        # Bounded ring: drop the lowest (oldest) seqs once over capacity.
        while len(d) > self._maxlen:
            del d[min(d)]

    def assemble(self) -> AlignedObs | None:
        """The freshest seq present on BOTH sides, or None if there is no such pair."""
        with self._lock:
            common = self._states.keys() & self._frames.keys()
            if not common:
                return None
            seq = max(common)
            t, joints = self._states[seq]
            return AlignedObs(seq=seq, t=t, joints=dict(joints), frame=self._frames[seq])

    def has_state(self) -> bool:
        with self._lock:
            return bool(self._states)
