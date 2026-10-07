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

"""Memory-bounded ring buffer for the Highlight Reel rollout strategy."""

from __future__ import annotations

from collections import deque

import numpy as np
import torch


class RolloutRingBuffer:
    """Fixed-capacity circular buffer for observation/action frames.

    Stores the last *N* seconds of telemetry in memory, bounded by both
    time (``max_frames``) and memory (``max_memory_bytes``).  When either
    limit is reached the oldest frames are evicted.

    .. note::
       This class is **single-threaded**.  ``append``/``drain``/``clear``
       must all be called from the same thread (the rollout main loop).
       Concurrent access from a background thread will corrupt
       ``_current_bytes`` accounting.

    Parameters
    ----------
    max_seconds:
        Maximum duration of buffered telemetry.
    max_memory_mb:
        Hard memory cap in MiB.  Frames are evicted when the estimated
        total size exceeds this.
    fps:
        Frames per second — used to convert ``max_seconds`` to a frame
        count.
    """

    def __init__(self, max_seconds: float = 30.0, max_memory_mb: int = 2048, fps: float = 30.0) -> None:
        self._max_frames = int(max_seconds * fps)
        self._max_bytes = int(max_memory_mb * 1024 * 1024)
        self._buffer: deque[dict] = deque(maxlen=self._max_frames)
        self._current_bytes: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(self, frame: dict) -> None:
        """Add *frame* to the buffer, evicting the oldest if at capacity."""
        frame_bytes = _estimate_frame_bytes(frame)

        # Evict oldest frames until we are under the memory cap
        while self._current_bytes + frame_bytes > self._max_bytes and self._buffer:
            evicted = self._buffer.popleft()
            self._current_bytes -= _estimate_frame_bytes(evicted)

        self._buffer.append(frame)
        self._current_bytes += frame_bytes

    def drain(self) -> list[dict]:
        """Return all buffered frames and clear the buffer."""
        frames = list(self._buffer)
        self._buffer.clear()
        self._current_bytes = 0
        return frames

    def clear(self) -> None:
        """Discard all buffered frames."""
        self._buffer.clear()
        self._current_bytes = 0

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def estimated_bytes(self) -> int:
        """Estimated total byte size of all buffered frames."""
        return self._current_bytes


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _estimate_frame_bytes(frame: dict) -> int:
    """Rough byte estimate for a single frame dictionary."""
    total = 0
    for v in frame.values():
        if isinstance(v, torch.Tensor):
            # ``torch.Tensor`` has no ``nbytes``; compute it explicitly so the
            # memory cap is honoured even when frames hold unconverted tensors.
            total += v.nelement() * v.element_size()
        elif isinstance(v, np.ndarray) or hasattr(v, "nbytes"):
            total += v.nbytes
        elif isinstance(v, (int, float)):
            total += 8
        elif isinstance(v, (str, bytes)):
            total += len(v)
    return max(total, 1)  # avoid zero-size frames
