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

"""Optional, time-windowed action latency measurements, independent of scheduling."""

from __future__ import annotations

from collections import deque
from threading import Lock


class LatencyWindow:
    """Keep completed action calls in a configurable window of monotonic time.

    Configure before starting the engine. Disabled by default, so users outside
    rollout do not collect diagnostics. The worker records and the control thread
    reads; neither holds the lock during inference or formatting.
    """

    def __init__(self, label: str = "infer") -> None:
        """Create a disabled window with the label used in health logs."""
        self.label = label
        self.interval_s = 0.0
        self._samples: deque[tuple[float, float]] = deque()
        self._lock = Lock()

    def configure(self, interval_s: float) -> None:
        """Set the retention window before engine startup; zero disables collection."""
        with self._lock:
            self.interval_s = interval_s
            self._samples.clear()

    def record(self, elapsed: float, completed_at: float) -> None:
        """Record a completed call, including cold start, when collection is enabled."""
        if self.interval_s == 0:
            return
        with self._lock:
            self._prune(completed_at)
            self._samples.append((completed_at, elapsed))

    def recent(self, now: float) -> list[float]:
        """Durations completed in ``(now - interval_s, now]``, in seconds."""
        with self._lock:
            self._prune(now)
            return [elapsed for completed, elapsed in self._samples if completed <= now]

    def _prune(self, now: float) -> None:
        while self._samples and self._samples[0][0] <= now - self.interval_s:
            self._samples.popleft()
