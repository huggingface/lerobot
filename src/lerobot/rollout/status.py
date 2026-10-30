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

"""Periodic rollout health logs, sampled on the control thread."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from statistics import median

import torch

from lerobot.inference import InferenceEngine
from lerobot.utils.cycle_timer import CadenceStats, CycleTimer

logger = logging.getLogger(__name__)
_GIB = 1024**3


def _process_rss_bytes() -> int | None:
    """Current resident memory on Linux; omit it where the probe is unavailable."""
    try:
        return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError, AttributeError):
        return None


def _gpu_memory_bytes(device: str | None) -> tuple[int, int] | None:
    """Device-wide CUDA memory; avoid initialization and explicit synchronization."""
    if device is None:
        return None
    try:
        selected = torch.device(device)
        if selected.type != "cuda" or not torch.cuda.is_initialized():
            return None
        free, total = torch.cuda.mem_get_info(selected)
        return total - free, total
    except (RuntimeError, ValueError, AssertionError, OSError):
        return None


class RolloutStatusReporter:
    """Own the reporting schedule and baseline; call after each control-loop tick starts."""

    def __init__(self, engine: InferenceEngine, interval_s: float, device: str | None) -> None:
        self._engine = engine
        self._interval_s = interval_s
        self._device = device
        self._timer: CycleTimer | None = None
        self._previous: CadenceStats | None = None
        self._due_at = 0.0

    def report_if_due(self, timer: CycleTimer) -> None:
        """Start a fresh interval for each timer and emit at most one log after a stall."""
        if self._interval_s == 0:
            return
        now = time.monotonic()
        if timer is not self._timer or self._previous is None:
            self._timer = timer
            self._previous = timer.snapshot()
            self._due_at = now + self._interval_s
            return
        if now >= self._due_at:
            self._due_at = now + self._interval_s
            self._previous = _log_status(timer, self._previous, self._engine, self._device, now)


def _log_status(
    timer: CycleTimer,
    previous: CadenceStats,
    engine: InferenceEngine,
    device: str | None,
    now: float,
) -> CadenceStats:
    """Log interval cadence, recent latency and available memory; return the new baseline."""
    current = timer.snapshot()
    if not logger.isEnabledFor(logging.INFO):
        return current
    parts = []
    span = current.span - previous.span
    if span > 0:
        hz = (current.span_ticks - previous.span_ticks) / span
        # Policy rate, as in the cadence summaries (equal to the command rate without interpolation).
        parts.append(f"loop {hz / timer.multiplier:.1f}/{timer.fps:g} Hz")
    parts.append(
        f"over budget {current.groups_over - previous.groups_over}/"
        f"{current.groups_judged - previous.groups_judged}"
    )
    parts.append(f"starved {current.starved_ticks - previous.starved_ticks}")
    if samples := engine.latency.recent(now):
        parts.append(
            f"{engine.latency.label} {median(samples) * 1000:.1f} ms (worst {max(samples) * 1000:.1f})"
        )
    if (rss := _process_rss_bytes()) is not None:
        parts.append(f"process {rss / _GIB:.2f} GiB")
    if (gpu := _gpu_memory_bytes(device)) is not None:
        used, total = gpu
        parts.append(f"GPU {used / _GIB:.2f}/{total / _GIB:.1f} GiB")
    logger.info("Rollout: %s", " · ".join(parts))
    return current
