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

"""Experiment metrics collection for async inference.

Provides per-tick metrics collection and CSV export for analyzing
async inference experiments.
"""

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ExperimentTick:
    """Single tick of experiment data."""

    t: float  # Wall-clock timestamp (Unix seconds)
    step: int  # Action step n(t)
    schedule_size: int  # |ψ(t)|
    latency_estimate_steps: int  # ℓ̂_Δ
    cooldown: int  # O^c(t)
    stall: int  # 1 if schedule_size == 0, else 0
    obs_sent: int  # 1 if obs request triggered this tick
    action_received: int  # 1 if action chunk merged this tick
    measured_latency_ms: float | None  # RTT of received chunk (if any)
    # Action discontinuity metrics (L2 distance between overlapping chunks)
    chunk_overlap_count: int | None  # Number of overlapping actions compared
    chunk_mean_l2: float | None  # Mean L2 distance across overlapping actions
    chunk_max_l2: float | None  # Max L2 distance across overlapping actions


class ExperimentMetricsWriter:
    """Collects per-tick experiment metrics and writes to CSV."""

    def __init__(self):
        self._ticks: list[ExperimentTick] = []

    def record_tick(
        self,
        *,
        step: int,
        schedule_size: int,
        latency_estimate_steps: int,
        cooldown: int,
        obs_sent: bool = False,
        action_received: bool = False,
        measured_latency_ms: float | None = None,
        chunk_overlap_count: int | None = None,
        chunk_mean_l2: float | None = None,
        chunk_max_l2: float | None = None,
    ) -> None:
        """Record a single tick of experiment data."""
        tick = ExperimentTick(
            t=time.time(),
            step=step,
            schedule_size=schedule_size,
            latency_estimate_steps=latency_estimate_steps,
            cooldown=cooldown,
            stall=1 if schedule_size == 0 else 0,
            obs_sent=1 if obs_sent else 0,
            action_received=1 if action_received else 0,
            measured_latency_ms=measured_latency_ms,
            chunk_overlap_count=chunk_overlap_count,
            chunk_mean_l2=chunk_mean_l2,
            chunk_max_l2=chunk_max_l2,
        )
        self._ticks.append(tick)

    def flush(self, path: str | Path) -> None:
        """Write collected metrics to CSV file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "t",
            "step",
            "schedule_size",
            "latency_estimate_steps",
            "cooldown",
            "stall",
            "obs_sent",
            "action_received",
            "measured_latency_ms",
            "chunk_overlap_count",
            "chunk_mean_l2",
            "chunk_max_l2",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for tick in self._ticks:
                row = {
                    "t": tick.t,
                    "step": tick.step,
                    "schedule_size": tick.schedule_size,
                    "latency_estimate_steps": tick.latency_estimate_steps,
                    "cooldown": tick.cooldown,
                    "stall": tick.stall,
                    "obs_sent": tick.obs_sent,
                    "action_received": tick.action_received,
                    "measured_latency_ms": tick.measured_latency_ms
                    if tick.measured_latency_ms is not None
                    else "",
                    "chunk_overlap_count": tick.chunk_overlap_count
                    if tick.chunk_overlap_count is not None
                    else "",
                    "chunk_mean_l2": tick.chunk_mean_l2
                    if tick.chunk_mean_l2 is not None
                    else "",
                    "chunk_max_l2": tick.chunk_max_l2
                    if tick.chunk_max_l2 is not None
                    else "",
                }
                writer.writerow(row)

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics from collected data."""
        if not self._ticks:
            return {}

        stall_count = sum(t.stall for t in self._ticks)
        total_count = len(self._ticks)

        # Collect L2 discrepancy values (only from ticks where action was received)
        mean_l2_values = [t.chunk_mean_l2 for t in self._ticks if t.chunk_mean_l2 is not None]
        max_l2_values = [t.chunk_max_l2 for t in self._ticks if t.chunk_max_l2 is not None]

        summary: dict[str, Any] = {
            "total_ticks": total_count,
            "stall_count": stall_count,
            "stall_fraction": stall_count / total_count if total_count > 0 else 0.0,
            "obs_sent_count": sum(t.obs_sent for t in self._ticks),
            "action_received_count": sum(t.action_received for t in self._ticks),
        }

        # Add L2 discrepancy summary if we have data
        if mean_l2_values:
            summary["mean_l2_avg"] = sum(mean_l2_values) / len(mean_l2_values)
            summary["mean_l2_max"] = max(mean_l2_values)
            summary["chunk_count"] = len(mean_l2_values)
        if max_l2_values:
            summary["max_l2_max"] = max(max_l2_values)

        return summary
