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
async inference experiments. Also supports trajectory data (action chunks
and executed actions) for visualization.
"""

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


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


@dataclass
class TrajectoryChunk:
    """Recorded action chunk for trajectory visualization."""

    src_action_step: int  # Chunk provenance (observation step that triggered inference)
    actions: list[list[float]]  # (T, A) action chunk as nested list
    frozen_len: int  # Number of frozen actions in this chunk
    t: float  # Timestamp (Unix seconds)


@dataclass
class ExecutedAction:
    """Recorded executed action for trajectory visualization."""

    step: int  # Action step number
    action: list[float]  # Action values (one per joint)
    t: float  # Timestamp (Unix seconds)


class ExperimentMetricsWriter:
    """Collects per-tick experiment metrics and writes to CSV.

    Also collects trajectory data (action chunks and executed actions)
    for post-hoc visualization of how chunks overlap and transition.
    """

    def __init__(self):
        self._ticks: list[ExperimentTick] = []
        self._chunks: list[TrajectoryChunk] = []
        self._executed: list[ExecutedAction] = []

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

    def record_chunk(
        self,
        *,
        src_action_step: int,
        actions: list[np.ndarray] | list[list[float]],
        frozen_len: int,
    ) -> None:
        """Record an action chunk for trajectory visualization.

        Args:
            src_action_step: The observation step that triggered this chunk's inference.
            actions: List of action arrays (T, A) - can be numpy arrays or lists.
            frozen_len: Number of frozen actions in this chunk.
        """
        # Convert numpy arrays to lists for JSON serialization
        actions_list: list[list[float]] = []
        for action in actions:
            if isinstance(action, np.ndarray):
                actions_list.append(action.tolist())
            else:
                actions_list.append(list(action))

        chunk = TrajectoryChunk(
            src_action_step=src_action_step,
            actions=actions_list,
            frozen_len=frozen_len,
            t=time.time(),
        )
        self._chunks.append(chunk)

    def record_executed_action(
        self,
        *,
        step: int,
        action: np.ndarray | list[float],
    ) -> None:
        """Record an executed action for trajectory visualization.

        Args:
            step: The action step number.
            action: The action values sent to the robot.
        """
        if isinstance(action, np.ndarray):
            action_list = action.tolist()
        else:
            action_list = list(action)

        executed = ExecutedAction(
            step=step,
            action=action_list,
            t=time.time(),
        )
        self._executed.append(executed)

    def flush(self, path: str | Path) -> None:
        """Write collected metrics to CSV file and trajectory data to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write per-tick metrics to CSV
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

        # Write trajectory data to JSON (alongside CSV)
        if self._chunks or self._executed:
            trajectory_path = path.with_suffix(".trajectory.json")
            trajectory_data = {
                "chunks": [
                    {
                        "source_step": c.src_action_step,  # JSON key stays for compat
                        "actions": c.actions,
                        "frozen_len": c.frozen_len,
                        "t": c.t,
                    }
                    for c in self._chunks
                ],
                "executed": [
                    {
                        "step": e.step,
                        "action": e.action,
                        "t": e.t,
                    }
                    for e in self._executed
                ],
            }
            with open(trajectory_path, "w") as f:
                json.dump(trajectory_data, f)

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

        # Add trajectory stats
        summary["trajectory_chunks"] = len(self._chunks)
        summary["trajectory_executed"] = len(self._executed)

        return summary
