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

"""Metrics collection for async inference.

This module provides two categories of metrics:
- **experiment**: per-tick metrics + trajectory data written to disk (CSV + JSON).
  This is the current/default behavior and should remain stable for experiments.
- **diagnostic**: lightweight timing/counter summaries printed to the console
  (avg/max only; no percentiles).
"""

from __future__ import annotations

import csv
import json
import logging
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Trajectory-viz event types (runtime streaming)
# =============================================================================


class EvActionChunk(tuple):
    """Action chunk event for real-time trajectory visualization.

    Kept as a lightweight tuple-like object for low-overhead passing between threads.
    """

    __slots__ = ()
    _fields = ("src_control_step", "actions", "frozen_len", "timestamp", "rtc_params", "prefix_weights")

    def __new__(
        cls,
        *,
        src_control_step: int,
        actions: list[list[float]],
        frozen_len: int,
        timestamp: float,
        rtc_params: dict | None = None,
        prefix_weights: list[float] | None = None,
    ):
        return tuple.__new__(
            cls,
            (src_control_step, actions, frozen_len, timestamp, rtc_params, prefix_weights),
        )

    @property
    def src_control_step(self) -> int:  # noqa: D401
        return self[0]

    @property
    def actions(self) -> list[list[float]]:
        return self[1]

    @property
    def frozen_len(self) -> int:
        return self[2]

    @property
    def timestamp(self) -> float:
        return self[3]

    @property
    def rtc_params(self) -> dict | None:
        return self[4]

    @property
    def prefix_weights(self) -> list[float] | None:
        return self[5]


class EvExecutedAction(tuple):
    """Single executed action event for real-time visualization."""

    __slots__ = ()
    _fields = ("step", "action", "timestamp")

    def __new__(cls, *, step: int, action: list[float], timestamp: float):
        return tuple.__new__(cls, (step, action, timestamp))

    @property
    def step(self) -> int:
        return self[0]

    @property
    def action(self) -> list[float]:
        return self[1]

    @property
    def timestamp(self) -> float:
        return self[2]


# =============================================================================
# Experiment metrics (disk output; keep stable)
# =============================================================================


@dataclass
class ExperimentTick:
    """Single tick of experiment data."""

    t: float  # Wall-clock timestamp (Unix seconds)
    step: int  # Action step n(t)
    schedule_size: int  # |ψ(t)|
    latency_estimate_steps: int  # ℓ̂_Δ
    latency_estimate_ms: float  # ℓ̂ in milliseconds (unquantized)
    cooldown: int  # O^c(t)
    stall: int  # 1 if schedule_size == 0, else 0
    obs_triggered: int  # 1 if obs request triggered this tick
    action_received: int  # 1 if action chunk merged this tick
    measured_latency_ms: float | None  # RTT of received chunk (if any)
    # Raw wall-clock timestamps for the round-trip journey (Unix seconds).
    # Durations can be derived: c2s = server_obs_received_ts - obs_sent_ts, etc.
    obs_sent_ts: float | None  # Client observation send timestamp
    server_obs_received_ts: float | None  # Server observation receive timestamp
    server_action_sent_ts: float | None  # Server action send timestamp
    action_received_ts: float | None  # Client action receive timestamp
    # Action discontinuity metrics (L2 distance between overlapping chunks)
    chunk_overlap_count: int | None  # Number of overlapping actions compared
    chunk_mean_l2: float | None  # Mean L2 distance across overlapping actions
    chunk_max_l2: float | None  # Max L2 distance across overlapping actions


@dataclass
class TrajectoryChunk:
    """Recorded action chunk for trajectory visualization."""

    src_control_step: int  # Chunk provenance (control step t that triggered inference)
    actions: list[list[float]]  # (T, A) action chunk as nested list
    frozen_len: int  # Number of frozen actions in this chunk
    t: float  # Timestamp (Unix seconds)
    chunk_start_step: int | None = None  # Start step of this chunk (provenance)


@dataclass
class ExecutedAction:
    """Recorded executed action for trajectory visualization."""

    step: int  # Action step number
    action: list[float]  # Action values (one per joint)
    src_control_step: int  # Control step t that produced this action (provenance)
    chunk_start_step: int  # Start step of the source chunk (provenance)
    t: float  # Timestamp (Unix seconds)


@dataclass
class SimEvent:
    """A recorded simulation event (drop, reorder, duplicate, etc.)."""

    event_type: str  # e.g. "obs_dropped", "action_dropped", "obs_reorder_held", etc.
    t: float  # Timestamp (Unix seconds)


@dataclass
class RegisterEvent:
    """A recorded LWW register write attempt (client-side)."""

    t: float  # Wall-clock timestamp (Unix seconds)
    register_name: str  # e.g. "client_obs_request", "client_action"
    control_step: int  # The control_step used as the LWW key
    chunk_start_step: int | None  # Only meaningful for action registers
    accepted: bool  # Whether update_if_newer accepted the write


class ExperimentMetricsWriter:
    """Collects per-tick experiment metrics and writes to CSV.

    Also collects trajectory data (action chunks and executed actions)
    for post-hoc visualization of how chunks overlap and transition.

    Memory is bounded:
    - ``_ticks`` are auto-flushed to CSV when the buffer exceeds
      ``auto_flush_threshold`` (default 50 000 ≈ 16 min @ 50 Hz).
    - ``_chunks`` and ``_executed`` use bounded deques (most-recent data
      kept; oldest evicted).
    - Running summary counters survive auto-flushes so ``get_summary()``
      always covers the full run.
    """

    _CSV_FIELDNAMES = [
        "t",
        "step",
        "schedule_size",
        "latency_estimate_steps",
        "latency_estimate_ms",
        "cooldown",
        "stall",
        "obs_triggered",
        "action_received",
        "measured_latency_ms",
        "obs_sent_ts",
        "server_obs_received_ts",
        "server_action_sent_ts",
        "action_received_ts",
        "chunk_overlap_count",
        "chunk_mean_l2",
        "chunk_max_l2",
    ]

    def __init__(
        self,
        path: str | Path | None = None,
        auto_flush_threshold: int = 50_000,
        max_trajectory_entries: int = 10_000,
        simulation_config: dict | None = None,
        experiment_config: dict | None = None,
    ):
        self._path: Path | None = Path(path) if path else None
        self._auto_flush_threshold = auto_flush_threshold
        self._simulation_config: dict = simulation_config or {}
        self._experiment_config: dict = experiment_config or {}

        # Lock to serialise flush operations.  signal_stop() and stop()
        # can both call flush() from different threads; without a lock the
        # same ticks are written twice (once in "w" mode, once in "a") and
        # the resulting CSV contains duplicate/corrupted rows.
        self._flush_lock = threading.Lock()

        # Tick buffer (flushed periodically to CSV)
        self._ticks: list[ExperimentTick] = []
        self._csv_header_written = False

        # Trajectory buffers (bounded deques — most recent data kept)
        self._chunks: deque[TrajectoryChunk] = deque(maxlen=max_trajectory_entries)
        self._executed: deque[ExecutedAction] = deque(maxlen=max_trajectory_entries)

        # Simulation event log (bounded deque)
        self._sim_events: deque[SimEvent] = deque(maxlen=max_trajectory_entries)

        # LWW register event log (bounded deque)
        self._register_events: deque[RegisterEvent] = deque(maxlen=max_trajectory_entries)

        # Running summary counters (survive auto-flushes)
        self._total_ticks: int = 0
        self._total_stalls: int = 0
        self._total_obs_triggered: int = 0
        self._total_action_received: int = 0
        self._l2_count: int = 0
        self._l2_mean_sum: float = 0.0
        self._l2_mean_max: float = 0.0
        self._l2_max_max: float = 0.0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_tick(
        self,
        *,
        step: int,
        schedule_size: int,
        latency_estimate_steps: int,
        latency_estimate_ms: float,
        cooldown: int,
        obs_triggered: bool = False,
        action_received: bool = False,
        measured_latency_ms: float | None = None,
        obs_sent_ts: float | None = None,
        server_obs_received_ts: float | None = None,
        server_action_sent_ts: float | None = None,
        action_received_ts: float | None = None,
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
            latency_estimate_ms=latency_estimate_ms,
            cooldown=cooldown,
            stall=1 if schedule_size == 0 else 0,
            obs_triggered=1 if obs_triggered else 0,
            action_received=1 if action_received else 0,
            measured_latency_ms=measured_latency_ms,
            obs_sent_ts=obs_sent_ts,
            server_obs_received_ts=server_obs_received_ts,
            server_action_sent_ts=server_action_sent_ts,
            action_received_ts=action_received_ts,
            chunk_overlap_count=chunk_overlap_count,
            chunk_mean_l2=chunk_mean_l2,
            chunk_max_l2=chunk_max_l2,
        )
        self._ticks.append(tick)

        # Update running summary counters
        self._total_ticks += 1
        if schedule_size == 0:
            self._total_stalls += 1
        if obs_triggered:
            self._total_obs_triggered += 1
        if action_received:
            self._total_action_received += 1
        if chunk_mean_l2 is not None:
            self._l2_count += 1
            self._l2_mean_sum += chunk_mean_l2
            self._l2_mean_max = max(self._l2_mean_max, chunk_mean_l2)
        if chunk_max_l2 is not None:
            self._l2_max_max = max(self._l2_max_max, chunk_max_l2)

        # Auto-flush when buffer exceeds threshold
        if len(self._ticks) >= self._auto_flush_threshold:
            self._auto_flush_ticks()

    def record_chunk(
        self,
        *,
        src_control_step: int,
        actions: list[np.ndarray] | list[list[float]],
        frozen_len: int,
        chunk_start_step: int | None = None,
    ) -> None:
        """Record an action chunk for trajectory visualization.

        Args:
            src_control_step: The control step t that triggered this chunk's inference.
            actions: List of action arrays (T, A) - can be numpy arrays or lists.
            frozen_len: Number of frozen actions in this chunk.
            chunk_start_step: The start step of this chunk (provenance).
        """
        # Convert numpy arrays to lists for JSON serialization
        actions_list: list[list[float]] = []
        for action in actions:
            if isinstance(action, np.ndarray):
                actions_list.append(action.tolist())
            else:
                actions_list.append(list(action))

        chunk = TrajectoryChunk(
            src_control_step=src_control_step,
            actions=actions_list,
            frozen_len=frozen_len,
            t=time.time(),
            chunk_start_step=chunk_start_step,
        )
        self._chunks.append(chunk)  # deque evicts oldest automatically

    def record_executed_action(
        self,
        *,
        step: int,
        action: np.ndarray | list[float],
        src_control_step: int,
        chunk_start_step: int,
    ) -> None:
        """Record an executed action for trajectory visualization.

        Args:
            step: The action step number.
            action: The action values sent to the robot.
            src_control_step: The control step t that produced this action.
            chunk_start_step: The start step of the source chunk.
        """
        if isinstance(action, np.ndarray):
            action_list = action.tolist()
        else:
            action_list = list(action)

        executed = ExecutedAction(
            step=step,
            action=action_list,
            src_control_step=src_control_step,
            chunk_start_step=chunk_start_step,
            t=time.time(),
        )
        self._executed.append(executed)  # deque evicts oldest automatically

    def record_sim_event(self, event_type: str) -> None:
        """Record a simulation event (drop, reorder, duplicate, etc.).

        Args:
            event_type: Event identifier, e.g. ``"obs_dropped"``, ``"action_dropped"``.
        """
        self._sim_events.append(SimEvent(event_type=event_type, t=time.time()))

    def record_register_event(
        self,
        *,
        register_name: str,
        control_step: int,
        accepted: bool,
        chunk_start_step: int | None = None,
    ) -> None:
        """Record an LWW register write attempt.

        Args:
            register_name: Identifier for the register, e.g. ``"client_obs_request"``.
            control_step: The control_step used as the LWW key.
            accepted: Whether ``update_if_newer`` accepted the write.
            chunk_start_step: Start step of the source chunk (action registers only).
        """
        self._register_events.append(
            RegisterEvent(
                t=time.time(),
                register_name=register_name,
                control_step=control_step,
                chunk_start_step=chunk_start_step,
                accepted=accepted,
            )
        )

    # ------------------------------------------------------------------
    # Flushing
    # ------------------------------------------------------------------

    @staticmethod
    def _tick_to_row(tick: ExperimentTick) -> dict[str, Any]:
        """Convert an ExperimentTick to a CSV-row dict."""
        return {
            "t": tick.t,
            "step": tick.step,
            "schedule_size": tick.schedule_size,
            "latency_estimate_steps": tick.latency_estimate_steps,
            "latency_estimate_ms": tick.latency_estimate_ms,
            "cooldown": tick.cooldown,
            "stall": tick.stall,
            "obs_triggered": tick.obs_triggered,
            "action_received": tick.action_received,
            "measured_latency_ms": tick.measured_latency_ms
            if tick.measured_latency_ms is not None
            else "",
            "obs_sent_ts": tick.obs_sent_ts
            if tick.obs_sent_ts is not None
            else "",
            "server_obs_received_ts": tick.server_obs_received_ts
            if tick.server_obs_received_ts is not None
            else "",
            "server_action_sent_ts": tick.server_action_sent_ts
            if tick.server_action_sent_ts is not None
            else "",
            "action_received_ts": tick.action_received_ts
            if tick.action_received_ts is not None
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

    def _auto_flush_ticks(self) -> None:
        """Incrementally flush buffered ticks to CSV and clear the buffer.

        Thread-safe: uses ``_flush_lock`` so that concurrent calls from
        ``signal_stop()`` (timer thread) and ``stop()`` (main thread) are
        serialised and each tick is written exactly once.
        """
        with self._flush_lock:
            if not self._path or not self._ticks:
                return
            self._path.parent.mkdir(parents=True, exist_ok=True)

            # Snapshot and clear under lock so no tick is written twice.
            ticks_to_write = list(self._ticks)
            self._ticks.clear()

            mode = "a" if self._csv_header_written else "w"
            with open(self._path, mode, newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._CSV_FIELDNAMES)
                if not self._csv_header_written:
                    writer.writeheader()
                    self._csv_header_written = True
                for tick in ticks_to_write:
                    writer.writerow(self._tick_to_row(tick))

    def flush(self, path: str | Path | None = None) -> None:
        """Write remaining metrics to disk.

        Args:
            path: Override path (updates the stored path). If *None*, uses
                  the path provided at construction time.
        """
        if path is not None:
            self._path = Path(path)
        if self._path is None:
            return

        # Drain any remaining ticks to CSV
        self._auto_flush_ticks()

        # Write trajectory data to JSON (from bounded deques)
        if self._chunks or self._executed:
            trajectory_path = self._path.with_suffix(".trajectory.json")
            trajectory_data: dict[str, Any] = {
                "chunks": [
                    {
                        "source_step": c.src_control_step,
                        "actions": c.actions,
                        "frozen_len": c.frozen_len,
                        "t": c.t,
                        "chunk_start_step": c.chunk_start_step,
                    }
                    for c in self._chunks
                ],
                "executed": [
                    {
                        "step": e.step,
                        "action": e.action,
                        "src_control_step": e.src_control_step,
                        "chunk_start_step": e.chunk_start_step,
                        "t": e.t,
                    }
                    for e in self._executed
                ],
                "sim_events": [
                    {"event_type": ev.event_type, "t": ev.t}
                    for ev in self._sim_events
                ],
                "register_events": [
                    {
                        "t": rev.t,
                        "register_name": rev.register_name,
                        "control_step": rev.control_step,
                        "chunk_start_step": rev.chunk_start_step,
                        "accepted": rev.accepted,
                    }
                    for rev in self._register_events
                ],
            }
            # Include the full simulation config so the plotter can overlay
            # configured windows/events (drops, spikes, duplicates, reorder).
            if self._simulation_config:
                trajectory_data["simulation_config"] = self._simulation_config
            # Include experiment config (policy, latency, filter params) for
            # the plotter to render a configuration table in the LaTeX output.
            if self._experiment_config:
                trajectory_data["experiment_config"] = self._experiment_config
            with open(trajectory_path, "w") as f:
                json.dump(trajectory_data, f)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics from collected data.

        Uses running counters so the summary covers the full run even
        after auto-flushes have cleared the tick buffer.
        """
        if self._total_ticks == 0:
            return {}

        summary: dict[str, Any] = {
            "total_ticks": self._total_ticks,
            "stall_count": self._total_stalls,
            "stall_fraction": self._total_stalls / self._total_ticks,
            "obs_triggered_count": self._total_obs_triggered,
            "action_received_count": self._total_action_received,
        }

        # Add L2 discrepancy summary if we have data
        if self._l2_count > 0:
            summary["mean_l2_avg"] = self._l2_mean_sum / self._l2_count
            summary["mean_l2_max"] = self._l2_mean_max
            summary["chunk_count"] = self._l2_count
            summary["max_l2_max"] = self._l2_max_max

        # Add trajectory stats (bounded deque sizes, not full-run counts)
        summary["trajectory_chunks"] = len(self._chunks)
        summary["trajectory_executed"] = len(self._executed)

        return summary


# =============================================================================
# Diagnostic metrics (console output; avg/max only)
# =============================================================================


def _format_avg_max(values: list[float]) -> str:
    if not values:
        return "n/a"
    avg = float(sum(values) / len(values))
    vmax = float(max(values))
    return f"{avg:.2f}/{vmax:.2f}"


class _EvTiming(tuple):
    __slots__ = ()

    def __new__(cls, name: str, ms: float):
        return tuple.__new__(cls, (name, float(ms)))

    @property
    def name(self) -> str:
        return self[0]

    @property
    def ms(self) -> float:
        return self[1]


class _EvCounter(tuple):
    __slots__ = ()

    def __new__(cls, name: str, inc: int):
        return tuple.__new__(cls, (name, int(inc)))

    @property
    def name(self) -> str:
        return self[0]

    @property
    def inc(self) -> int:
        return self[1]


class _EvContext(tuple):
    __slots__ = ()

    def __new__(cls, ctx: dict[str, Any]):
        return tuple.__new__(cls, (ctx,))

    @property
    def ctx(self) -> dict[str, Any]:
        return self[0]


class DiagnosticMetrics:
    """Lossy, queue-based diagnostic metrics with periodic console summaries."""

    def __init__(
        self,
        *,
        fps: int,
        window_s: float = 10.0,
        interval_s: float = 2.0,
        enabled: bool = True,
        verbose: bool = False,
        prefix: str = "DIAG",
    ):
        self._enabled = bool(enabled)
        self._fps = int(fps)
        self._window_s = float(window_s)
        self._interval_s = float(interval_s)
        self._verbose = bool(verbose)
        self._prefix = str(prefix)

        self._shutdown = threading.Event()
        self._queue: Queue = Queue(maxsize=4096)
        self._thread: threading.Thread | None = None

    @staticmethod
    def _ms(seconds: float) -> float:
        return seconds * 1000.0

    def start(self) -> None:
        if not self._enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._consumer_loop,
            name="metrics_diagnostic_consumer",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_s: float = 1.0) -> None:
        if not self._enabled:
            return
        self._shutdown.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)

    def timing_ms(self, name: str, ms: float) -> None:
        if not self._enabled:
            return
        try:
            self._queue.put_nowait(_EvTiming(str(name), float(ms)))
        except Full:
            pass

    def timing_s(self, name: str, seconds: float) -> None:
        self.timing_ms(name, self._ms(seconds))

    def counter(self, name: str, inc: int = 1) -> None:
        if not self._enabled:
            return
        try:
            self._queue.put_nowait(_EvCounter(str(name), int(inc)))
        except Full:
            pass

    def set_context(self, **ctx: Any) -> None:
        if not self._enabled:
            return
        try:
            self._queue.put_nowait(_EvContext(dict(ctx)))
        except Full:
            pass

    @contextmanager
    def time_block(self, name: str):
        if not self._enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.timing_s(name, time.perf_counter() - t0)

    def _consumer_loop(self) -> None:
        maxlen = max(10, int(self._fps * self._window_s))
        timings: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=maxlen))
        counters_total: dict[str, int] = defaultdict(int)
        latest_ctx: dict[str, Any] = {}

        last_emit = time.perf_counter()
        while not self._shutdown.is_set():
            try:
                ev = self._queue.get(timeout=0.1)
            except Empty:
                ev = None

            if isinstance(ev, _EvTiming):
                timings[ev.name].append(ev.ms)
            elif isinstance(ev, _EvCounter):
                counters_total[ev.name] += ev.inc
            elif isinstance(ev, _EvContext):
                latest_ctx = ev.ctx

            now = time.perf_counter()
            if (now - last_emit) < self._interval_s:
                continue

            last_emit = now

            # Default: compact summary of core fields + total RTT.
            core_keys = ["step", "schedule_size", "latency_steps", "cooldown", "chunk_size", "s_min", "fps"]
            core_ctx = " ".join(f"{k}={latest_ctx[k]}" for k in core_keys if k in latest_ctx)

            rtt_key = "total_latency_rtt_ms"
            rtt_part = ""
            if rtt_key in timings:
                rtt_part = f"{rtt_key}(avg/max)={_format_avg_max(list(timings[rtt_key]))}"
            else:
                rtt_part = f"{rtt_key}(avg/max)=n/a"

            if not self._verbose:
                # Skip emit when there is no data yet (startup period before
                # the control loop or action receiver have produced any events).
                if not core_ctx and rtt_key not in timings:
                    continue
                parts = [p for p in [core_ctx, rtt_part] if p]
                logger.info(f"{self._prefix} | " + " ".join(parts))
                continue

            # Verbose: include all context keys plus timing/counter details.
            # Skip emit when there is no data yet (startup period).
            if not latest_ctx and not timings and not counters_total:
                continue
            ctx_part = " ".join(f"{k}={v}" for k, v in latest_ctx.items())

            # Prefer a stable ordering for common names; append others alphabetically.
            preferred = [
                rtt_key,
                "loop_dt_ms",
                "phase_exec_ms",
                "phase_trigger_ms",
                "phase_merge_ms",
                "send_action_ms",
                "obs_wait_ms",
                "obs_capture_ms",
                "obs_encode_ms",
                "obs_send_ms",
                "rpc_ms",
                "deser_ms",
                "rtc_build_ms",
                "chunk_gap_ms",
                "policy_predict_ms",
                "infer_total_ms",
                "policy_load_ms",
                "obs_recv_ms",
                "obs_decode_ms",
            ]
            remaining = sorted([k for k in timings.keys() if k not in preferred])
            keys = [k for k in preferred if k in timings] + remaining

            timing_part = " ".join(f"{k}(avg/max)={_format_avg_max(list(timings[k]))}" for k in keys)
            counter_part = " ".join(f"{k}={v}" for k, v in sorted(counters_total.items()))

            parts = [p for p in [ctx_part, timing_part, counter_part] if p]
            logger.info(f"{self._prefix} | " + " | ".join(parts))


@dataclass
class Metrics:
    """Bundle of experiment (disk) + diagnostic (console) metrics."""

    experiment: ExperimentMetricsWriter | None = None
    diagnostic: DiagnosticMetrics | None = None

