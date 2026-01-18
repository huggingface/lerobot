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

"""Diagnostics utilities for async inference.

Provides queue-based, lock-free diagnostics for timing, latency jitter,
and action delta tracking.
"""

import logging
import threading
import time
from collections import deque
from queue import Empty, Full, Queue
from typing import Callable, NamedTuple, Protocol

import numpy as np


# =============================================================================
# Diagnostics Event Types
# =============================================================================


class EvLoopDt(NamedTuple):
    """Loop iteration timing event."""

    dt_ms: float


class EvSendAction(NamedTuple):
    """Action send timing event."""

    send_ms: float
    send_t: float


class EvActionDelta(NamedTuple):
    """Action delta metrics event."""

    mean_abs: float
    max_abs: float
    l2: float


class EvActionReceiver(NamedTuple):
    """Action receiver timing event."""

    rpc_ms: float
    deser_ms: float
    latency_ms: float


class EvObsSender(NamedTuple):
    """Observation sender timing event."""

    capture_ms: float
    encode_ms: float
    send_ms: float


class EvOverrun(NamedTuple):
    """Control loop overrun event."""

    pass


class EvLogContext(NamedTuple):
    """Log context for periodic diagnostics."""

    step: int
    schedule_size: int
    latency_steps: int
    cooldown: int


class EvStarvation(NamedTuple):
    """Schedule starvation event (sched=0)."""

    is_starved: bool  # True if schedule is empty this tick


class EvObsWait(NamedTuple):
    """Observation sender wait time event."""

    wait_ms: float  # Time waiting on mailbox.get()


class EvLoopPhases(NamedTuple):
    """Control loop phase timing event."""

    exec_ms: float  # Time executing action (Step 1)
    trigger_ms: float  # Time checking inference trigger (Step 2)
    merge_ms: float  # Time checking mailbox + merging (Step 3)


class EvRtcBuild(NamedTuple):
    """RTC metadata construction timing event."""

    build_ms: float  # Time to construct frozen prefix payload


class EvChunkGap(NamedTuple):
    """Action chunk arrival gap timing event."""

    gap_ms: float  # Time since last chunk arrival


class EvActionChunk(NamedTuple):
    """Action chunk for trajectory visualization.

    Used to visualize per-motor trajectories in real-time, showing how
    action chunks overlap and transition (for RTC inpainting assessment).
    """

    source_step: int  # Chunk provenance (observation step that triggered inference)
    actions: list[list[float]]  # (T, A) action chunk as nested list
    frozen_len: int  # Number of frozen actions in this chunk
    timestamp: float  # Arrival time (time.time())
    # RTC visualization fields (optional)
    rtc_params: dict | None = None  # {d, H, overlap_end, sigma_d, schedule, max_gw, ...}
    prefix_weights: list[float] | None = None  # c_i weights [0,1] for each action


class EvExecutedAction(NamedTuple):
    """Single executed action for trajectory visualization.

    Used to visualize the actual actions sent to the robot, for comparison
    with predicted action chunks.
    """

    step: int  # Action step number
    action: list[float]  # Action values (one per joint)
    timestamp: float  # Execution time (time.time())


# =============================================================================
# Utility Functions
# =============================================================================


def format_p50_p95_max(values: list[float]) -> str:
    """Format percentile summary for a list of values."""
    if not values:
        return "n/a"
    arr = np.asarray(values, dtype=np.float64)
    p50 = float(np.quantile(arr, 0.50))
    p95 = float(np.quantile(arr, 0.95))
    vmax = float(np.max(arr))
    return f"{p50:.2f}/{p95:.2f}/{vmax:.2f}"


# =============================================================================
# Rolling Window
# =============================================================================


class RollingWindow:
    """A simple rolling window of floats with percentile summaries."""

    def __init__(self, maxlen: int):
        self._buf: deque[float] = deque(maxlen=maxlen)

    def add(self, x: float) -> None:
        self._buf.append(float(x))

    def snapshot(self) -> list[float]:
        return list(self._buf)


# =============================================================================
# Diagnostics Config Protocol
# =============================================================================


class DiagnosticsConfigProtocol(Protocol):
    """Protocol for diagnostics configuration."""

    fps: int
    diagnostics_window_s: float
    diagnostics_interval_s: float


# =============================================================================
# Diagnostics Queue
# =============================================================================


class DiagnosticsQueue:
    """Lock-free producer API for diagnostics. Lossy under contention.

    This class provides a queue-based diagnostics system where producers
    (multiple threads) can emit events without blocking, and a single
    consumer thread aggregates and logs statistics periodically.
    """

    def __init__(self, cfg: DiagnosticsConfigProtocol, shutdown_event: threading.Event):
        self._fps = cfg.fps
        self._diagnostics_window_s = cfg.diagnostics_window_s
        self._diagnostics_interval_s = cfg.diagnostics_interval_s
        self._shutdown_event = shutdown_event
        self._queue: Queue = Queue(maxsize=2048)
        self._consumer_thread: threading.Thread | None = None

        # Callback for action chunk events (used by trajectory visualization)
        self._action_chunk_callback: Callable[[EvActionChunk], None] | None = None
        # Callback for sending chunks to external WebSocket server
        self._ws_sender_callback: Callable[[EvActionChunk], None] | None = None
        # Callback for executed action events (used by trajectory visualization)
        self._executed_action_callback: Callable[["EvExecutedAction"], None] | None = None

    def start_consumer(self, logger: logging.Logger) -> None:
        """Start the background consumer thread."""
        self._consumer_thread = threading.Thread(
            target=self._consumer_loop,
            args=(logger,),
            name="diagnostics_consumer",
            daemon=True,
        )
        self._consumer_thread.start()

    def stop(self) -> None:
        """Signal consumer to stop."""
        pass

    def reset(self) -> None:
        """Clear the queue for a new experiment."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Empty:
                break

    def _emit(self, event) -> None:
        """Non-blocking emit: drop if queue is full."""
        try:
            self._queue.put_nowait(event)
        except Full:
            pass

    def emit_loop_dt_ms(self, dt_ms: float) -> None:
        self._emit(EvLoopDt(dt_ms))

    def emit_send_action(self, send_ms: float, send_t: float) -> None:
        self._emit(EvSendAction(send_ms, send_t))

    def emit_action_delta(self, prev: np.ndarray, cur: np.ndarray) -> None:
        if prev.shape != cur.shape:
            return
        d = (cur - prev).astype(np.float64, copy=False)
        mean_abs = float(np.mean(np.abs(d)))
        max_abs = float(np.max(np.abs(d)))
        l2 = float(np.linalg.norm(d.ravel(), ord=2))
        self._emit(EvActionDelta(mean_abs, max_abs, l2))

    def emit_action_receiver(self, rpc_ms: float, deser_ms: float, latency_ms: float) -> None:
        self._emit(EvActionReceiver(rpc_ms, deser_ms, latency_ms))

    def emit_observation_sender(self, capture_ms: float, encode_ms: float, send_ms: float) -> None:
        self._emit(EvObsSender(capture_ms, encode_ms, send_ms))

    def emit_overrun(self) -> None:
        self._emit(EvOverrun())

    def emit_log_context(self, step: int, schedule_size: int, latency_steps: int, cooldown: int) -> None:
        self._emit(EvLogContext(step, schedule_size, latency_steps, cooldown))

    def emit_starvation(self, is_starved: bool) -> None:
        self._emit(EvStarvation(is_starved))

    def emit_obs_wait(self, wait_ms: float) -> None:
        self._emit(EvObsWait(wait_ms))

    def emit_loop_phases(self, exec_ms: float, trigger_ms: float, merge_ms: float) -> None:
        self._emit(EvLoopPhases(exec_ms, trigger_ms, merge_ms))

    def emit_rtc_build(self, build_ms: float) -> None:
        self._emit(EvRtcBuild(build_ms))

    def emit_chunk_gap(self, gap_ms: float) -> None:
        self._emit(EvChunkGap(gap_ms))

    def set_action_chunk_callback(self, callback: Callable[["EvActionChunk"], None] | None) -> None:
        """Set callback for action chunk events (used by trajectory visualization)."""
        self._action_chunk_callback = callback

    def set_ws_sender_callback(self, callback: Callable[["EvActionChunk"], None] | None) -> None:
        """Set callback for sending chunks to external WebSocket server."""
        self._ws_sender_callback = callback

    def emit_action_chunk(
        self,
        source_step: int,
        actions: list[list[float]],
        frozen_len: int,
        rtc_params: dict | None = None,
        prefix_weights: list[float] | None = None,
    ) -> None:
        """Emit an action chunk event for trajectory visualization."""
        event = EvActionChunk(
            source_step=source_step,
            actions=actions,
            frozen_len=frozen_len,
            timestamp=time.time(),
            rtc_params=rtc_params,
            prefix_weights=prefix_weights,
        )
        # Forward to callback if set (for real-time visualization)
        if self._action_chunk_callback is not None:
            try:
                self._action_chunk_callback(event)
            except Exception:
                pass  # Don't let visualization errors affect control loop
        # Forward to WebSocket sender if set
        if self._ws_sender_callback is not None:
            try:
                self._ws_sender_callback(event)
            except Exception:
                pass
        # Also emit to queue for logging
        self._emit(event)

    def set_executed_action_callback(
        self, callback: Callable[["EvExecutedAction"], None] | None
    ) -> None:
        """Set callback for executed action events (used by trajectory visualization)."""
        self._executed_action_callback = callback

    def emit_executed_action(self, step: int, action: list[float]) -> None:
        """Emit an executed action event for trajectory visualization."""
        event = EvExecutedAction(
            step=step,
            action=action,
            timestamp=time.time(),
        )
        # Forward to callback if set (for real-time visualization)
        if self._executed_action_callback is not None:
            try:
                self._executed_action_callback(event)
            except Exception:
                pass  # Don't let visualization errors affect control loop
        # Also emit to queue for logging
        self._emit(event)

    def _consumer_loop(self, logger: logging.Logger) -> None:
        """Background thread: drain queue, aggregate stats, emit periodic logs."""
        maxlen = max(10, int(self._fps * self._diagnostics_window_s))

        # Rolling windows (owned exclusively by consumer thread - no lock needed)
        loop_dt_ms = RollingWindow(maxlen=maxlen)
        send_action_ms = RollingWindow(maxlen=maxlen)
        send_action_dt_ms = RollingWindow(maxlen=maxlen)
        action_delta_mean_abs = RollingWindow(maxlen=maxlen)
        action_delta_max_abs = RollingWindow(maxlen=maxlen)
        action_delta_l2 = RollingWindow(maxlen=maxlen)
        rpc_ms = RollingWindow(maxlen=maxlen)
        deser_ms = RollingWindow(maxlen=maxlen)
        measured_latency_ms = RollingWindow(maxlen=maxlen)
        obs_capture_ms = RollingWindow(maxlen=maxlen)
        obs_encode_ms = RollingWindow(maxlen=maxlen)
        obs_send_ms = RollingWindow(maxlen=maxlen)

        # New granular timing rolling windows
        obs_wait_ms = RollingWindow(maxlen=maxlen)
        phase_exec_ms = RollingWindow(maxlen=maxlen)
        phase_trigger_ms = RollingWindow(maxlen=maxlen)
        phase_merge_ms = RollingWindow(maxlen=maxlen)
        rtc_build_ms = RollingWindow(maxlen=maxlen)
        chunk_gap_ms = RollingWindow(maxlen=maxlen)

        overrun_count = 0
        last_emit_t = time.perf_counter()
        last_action_send_t: float | None = None
        latest_ctx: EvLogContext | None = None

        # Starvation tracking (count and current streak)
        starvation_count = 0
        starvation_streak = 0
        starvation_streak_max = 0

        while not self._shutdown_event.is_set():
            # Drain queue with timeout
            try:
                event = self._queue.get(timeout=0.1)
            except Empty:
                continue

            # Dispatch event by type
            if isinstance(event, EvLoopDt):
                loop_dt_ms.add(event.dt_ms)
            elif isinstance(event, EvSendAction):
                send_action_ms.add(event.send_ms)
                if last_action_send_t is not None:
                    send_action_dt_ms.add((event.send_t - last_action_send_t) * 1000.0)
                last_action_send_t = event.send_t
            elif isinstance(event, EvActionDelta):
                action_delta_mean_abs.add(event.mean_abs)
                action_delta_max_abs.add(event.max_abs)
                action_delta_l2.add(event.l2)
            elif isinstance(event, EvActionReceiver):
                rpc_ms.add(event.rpc_ms)
                deser_ms.add(event.deser_ms)
                measured_latency_ms.add(event.latency_ms)
            elif isinstance(event, EvObsSender):
                obs_capture_ms.add(event.capture_ms)
                obs_encode_ms.add(event.encode_ms)
                obs_send_ms.add(event.send_ms)
            elif isinstance(event, EvOverrun):
                overrun_count += 1
            elif isinstance(event, EvLogContext):
                latest_ctx = event
            elif isinstance(event, EvStarvation):
                if event.is_starved:
                    starvation_count += 1
                    starvation_streak += 1
                    starvation_streak_max = max(starvation_streak_max, starvation_streak)
                else:
                    starvation_streak = 0
            elif isinstance(event, EvObsWait):
                obs_wait_ms.add(event.wait_ms)
            elif isinstance(event, EvLoopPhases):
                phase_exec_ms.add(event.exec_ms)
                phase_trigger_ms.add(event.trigger_ms)
                phase_merge_ms.add(event.merge_ms)
            elif isinstance(event, EvRtcBuild):
                rtc_build_ms.add(event.build_ms)
            elif isinstance(event, EvChunkGap):
                chunk_gap_ms.add(event.gap_ms)
            elif isinstance(event, EvActionChunk):
                # Action chunk events are handled by callback, just skip in consumer
                pass
            elif isinstance(event, EvExecutedAction):
                # Executed action events are handled by callback, just skip in consumer
                pass

            # Periodic logging (only when we have context)
            now = time.perf_counter()
            if (now - last_emit_t) >= self._diagnostics_interval_s and latest_ctx is not None:
                last_emit_t = now
                logger.info(
                    "DIAG | step=%s sched=%s cooldown=%s latency_steps=%s | "
                    "loop_dt_ms(p50/p95/max)=%s | phases(exec/trig/merge)=%s/%s/%s | "
                    "send_action_ms=%s | send_action_dt_ms=%s | "
                    "d_mean_abs=%s d_max_abs=%s d_l2=%s | "
                    "rpc_ms=%s deser_ms=%s latency_ms=%s | "
                    "obs_wait_ms=%s obs_cap_ms=%s obs_enc_ms=%s obs_send_ms=%s | "
                    "rtc_build_ms=%s chunk_gap_ms=%s | "
                    "starvation=%s/%s overruns=%s",
                    latest_ctx.step,
                    latest_ctx.schedule_size,
                    latest_ctx.cooldown,
                    latest_ctx.latency_steps,
                    format_p50_p95_max(loop_dt_ms.snapshot()),
                    format_p50_p95_max(phase_exec_ms.snapshot()),
                    format_p50_p95_max(phase_trigger_ms.snapshot()),
                    format_p50_p95_max(phase_merge_ms.snapshot()),
                    format_p50_p95_max(send_action_ms.snapshot()),
                    format_p50_p95_max(send_action_dt_ms.snapshot()),
                    format_p50_p95_max(action_delta_mean_abs.snapshot()),
                    format_p50_p95_max(action_delta_max_abs.snapshot()),
                    format_p50_p95_max(action_delta_l2.snapshot()),
                    format_p50_p95_max(rpc_ms.snapshot()),
                    format_p50_p95_max(deser_ms.snapshot()),
                    format_p50_p95_max(measured_latency_ms.snapshot()),
                    format_p50_p95_max(obs_wait_ms.snapshot()),
                    format_p50_p95_max(obs_capture_ms.snapshot()),
                    format_p50_p95_max(obs_encode_ms.snapshot()),
                    format_p50_p95_max(obs_send_ms.snapshot()),
                    format_p50_p95_max(rtc_build_ms.snapshot()),
                    format_p50_p95_max(chunk_gap_ms.snapshot()),
                    starvation_count,
                    starvation_streak_max,
                    overrun_count,
                )
