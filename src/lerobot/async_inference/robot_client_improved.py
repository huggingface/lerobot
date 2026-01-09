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

"""
Latency-Adaptive Asynchronous Inference Robot Client

This implementation follows the latency-adaptive async inference algorithm with:
- Jacobson-Karels latency estimation
- SPSC (single-producer/single-consumer) one-slot mailboxes
- Cool-down mechanism for inference triggering
- Frozen action invariant
- Freshest-observation-wins merging strategy

Threading model (3 threads):
- Main thread: control loop, executes actions, checks inference condition
- Observation sender thread: captures, encodes, sends observations
- Action receiver thread: receives action chunks from server

Example command:
```shell
python src/lerobot/async_inference/robot_client_improved.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --task="dummy" \
    --server_address=127.0.0.1:8080 \
    --policy_type=act \
    --pretrained_name_or_path=user/model \
    --policy_device=mps \
    --actions_per_chunk=50
```
"""

# ruff: noqa: E402, I001

import os as _os
import sys as _sys
import time as _time

_IMPORT_TIMING_ENABLED = _os.getenv("LEROBOT_IMPORT_TIMING", "0") == "1"
_IMPORT_T0 = _time.perf_counter() if _IMPORT_TIMING_ENABLED else 0.0

import logging
import math
import pickle  # nosec
import threading
import time
from collections import OrderedDict, deque
from contextlib import suppress
from dataclasses import dataclass, field
from pprint import pformat
from queue import Empty, Full, Queue
from typing import Any

import cv2  # type: ignore
import numpy as np
import grpc

from lerobot.robots.utils import make_robot_from_config
from lerobot.robots.config import RobotConfig
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks

from .constants import DEFAULT_FPS, SUPPORTED_ROBOTS
from .helpers import (
    FPSTracker,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
    visualize_action_queue_size,
)

if _IMPORT_TIMING_ENABLED:
    _sys.stderr.write(
        f"[import-timing] {__name__} imports: {(_time.perf_counter() - _IMPORT_T0) * 1000.0:.2f}ms\n"
    )


# Sentinel value for shutdown signaling (must not conflict with valid action steps)
_SHUTDOWN_SENTINEL = -999999


# =============================================================================
# Diagnostics (optional) - Queue-based, lock-free for producers
# =============================================================================

from typing import NamedTuple


class _EvLoopDt(NamedTuple):
    """Loop iteration timing event."""
    dt_ms: float


class _EvSendAction(NamedTuple):
    """Action send timing event."""
    send_ms: float
    send_t: float


class _EvActionDelta(NamedTuple):
    """Action delta metrics event."""
    mean_abs: float
    max_abs: float
    l2: float


class _EvActionReceiver(NamedTuple):
    """Action receiver timing event."""
    rpc_ms: float
    deser_ms: float
    latency_ms: float


class _EvObsSender(NamedTuple):
    """Observation sender timing event."""
    capture_ms: float
    encode_ms: float
    send_ms: float


class _EvOverrun(NamedTuple):
    """Control loop overrun event."""
    pass


class _EvLogContext(NamedTuple):
    """Log context for periodic diagnostics."""
    step: int
    schedule_size: int
    latency_steps: int
    cooldown: int


def _format_p50_p95_max(values: list[float]) -> str:
    if not values:
        return "n/a"
    arr = np.asarray(values, dtype=np.float64)
    p50 = float(np.quantile(arr, 0.50))
    p95 = float(np.quantile(arr, 0.95))
    vmax = float(np.max(arr))
    return f"{p50:.2f}/{p95:.2f}/{vmax:.2f}"


class _RollingWindow:
    """A simple rolling window of floats with percentile summaries."""

    def __init__(self, maxlen: int):
        self._buf: deque[float] = deque(maxlen=maxlen)

    def add(self, x: float) -> None:
        self._buf.append(float(x))

    def snapshot(self) -> list[float]:
        return list(self._buf)


class _DiagnosticsQueue:
    """Lock-free producer API for diagnostics. Lossy under contention."""

    def __init__(self, cfg: "RobotClientImprovedConfig", shutdown_event: threading.Event):
        self._cfg = cfg
        self._shutdown_event = shutdown_event
        self._queue: Queue = Queue(maxsize=2048)
        self._consumer_thread: threading.Thread | None = None

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

    def _emit(self, event) -> None:
        """Non-blocking emit: drop if queue is full."""
        try:
            self._queue.put_nowait(event)
        except Full:
            pass

    def emit_loop_dt_ms(self, dt_ms: float) -> None:
        self._emit(_EvLoopDt(dt_ms))

    def emit_send_action(self, send_ms: float, send_t: float) -> None:
        self._emit(_EvSendAction(send_ms, send_t))

    def emit_action_delta(self, prev: np.ndarray, cur: np.ndarray) -> None:
        if prev.shape != cur.shape:
            return
        d = (cur - prev).astype(np.float64, copy=False)
        mean_abs = float(np.mean(np.abs(d)))
        max_abs = float(np.max(np.abs(d)))
        l2 = float(np.linalg.norm(d.ravel(), ord=2))
        self._emit(_EvActionDelta(mean_abs, max_abs, l2))

    def emit_action_receiver(self, rpc_ms: float, deser_ms: float, latency_ms: float) -> None:
        self._emit(_EvActionReceiver(rpc_ms, deser_ms, latency_ms))

    def emit_observation_sender(self, capture_ms: float, encode_ms: float, send_ms: float) -> None:
        self._emit(_EvObsSender(capture_ms, encode_ms, send_ms))

    def emit_overrun(self) -> None:
        self._emit(_EvOverrun())

    def emit_log_context(self, step: int, schedule_size: int, latency_steps: int, cooldown: int) -> None:
        self._emit(_EvLogContext(step, schedule_size, latency_steps, cooldown))

    def _consumer_loop(self, logger: logging.Logger) -> None:
        """Background thread: drain queue, aggregate stats, emit periodic logs."""
        cfg = self._cfg
        maxlen = max(10, int(cfg.fps * cfg.diagnostics_window_s))

        # Rolling windows (owned exclusively by consumer thread - no lock needed)
        loop_dt_ms = _RollingWindow(maxlen=maxlen)
        send_action_ms = _RollingWindow(maxlen=maxlen)
        send_action_dt_ms = _RollingWindow(maxlen=maxlen)
        action_delta_mean_abs = _RollingWindow(maxlen=maxlen)
        action_delta_max_abs = _RollingWindow(maxlen=maxlen)
        action_delta_l2 = _RollingWindow(maxlen=maxlen)
        rpc_ms = _RollingWindow(maxlen=maxlen)
        deser_ms = _RollingWindow(maxlen=maxlen)
        measured_latency_ms = _RollingWindow(maxlen=maxlen)
        obs_capture_ms = _RollingWindow(maxlen=maxlen)
        obs_encode_ms = _RollingWindow(maxlen=maxlen)
        obs_send_ms = _RollingWindow(maxlen=maxlen)

        overrun_count = 0
        last_emit_t = time.perf_counter()
        last_action_send_t: float | None = None
        latest_ctx: _EvLogContext | None = None

        while not self._shutdown_event.is_set():
            # Drain queue with timeout
            try:
                event = self._queue.get(timeout=0.1)
            except Empty:
                continue

            # Dispatch event by type
            if isinstance(event, _EvLoopDt):
                loop_dt_ms.add(event.dt_ms)
            elif isinstance(event, _EvSendAction):
                send_action_ms.add(event.send_ms)
                if last_action_send_t is not None:
                    send_action_dt_ms.add((event.send_t - last_action_send_t) * 1000.0)
                last_action_send_t = event.send_t
            elif isinstance(event, _EvActionDelta):
                action_delta_mean_abs.add(event.mean_abs)
                action_delta_max_abs.add(event.max_abs)
                action_delta_l2.add(event.l2)
            elif isinstance(event, _EvActionReceiver):
                rpc_ms.add(event.rpc_ms)
                deser_ms.add(event.deser_ms)
                measured_latency_ms.add(event.latency_ms)
            elif isinstance(event, _EvObsSender):
                obs_capture_ms.add(event.capture_ms)
                obs_encode_ms.add(event.encode_ms)
                obs_send_ms.add(event.send_ms)
            elif isinstance(event, _EvOverrun):
                overrun_count += 1
            elif isinstance(event, _EvLogContext):
                latest_ctx = event

            # Periodic logging (only when we have context)
            now = time.perf_counter()
            if (now - last_emit_t) >= cfg.diagnostics_interval_s and latest_ctx is not None:
                last_emit_t = now
                logger.info(
                    "DIAG | step=%s sched=%s cooldown=%s latency_steps=%s | "
                    "loop_dt_ms(p50/p95/max)=%s | send_action_ms=%s | send_action_dt_ms=%s | "
                    "d_mean_abs=%s d_max_abs=%s d_l2=%s | "
                    "rpc_ms=%s deser_ms=%s latency_ms=%s | "
                    "obs_cap_ms=%s obs_enc_ms=%s obs_send_ms=%s | overruns=%s",
                    latest_ctx.step,
                    latest_ctx.schedule_size,
                    latest_ctx.cooldown,
                    latest_ctx.latency_steps,
                    _format_p50_p95_max(loop_dt_ms.snapshot()),
                    _format_p50_p95_max(send_action_ms.snapshot()),
                    _format_p50_p95_max(send_action_dt_ms.snapshot()),
                    _format_p50_p95_max(action_delta_mean_abs.snapshot()),
                    _format_p50_p95_max(action_delta_max_abs.snapshot()),
                    _format_p50_p95_max(action_delta_l2.snapshot()),
                    _format_p50_p95_max(rpc_ms.snapshot()),
                    _format_p50_p95_max(deser_ms.snapshot()),
                    _format_p50_p95_max(measured_latency_ms.snapshot()),
                    _format_p50_p95_max(obs_capture_ms.snapshot()),
                    _format_p50_p95_max(obs_encode_ms.snapshot()),
                    _format_p50_p95_max(obs_send_ms.snapshot()),
                    overrun_count,
                )


# =============================================================================
# Latency Estimation (Abstract Base + Implementations)
# =============================================================================

from abc import ABC, abstractmethod


class LatencyEstimatorBase(ABC):
    """Abstract base class for latency estimators."""

    def __init__(self, fps: float):
        self._fps = fps

    @property
    def fps(self) -> float:
        return self._fps

    @abstractmethod
    def update(self, measured_rtt: float) -> None:
        """Update the latency estimate with a new RTT measurement."""
        ...

    @property
    @abstractmethod
    def estimate_seconds(self) -> float:
        """Get the latency estimate in seconds."""
        ...

    @property
    def estimate_steps(self) -> int:
        """Get the latency estimate quantized to action steps."""
        return max(1, math.ceil(self.estimate_seconds * self._fps))

    @abstractmethod
    def reset(self) -> None:
        """Reset the estimator state."""
        ...


class JKLatencyEstimator(LatencyEstimatorBase):
    """Jacobson-Karels style latency estimator with exponential smoothing.

    Maintains a smoothed mean and deviation estimate of round-trip latency,
    combining them to produce a conservative estimate that adapts to variance.

    Attributes:
        fps: Control loop frequency for quantizing to action steps.
        alpha: Smoothing factor for mean (default 0.125 per RFC 6298).
        beta: Smoothing factor for deviation (default 0.25 per RFC 6298).
        k: Scaling factor for deviation in estimate (paper suggests K=1 for faster recovery).
    """

    def __init__(
        self,
        fps: float,
        alpha: float = 0.125,
        beta: float = 0.25,
        k: float = 1.0,
    ):
        super().__init__(fps)
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.smoothed_rtt: float = 0.0
        self.rtt_deviation: float = 0.0
        self._initialized: bool = False

    def update(self, measured_rtt: float) -> None:
        """Update the latency estimate with a new RTT measurement."""
        if not self._initialized:
            self.smoothed_rtt = measured_rtt
            self.rtt_deviation = measured_rtt / 2.0
            self._initialized = True
            return

        error = measured_rtt - self.smoothed_rtt
        self.smoothed_rtt = (1 - self.alpha) * self.smoothed_rtt + self.alpha * measured_rtt
        self.rtt_deviation = (1 - self.beta) * self.rtt_deviation + self.beta * abs(error)

    @property
    def estimate_seconds(self) -> float:
        """Get the latency estimate in seconds: ℓ̂ = ℓ̄ + K·σ"""
        if not self._initialized:
            return 1.0 / self._fps * 5  # 5 action steps as initial guess
        return self.smoothed_rtt + self.k * self.rtt_deviation

    def reset(self) -> None:
        """Reset the estimator state."""
        self.smoothed_rtt = 0.0
        self.rtt_deviation = 0.0
        self._initialized = False


class MaxLast10Estimator(LatencyEstimatorBase):
    """Conservative latency estimator using max of last 10 measurements (RTC-style).

    Returns the maximum RTT observed in the last 10 measurements, providing a
    conservative bound that is less adaptive but more stable under spikes.
    """

    def __init__(self, fps: float, window_size: int = 10):
        super().__init__(fps)
        self._window_size = window_size
        self._buffer: deque[float] = deque(maxlen=window_size)

    def update(self, measured_rtt: float) -> None:
        """Add a new RTT measurement to the window."""
        self._buffer.append(measured_rtt)

    @property
    def estimate_seconds(self) -> float:
        """Get the latency estimate as max of last N measurements."""
        if not self._buffer:
            return 1.0 / self._fps * 5  # 5 action steps as initial guess
        return max(self._buffer)

    def reset(self) -> None:
        """Reset the estimator state."""
        self._buffer.clear()


# Backwards compatibility alias
LatencyEstimator = JKLatencyEstimator


def make_latency_estimator(
    kind: str,
    fps: float,
    alpha: float = 0.125,
    beta: float = 0.25,
    k: float = 1.0,
) -> LatencyEstimatorBase:
    """Factory function to create a latency estimator.

    Args:
        kind: Type of estimator - "jk" for Jacobson-Karels, "max_last_10" for max of last 10.
        fps: Control loop frequency.
        alpha: JK smoothing factor for mean.
        beta: JK smoothing factor for deviation.
        k: JK scaling factor for deviation.

    Returns:
        A latency estimator instance.
    """
    if kind == "jk":
        return JKLatencyEstimator(fps=fps, alpha=alpha, beta=beta, k=k)
    elif kind == "max_last_10":
        return MaxLast10Estimator(fps=fps)
    else:
        raise ValueError(f"Unknown latency estimator type: {kind}. Use 'jk' or 'max_last_10'.")


# =============================================================================
# Action Schedule (OrderedDict-based with merge logic)
# =============================================================================


@dataclass
class ScheduledAction:
    """An action scheduled for execution at a specific step.

    Attributes:
        action: The action tensor/array to execute.
        source_step: The action step at which the source observation was captured.
    """

    action: np.ndarray
    source_step: int


class ActionSchedule:
    """Action schedule using OrderedDict for O(1) lookups and ordered iteration.

    The schedule stores (action_step -> ScheduledAction) mappings, enabling efficient
    merging of incoming action chunks with freshest-observation-wins semantics.

    Note: This class is NOT thread-safe. Per the latency-adaptive async inference
    algorithm, only the main control loop thread should read from and write to the
    action schedule. The action receiver thread communicates with the main thread
    via the SPSC action mailbox, and the main thread performs all merge operations.
    """

    def __init__(self):
        self._schedule: OrderedDict[int, ScheduledAction] = OrderedDict()

    def __len__(self) -> int:
        return len(self._schedule)

    def pop_front(self) -> tuple[int, np.ndarray, int] | None:
        """Pop and return the first (lowest action step) scheduled action.

        Returns:
            Tuple of (action, source_step) or None if empty.
        """
        if not self._schedule:
            return None
        # OrderedDict maintains insertion order; pop first item
        step, scheduled = self._schedule.popitem(last=False)
        return step, scheduled.action, scheduled.source_step

    def get_exec_prefix(self, *, current_step: int, max_len: int) -> np.ndarray | None:
        """Get up to `max_len` executable actions immediately after `current_step`.

        This is used to build the frozen-prefix payload for server-side RTC.
        """
        if max_len <= 0:
            return None
        out: list[np.ndarray] = []
        for step, scheduled in self._schedule.items():
            if step <= current_step:
                continue
            out.append(scheduled.action.astype(np.float32, copy=False))
            if len(out) >= max_len:
                break
        if not out:
            return None
        return np.asarray(out, dtype=np.float32, order="C")

    def get_size(self) -> int:
        """Get the current schedule size."""
        return len(self._schedule)

    def is_empty(self) -> bool:
        """Check if schedule is empty."""
        return len(self._schedule) == 0

    def merge(
        self,
        incoming_actions: list[TimedAction],
        source_step: int,
        current_action_step: int,
        latency_steps: int,
        logger: logging.Logger | None = None,
    ) -> None:
        """Merge incoming actions using freshest-observation-wins strategy.

        Respects the frozen action invariant: actions within latency_steps of
        the current execution point cannot be modified.

        Args:
            incoming_actions: List of TimedAction from the server.
            source_step: The action step at which the source observation was captured.
            current_action_step: The most recently executed action step (n*).
            latency_steps: Current latency estimate in action steps (ℓ̂_Δ).
            logger: Optional logger for debug output.
        """
        # Use counters instead of per-action logging to avoid ~1ms per log call
        stale_count = 0
        frozen_count = 0
        inserted_count = 0
        updated_count = 0

        # Track if we need to re-sort (only if inserting out of order)
        max_existing_step = max(self._schedule.keys()) if self._schedule else -1
        needs_sort = False

        for timed_action in incoming_actions:
            step = timed_action.get_timestep()
            action = timed_action.get_action()

            # Skip stale actions (already executed)
            if step <= current_action_step:
                stale_count += 1
                continue

            existing = self._schedule.get(step)

            if existing is None:
                # New action step: always schedule it, even if it's in the frozen window.
                # The frozen-action invariant only prevents *modifying* already-scheduled actions.
                self._schedule[step] = ScheduledAction(action=action, source_step=source_step)
                inserted_count += 1
                # Check if this insertion is out of order
                if step < max_existing_step:
                    needs_sort = True
                else:
                    max_existing_step = step
                continue

            # Existing action: do not modify if frozen.
            if self._is_frozen(step, current_action_step, latency_steps):
                frozen_count += 1
                continue

            if source_step > existing.source_step:
                # Fresher observation wins (only for non-frozen actions)
                self._schedule[step] = ScheduledAction(action=action, source_step=source_step)
                updated_count += 1

        # Re-sort only if we inserted out of order
        if needs_sort:
            sorted_items = sorted(self._schedule.items(), key=lambda x: x[0])
            self._schedule = OrderedDict(sorted_items)

        # Single summary log instead of per-action logs (saves ~20ms for 23 log calls)
        if logger and (stale_count or frozen_count):
            logger.debug(
                f"Merge stats: {stale_count} stale, {frozen_count} frozen, "
                f"{inserted_count} inserted, {updated_count} updated, resort={needs_sort}"
            )

    @staticmethod
    def _is_frozen(action_step: int, current_step: int, latency_steps: int) -> bool:
        """Check if an action step is frozen (cannot be modified).

        Frozen(j, t) ≡ j ≤ n*(t) + ℓ̂_Δ

        Actions within the frozen window will be executed before any new
        inference result can possibly arrive and be merged.
        """
        return action_step <= current_step + latency_steps

    def get_step_range(self) -> tuple[int, int] | None:
        """Get the range of action steps in the schedule.

        Returns:
            Tuple of (min_step, max_step) or None if empty.
        """
        if not self._schedule:
            return None
        steps = list(self._schedule.keys())
        return min(steps), max(steps)

    def clear(self) -> None:
        """Clear all scheduled actions."""
        self._schedule.clear()


# =============================================================================
# Observation Request (for SPSC mailbox)
# =============================================================================


@dataclass
class ObservationRequest:
    """Request for an observation capture, sent from main thread to obs sender.

    Attributes:
        action_step: The current action step when the request was made.
        task: The task description string.
    """

    action_step: int
    task: str
    rtc_meta: dict[str, Any] | None = None


# =============================================================================
# Action Chunk (received from server)
# =============================================================================


@dataclass
class ReceivedActionChunk:
    """Action chunk received from the server with metadata.

    Attributes:
        actions: List of TimedAction from the server.
        source_step: The action step at which the source observation was captured.
        measured_latency: Measured round-trip time for this chunk.
    """

    actions: list[TimedAction]
    source_step: int
    measured_latency: float


# =============================================================================
# Improved Robot Client Configuration
# =============================================================================


@dataclass
class RobotClientImprovedConfig:
    """Configuration for the improved latency-adaptive robot client.

    This configuration follows the latency-adaptive async inference algorithm
    from the paper, with proper SPSC mailboxes, Jacobson-Karels latency estimation,
    cool-down mechanism, and freshest-observation-wins merging.
    """

    # Policy configuration
    policy_type: str = field(metadata={"help": "Type of policy to use (e.g., 'act', 'smolvla')"})
    pretrained_name_or_path: str = field(metadata={"help": "Pretrained model name or path"})

    # Robot configuration
    robot: RobotConfig = field(metadata={"help": "Robot configuration"})

    # Actions per chunk (should be <= policy's max action horizon)
    actions_per_chunk: int = field(metadata={"help": "Number of actions per chunk (H in the paper)"})

    # Task instruction for the robot
    task: str = field(default="", metadata={"help": "Task instruction for the robot to execute"})

    # Network configuration
    server_address: str = field(default="localhost:8080", metadata={"help": "Server address to connect to"})

    # Device configuration (for policy inference on server)
    policy_device: str = field(default="cpu", metadata={"help": "Device for policy inference"})

    # Control frequency
    fps: int = field(default=DEFAULT_FPS, metadata={"help": "Control loop frequency in Hz"})

    # Latency-adaptive parameters
    epsilon: int = field(default=5, metadata={"help": "Safety margin in action steps (ε)"})
    latency_estimator_type: str = field(
        default="jk",
        metadata={"help": "Latency estimator type: 'jk' (Jacobson-Karels) or 'max_last_10'"},
    )
    latency_alpha: float = field(
        default=0.125, metadata={"help": "Jacobson-Karels smoothing factor for RTT mean"}
    )
    latency_beta: float = field(
        default=0.25, metadata={"help": "Jacobson-Karels smoothing factor for RTT deviation"}
    )
    latency_k: float = field(
        default=4.0, metadata={"help": "Jacobson-Karels scaling factor for deviation (K)"}
    )

    # Debug configuration
    debug_visualize_queue_size: bool = field(
        default=False, metadata={"help": "Visualize the action queue size after stopping"}
    )

    # RTC (client-driven, server-side inpainting; flow policies only)
    rtc_enabled: bool = field(
        default=True,
        metadata={"help": "Enable RTC-style inpainting on the policy server (flow policies only)"},
    )
    rtc_execution_horizon: int = field(
        default=10,
        metadata={"help": "RTC execution horizon (prefix blending horizon)"},
    )
    rtc_max_guidance_weight: float = field(
        default=1.0,
        metadata={"help": "RTC max guidance weight (clamp)"},
    )
    rtc_prefix_attention_schedule: str = field(
        default="linear",
        metadata={"help": "RTC prefix attention schedule: zeros|ones|linear|exp"},
    )

    # Diagnostics configuration (off by default)
    diagnostics_enabled: bool = field(
        default=False,
        metadata={"help": "Enable periodic diagnostics logs (timing, latency jitter, action deltas)"},
    )
    diagnostics_interval_s: float = field(
        default=2.0, metadata={"help": "How often to emit a diagnostics summary (seconds)"}
    )
    diagnostics_window_s: float = field(
        default=10.0, metadata={"help": "Rolling window for diagnostics stats (seconds)"}
    )

    # Control-loop clocking (optional)
    control_use_deadline_clock: bool = field(
        default=True,
        metadata={"help": "Use a deadline-based control clock (reduces jitter under overruns)"},
    )

    # Observation sender robustness
    obs_fallback_on_failure: bool = field(
        default=True,
        metadata={"help": "If robot observation capture fails, reuse the last good observation to avoid stalling"},
    )
    obs_fallback_max_age_s: float = field(
        default=2.0,
        metadata={"help": "Max age (seconds) of the last good observation that may be reused on failure"},
    )

    # Simulation mode (for experiments)
    simulation_mode: bool = field(
        default=False,
        metadata={"help": "Use mock robot instead of real hardware (for experiments)"},
    )
    cooldown_enabled: bool = field(
        default=True,
        metadata={"help": "Enable cooldown mechanism (set False for classic async baseline)"},
    )

    # Drop injection (for experiments)
    drop_obs_p: float = field(
        default=0.0,
        metadata={"help": "Random probability of dropping an observation (0.0-1.0)"},
    )
    drop_obs_burst_pattern: str | None = field(
        default=None,
        metadata={"help": "Deterministic drop burst pattern, e.g. '1s@20s' = drop for 1s every 20s"},
    )
    drop_action_p: float = field(
        default=0.0,
        metadata={"help": "Random probability of dropping an action chunk (0.0-1.0)"},
    )
    drop_action_burst_pattern: str | None = field(
        default=None,
        metadata={"help": "Deterministic drop burst pattern, e.g. '1s@20s' = drop for 1s every 20s"},
    )

    # Experiment metrics (CSV export)
    experiment_metrics_path: str | None = field(
        default=None,
        metadata={"help": "Path to write experiment metrics CSV (None = disabled)"},
    )

    @property
    def environment_dt(self) -> float:
        """Environment time step in seconds."""
        return 1.0 / self.fps

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.server_address:
            raise ValueError("server_address cannot be empty")
        if not self.policy_type:
            raise ValueError("policy_type cannot be empty")
        if not self.pretrained_name_or_path:
            raise ValueError("pretrained_name_or_path cannot be empty")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.actions_per_chunk <= 0:
            raise ValueError(f"actions_per_chunk must be positive, got {self.actions_per_chunk}")
        if self.epsilon < 0:
            raise ValueError(f"epsilon must be non-negative, got {self.epsilon}")
        if self.latency_estimator_type not in ("jk", "max_last_10"):
            raise ValueError(
                f"latency_estimator_type must be 'jk' or 'max_last_10', got {self.latency_estimator_type}"
            )
        if self.diagnostics_interval_s <= 0:
            raise ValueError(
                f"diagnostics_interval_s must be positive, got {self.diagnostics_interval_s}"
            )
        if self.diagnostics_window_s <= 0:
            raise ValueError(f"diagnostics_window_s must be positive, got {self.diagnostics_window_s}")
        if self.obs_fallback_max_age_s <= 0:
            raise ValueError(f"obs_fallback_max_age_s must be positive, got {self.obs_fallback_max_age_s}")
        if self.rtc_execution_horizon <= 0:
            raise ValueError(f"rtc_execution_horizon must be positive, got {self.rtc_execution_horizon}")
        if self.rtc_max_guidance_weight <= 0:
            raise ValueError(f"rtc_max_guidance_weight must be positive, got {self.rtc_max_guidance_weight}")


# =============================================================================
# Simulation Helpers (Mock Robot, Drop Simulator)
# =============================================================================

import random
import re


class MockRobot:
    """Mock robot for simulation experiments (no hardware required)."""

    def __init__(self, action_dim: int = 6, state_dim: int = 6):
        self._action_dim = action_dim
        self._state_dim = state_dim
        self._connected = False
        self._step = 0

    @property
    def action_features(self) -> list[str]:
        return [f"joint_{i}" for i in range(self._action_dim)]

    @property
    def state_features(self) -> list[str]:
        return [f"state_{i}" for i in range(self._state_dim)]

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def get_observation(self) -> dict[str, Any]:
        """Return synthetic observation (random state, placeholder images)."""
        self._step += 1
        obs = {}
        # Random joint state
        for feat in self.state_features:
            obs[feat] = np.random.randn().astype(np.float32)
        # Placeholder image (small random RGB)
        obs["camera1"] = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        return obs

    def send_action(self, action: dict[str, float]) -> None:
        """No-op for mock robot."""
        pass


class DropSimulator:
    """Simulates random and burst drops for observations/actions."""

    def __init__(
        self,
        random_drop_p: float = 0.0,
        burst_pattern: str | None = None,
    ):
        self._random_drop_p = random_drop_p
        self._burst_duration_s: float = 0.0
        self._burst_period_s: float = 0.0
        self._start_time: float | None = None

        # Parse burst pattern like "1s@20s" -> drop for 1s every 20s
        if burst_pattern:
            match = re.match(r"(\d+(?:\.\d+)?)\s*s?\s*@\s*(\d+(?:\.\d+)?)\s*s?", burst_pattern)
            if match:
                self._burst_duration_s = float(match.group(1))
                self._burst_period_s = float(match.group(2))

    def should_drop(self) -> bool:
        """Check if the current event should be dropped."""
        now = time.time()
        if self._start_time is None:
            self._start_time = now

        # Check burst drop
        if self._burst_period_s > 0:
            elapsed = now - self._start_time
            time_in_period = elapsed % self._burst_period_s
            if time_in_period < self._burst_duration_s:
                return True

        # Check random drop
        if self._random_drop_p > 0 and random.random() < self._random_drop_p:
            return True

        return False


# =============================================================================
# Experiment Metrics Collector
# =============================================================================

import csv
from pathlib import Path


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
        )
        self._ticks.append(tick)

    def flush(self, path: str | Path) -> None:
        """Write collected metrics to CSV file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "t", "step", "schedule_size", "latency_estimate_steps",
            "cooldown", "stall", "obs_sent", "action_received", "measured_latency_ms",
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
                    "measured_latency_ms": tick.measured_latency_ms if tick.measured_latency_ms is not None else "",
                }
                writer.writerow(row)

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics from collected data."""
        if not self._ticks:
            return {}

        stall_count = sum(t.stall for t in self._ticks)
        total_count = len(self._ticks)

        return {
            "total_ticks": total_count,
            "stall_count": stall_count,
            "stall_fraction": stall_count / total_count if total_count > 0 else 0.0,
            "obs_sent_count": sum(t.obs_sent for t in self._ticks),
            "action_received_count": sum(t.action_received for t in self._ticks),
        }


# =============================================================================
# Improved Robot Client
# =============================================================================


class RobotClientImproved:
    """Latency-adaptive asynchronous inference robot client.

    This implementation follows the latency-adaptive async inference algorithm with:
    - 3-thread architecture (main, observation sender, action receiver)
    - SPSC one-slot mailboxes for thread communication
    - Jacobson-Karels latency estimation
    - Cool-down mechanism for inference triggering
    - Frozen action invariant
    - Freshest-observation-wins merging strategy
    """

    prefix = "robot_client_improved"
    logger = get_logger(prefix)

    @staticmethod
    def _ms(seconds: float) -> float:
        return seconds * 1000.0

    def __init__(self, config: RobotClientImprovedConfig):
        """Initialize the improved robot client.

        Args:
            config: Configuration for the robot client.
        """
        self.config = config

        # Use mock robot in simulation mode, real robot otherwise
        if config.simulation_mode:
            self.robot = MockRobot()
            self.robot.connect()
            # Mock features for simulation
            lerobot_features = {
                "observation.state": list(self.robot.state_features),
                "action": list(self.robot.action_features),
            }
            self.logger.info("Simulation mode: using MockRobot")
        else:
            self.robot = make_robot_from_config(config.robot)
            self.robot.connect()
            lerobot_features = map_robot_keys_to_lerobot_features(self.robot)

        # Drop simulators for experiments
        self._obs_drop_sim = DropSimulator(
            random_drop_p=config.drop_obs_p,
            burst_pattern=config.drop_obs_burst_pattern,
        )
        self._action_drop_sim = DropSimulator(
            random_drop_p=config.drop_action_p,
            burst_pattern=config.drop_action_burst_pattern,
        )

        self.server_address = config.server_address
        self.policy_config = RemotePolicyConfig(
            config.policy_type,
            config.pretrained_name_or_path,
            lerobot_features,
            config.actions_per_chunk,
            config.policy_device,
            rtc_enabled=config.rtc_enabled,
            rtc_execution_horizon=config.rtc_execution_horizon,
            rtc_max_guidance_weight=config.rtc_max_guidance_weight,
            rtc_prefix_attention_schedule=config.rtc_prefix_attention_schedule,
        )

        self.channel = grpc.insecure_channel(
            self.server_address, grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.logger.info(f"Initializing client to connect to server at {self.server_address}")

        # Shutdown coordination
        self.shutdown_event = threading.Event()

        # Action state: n(t), initialized to -1 per algorithm.
        # Note: Only the main control loop thread reads/writes action_step.
        self.action_step: int = -1

        # Latency estimation (configurable: JK or max_last_10)
        self.latency_estimator = make_latency_estimator(
            kind=config.latency_estimator_type,
            fps=config.fps,
            alpha=config.latency_alpha,
            beta=config.latency_beta,
            k=config.latency_k,
        )

        # Action schedule (replaces Queue with OrderedDict)
        self.action_schedule = ActionSchedule()

        # Cool-down counter O^c(t).
        # Note: Only the main control loop thread reads/writes obs_cooldown.
        self.obs_cooldown: int = 0

        # SPSC Mailboxes (one-slot queues)
        # Observation request mailbox: main thread -> observation sender
        self._obs_request_mailbox: Queue[ObservationRequest] = Queue(maxsize=1)

        # Action mailbox: action receiver -> main thread
        self._action_mailbox: Queue[ReceivedActionChunk] = Queue(maxsize=1)

        # Synchronization barrier for thread startup
        self.start_barrier = threading.Barrier(3)  # 3 threads: main, obs sender, action receiver

        # Debug tracking
        self.action_queue_sizes: list[int] = []

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        # Optional diagnostics (queue-based, lock-free for producers)
        self._diagnostics: _DiagnosticsQueue | None = None
        if config.diagnostics_enabled:
            self._diagnostics = _DiagnosticsQueue(config, self.shutdown_event)
            self._diagnostics.start_consumer(self.logger)

        # Experiment metrics collector (CSV export)
        self._experiment_metrics: ExperimentMetricsWriter | None = None
        if config.experiment_metrics_path:
            self._experiment_metrics = ExperimentMetricsWriter()
            self.logger.info(f"Experiment metrics enabled, will write to: {config.experiment_metrics_path}")

        self.logger.info("Robot connected and ready")

    @property
    def running(self) -> bool:
        return not self.shutdown_event.is_set()

    @property
    def current_action_step(self) -> int:
        """Get the most recently executed action step n*(t).

        Note: Only the main control loop thread should access this property.
        """
        return max(self.action_step, -1)

    def start(self) -> bool:
        """Start the robot client and connect to the policy server."""
        try:
            t_total_start = time.perf_counter()

            # Server handshake
            t_ready_start = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            t_ready_done = time.perf_counter()
            self.logger.debug(
                "Connected to policy server (Ready RPC) in %.2fms",
                self._ms(t_ready_done - t_ready_start),
            )

            # Send policy configuration
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            self.logger.info("Sending policy instructions to policy server")
            self.logger.debug(
                f"Policy type: {self.policy_config.policy_type} | "
                f"Pretrained: {self.policy_config.pretrained_name_or_path} | "
                f"Device: {self.policy_config.device}"
            )

            t_policy_rpc_start = time.perf_counter()
            self.stub.SendPolicyInstructions(policy_setup)
            t_policy_rpc_done = time.perf_counter()

            self.shutdown_event.clear()

            self.logger.info(
                "Client init complete | Ready: %.2fms | Policy RPC: %.2fms | Total: %.2fms",
                self._ms(t_ready_done - t_ready_start),
                self._ms(t_policy_rpc_done - t_policy_rpc_start),
                self._ms(time.perf_counter() - t_total_start),
            )

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    def stop(self) -> None:
        """Stop the robot client."""
        self.shutdown_event.set()

        # Unblock any waiting threads with sentinel values
        with suppress(Full):
            self._obs_request_mailbox.put_nowait(
                ObservationRequest(action_step=_SHUTDOWN_SENTINEL, task="")
            )
        with suppress(Full):
            self._action_mailbox.put_nowait(
                ReceivedActionChunk(actions=[], source_step=_SHUTDOWN_SENTINEL, measured_latency=0.0)
            )

        # Flush experiment metrics if enabled
        if self._experiment_metrics is not None and self.config.experiment_metrics_path:
            self._experiment_metrics.flush(self.config.experiment_metrics_path)
            summary = self._experiment_metrics.get_summary()
            self.logger.info(
                f"Experiment metrics written to {self.config.experiment_metrics_path}: {summary}"
            )

        self.robot.disconnect()
        self.logger.debug("Robot disconnected")

        self.channel.close()
        self.logger.debug("Client stopped, channel closed")

    # -------------------------------------------------------------------------
    # Observation Sender Thread
    # -------------------------------------------------------------------------

    def observation_sender(self) -> None:
        """Captures, encodes, and sends observations to the policy server."""
        self.start_barrier.wait()
        self.logger.info("Observation sender thread starting")

        last_good_observation: RawObservation | None = None
        last_good_observation_time: float | None = None
        consecutive_capture_failures = 0

        while self.running:
            try:
                # Wait for an observation request from the main thread
                try:
                    request = self._obs_request_mailbox.get(timeout=0.1)
                except Empty:
                    continue

                # Sentinel value to unblock on shutdown
                if request.action_step == _SHUTDOWN_SENTINEL:
                    continue

                t_capture_start = time.perf_counter()

                # Capture observation from robot
                used_fallback = False
                try:
                    raw_observation = self.robot.get_observation()
                    last_good_observation = raw_observation
                    last_good_observation_time = time.time()
                    consecutive_capture_failures = 0
                except Exception as e:
                    consecutive_capture_failures += 1
                    if (
                        self.config.obs_fallback_on_failure
                        and last_good_observation is not None
                        and last_good_observation_time is not None
                        and (time.time() - last_good_observation_time) <= self.config.obs_fallback_max_age_s
                    ):
                        used_fallback = True
                        raw_observation = last_good_observation
                        self.logger.warning(
                            "Observation capture failed (%s). Reusing last good observation (age=%.2fs, consecutive_failures=%s).",
                            e,
                            time.time() - last_good_observation_time,
                            consecutive_capture_failures,
                        )
                    else:
                        self.logger.error(
                            "Observation capture failed (%s). No usable fallback (consecutive_failures=%s).",
                            e,
                            consecutive_capture_failures,
                        )
                        continue

                # Avoid mutating cached observation dict if we are reusing it.
                if used_fallback:
                    raw_observation = dict(raw_observation)
                raw_observation["task"] = request.task
                if request.rtc_meta is not None:
                    raw_observation["__rtc__"] = request.rtc_meta

                t_capture_done = time.perf_counter()

                # Encode images for transport
                t_encode_start = time.perf_counter()
                encoded_observation, encode_stats = _encode_images_for_transport(
                    raw_observation, jpeg_quality=60
                )
                t_encode_done = time.perf_counter()

                if encode_stats["images_encoded"] > 0:
                    self.logger.debug(
                        "Encoded %s images in %.2fms | raw=%s -> encoded=%s",
                        encode_stats["images_encoded"],
                        self._ms(t_encode_done - t_encode_start),
                        encode_stats["raw_bytes_total"],
                        encode_stats["encoded_bytes_total"],
                    )

                # Create timed observation
                timed_obs = TimedObservation(
                    timestamp=time.time(),
                    observation=encoded_observation,
                    timestep=request.action_step,
                    must_go=True,  # All observations we send should be processed
                )

                # Check if observation should be dropped (simulation/experiments)
                if self._obs_drop_sim.should_drop():
                    self.logger.debug("Dropping observation #%s (simulated drop)", request.action_step)
                    continue

                # Send to server
                t_send_start = time.perf_counter()
                self._send_observation(timed_obs)
                t_send_done = time.perf_counter()

                if self._diagnostics is not None:
                    self._diagnostics.emit_observation_sender(
                        capture_ms=self._ms(t_capture_done - t_capture_start),
                        encode_ms=self._ms(t_encode_done - t_encode_start),
                        send_ms=self._ms(t_send_done - t_send_start),
                    )

                self.logger.debug(
                    "Observation #%s sent | capture: %.2fms | encode: %.2fms | send: %.2fms",
                    request.action_step,
                    self._ms(t_capture_done - t_capture_start),
                    self._ms(t_encode_done - t_encode_start),
                    self._ms(t_send_done - t_send_start),
                )

            except Exception as e:
                self.logger.error(f"Error in observation sender: {e}")

    def _send_observation(self, obs: TimedObservation) -> bool:
        """Send a timed observation to the policy server via gRPC."""
        try:
            observation_bytes = pickle.dumps(obs)
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            return True
        except grpc.RpcError as e:
            self.logger.error(f"Error sending observation: {e}")
            return False

    # -------------------------------------------------------------------------
    # Action Receiver Thread
    # -------------------------------------------------------------------------

    def action_receiver(self) -> None:
        """Receives actions from the server via streaming."""
        self.start_barrier.wait()
        self.logger.info("Action receiver thread starting")
        while self.running:
            try:
                t_rpc_start = time.perf_counter()
                stream = self.stub.StreamActionsDense(services_pb2.Empty())
                t_rpc_done = time.perf_counter()
                if self._diagnostics is not None:
                    self._diagnostics.emit_action_receiver(
                        rpc_ms=self._ms(t_rpc_done - t_rpc_start),
                        deser_ms=0.0,
                        latency_ms=0.0,
                    )

                for dense in stream:
                    if not self.running:
                        break
                    self._handle_actions_dense(dense, rpc_ms=0.0)

            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.UNIMPLEMENTED:
                    self.logger.error(
                        "Server does not implement StreamActionsDense. "
                        "This client is streaming-only for actions; please update the server."
                    )
                    self.stop()
                    return
                self.logger.error(f"Error in StreamActionsDense: {e}")
                time.sleep(0.1)

    def _handle_actions_dense(self, dense: services_pb2.ActionsDense, rpc_ms: float) -> None:
        """Decode a dense action chunk into TimedAction list and publish to main thread."""
        receive_time = time.time()

        t = int(dense.t)
        a = int(dense.a)
        if t <= 0 or a <= 0:
            return

        t_deser_start = time.perf_counter()
        actions = np.frombuffer(dense.actions_f32, dtype=np.float32)
        if actions.size != t * a:
            raise ValueError(f"ActionsDense buffer size mismatch: {actions.size} != {t*a}")
        actions = actions.reshape(t, a)
        t_deser_done = time.perf_counter()

        t0 = float(dense.t0)
        i0 = int(dense.i0)
        dt = float(dense.dt)

        measured_latency = receive_time - t0
        timed_actions = [
            TimedAction(timestamp=t0 + i * dt, timestep=i0 + i, action=actions[i]) for i in range(t)
        ]

        self.logger.debug(
            "Received %s dense actions for step #%s | RPC: %.2fms | decode: %.2fms | latency: %.2fms",
            t,
            i0,
            rpc_ms,
            self._ms(t_deser_done - t_deser_start),
            self._ms(measured_latency),
        )

        if self._diagnostics is not None:
            self._diagnostics.emit_action_receiver(
                rpc_ms=rpc_ms,
                deser_ms=self._ms(t_deser_done - t_deser_start),
                latency_ms=self._ms(measured_latency),
            )

        # Check if action chunk should be dropped (simulation/experiments)
        if self._action_drop_sim.should_drop():
            self.logger.debug("Dropping action chunk for step #%s (simulated drop)", i0)
            return

        self._publish_received_actions(
            timed_actions=timed_actions,
            source_step=i0,
            measured_latency=measured_latency,
        )

    def _publish_received_actions(
        self,
        *,
        timed_actions: list[TimedAction],
        source_step: int,
        measured_latency: float,
    ) -> None:
        # Put in action mailbox (overwrite if full - one-slot mailbox)
        chunk = ReceivedActionChunk(
            actions=timed_actions,
            source_step=source_step,
            measured_latency=measured_latency,
        )

        if self._action_mailbox.full():
            with suppress(Empty):
                _ = self._action_mailbox.get_nowait()

        with suppress(Full):
            self._action_mailbox.put_nowait(chunk)

    # -------------------------------------------------------------------------
    # Main Thread: Control Loop
    # -------------------------------------------------------------------------

    def control_loop(self, task: str | None = None) -> None:
        """Main control loop following Algorithm 1 from the paper.

        This loop:
        1. Executes actions if available
        2. Checks inference trigger condition and requests observations
        3. Processes incoming action chunks
        4. Maintains control frequency

        Args:
            task: Optional task override (uses config.task if not provided).
        """
        self.start_barrier.wait()
        self.logger.info("Control loop starting")

        task = task or self.config.task

        prev_loop_start: float | None = None
        prev_action: np.ndarray | None = None
        next_tick: float | None = time.perf_counter() if self.config.control_use_deadline_clock else None

        while self.running:
            t_loop_start = time.perf_counter()
            if prev_loop_start is not None and self._diagnostics is not None:
                self._diagnostics.emit_loop_dt_ms(self._ms(t_loop_start - prev_loop_start))
            prev_loop_start = t_loop_start

            # Experiment metrics tracking for this tick
            _tick_obs_sent = False
            _tick_action_received = False
            _tick_measured_latency_ms: float | None = None

            # ---------------------------------------------------------------------
            # Step 1: Execute action if available
            # ---------------------------------------------------------------------
            if not self.action_schedule.is_empty():
                result = self.action_schedule.pop_front()
                if result is not None:
                    step, action, _ = result
                    t_send_start = time.perf_counter()
                    self.robot.send_action(self._action_array_to_dict(action))
                    t_send_done = time.perf_counter()

                    # Keep action_step aligned with the schedule's action-step keys.
                    # Only the main control loop thread writes this.
                    self.action_step = step

                    self.logger.debug(
                        "Executed action #%s | send_action: %.2fms",
                        self.action_step,
                        self._ms(t_send_done - t_send_start),
                    )

                    if self._diagnostics is not None:
                        self._diagnostics.emit_send_action(
                            send_ms=self._ms(t_send_done - t_send_start),
                            send_t=t_send_done,
                        )
                        if prev_action is not None:
                            self._diagnostics.emit_action_delta(prev_action, action)
                        prev_action = action

            # Track queue size for debugging
            self.action_queue_sizes.append(self.action_schedule.get_size())

            # ---------------------------------------------------------------------
            # Step 2: Check inference trigger condition
            # ---------------------------------------------------------------------
            schedule_size = self.action_schedule.get_size()
            latency_steps = self.latency_estimator.estimate_steps
            epsilon = self.config.epsilon

            # Inference condition: |ψ(t)| ≤ ℓ̂_Δ + ε AND O^c(t) = 0 (if cooldown enabled)
            trigger_threshold = latency_steps + epsilon
            if self.config.cooldown_enabled:
                should_trigger = schedule_size <= trigger_threshold and self.obs_cooldown == 0
            else:
                # Classic async baseline: always trigger when schedule is low
                should_trigger = schedule_size <= trigger_threshold

            if should_trigger:
                current_step = self.current_action_step

                # Put observation request in mailbox
                # Clamp to 0 so the server produces chunks starting at 0 on startup (consistent with the
                # original async inference implementation that uses max(latest_action, 0)).
                rtc_meta: dict[str, Any] | None = None
                if self.config.rtc_enabled:
                    frozen_len = max(
                        0, min(int(latency_steps), int(self.config.actions_per_chunk))
                    )
                    frozen_exec = self.action_schedule.get_exec_prefix(
                        current_step=current_step, max_len=frozen_len
                    )
                    if frozen_exec is not None:
                        rtc_meta = {
                            "enabled": True,
                            "latency_steps": int(latency_steps),
                            "frozen_t": int(frozen_exec.shape[0]),
                            "frozen_a": int(frozen_exec.shape[1]),
                            "frozen_actions_f32": frozen_exec.tobytes(order="C"),
                        }
                    else:
                        rtc_meta = {
                            "enabled": True,
                            "latency_steps": int(latency_steps),
                            "frozen_t": 0,
                            "frozen_a": 0,
                            "frozen_actions_f32": b"",
                        }

                request = ObservationRequest(
                    action_step=max(current_step, 0),
                    task=task,
                    rtc_meta=rtc_meta,
                )

                if self._obs_request_mailbox.full():
                    with suppress(Empty):
                        _ = self._obs_request_mailbox.get_nowait()

                with suppress(Full):
                    self._obs_request_mailbox.put_nowait(request)
                    # Reset cooldown: O^c(t+1) = ℓ̂_Δ + ε (if enabled)
                    if self.config.cooldown_enabled:
                        self.obs_cooldown = trigger_threshold
                    _tick_obs_sent = True
                    self.logger.debug(
                        "Triggered inference | step: %s | schedule: %s | latency_steps: %s | cooldown set to: %s",
                        current_step,
                        schedule_size,
                        latency_steps,
                        self.obs_cooldown,
                    )
            else:
                # Decrement cooldown: O^c(t+1) = max(O^c(t) - 1, 0) (if enabled)
                if self.config.cooldown_enabled:
                    self.obs_cooldown = max(self.obs_cooldown - 1, 0)

            # ---------------------------------------------------------------------
            # Step 3: Check for incoming action chunks
            # ---------------------------------------------------------------------
            try:
                chunk = self._action_mailbox.get_nowait()
                if chunk.source_step != _SHUTDOWN_SENTINEL:  # Not a sentinel
                    current_step = self.current_action_step
                    latency_steps = self.latency_estimator.estimate_steps

                    # Update latency estimate
                    self.latency_estimator.update(chunk.measured_latency)

                    # Merge actions into schedule
                    t_merge_start = time.perf_counter()
                    self.action_schedule.merge(
                        incoming_actions=chunk.actions,
                        source_step=chunk.source_step,
                        current_action_step=current_step,
                        latency_steps=latency_steps,
                        logger=self.logger,
                    )
                    t_merge_done = time.perf_counter()

                    new_estimate = self.latency_estimator.estimate_steps
                    _tick_action_received = True
                    _tick_measured_latency_ms = self._ms(chunk.measured_latency)
                    self.logger.info(
                        "Merged %s actions from step #%s | latency: %.2fms | new estimate: %s steps | "
                        "schedule size: %s | merge time: %.2fms",
                        len(chunk.actions),
                        chunk.source_step,
                        self._ms(chunk.measured_latency),
                        new_estimate,
                        self.action_schedule.get_size(),
                        self._ms(t_merge_done - t_merge_start),
                    )

            except Empty:
                pass

            # ---------------------------------------------------------------------
            # Step 4: Maintain control frequency
            # ---------------------------------------------------------------------
            elapsed = time.perf_counter() - t_loop_start
            if next_tick is None:
                sleep_s = max(0.0, self.config.environment_dt - elapsed)
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    if self._diagnostics is not None:
                        self._diagnostics.emit_overrun()
                    self.logger.debug(
                        "Control loop overran | elapsed: %.2fms | target: %.2fms",
                        self._ms(elapsed),
                        self._ms(self.config.environment_dt),
                    )
            else:
                # Deadline-based clock: reduces drift and jitter when occasional overruns happen.
                next_tick += self.config.environment_dt
                now = time.perf_counter()
                sleep_s = next_tick - now
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    # If we're behind, count an overrun and re-anchor to now to avoid runaway lag.
                    if self._diagnostics is not None:
                        self._diagnostics.emit_overrun()
                    self.logger.debug(
                        "Control loop overran (deadline clock) | late_by: %.2fms | target: %.2fms",
                        self._ms(-sleep_s),
                        self._ms(self.config.environment_dt),
                    )
                    next_tick = now

            if self._diagnostics is not None:
                self._diagnostics.emit_log_context(
                    step=self.current_action_step,
                    schedule_size=self.action_schedule.get_size(),
                    latency_steps=self.latency_estimator.estimate_steps,
                    cooldown=self.obs_cooldown,
                )

            # Record experiment metrics for this tick
            if self._experiment_metrics is not None:
                self._experiment_metrics.record_tick(
                    step=self.current_action_step,
                    schedule_size=self.action_schedule.get_size(),
                    latency_estimate_steps=self.latency_estimator.estimate_steps,
                    cooldown=self.obs_cooldown,
                    obs_sent=_tick_obs_sent,
                    action_received=_tick_action_received,
                    measured_latency_ms=_tick_measured_latency_ms,
                )

    def _action_array_to_dict(self, action_array: np.ndarray) -> dict[str, float]:
        """Convert action array to dictionary keyed by robot action features."""
        return {key: action_array[i].item() for i, key in enumerate(self.robot.action_features)}


# =============================================================================
# Image Encoding for Transport
# =============================================================================


def _is_uint8_hwc3_image(x: Any) -> bool:
    if not isinstance(x, np.ndarray):
        return False
    if x.dtype != np.uint8:
        return False
    if x.ndim != 3:
        return False
    h, w, c = x.shape
    if h <= 0 or w <= 0:
        return False
    return c == 3


def _encode_images_for_transport(
    observation: Any,
    jpeg_quality: int,
) -> tuple[Any, dict[str, int]]:
    """Recursively JPEG-encode uint8 HWC3 images inside an observation structure."""
    stats = {"images_encoded": 0, "raw_bytes_total": 0, "encoded_bytes_total": 0}

    def _encode_any(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: _encode_any(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_encode_any(v) for v in x]
        if isinstance(x, tuple):
            return tuple(_encode_any(v) for v in x)

        if not _is_uint8_hwc3_image(x):
            return x

        bgr = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(
            ".jpg",
            bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        )
        if not ok:
            raise RuntimeError("OpenCV failed to JPEG-encode image for transport")

        payload = bytes(buf)
        stats["images_encoded"] += 1
        stats["raw_bytes_total"] += int(x.nbytes)
        stats["encoded_bytes_total"] += len(payload)
        return {"__lerobot_image_encoding__": "jpeg", "quality": int(jpeg_quality), "data": payload}

    return _encode_any(observation), stats


# =============================================================================
# Entry Point
# =============================================================================


def async_client_improved(cfg: RobotClientImprovedConfig) -> None:
    """Run the improved async inference client."""
    logging.info(pformat(cfg.__dict__))

    if cfg.robot.type not in SUPPORTED_ROBOTS:
        raise ValueError(f"Robot {cfg.robot.type} not yet supported!")

    client = RobotClientImproved(cfg)

    if client.start():
        # Start observation sender thread
        obs_sender_thread = threading.Thread(
            target=client.observation_sender,
            name="observation_sender",
            daemon=True,
        )

        # Start action receiver thread
        action_receiver_thread = threading.Thread(
            target=client.action_receiver,
            name="action_receiver",
            daemon=True,
        )

        obs_sender_thread.start()
        action_receiver_thread.start()

        try:
            # Main thread runs the control loop
            client.control_loop()

        finally:
            client.stop()
            obs_sender_thread.join(timeout=2.0)
            action_receiver_thread.join(timeout=2.0)

            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_sizes)

            client.logger.info("Client stopped")


if __name__ == "__main__":
    import draccus

    draccus.wrap()(async_client_improved)()
