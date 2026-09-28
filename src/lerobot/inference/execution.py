# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Shared chunk scheduling, atomic continuation and local motion permission.

The control thread never waits for an executor. A queue pop commits a policy
endpoint to interpolation; freshness remains attached to that endpoint on every
motor tick, including ticks without a queue pop. All clocks here are client-local.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from threading import RLock
from uuid import uuid4

import torch

from lerobot.policies.rtc.action_queue import ActionQueue, QueueSnapshot
from lerobot.policies.rtc.configuration_rtc import RTCConfig

from .contracts import ActionChunk, ActionProvenance, ExecutionMode, ObservationSnapshot


@dataclass(frozen=True)
class ChunkRequest:
    """Immutable observation/continuation binding for one executor operation."""

    observation: ObservationSnapshot
    continuation: QueueSnapshot
    request_id: str
    generation: int
    mode: ExecutionMode
    delay: int
    submitted_at: float


def estimate_delay(
    turnaround: float, interval: float, mode: ExecutionMode, training_max_delay: int, available: int
) -> int:
    """Estimate overlap from full turnaround, bounded by real trained-prefix capacity."""
    delay = math.ceil(turnaround / interval) if turnaround else 0
    if mode is ExecutionMode.RTC_TRAINED:
        delay = min(delay or training_max_delay, training_max_delay, available)
    return delay if available else 0


def trained_overlap_valid(conditioned: int, measured: int, maximum: int, has_previous: bool) -> bool:
    """Require elapsed overlap to fit both the conditioning and training range."""
    return not has_previous or measured <= min(conditioned, maximum)


class ChunkRuntime:
    """Capacity-one executor scheduling with per-action provenance and latched faults."""

    def __init__(
        self,
        *,
        mode: ExecutionMode,
        action_interval: float,
        refill_seconds: float,
        max_observation_age_s: float,
        action_timeout_s: float,
        startup_timeout_s: float,
        training_max_delay: int = 0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Allocate one action queue and a bounded steady-state latency window."""
        for value in (action_interval, max_observation_age_s, action_timeout_s, startup_timeout_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Execution intervals, ages and deadlines must be finite and positive")
        if not math.isfinite(refill_seconds) or refill_seconds < 0:
            raise ValueError("Refill playback threshold must be finite and nonnegative")
        self.mode = mode
        self.interval = action_interval
        self.refill_seconds = refill_seconds
        self.max_age = max_observation_age_s
        self.action_timeout = action_timeout_s
        self.startup_timeout = startup_timeout_s
        self.training_max_delay = training_max_delay
        self.clock = clock
        self.lock = RLock()
        self.queue = ActionQueue(RTCConfig(enabled=mode is not ExecutionMode.CHUNK))
        self.generation = 0
        self.pending: ChunkRequest | None = None
        self.current: ActionProvenance | None = None
        self.failure: str | None = None
        self.active = False
        self.held = False
        self.has_executed = False
        self.started_at = clock()
        self.turnarounds: deque[float] = deque(maxlen=100)
        self._completed = 0

    @property
    def turnaround(self) -> float:
        """Largest complete action turnaround in the recent steady-state window."""
        return max(self.turnarounds, default=0.0)

    def invalidate(self, *, held: bool = False) -> int:
        """Revoke old motion locally and return the new execution generation."""
        with self.lock:
            self.generation += 1
            self.queue.clear()
            self.pending = None
            self.current = None
            self.held = held
            self.has_executed = False
            self.started_at = self.clock()
            return self.generation

    def fault(self, reason: str) -> None:
        """Latch a terminal fault; invalidate does not clear this latch."""
        with self.lock:
            if self.failure is None:
                self.failure = reason
                self.invalidate(held=True)
                self.active = False

    def check_deadlines(self) -> None:
        """Enforce local action/startup bounds even while network work is blocked."""
        with self.lock:
            if self.failure or not self.active or self.held:
                return
            now = self.clock()
            if self.pending is not None and now - self.pending.submitted_at > self.action_timeout:
                self.fault("Action request deadline exceeded")
            elif not self.has_executed and now - self.started_at > self.startup_timeout:
                self.fault("Initial action deadline exceeded")

    def should_request(self) -> bool:
        """Allow refill only with a free request/successor slot and playback need."""
        with self.lock:
            self.check_deadlines()
            if self.failure or not self.active or self.held or self.pending is not None:
                return False
            snapshot = self.queue.snapshot()
            if self.mode is ExecutionMode.CHUNK:
                # One accepted successor beyond the executing chunk, never a chain
                # of old-observation predictions hidden behind newer provenance.
                sources = {p.request_id for p in snapshot.provenance if p is not None}
                if len(sources) > 1:
                    return False
                if self.current and sources and self.current.request_id not in sources:
                    return False
            playback = self.queue.qsize() * self.interval
            return playback <= max(self.refill_seconds, self.turnaround + self.interval) + 1e-9

    def begin(self, observation: ObservationSnapshot) -> ChunkRequest | None:
        """Reserve inference against one atomic cursor/continuation snapshot."""
        with self.lock:
            if not self.should_request():
                return None
            age = self.clock() - observation.capture_time
            if age < 0 or age > self.max_age:
                return None  # wait for a fresh capture; startup/dispatch deadlines still apply
            snapshot = self.queue.snapshot()
            available = 0 if snapshot.model_actions is None else len(snapshot.model_actions)
            self.pending = ChunkRequest(
                observation,
                snapshot,
                uuid4().hex,
                self.generation,
                self.mode,
                estimate_delay(self.turnaround, self.interval, self.mode, self.training_max_delay, available),
                self.clock(),
            )
            return self.pending

    def accept(self, request: ChunkRequest, chunk: ActionChunk, *, task_version: int) -> bool:
        """Validate context and timing, then atomically merge with original provenance."""
        with self.lock:
            self.check_deadlines()
            if self.failure or self.pending is not request or request.generation != self.generation:
                return False
            self.pending = None
            if request.observation.task_version != task_version:
                return False
            now = self.clock()
            elapsed = now - request.submitted_at
            self._completed += 1
            if self._completed > 1:  # cold start is not a steady-state action latency
                self.turnarounds.append(elapsed)
            if now - request.observation.capture_time > self.max_age:
                self.fault("Action result source observation is too old")
                return False
            actions = chunk.canonical_actions
            model = chunk.model_actions if chunk.model_actions is not None else actions
            if (
                actions.ndim != 2
                or model.ndim != 2
                or len(actions) != len(model)
                or not len(actions)
                or not torch.isfinite(actions).all()
                or not torch.isfinite(model).all()
            ):
                self.fault("Invalid action chunk")
                return False
            progress = self.queue.snapshot().cursor - request.continuation.cursor
            measured = math.ceil(elapsed / self.interval)
            available = request.continuation.model_actions
            has_previous = available is not None and len(available) > 0
            if self.mode is ExecutionMode.RTC_TRAINED and not trained_overlap_valid(
                request.delay, max(measured, progress), self.training_max_delay, has_previous
            ):
                return False
            trim = min(measured, progress) if self.mode is not ExecutionMode.CHUNK else 0
            if trim >= len(actions):
                self.fault("Inference completed beyond its usable continuation horizon")
                return False
            provenance = replace(
                chunk.provenance,
                request_id=request.request_id,
                generation=request.generation,
                capture_time=request.observation.capture_time,
                task=request.observation.task,
                task_version=request.observation.task_version,
                observation_id=request.observation.observation_id,
            )
            return self.queue.merge(
                model,
                actions,
                measured,
                task=provenance.task,
                provenance=provenance,
                snapshot=request.continuation,
            )

    def dispatch_allowed(self) -> bool:
        """Check permission and the committed endpoint's age on every motor tick."""
        with self.lock:
            self.check_deadlines()
            if self.failure or not self.active or self.held:
                return False
            if self.current is not None and self.clock() - self.current.capture_time > self.max_age:
                self.fault("Dispatched action source observation is too old")
                return False
            return self.current is not None or not self.queue.empty()

    def pop(self) -> tuple[torch.Tensor, ActionProvenance] | None:
        """Commit an endpoint to interpolation; its provenance remains age-checked."""
        with self.lock:
            if not self.dispatch_allowed():
                return None
            item = self.queue.get_with_provenance()
            if item is None:
                if self.has_executed:
                    self.fault("Active motion buffer exhausted")
                return None
            action, _, provenance = item
            if provenance is None:
                self.fault("Action lacks request provenance")
                return None
            self.current = provenance
            self.has_executed = True
            return (action, provenance) if self.dispatch_allowed() else None
