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

"""Shared chunk scheduling, atomic continuation and local motion permission.

The control thread never waits for an executor. A queue pop commits a policy
endpoint to interpolation; freshness remains attached to that endpoint on every
motor tick, including ticks without a queue pop. All clocks here are client-local.
"""

from __future__ import annotations

import hashlib
import json
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

from .contracts import ActionChunk, ActionProvenance, ActionSource, ExecutionMode, ObservationSnapshot


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
    playback_at_submission: float = 0.0


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
        chunk_merge: str = "append",
        blend_steps: int = 0,
        blend_weight: float = 0.5,
        blend_indices: tuple[int, ...] = (),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Allocate one action queue and a bounded steady-state latency window."""
        for value in (action_interval, max_observation_age_s, action_timeout_s, startup_timeout_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Execution intervals, ages and deadlines must be finite and positive")
        if not math.isfinite(refill_seconds) or refill_seconds < 0:
            raise ValueError("Refill playback threshold must be finite and nonnegative")
        if chunk_merge not in {"append", "aligned"} or (
            chunk_merge != "append" and mode is not ExecutionMode.CHUNK
        ):
            raise ValueError("Aligned replacement is supported only for plain chunks")
        if type(blend_steps) is not int or blend_steps < 0:
            raise ValueError("Blend steps must be a nonnegative integer")
        if not math.isfinite(blend_weight) or not 0 < blend_weight <= 1:
            raise ValueError("Incoming blend weight must be in (0, 1]")
        if len(set(blend_indices)) != len(blend_indices) or any(
            type(index) is not int or index < 0 for index in blend_indices
        ):
            raise ValueError("Blend indices must be distinct nonnegative integers")
        if blend_steps and (chunk_merge != "aligned" or not blend_indices):
            raise ValueError("Blending requires alignment and explicit continuous components")
        if blend_indices and not blend_steps:
            raise ValueError("Blend components require a positive blend window")
        self.mode = mode
        self.chunk_merge = chunk_merge
        self.blend_steps = blend_steps
        self.blend_weight = blend_weight
        self.blend_indices = blend_indices
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
        self._initial_turnaround = 0.0
        self._completed = 0
        self._last_anchor: tuple[int, float, int] | None = None
        self._last_action: torch.Tensor | None = None
        self._last_dispatch_request = ""
        self.dispatch_events: deque[dict] = deque(maxlen=128)
        self.last_accept: dict = {}

    @property
    def turnaround(self) -> float:
        """Largest complete action turnaround in the recent steady-state window."""
        return max(self.turnarounds, default=self._initial_turnaround)

    @property
    def effective_refill(self) -> float:
        """Playback threshold including measured turnaround headroom."""
        return max(self.refill_seconds, self.turnaround + self.interval)

    def anchor_observation(self, observation: ObservationSnapshot) -> ObservationSnapshot:
        """Bind a control-thread sample before committing the next endpoint.

        The sampling caller must not pop actions between capture and this call.
        The rollout control thread owns both operations; a worker may replace
        the future concurrently but cannot advance this commitment cursor.
        """
        with self.lock:
            return replace(
                observation,
                action_cursor=self.queue.snapshot().cursor,
                execution_generation=self.generation,
            )

    def invalidate(self, *, held: bool = False) -> int:
        """Revoke old motion locally and return the new execution generation."""
        with self.lock:
            self.generation += 1
            self.queue.clear()
            self.pending = None
            self.current = None
            self.held = held
            self.has_executed = False
            self._last_anchor = None
            self._last_action = None
            self._last_dispatch_request = ""
            self.started_at = self.clock()
            return self.generation

    def activate(self, *, held: bool = False) -> bool:
        """Activate a healthy run and atomically arm its startup budget."""
        with self.lock:
            if self.failure is not None:
                return False
            self.active = True
            self.held = held
            self.started_at = self.clock()
            return True

    def deactivate(self) -> None:
        """Revoke motion and pending work atomically without clearing a fault."""
        with self.lock:
            self.active = False
            self.invalidate(held=True)

    def release_hold(self, expected_generation: int) -> bool:
        """Arm fresh resumption only for the same healthy, active generation."""
        with self.lock:
            if self.failure is not None or not self.active or self.generation != expected_generation:
                return False
            self.started_at = self.clock()
            self.held = False
            return True

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

    def should_request(self, *, task_version: int | None = None) -> bool:
        """Check permission and playback need without reserving an observation.

        Aligned task changes may bypass the playback threshold. ``begin`` still
        validates their fresh capture, generation and commitment anchor.
        """
        with self.lock:
            self.check_deadlines()
            if self.failure or not self.active or self.held or self.pending is not None:
                return False
            snapshot = self.queue.snapshot()
            if (
                self.chunk_merge == "aligned"
                and task_version is not None
                and self._last_anchor is not None
                and task_version != self._last_anchor[2]
            ):
                return True
            if self.mode is ExecutionMode.CHUNK and self.chunk_merge == "append":
                # One accepted successor beyond the executing chunk, never a chain
                # of old-observation predictions hidden behind newer provenance.
                sources = {p.request_id for p in snapshot.provenance if p is not None}
                if len(sources) > 1:
                    return False
                if self.current and sources and self.current.request_id not in sources:
                    return False
            playback = self.queue.qsize() * self.interval
            return playback <= self.effective_refill + 1e-9

    def begin(self, observation: ObservationSnapshot) -> ChunkRequest | None:
        """Reserve inference against one atomic cursor/continuation snapshot."""
        with self.lock:
            if not self.should_request(task_version=observation.task_version):
                return None
            age = self.clock() - observation.capture_time
            if age < 0 or age > self.max_age:
                return None  # wait for a fresh capture; startup/dispatch deadlines still apply
            snapshot = self.queue.snapshot()
            if self.chunk_merge == "aligned":
                if (
                    observation.execution_generation != self.generation
                    or observation.action_cursor is None
                    or observation.action_cursor > snapshot.cursor
                    or observation.capture_time < self.started_at
                ):
                    return None
                if self._last_anchor is not None and (
                    (
                        observation.task_version == self._last_anchor[2]
                        and observation.action_cursor <= self._last_anchor[0]
                    )
                    or observation.capture_time <= self._last_anchor[1]
                ):
                    return None
                self._last_anchor = (
                    observation.action_cursor,
                    observation.capture_time,
                    observation.task_version,
                )
            available = 0 if snapshot.model_actions is None else len(snapshot.model_actions)
            self.pending = ChunkRequest(
                observation,
                snapshot,
                uuid4().hex,
                self.generation,
                self.mode,
                estimate_delay(self.turnaround, self.interval, self.mode, self.training_max_delay, available),
                self.clock(),
                len(snapshot.provenance) * self.interval,
            )
            return self.pending

    def accept(self, request: ChunkRequest, chunk: ActionChunk, *, task_version: int) -> bool:
        """Validate context and timing, then atomically merge with original provenance."""
        with self.lock:
            self.last_accept = {"accepted": False}
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
            else:
                # Until a warm measurement exists, reserving less than the only
                # observed turnaround can deterministically starve request two.
                self._initial_turnaround = elapsed
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
                or type(chunk.execution_steps) is not int
                or not 0 < chunk.execution_steps <= len(actions)
                or not torch.isfinite(actions).all()
                or not torch.isfinite(model).all()
            ):
                self.fault("Invalid action chunk")
                return False
            # The runner/client already enforces the slice. Keep the shared
            # executor strict even for independently supplied ActionChunks.
            if self.mode is ExecutionMode.CHUNK:
                actions = actions[: chunk.execution_steps]
                model = model[: chunk.execution_steps]
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
            self.last_accept.update(
                turnaround_s=elapsed,
                playback_at_submission_s=request.playback_at_submission,
                timing_margin_s=request.playback_at_submission - elapsed if self.has_executed else None,
                cursor_advance=progress,
                trimmed_actions=trim,
                overlap_steps=0,
                blended_steps=0,
                source_age_at_acceptance_s=now - provenance.capture_time,
                remaining_old_playback_s=self.queue.qsize() * self.interval,
                estimated_first_dispatch_source_age_s=now
                - provenance.capture_time
                + (
                    self.queue.qsize() * self.interval
                    if self.chunk_merge == "append" and self.mode is ExecutionMode.CHUNK
                    else 0
                ),
            )
            if self.chunk_merge == "aligned":
                return self._accept_aligned(request, actions, provenance, now)
            accepted = self.queue.merge(
                model,
                actions,
                measured,
                task=provenance.task,
                provenance=provenance,
                snapshot=request.continuation,
            )
            self.last_accept["accepted"] = accepted
            return accepted

    @staticmethod
    def _same_context(old: ActionProvenance, new: ActionProvenance) -> bool:
        return all(
            getattr(old, key) == getattr(new, key)
            for key in (
                "task",
                "task_version",
                "generation",
                "session_id",
                "server_instance_id",
                "artifact_identity",
            )
        )

    @staticmethod
    def _blended_provenance(old: ActionProvenance, new: ActionProvenance) -> ActionProvenance:
        """Summarize all contributing history in constant storage, without refreshing age."""
        oldest = old.oldest_contributor or ActionSource(old.capture_time, old.observation_id, old.request_id)
        if new.capture_time < oldest.capture_time:
            oldest = ActionSource(new.capture_time, new.observation_id, new.request_id)
        digest = hashlib.sha256(
            json.dumps(
                [old.contributor_digest, old.request_id, old.capture_time, new.request_id, new.capture_time],
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        return replace(
            new,
            capture_time=oldest.capture_time,
            oldest_contributor=oldest,
            contributor_count=min(old.contributor_count + 1, 2**63 - 1),
            contributor_digest=digest,
        )

    def _accept_aligned(
        self, request: ChunkRequest, actions: torch.Tensor, provenance: ActionProvenance, now: float
    ) -> bool:
        snapshot = self.queue.snapshot()
        anchor = request.observation.action_cursor
        if anchor is None or request.observation.execution_generation != self.generation:
            self.fault("Aligned result lacks its observation commitment anchor")
            return False
        trim = snapshot.cursor - anchor
        self.last_accept["trimmed_actions"] = trim
        if trim < 0 or trim >= len(actions):
            self.last_accept["rejection"] = "no usable aligned suffix"
            return False
        future = actions[trim:].clone()
        sources = [provenance] * len(future)
        previous = snapshot.canonical_actions
        overlap = min(len(previous), len(future)) if previous is not None else 0
        self.last_accept["overlap_steps"] = overlap
        if self.blend_indices and max(self.blend_indices) >= future.shape[1]:
            self.fault("Blend component index exceeds canonical action dimensions")
            return False
        if previous is not None and self.blend_weight < 1:
            for index in range(min(overlap, self.blend_steps)):
                old = snapshot.provenance[index]
                if (
                    old is None
                    or not self._same_context(old, provenance)
                    or not 0 <= now - old.capture_time <= self.max_age
                ):
                    continue
                coordinates = list(self.blend_indices)
                future[index, coordinates] = (1 - self.blend_weight) * previous[
                    index, coordinates
                ] + self.blend_weight * future[index, coordinates]
                sources[index] = self._blended_provenance(old, provenance)
                self.last_accept["blended_steps"] += 1
        if self._last_action is not None:
            self.last_accept["transition_delta"] = (future[0] - self._last_action).tolist()
        if overlap and previous is not None:
            self.last_accept["replacement_delta"] = (future[0] - previous[0]).tolist()
        accepted = self.queue.replace_future(future, sources, snapshot=snapshot)
        self.last_accept.update(
            accepted=accepted,
            usable_actions=len(future),
            estimated_first_dispatch_source_age_s=now - sources[0].capture_time,
        )
        return accepted

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

    def pop(self, *, control_tick: int | None = None) -> tuple[torch.Tensor, ActionProvenance] | None:
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
            if not self.dispatch_allowed():
                return None
            if provenance.request_id != self._last_dispatch_request:
                self._last_dispatch_request = provenance.request_id
                self.dispatch_events.append(
                    {
                        "time": self.clock(),
                        "generation": provenance.generation,
                        **({"control_tick": control_tick} if control_tick is not None else {}),
                        "request_id": provenance.request_id,
                        "source_age_s": self.clock() - provenance.capture_time,
                        "contributor_count": provenance.contributor_count,
                        "action_cursor": self.queue.snapshot().cursor,
                        "transition_delta": None
                        if self._last_action is None
                        else (action - self._last_action).tolist(),
                    }
                )
            self._last_action = action.clone()
            return action, provenance
