# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Remote rollout backend: worker-owned network exchanges, local motor permission."""

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from threading import Event, RLock, Thread
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
import torch

from lerobot.inference.contracts import ExecutionMode, ObservationSnapshot
from lerobot.inference.events import EventWriter
from lerobot.inference.execution import ChunkRuntime
from lerobot.remote_inference.client import RemoteClient, RequestCancelled
from lerobot.remote_inference.protocol import ErrorCode, ProtocolError
from lerobot.utils.feature_utils import build_dataset_frame

from .base import InferenceEngine, PolicyQuery, QueryAnswer, QueryKind

if TYPE_CHECKING:
    from ..robot_wrapper import ThreadSafeRobot
    from .factory import RemoteInferenceConfig

logger = logging.getLogger(__name__)


class RemoteInferenceEngine(InferenceEngine):
    def __init__(
        self,
        client: RemoteClient,
        config: RemoteInferenceConfig,
        dataset_features: dict,
        rename_map: dict[str, str],
        robot_wrapper: ThreadSafeRobot,
        task: str,
        shutdown_event: Event | None = None,
    ) -> None:
        super().__init__(task)
        self.client = client
        self.config = config
        self._features = dataset_features
        self._rename_map = rename_map
        self._robot = robot_wrapper
        self._global_shutdown = shutdown_event
        self.runtime = ChunkRuntime(
            mode=ExecutionMode(config.mode),
            action_interval=client.capabilities.action_interval,
            refill_seconds=config.refill_seconds,
            max_observation_age_s=config.max_observation_age_s,
            action_timeout_s=config.action_timeout_s,
            startup_timeout_s=config.startup_timeout_s,
            training_max_delay=client.capabilities.training_max_delay,
        )
        self._lock = RLock()
        self._observation: ObservationSnapshot | None = None
        self._control: tuple[str, int] | None = None
        self._hold_requested = False
        self._hold_acknowledged = False
        self._hold_started = 0.0
        self._query_observation: ObservationSnapshot | None = None
        self._query_generation = 0
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._traceback: str | None = None
        self._event_writer: EventWriter | None = None
        self._tick = 0
        self._fault_reported = False

    def configure_event_log(self, path: Path) -> None:
        self._event_writer = EventWriter(path)

    def _event(self, name: str, **values) -> None:
        event = {
            "event": name,
            "time": time.monotonic(),
            "deployment": self.config.deployment,
            "instance": self.client.instance_id,
            "session": self.client.session_id,
            "generation": self.runtime.generation,
            "control_tick": self._tick,
            **values,
        }
        if self._event_writer is not None:
            self._event_writer.write(event)
        if name != "dispatch":
            logger.info("Remote inference %s", event)

    @property
    def control_thread_owns_policy(self) -> bool:
        return False

    @property
    def supports_text_queries(self) -> bool:
        return self.client.capabilities.language

    @property
    def ready(self) -> bool:
        return bool(self.client.session_id) and not self.failed

    @property
    def failed(self) -> bool:
        return self.runtime.failure is not None

    @property
    def failure_traceback(self) -> str | None:
        return self._traceback or self.runtime.failure

    def _fault(self, reason: str) -> None:
        if not self.failed:
            self.runtime.fault(reason)
        if not self._fault_reported:
            self._fault_reported = True
            self._event("fault", reason=reason)
        # Keep the control loop alive until it has actually applied the local hold.
        # acknowledge_hold then signals shutdown; otherwise teardown could preempt it.

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Remote inference engine has already started")
        self._thread = Thread(target=self._loop, daemon=True, name="RemoteInference")
        self._thread.start()

    def stop(self) -> None:
        self.pause()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._event_writer is not None:
            self._event_writer.close()

    def _invalidate(self, operation: str) -> None:
        with self._lock:
            generation = self.runtime.invalidate()
            self._observation = None
            self._hold_requested = False
            self._hold_acknowledged = False
            # A later motion invalidation must not erase a pending full reset.
            if self._control is not None and self._control[0] == "reset":
                operation = "reset"
            self._control = (operation, generation)
        self._event(operation)

    def pause(self) -> None:
        self.runtime.active = False
        self._invalidate("invalidate")

    def resume(self) -> None:
        if self.failed or self._stop_event.is_set():
            raise RuntimeError("Faulted/closed remote sessions require a new rollout")
        with self.runtime.lock:
            self.runtime.active = True
            self.runtime.started_at = time.monotonic()

    def reset(self) -> None:
        self.drop_pending_query()
        self._invalidate("reset")
        self._discard_task_change()

    def set_task(self, task: str) -> bool:
        if len(task) > self.client.descriptor["limits"]["max_input_chars"]:
            raise ValueError("Instruction exceeds deployed text limit")
        changed = super().set_task(task)
        if changed:
            self._event("task", task=task, task_version=self.task_version)
        return changed

    def _queue_query(self, query: PolicyQuery) -> bool:
        if (
            not self.supports_text_queries
            or self.failed
            or not 0 < len(query.text) <= self.client.descriptor["limits"]["max_input_chars"]
        ):
            return False
        queued = super()._queue_query(query)
        if queued:
            self._request_language_hold()
        return queued

    def pump_query(self, obs_processed: dict | None = None) -> bool:
        """Stop submitting actions as soon as the sequencer queues a text request."""
        served = super().pump_query(obs_processed)
        if self.has_pending_query:
            self._request_language_hold()
        return served

    def start_autosteer(self, goal: str, interval_s: float) -> None:
        if (
            not self.supports_text_queries
            or not 0 < len(goal) <= self.client.descriptor["limits"]["max_input_chars"]
        ):
            raise ValueError("Unsupported or oversized autosteering query")
        super().start_autosteer(goal, interval_s)

    def notify_observation(self, obs: dict) -> None:
        sampled = self._robot.observation_time
        if sampled is None:
            sampled = time.monotonic()
        frame = build_dataset_frame(self._features, obs, prefix="observation")
        mapped = {self._rename_map.get(key, key): value for key, value in frame.items()}
        if len(mapped) != len(frame):
            self._fault("Feature mapping contains a collision")
            return
        # Mapping is applied once here; server processors require an empty rename map.
        with self._task_lock:
            task, version = self._task, self._task_version
        snapshot = ObservationSnapshot(mapped, sampled, task, version, uuid4().hex)
        with self._lock:
            self._observation = snapshot

    def dispatch_allowed(self) -> bool:
        # Called on every motor tick, before any interpolated target is sent.
        if self.has_pending_query:
            self._request_language_hold()
        with self._lock:
            if self._hold_requested and time.monotonic() - self._hold_started > (
                self.config.language_timeout_s
                + self.config.action_timeout_s
                + self.config.handshake_timeout_s
            ):
                self._fault("Planned language hold deadline exceeded")
        allowed = self.runtime.dispatch_allowed()
        if self.runtime.failure is not None:
            self._fault(self.runtime.failure)
        return allowed

    def _request_language_hold(self) -> None:
        with self._lock:
            if not self.runtime.active or self._hold_requested or self.failed:
                return
            generation = self.runtime.invalidate(held=True)
            self._hold_requested = True
            self._hold_acknowledged = False
            self._hold_started = time.monotonic()
            self._observation = None
            operation = "reset" if self._control is not None and self._control[0] == "reset" else "invalidate"
            self._control = (operation, generation)
            self._event("planned_hold")

    def acknowledge_hold(self) -> None:
        with self._lock:
            if self._hold_requested and not self._hold_acknowledged:
                self._hold_acknowledged = True
                self._hold_started = time.monotonic()
                self._observation = None  # require capture after the control-thread hold
        if self.failed and self._global_shutdown is not None:
            self._global_shutdown.set()

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        item = self.runtime.pop()
        if item is None:
            return None
        action, provenance = item
        self._set_dispatched_task(provenance.task)
        return action

    def _generate_text(self, obs_processed: dict, query: PolicyQuery) -> str:
        observation = self._query_observation
        generation = self._query_generation
        if observation is None:
            raise RuntimeError("Language requires a fresh held observation")
        if generation != self.runtime.generation or not self.runtime.active:
            raise RequestCancelled("Query superseded before submission")
        try:
            answer = self.client.query_language(
                observation,
                kind=query.kind.value,
                text=query.text,
                intent_generation=query.intent_generation,
                generation=generation,
                cancelled=lambda: self._stop_event.is_set() or self.runtime.generation != generation,
            )
        except ProtocolError as exc:
            # A completed text call returning an invalid answer is a query error.
            # Timeout or malformed wire context makes further model execution uncertain.
            if exc.code not in {ErrorCode.EXECUTION, ErrorCode.UNSUPPORTED}:
                self._query_fault(query, exc)
            raise
        except RequestCancelled:
            raise
        except Exception as exc:
            self._query_fault(query, exc)
            raise
        if (
            self.runtime.generation != generation
            or query.task_version != self.task_version
            or (
                query.kind is QueryKind.NEXT_SUBTASK
                and query.intent_generation != self.query_intent_generation
            )
        ):
            raise RequestCancelled("Query superseded by newer operator intent")
        self._event("language_result", intent_generation=query.intent_generation, kind=query.kind.value)
        return answer

    def _query_fault(self, query: PolicyQuery, error: Exception) -> None:
        current = self._query_context_valid(query)
        self._fault(str(error))
        if current:
            # Fault invalidation makes the base service discard its late result;
            # retain this one terminal query error for control-thread delivery.
            self._publish_answer(
                QueryAnswer(question=query.text, error=f"{type(error).__name__}: {error}", kind=query.kind)
            )

    def _query_context_valid(self, query: PolicyQuery) -> bool:
        return (
            self.runtime.generation == self._query_generation
            and query.task_version == self.task_version
            and (
                query.kind is not QueryKind.NEXT_SUBTASK
                or query.intent_generation == self.query_intent_generation
            )
            and self.runtime.active
            and not self.failed
            and not self._stop_event.is_set()
        )

    def _loop(self) -> None:
        try:
            while not self._stop_event.is_set() and not self.failed:
                with self._lock:
                    control = self._control
                    observation = self._observation
                    hold_ready = self._hold_requested and self._hold_acknowledged
                if control is not None:
                    # Language invalidation must be acknowledged by the motor thread
                    # before submitting any server-side generation operation.
                    if self._hold_requested and not hold_ready:
                        self._stop_event.wait(0.002)
                        continue
                    operation, generation = control
                    self.client.control(operation, generation)
                    with self._lock:
                        if self._control == control:
                            self._control = None
                    continue
                if not self.runtime.active or observation is None:
                    self.runtime.check_deadlines()
                    self._stop_event.wait(0.002)
                    continue
                if hold_ready:
                    with self._lock:
                        # Reset/pause may arrive after the initial loop snapshot.
                        # A new-generation query must wait for its ordered control
                        # acknowledgement, just like an action request.
                        eligible = (
                            self._control is None
                            and self.runtime.active
                            and self._hold_requested
                            and self._hold_acknowledged
                            and observation is self._observation
                            and observation.capture_time >= self._hold_started
                        )
                        generation = self.runtime.generation
                        if eligible:
                            self._query_observation, self._query_generation = observation, generation
                    if not eligible:
                        self._stop_event.wait(0.002)
                        continue
                    self._service_query({})
                    with self._lock:
                        if self.runtime.generation == generation and not self.failed:
                            self.runtime.held = False
                            self.runtime.started_at = time.monotonic()
                            self._hold_requested = False
                            self._hold_acknowledged = False
                            self._observation = None  # action always gets a post-query capture
                    continue
                if self.has_pending_query or not self.client.present:
                    self.runtime.check_deadlines()
                    self._stop_event.wait(0.002)
                    continue
                with self._lock:
                    if (
                        self._control is None
                        and observation is self._observation
                        and not self._hold_requested
                    ):
                        with self._task_lock:
                            observation = replace(
                                observation, task=self._task, task_version=self._task_version
                            )
                        request = self.runtime.begin(observation)
                    else:
                        request = None
                if request is None:
                    self._stop_event.wait(0.002)
                    continue
                self._event(
                    "request",
                    request_id=request.request_id,
                    capture_time=observation.capture_time,
                    task_version=observation.task_version,
                    delay=request.delay,
                )
                try:
                    request_generation = request.generation

                    def request_cancelled(request_generation: int = request_generation) -> bool:
                        return self._stop_event.is_set() or self.runtime.generation != request_generation

                    result = self.client.infer(
                        request,
                        cancelled=request_cancelled,
                    )
                except RequestCancelled:
                    continue
                accepted = self.runtime.accept(request, result, task_version=self.task_version)
                self._event(
                    "result",
                    request_id=request.request_id,
                    accepted=accepted,
                    turnaround_s=time.monotonic() - request.submitted_at,
                    source_age_s=time.monotonic() - observation.capture_time,
                    queue_playback_s=self.runtime.queue.qsize() * self.runtime.interval,
                    server_durations=result.server_durations,
                )
        except Exception as exc:
            self._traceback = traceback.format_exc()
            self._fault(str(exc))
            logger.exception("Remote inference worker fault")
        finally:
            if self.runtime.failure is not None:
                self._fault(self.runtime.failure)
            try:
                self.client.close()
            except Exception:
                logger.exception("Remote session close failed; presence/idle cleanup will release it")

    def begin_control_tick(self) -> None:
        """Count every motor tick, including holds, for the event sidecar."""
        self._tick += 1

    def record_dispatch(self, canonical: dict, command: dict, measured: dict) -> None:
        super().record_dispatch(canonical, command, measured)
        provenance = self.runtime.current
        if provenance is None:
            return
        scalar_state = {
            key: float(value)
            for key, value in measured.items()
            if isinstance(value, (float, int, np.number)) and np.isfinite(value)
        }
        self._event(
            "dispatch",
            provenance=asdict(provenance),
            canonical=canonical,
            command=command,
            measured=scalar_state,
            source_age_s=time.monotonic() - provenance.capture_time,
        )
