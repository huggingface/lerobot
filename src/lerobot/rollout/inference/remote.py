# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Remote rollout backend: worker-owned network exchanges, local motor permission."""

from __future__ import annotations

import logging
import time
import traceback
from collections import deque
from dataclasses import asdict, replace
from pathlib import Path
from queue import Empty, Full, Queue
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
            chunk_merge=config.chunk_merge,
            blend_steps=config.blend_steps,
            blend_weight=config.blend_weight,
            blend_indices=tuple(
                client.capabilities.action_feature.names.index(name) for name in config.blend_components
            ),
        )
        self._lock = RLock()
        self._observation: ObservationSnapshot | None = None
        self._control: tuple[str, int] | None = None
        self._hold_requested = False
        self._hold_acknowledged = False
        self._hold_started = 0.0
        self._hold_reason = "language"
        self._query_observation: ObservationSnapshot | None = None
        self._query_generation = 0
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._traceback: str | None = None
        self._event_writer: EventWriter | None = None
        self._tick = 0
        self._fault_reported = False
        self._refill_horizon_warned = False
        self._age_budget_warned = False
        # Bounded scalar history for cadence diagnostics; do not retain captures.
        self._last_request: tuple[int, float, int, int] | None = None
        # Console handlers may block. Control-thread events only enter this bounded
        # handoff; the network worker formats and writes all console diagnostics.
        self._log_events: Queue[dict] = Queue(maxsize=128)
        self._dropped_log_events = 0
        self._fault_logged = False
        self._last_summary_at = time.monotonic()
        self._recent_results: deque[tuple[float, float | None]] = deque(maxlen=100)
        self._accepted_since_summary = 0
        self._rejected_since_summary = 0
        self._request_diagnostics: dict = {}

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
            try:
                self._log_events.put_nowait(event)
            except Full:
                self._dropped_log_events += 1

    def _drain_log_events(self) -> None:
        """Worker-only console output; dropping diagnostics never changes execution."""
        # Limit work even if the control thread keeps producing new events.
        for _ in range(self._log_events.maxsize):
            try:
                event = self._log_events.get_nowait()
            except Empty:
                break
            logger.debug("Remote inference %s", event)
            name = event["event"]
            if name == "scheduling":
                logger.info(
                    "Remote execution: mode=%s merge=%s blend_steps=%s incoming_weight=%.2f; "
                    "action_rate=%.1f Hz horizon=%.3fs refill=%.3fs (effective=%.3fs); "
                    "max_source_age=%.3fs action_timeout=%.3fs startup_timeout=%.3fs",
                    self.config.mode,
                    self.config.chunk_merge,
                    self.config.blend_steps,
                    self.config.blend_weight,
                    1 / self.runtime.interval,
                    event["execution_horizon_s"],
                    event["configured_refill_s"],
                    event["effective_refill_s"],
                    self.config.max_observation_age_s,
                    self.config.action_timeout_s,
                    self.config.startup_timeout_s,
                )
            elif name == "planned_hold":
                logger.info("Remote %s: requesting a local hold before fresh resumption", event["reason"])
            elif name == "language_result":
                logger.info(
                    "Remote language query completed (%s, %.3fs); action resumption requires a fresh observation",
                    event["kind"],
                    event["duration_s"],
                )
            elif name == "task":
                logger.info("Remote instruction updated (version=%s)", event["task_version"])
            elif name == "reset":
                logger.info("Remote session reset requested; previous motion is invalidated")
            elif name == "request":
                self._request_diagnostics = event
            elif name == "result":
                self._recent_results.append((event["turnaround_s"], event["timing_margin_s"]))
                self._accepted_since_summary += bool(event["accepted"])
                self._rejected_since_summary += not event["accepted"]
        if self.runtime.failure is not None and not self._fault_logged:
            self._fault_logged = True
            self._log_fault(self.runtime.failure)
        now = time.monotonic()
        if now - self._last_summary_at >= 5 and (
            self._accepted_since_summary or self._rejected_since_summary
        ):
            turnarounds = [sample[0] for sample in self._recent_results]
            margins = [sample[1] for sample in self._recent_results if sample[1] is not None]
            logger.info(
                "Remote progress: accepted=%d rejected=%d; turnaround mean/max=%.3f/%.3fs "
                "(last %d results), estimated submission headroom min=%s; "
                "queued playback=%.3fs effective_refill=%.3fs; diagnostic_events_dropped=%d",
                self._accepted_since_summary,
                self._rejected_since_summary,
                sum(turnarounds) / len(turnarounds),
                max(turnarounds),
                len(turnarounds),
                f"{min(margins):.3f}s" if margins else "unavailable (no buffered submission)",
                self.runtime.queue.qsize() * self.runtime.interval,
                self.runtime.effective_refill,
                self._dropped_log_events,
            )
            self._last_summary_at = now
            self._accepted_since_summary = self._rejected_since_summary = 0

    def _log_fault(self, reason: str) -> None:
        """Explain a terminal failure without implying that requested motion completed."""
        if "exhaust" in reason.lower() or "continuation horizon" in reason.lower():
            hint = (
                "Check server completion/connection, turnaround tails and the usable suffix after trimming; "
                "compare refill headroom with that delay before changing the setting."
            )
        elif "observation" in reason.lower() or "stale" in reason.lower():
            hint = (
                "Check max_observation_age_s against execution horizon plus queued playback/turnaround; "
                "also inspect acquisition delay and oldest blended-contributor age."
            )
        elif "deadline" in reason.lower() or "timeout" in reason.lower():
            hint = "Check server completion and connectivity; increasing a timeout cannot replenish an empty buffer."
        else:
            hint = "Check the client/server error details and loaded contract before starting a new rollout."
        playback = self._request_diagnostics.get("playback_at_submission_s")
        logger.error(
            "Remote inference stopped: %s. Policy motion is revoked; rollout requests a local hold "
            "and follows its configured shutdown/return procedure. Last submission playback=%s; "
            "recent completed turnaround max=%s; refill configured/effective=%.3f/%.3fs. %s "
            "Request details: --inference.log_level=DEBUG; deployment=%s session=%s",
            reason,
            f"{playback:.3f}s" if playback is not None else "unavailable",
            f"{max(sample[0] for sample in self._recent_results):.3f}s"
            if self._recent_results
            else "unavailable",
            self.runtime.refill_seconds,
            self.runtime.effective_refill,
            hint,
            self.config.deployment,
            self.client.session_id,
        )

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
        self._event(
            "scheduling",
            scheduling="playback_threshold",
            mode=self.config.mode,
            chunk_merge=self.config.chunk_merge,
            blend_steps=self.config.blend_steps,
            configured_refill_s=self.runtime.refill_seconds,
            effective_refill_s=self.runtime.effective_refill,
            execution_horizon_s=self.client.capabilities.execution_steps * self.runtime.interval,
            aligned_task_change_bypasses_refill=self.config.chunk_merge == "aligned",
        )
        self._warn_refill_horizon()
        self._thread = Thread(target=self._loop, daemon=True, name="RemoteInference")
        self._thread.start()

    def _warn_refill_horizon(self) -> None:
        """Report saturation once, including when the measured latency floor grows."""
        horizon = self.client.capabilities.execution_steps * self.runtime.interval
        minimum_age = (
            horizon + min(horizon, self.runtime.effective_refill)
            if self.config.chunk_merge == "append" and self.config.mode == "chunk"
            else max(0.0, horizon - self.runtime.effective_refill)
        )
        if not self._age_budget_warned and self.runtime.max_age <= minimum_age + self.runtime.interval:
            self._age_budget_warned = True
            logger.warning(
                "max_observation_age_s=%.3fs leaves insufficient playback margin: mode=%s merge=%s "
                "horizon=%.3fs effective_refill=%.3fs estimated playback source age=%.3fs before "
                "capture/inference/jitter margin. Increase the explicit age budget or shorten the "
                "execution slice; aligned replacement and blending require their own measured margin.",
                self.runtime.max_age,
                self.config.mode,
                self.config.chunk_merge,
                horizon,
                self.runtime.effective_refill,
                minimum_age,
            )
        if (
            self.config.chunk_merge == "aligned"
            and not self._refill_horizon_warned
            and self.runtime.effective_refill + 1e-9 >= horizon
        ):
            self._refill_horizon_warned = True
            logger.warning(
                "Aligned effective refill %.3fs (configured %.3fs, recent turnaround %.3fs) "
                "covers the full execution horizon %.3fs; requests may follow every fresh advanced "
                "observation. Inspect request_spacing_s and committed_actions_since_request; "
                "lower configured refill only within measured latency headroom, or use a supported "
                "longer execution horizon. No setting was changed.",
                self.runtime.effective_refill,
                self.runtime.refill_seconds,
                self.runtime.turnaround,
                horizon,
            )

    def stop(self) -> None:
        self.pause()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                logger.warning("Remote worker did not stop within 2s; local hardware shutdown continues")
        if self._event_writer is not None:
            self._event_writer.close()

    def _invalidate(self, operation: str) -> None:
        with self._lock:
            generation = self.runtime.invalidate(held=True)
            self._observation = None
            self._hold_requested = False
            self._hold_acknowledged = False
            # A later motion invalidation must not erase a pending full reset.
            if self._control is not None and self._control[0] == "reset":
                operation = "reset"
            self._control = (operation, generation)
        self._event(operation)

    def pause(self) -> None:
        self.drop_pending_query()
        with self.runtime.lock:
            self.runtime.active = False
        self._invalidate("invalidate")

    def resume(self) -> None:
        if self.failed or self._stop_event.is_set():
            raise RuntimeError("Faulted/closed remote sessions require a new rollout")
        with self._lock:
            self.runtime.activate(held=self._control is not None)

    def reset(self) -> None:
        self.drop_pending_query()
        self._invalidate("reset")
        self._discard_task_change()

    def set_task(self, task: str) -> bool:
        if error := self.text_input_error(task, instruction=True):
            logger.warning("Instruction rejected: %s", error)
            return False
        # Acceptance takes the task lock before deciding eligibility. A result
        # accepted before the change remains valid buffered continuity.
        with self._lock, self._task_lock:
            if task == self._task:
                return False
            self._task = task
            self._task_changed = True
            self._task_version += 1
            if self.runtime.pending is not None:
                self._request_hold("instruction change")
            self._event("task", task=task, task_version=self._task_version)
        return True

    def text_input_error(self, text: str, *, instruction: bool = False) -> str | None:
        limit = self.client.descriptor["limits"]["max_input_chars"]
        if not instruction and not text.strip():
            return "Enter a non-empty question or goal."
        if len(text) > limit:
            return (
                f"Text contains {len(text)} characters; this deployment's limit is {limit}. "
                "Shorten it and try again."
            )
        return None

    def _queue_query(self, query: PolicyQuery) -> bool:
        if not self.supports_text_queries or self.failed or self.text_input_error(query.text) is not None:
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
        if not self.supports_text_queries:
            raise ValueError("This deployment does not support autosteering queries")
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
        expected = {feature.name for feature in self.client.capabilities.features}
        mapped = {key: value for key, value in mapped.items() if key in expected}
        # Mapping is applied once here; server processors require an empty rename map.
        with self._task_lock:
            task, version = self._task, self._task_version
        snapshot = ObservationSnapshot(mapped, sampled, task, version, uuid4().hex)
        with self._lock:
            # Sampling and endpoint commits both belong to this control thread;
            # no pop occurs between the hardware read and this anchor. Workers
            # may replace the future, but never advance the commitment cursor.
            if self.config.chunk_merge == "aligned":
                snapshot = self.runtime.anchor_observation(snapshot)
            self._observation = snapshot

    def dispatch_allowed(self) -> bool:
        # Called on every motor tick, before any interpolated target is sent.
        if self.has_pending_query:
            self._request_language_hold()
        with self._lock:
            hold_timeout = self.config.action_timeout_s + self.config.handshake_timeout_s
            if self._hold_reason == "language":
                hold_timeout += self.config.language_timeout_s
            if self._hold_requested and time.monotonic() - self._hold_started > hold_timeout:
                self._fault(f"Planned {self._hold_reason} hold deadline exceeded")
        allowed = self.runtime.dispatch_allowed()
        if self.runtime.failure is not None:
            self._fault(self.runtime.failure)
        return allowed

    def _request_language_hold(self) -> None:
        self._request_hold("language")

    def _request_hold(self, reason: str) -> None:
        """Invalidate an operator-triggered transition before fresh resumption."""
        with self._lock:
            if not self.runtime.active or self._hold_requested or self.failed:
                return
            generation = self.runtime.invalidate(held=True)
            self._hold_requested = True
            self._hold_acknowledged = False
            self._hold_started = time.monotonic()
            self._hold_reason = reason
            self._observation = None
            operation = "reset" if self._control is not None and self._control[0] == "reset" else "invalidate"
            self._control = (operation, generation)
            self._event("planned_hold", reason=reason)

    def acknowledge_hold(self) -> None:
        with self._lock:
            if self._hold_requested and not self._hold_acknowledged:
                self._hold_acknowledged = True
                self._hold_started = time.monotonic()
                self._observation = None  # require capture after the control-thread hold
        if self.failed and self._global_shutdown is not None:
            self._global_shutdown.set()

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        item = self.runtime.pop(control_tick=self._tick)
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
        started = time.monotonic()
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
        # A completed language response can fit the output budget yet be too
        # large to use as the next instruction. Reject inside the query service
        # path, outside transport-fault handling, before changing any task state.
        if query.kind is QueryKind.NEXT_SUBTASK and (
            error := self.text_input_error(answer, instruction=True)
        ):
            raise ValueError(f"Generated subtask cannot be used as an instruction: {error}")
        self._event(
            "language_result",
            intent_generation=query.intent_generation,
            kind=query.kind.value,
            duration_s=time.monotonic() - started,
        )
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
                self._drain_log_events()
                # Control-thread dispatch only queues bounded diagnostics. Log
                # from this worker so first-dispatch reporting adds no motor I/O.
                with self.runtime.lock:
                    dispatch_events = list(self.runtime.dispatch_events)
                    self.runtime.dispatch_events.clear()
                for event in dispatch_events:
                    self._event("first_dispatch", **event)
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
                    try:
                        self.client.control(
                            operation,
                            generation,
                            cancelled=lambda: self._stop_event.is_set() or self.failed,
                        )
                    except RequestCancelled:
                        continue
                    with self._lock:
                        if self._control == control:
                            self._control = None
                            if not self._hold_requested:
                                self.runtime.release_hold(generation)
                                self._observation = None
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
                        if self.runtime.release_hold(generation):
                            self._hold_requested = False
                            self._hold_acknowledged = False
                            self._observation = None  # action always gets a post-query capture
                    continue
                if self.has_pending_query or not self.client.present:
                    self.runtime.check_deadlines()
                    self._stop_event.wait(0.002)
                    continue
                if not self.runtime.should_request(task_version=self.task_version):
                    self._stop_event.wait(0.002)
                    continue
                with self._lock:
                    if (
                        self._control is None
                        and observation is self._observation
                        and not self._hold_requested
                    ):
                        with self._task_lock:
                            if observation.task_version != self._task_version:
                                observation = replace(
                                    observation, task=self._task, task_version=self._task_version
                                )
                        request = self.runtime.begin(observation)
                    else:
                        request = None
                if request is None:
                    self._stop_event.wait(0.002)
                    continue
                previous = self._last_request
                request_spacing = committed_since_request = None
                task_changed = False
                if previous is not None and previous[0] == request.generation:
                    request_spacing = request.submitted_at - previous[1]
                    committed_since_request = request.continuation.cursor - previous[2]
                    task_changed = observation.task_version != previous[3]
                self._event(
                    "request",
                    request_id=request.request_id,
                    capture_time=observation.capture_time,
                    task_version=observation.task_version,
                    delay=request.delay,
                    chunk_merge=self.config.chunk_merge,
                    configured_refill_s=self.runtime.refill_seconds,
                    effective_refill_s=self.runtime.effective_refill,
                    scheduling="playback_threshold",
                    request_spacing_s=request_spacing,
                    committed_actions_since_request=committed_since_request,
                    task_changed_since_request=task_changed,
                    recent_turnaround_s=self.runtime.turnaround,
                    playback_at_submission_s=request.playback_at_submission,
                    observation_age_at_submission_s=request.submitted_at - observation.capture_time,
                    observation_cursor=observation.action_cursor,
                    submission_cursor=request.continuation.cursor,
                )
                self._last_request = (
                    request.generation,
                    request.submitted_at,
                    request.continuation.cursor,
                    observation.task_version,
                )
                self._drain_log_events()
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
                # Acceptance and an operator retarget must have one order: an
                # old in-flight result cannot pass using a previously read version.
                with self._task_lock:
                    accepted = self.runtime.accept(request, result, task_version=self._task_version)
                self._warn_refill_horizon()
                turnaround = time.monotonic() - request.submitted_at
                self._event(
                    "result",
                    request_id=request.request_id,
                    accepted=accepted,
                    turnaround_s=turnaround,
                    # Diagnostic estimate only: the committed interpolation endpoint
                    # is excluded, and cursor trimming can further reduce usable work.
                    timing_margin_s=request.playback_at_submission - turnaround
                    if request.playback_at_submission > 0
                    else None,
                    source_age_s=time.monotonic() - observation.capture_time,
                    queue_playback_s=self.runtime.queue.qsize() * self.runtime.interval,
                    server_durations=result.server_durations,
                    merge=self.runtime.last_accept.copy(),
                )
        except Exception as exc:
            self._traceback = traceback.format_exc()
            self._fault(str(exc))
            logger.exception("Remote inference worker fault")
        finally:
            if self.runtime.failure is not None:
                self._fault(self.runtime.failure)
            self._drain_log_events()
            try:
                acknowledged = self.client.close()
                if acknowledged is False:
                    logger.info("Remote transport closed without a server acknowledgement")
                else:
                    logger.info("Remote session closed; server acknowledged session release")
            except (TimeoutError, ProtocolError) as exc:
                logger.warning(
                    "Remote session close was not acknowledged (%s); transport was closed and server "
                    "presence/idle cleanup will release ownership after pending model work completes",
                    exc,
                )
                logger.debug("Remote session close details", exc_info=True)
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
