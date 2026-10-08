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

"""Exclusive session ownership and a single ordered policy worker.

Model calls and all processor resets happen on one worker. A hung call keeps the
process occupied, including after close or presence loss: no replacement worker
can accidentally execute against that same mutable model.
"""

from __future__ import annotations

import logging
import math
import time
from collections import OrderedDict, deque
from concurrent.futures import Future
from dataclasses import asdict, dataclass, field
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Any
from uuid import uuid4

import numpy as np
import torch

from lerobot.inference import ExecutionMode, FeatureSpec, ObservationSnapshot, PolicyRunner, QueryKind
from lerobot.transport.zenoh import (
    BoundedQueryable,
    BoundedSubscriber,
    PendingQuery,
    PresenceEvent,
    ZenohTransport,
)

from .build_info import SOFTWARE_BUILD
from .chunk_contract import (
    CHUNK_ALIGNMENT,
    RTC_MODEL_SPACE,
    default_chunk_settings,
    required_chunk_capabilities,
    validate_chunk_contract,
)
from .codec import CodecLimits, decode_message, encode_message
from .protocol import (
    IDENTITY_KEYS,
    Envelope,
    ErrorCode,
    MessageType,
    ProtocolError,
    deployment_prefix,
    instance_prefix,
    session_prefix,
)

logger = logging.getLogger(__name__)


def _remember(mapping: OrderedDict, key: Any, value: Any, capacity: int = 128) -> None:
    mapping[key] = value
    while len(mapping) > capacity:
        mapping.popitem(last=False)


@dataclass
class _Session:
    identity: str
    generation: int = 0
    mode: ExecutionMode = ExecutionMode.CHUNK
    chunk_settings: dict[str, Any] = field(default_factory=default_chunk_settings)
    busy: bool = False
    ready: bool = False
    closing: bool = False
    faulted: bool = False
    present: bool | None = None
    presence_established: bool = False
    absent_since: float | None = None
    cleanup_reason: str | None = None
    cleanup_queue_blocked: bool = False
    last_sequence: int = -1


@dataclass
class _SessionChannels:
    session_id: str
    key: str
    control: BoundedQueryable
    observations: BoundedSubscriber[bytes]
    language: BoundedSubscriber[bytes]
    presence: BoundedSubscriber[PresenceEvent]
    present: bool = False

    def close(self) -> None:
        self.control.close()
        self.observations.close()
        self.language.close()
        self.presence.close()


@dataclass(frozen=True)
class _PendingReply:
    future: Future[Envelope]
    target: PendingQuery | str
    # Only session controls retain their queryable until a reply has been flushed.
    session_id: str | None = None


class SessionWorker:
    """Bounded admission surface. ``submit`` never invokes or waits for the policy."""

    def __init__(
        self,
        runner: PolicyRunner,
        *,
        deployment: str,
        artifact_identity: str,
        semantics: str,
        action_deadline_s: float = 5.0,
        language_deadline_s: float = 60.0,
        idle_timeout_s: float = 10.0,
        max_input_chars: int = 4096,
        max_output_chars: int = 8192,
    ) -> None:
        """Start the single policy worker with bounded command and identity bookkeeping."""
        self.runner = runner
        self.instance_id = uuid4().hex
        self.deployment = deployment
        self.artifact_identity = artifact_identity
        self.semantics = semantics
        self.action_deadline_s = action_deadline_s
        self.language_deadline_s = language_deadline_s
        self.idle_timeout_s = idle_timeout_s
        self.max_input_chars = max_input_chars
        self.max_output_chars = max_output_chars
        self.execution_contracts = (
            [CHUNK_ALIGNMENT]
            if ExecutionMode.CHUNK in runner.capabilities.modes
            and runner.capabilities.action_representation == "canonical"
            else []
        )
        if runner.capabilities.model_action_dim not in (None, runner.capabilities.action_feature.shape[0]):
            self.execution_contracts.append(RTC_MODEL_SPACE)
        self._lock = Lock()
        self._commands: Queue[tuple[Envelope, Future[Envelope], float]] = Queue(maxsize=8)
        self._session: _Session | None = None
        self._failure: str | None = None
        self._opens: OrderedDict[str, Future[Envelope]] = OrderedDict()
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._controls: OrderedDict[str, Future[Envelope]] = OrderedDict()
        self._closed_controls: OrderedDict[tuple[str, str], Future[Envelope]] = OrderedDict()
        self._recent_operations: deque[tuple[float, float]] = deque(maxlen=100)
        self._last_summary_at = time.monotonic()
        self._actions_since_summary = self._language_since_summary = 0
        self._errors_since_summary = self._stale_since_summary = 0
        self._invalidations_since_summary = self._resets_since_summary = 0
        self._control_wait_max = 0.0
        self._active_operation: str | None = None
        self._stopping = Event()
        self._thread = Thread(target=self._run, name="PolicyWorker", daemon=True)
        self._thread.start()

    @property
    def descriptor(self) -> dict[str, Any]:
        """Describe the preloaded artifact and its effective serving contract."""
        with self._lock:
            return self._descriptor_locked()

    def _session_diagnostics_locked(self) -> dict[str, Any]:
        session = self._session
        remaining = (
            None
            if session is None or session.absent_since is None
            else max(0.0, self.idle_timeout_s - (time.monotonic() - session.absent_since))
        )
        return {
            "owner": None if session is None else session.identity,
            "client_present": None if session is None else session.present,
            "absence_grace_remaining_s": remaining,
            "cleanup_pending": session is not None and session.closing,
            "cleanup_reason": None if session is None else session.cleanup_reason,
            "inference_pending": session is not None and session.busy,
            "admission_blocker": self._admission_blocker_locked(),
        }

    def _admission_blocker_locked(self) -> str | None:
        session = self._session
        if session is None:
            return None
        if session.closing:
            return "unfinished_inference" if session.busy else "worker_cleanup_pending"
        if session.cleanup_queue_blocked:
            return "cleanup_queue_full"
        if session.absent_since is not None:
            return "absence_grace" if session.presence_established else "awaiting_initial_presence"
        return "active_session"

    def _session_status_locked(self) -> str:
        session = self._session
        if self._failure is not None:
            return "unhealthy"
        if session is None:
            return "idle"
        if session.faulted:
            return "faulted"
        if session.closing:
            return "closing"
        if session.busy:
            return "busy"
        return "active"

    def _descriptor_locked(self) -> dict[str, Any]:
        capabilities = asdict(self.runner.capabilities)
        if capabilities["model_action_dim"] in (None, self.runner.capabilities.action_feature.shape[0]):
            capabilities.pop("model_action_dim")
        return {
            "deployment": self.deployment,
            "instance_id": self.instance_id,
            "artifact_identity": self.artifact_identity,
            "software": asdict(SOFTWARE_BUILD),
            "semantics": self.semantics,
            "capabilities": capabilities,
            "execution_contracts": list(self.execution_contracts),
            "ready": self._failure is None and not self._stopping.is_set(),
            "available": self._failure is None and self._session is None and not self._stopping.is_set(),
            "failure": self._failure,
            "session": self._session_diagnostics_locked(),
            "session_status": self._session_status_locked(),
            "limits": {
                "action_deadline_s": self.action_deadline_s,
                "language_deadline_s": self.language_deadline_s,
                "idle_timeout_s": self.idle_timeout_s,
                "max_input_chars": self.max_input_chars,
                "max_output_chars": self.max_output_chars,
                "codec": asdict(CodecLimits()),
                "inflight_inference": 1,
            },
        }

    @property
    def session_id(self) -> str | None:
        """Return the exclusive owner, including while its close waits behind a model call."""
        with self._lock:
            return None if self._session is None else self._session.identity

    def submit(self, message: Envelope) -> Future[Envelope]:
        """Validate and enqueue work without running policy code on the caller's thread."""
        future: Future[Envelope] = Future()
        with self._lock:
            try:
                if message.message_type is MessageType.DESCRIBE:
                    future.set_result(
                        Envelope(
                            MessageType.DESCRIPTOR,
                            self.instance_id,
                            request_id=message.request_id,
                            body=self._descriptor_locked(),
                        )
                    )
                    return future
                if message.instance_id != self.instance_id:
                    raise ProtocolError(ErrorCode.STALE, "Server instance does not match")
                if self._failure is not None:
                    raise ProtocolError(ErrorCode.EXECUTION, self._failure)
                if message.message_type is MessageType.OPEN:
                    if not message.request_id:
                        raise ProtocolError(ErrorCode.MALFORMED, "Open operation ID is required")
                    if message.request_id in self._opens:
                        return self._opens[message.request_id]
                    if self._session is not None:
                        state = self._session_diagnostics_locked()
                        detail = (
                            f"Deployment owned by session={state['owner']}; "
                            f"admission_blocker={state['admission_blocker']} "
                            f"client_present={state['client_present']} "
                            f"absence_grace_remaining_s={state['absence_grace_remaining_s']} "
                            f"cleanup_pending={state['cleanup_pending']} "
                            f"inference_pending={state['inference_pending']}. "
                            "A new session requires the previous owner's close/reset to finish; "
                            "unfinished model calls are never taken over."
                        )
                        logger.info("Session admission blocked instance=%s %s", self.instance_id, detail)
                        raise ProtocolError(
                            ErrorCode.BUSY,
                            detail,
                            details={
                                "admission_blocker": state["admission_blocker"],
                                "absence_grace_remaining_s": state["absence_grace_remaining_s"],
                            },
                        )
                    if self._commands.full():
                        raise ProtocolError(ErrorCode.BUSY, "Policy command queue is full")
                    self._validate_open(message.body)
                    self._session = _Session(
                        uuid4().hex,
                        mode=ExecutionMode(message.body["mode"]),
                        chunk_settings=message.body.get("chunk_settings", default_chunk_settings()),
                    )
                    self._seen.clear()
                    self._controls.clear()
                    _remember(self._opens, message.request_id, future, 32)
                else:
                    if (
                        message.message_type is MessageType.CONTROL
                        and message.body.get("operation") == "close"
                    ):
                        previous = self._closed_controls.get((message.session_id, message.request_id))
                        if previous is not None:
                            return previous
                    session = self._session
                    if session is None or message.session_id != session.identity:
                        raise ProtocolError(ErrorCode.STALE, "Session does not match")
                    if message.message_type is MessageType.CONTROL:
                        if message.request_id in self._controls:
                            return self._controls[message.request_id]
                        if self._commands.full():
                            raise ProtocolError(ErrorCode.BUSY, "Policy command queue is full")
                        operation = message.body.get("operation")
                        if operation not in {"reset", "invalidate", "close", "status"}:
                            raise ProtocolError(ErrorCode.MALFORMED, "Unknown control operation")
                        if operation != "close" and message.generation < session.generation:
                            raise ProtocolError(ErrorCode.STALE, "Obsolete control generation")
                        if operation == "close":
                            self._mark_closing_locked(session, reason="client_close")
                            _remember(
                                self._closed_controls, (message.session_id, message.request_id), future, 32
                            )
                        _remember(self._controls, message.request_id, future, 32)
                    elif message.message_type in {MessageType.OBSERVATION, MessageType.LANGUAGE_REQUEST}:
                        if not session.ready or session.closing or session.faulted:
                            raise ProtocolError(ErrorCode.STALE, "Session is not accepting inference")
                        if message.generation != session.generation:
                            raise ProtocolError(ErrorCode.STALE, "Generation has not been acknowledged")
                        if message.request_id in self._seen:
                            raise ProtocolError(ErrorCode.STALE, "Duplicate or completed request")
                        if session.busy:
                            raise ProtocolError(
                                ErrorCode.BUSY, "One inference operation is already outstanding"
                            )
                        self._validate_data(message)
                        if message.body["sequence"] <= session.last_sequence:
                            raise ProtocolError(ErrorCode.STALE, "Request sequence was already consumed")
                        if self._commands.full():
                            raise ProtocolError(ErrorCode.BUSY, "Policy command queue is full")
                        session.busy = True
                        session.last_sequence = message.body["sequence"]
                        _remember(self._seen, message.request_id, None)
                    else:
                        raise ProtocolError(ErrorCode.MALFORMED, "Unexpected request message type")
                self._commands.put_nowait((message, future, time.monotonic()))
                if message.message_type is MessageType.LANGUAGE_REQUEST:
                    logger.info(
                        "Language request accepted: deployment=%s kind=%s",
                        self.deployment,
                        message.body["kind"],
                    )
            except (
                ProtocolError,
                ValueError,
                KeyError,
                TypeError,
                AttributeError,
                OverflowError,
                Full,
            ) as exc:
                code = (
                    exc.code
                    if isinstance(exc, ProtocolError)
                    else ErrorCode.BUSY
                    if isinstance(exc, Full)
                    else ErrorCode.MALFORMED
                )
                details = exc.details if isinstance(exc, ProtocolError) else None
                future.set_result(message.error(code, str(exc), details=details))
        return future

    def _mark_closing_locked(self, session: _Session, *, reason: str) -> None:
        if not session.closing:
            session.closing = True
            session.cleanup_reason = reason
            logger.info(
                "Session cleanup queued instance=%s session=%s reason=%s inference_pending=%s; "
                "ownership releases only after worker close/reset completes",
                self.instance_id,
                session.identity,
                reason,
                session.busy,
            )

    def _validate_open(self, body: dict[str, Any]) -> None:
        caps = self.runner.capabilities
        required = body.get("required_capabilities", [])
        if not isinstance(required, list) or any(
            not isinstance(name, str) or name not in self.execution_contracts for name in required
        ):
            raise ProtocolError(ErrorCode.UNSUPPORTED, "Unknown required protocol capabilities")
        settings = body.get("chunk_settings", default_chunk_settings())
        try:
            validate_chunk_contract(settings, caps)
        except ValueError as exc:
            raise ProtocolError(ErrorCode.INCOMPATIBLE, str(exc)) from exc
        expected = required_chunk_capabilities(settings)
        if body.get("mode") != ExecutionMode.CHUNK and caps.model_action_dim not in (
            None,
            caps.action_feature.shape[0],
        ):
            expected.append(RTC_MODEL_SPACE)
        if required != expected:
            raise ProtocolError(ErrorCode.INCOMPATIBLE, "Required capabilities differ from chunk_settings")
        if settings["chunk_merge"] == "aligned" and body.get("mode") != ExecutionMode.CHUNK:
            raise ProtocolError(ErrorCode.INCOMPATIBLE, "Aligned merge requires chunk execution")
        if body.get("expected_artifact") not in (None, self.artifact_identity):
            raise ProtocolError(ErrorCode.INCOMPATIBLE, "Deployed artifact identity differs")
        if body.get("semantics") != self.semantics:
            raise ProtocolError(ErrorCode.INCOMPATIBLE, "Robot/action semantic profile differs")
        if ExecutionMode(body["mode"]) not in caps.modes:
            raise ProtocolError(ErrorCode.UNSUPPORTED, "Requested execution mode is unavailable")
        if not math.isclose(float(body["action_interval"]), caps.action_interval, rel_tol=1e-6):
            raise ProtocolError(ErrorCode.INCOMPATIBLE, "Policy action interval differs")
        features = tuple(FeatureSpec(**value) for value in body["features"])
        action = FeatureSpec(**body["action_feature"])
        if features != caps.features or action != caps.action_feature:
            raise ProtocolError(ErrorCode.INCOMPATIBLE, "Ordered feature schema or semantics differ")
        if body.get("encoding") not in {"raw", "jpeg"}:
            raise ProtocolError(ErrorCode.UNSUPPORTED, "Unsupported image encoding")

    def _validate_data(self, message: Envelope) -> None:
        body = message.body
        if type(body.get("sequence")) is not int or not 0 <= body["sequence"] < 2**63:
            raise ProtocolError(ErrorCode.MALFORMED, "A bounded monotonic request sequence is required")
        if body.get("artifact_identity") != self.artifact_identity:
            raise ProtocolError(ErrorCode.INCOMPATIBLE, "Artifact does not match")
        if not isinstance(body.get("task"), str) or len(body["task"]) > self.max_input_chars:
            raise ProtocolError(ErrorCode.MALFORMED, "Invalid or oversized task")
        if type(body.get("task_version")) is not int or body["task_version"] < 0:
            raise ProtocolError(ErrorCode.MALFORMED, "Invalid task version")
        if not isinstance(body.get("observation_id"), str) or not body["observation_id"]:
            raise ProtocolError(ErrorCode.MALFORMED, "Observation identity required")
        if not isinstance(body.get("capture_time"), (int, float)) or not math.isfinite(body["capture_time"]):
            raise ProtocolError(ErrorCode.MALFORMED, "Invalid opaque capture time")
        supplied = body.get("features")
        caps = self.runner.capabilities
        if not isinstance(supplied, dict) or set(supplied) != {f.name for f in caps.features}:
            raise ProtocolError(ErrorCode.INCOMPATIBLE, "Observation feature keys differ")
        for feature in caps.features:
            value = supplied[feature.name]
            if (
                not isinstance(value, np.ndarray)
                or value.shape != feature.shape
                or value.dtype.name != feature.dtype
            ):
                raise ProtocolError(ErrorCode.INCOMPATIBLE, f"Invalid feature {feature.name}")
            if not np.isfinite(value).all():
                raise ProtocolError(ErrorCode.MALFORMED, "Non-finite observation values")
        if message.message_type is MessageType.LANGUAGE_REQUEST:
            if not caps.language:
                raise ProtocolError(ErrorCode.UNSUPPORTED, "Deployment has no text capability")
            try:
                QueryKind(body.get("kind"))
            except (ValueError, TypeError) as exc:
                raise ProtocolError(ErrorCode.UNSUPPORTED, "Unknown query kind") from exc
            if not isinstance(body.get("text"), str) or not 0 < len(body["text"]) <= self.max_input_chars:
                raise ProtocolError(ErrorCode.MALFORMED, "Invalid or oversized query")
            if type(body.get("intent_generation")) is not int or body["intent_generation"] < 0:
                raise ProtocolError(ErrorCode.MALFORMED, "Invalid query intent")
        else:
            if self._session is not None and self._session.chunk_settings["chunk_merge"] == "aligned":
                if any(
                    type(body.get(key)) is not int or not 0 <= body[key] < 2**63
                    for key in ("observation_cursor", "cursor")
                ):
                    raise ProtocolError(ErrorCode.MALFORMED, "Aligned chunks require bounded action cursors")
                if body["cursor"] < body["observation_cursor"]:
                    raise ProtocolError(ErrorCode.MALFORMED, "Request cursor predates the observation")
            if type(body.get("delay")) is not int or not 0 <= body["delay"] <= caps.prediction_steps:
                raise ProtocolError(ErrorCode.MALFORMED, "Invalid RTC delay")
            for name in ("model_continuation", "canonical_continuation"):
                value = body.get(name)
                width = (
                    caps.model_action_dim or caps.action_feature.shape[0]
                    if name == "model_continuation"
                    else caps.action_feature.shape[0]
                )
                if value is not None and (
                    not isinstance(value, np.ndarray)
                    or value.ndim != 2
                    or value.shape[1:] != (width,)
                    or value.dtype.name != caps.action_feature.dtype
                    or len(value) > caps.prediction_steps
                    or not np.isfinite(value).all()
                ):
                    raise ProtocolError(ErrorCode.MALFORMED, "Invalid continuation")
            model, canonical = body.get("model_continuation"), body.get("canonical_continuation")
            if model is not None and canonical is not None and len(model) != len(canonical):
                raise ProtocolError(ErrorCode.MALFORMED, "Continuation spaces must cover identical steps")

    def _reset_runner(self, *, full: bool) -> None:
        """A failed reset leaves model state unknown; only a server restart recovers it."""
        try:
            self.runner.reset(full=full)
        except Exception as exc:
            with self._lock:
                self._failure = "Policy reset failed; deployment is unavailable until the server restarts"
                if self._session is not None:
                    self._session.faulted = True
            raise RuntimeError(self._failure) from exc

    def _execute(self, message: Envelope) -> Envelope:
        if self._failure is not None:
            return message.error(ErrorCode.EXECUTION, self._failure)
        session = self._session
        if session is None:
            return message.error(ErrorCode.STALE, "Session closed")
        if message.message_type is not MessageType.OPEN and message.session_id != session.identity:
            return message.error(ErrorCode.STALE, "Session ownership changed before execution")
        if message.message_type is MessageType.OPEN:
            self._reset_runner(full=True)
            session.ready = True
            self._recent_operations.clear()
            self._actions_since_summary = self._language_since_summary = 0
            self._errors_since_summary = self._stale_since_summary = 0
            self._invalidations_since_summary = self._resets_since_summary = 0
            self._control_wait_max = 0.0
            self._last_summary_at = time.monotonic()
            logger.info("Session admitted instance=%s session=%s", self.instance_id, session.identity)
            return Envelope(
                MessageType.ACCEPTED,
                self.instance_id,
                session.identity,
                0,
                message.request_id,
                {**self.descriptor, "mode": session.mode.value, "chunk_settings": session.chunk_settings},
            )
        if message.message_type is MessageType.CONTROL:
            operation = message.body["operation"]
            if operation != "close" and message.generation < session.generation:
                return message.error(ErrorCode.STALE, "Control generation became obsolete while queued")
            if operation == "status" and message.generation != session.generation:
                return message.error(ErrorCode.STALE, "Status generation does not match")
            if operation == "close" or (
                operation in {"reset", "invalidate"} and message.generation > session.generation
            ):
                self._reset_runner(full=operation != "invalidate")
                session.generation = message.generation
            response = message.reply(
                MessageType.ACK, {"operation": operation, "applied_generation": session.generation}
            )
            if operation == "status":
                with self._lock:
                    response.body["session"] = self._session_diagnostics_locked()
            if operation == "close":
                with self._lock:
                    self._session = None
                logger.info(
                    "Session released instance=%s session=%s reason=%s; deployment available for a new session",
                    self.instance_id,
                    session.identity,
                    session.cleanup_reason,
                )
            return response
        if session.faulted or session.closing or message.generation != session.generation:
            return message.error(ErrorCode.STALE, "Motion generation invalidated")
        body = message.body
        observation = ObservationSnapshot(
            body["features"], body["capture_time"], body["task"], body["task_version"], body["observation_id"]
        )
        started = time.monotonic()
        identity = {key: body[key] for key in IDENTITY_KEYS}
        if (
            message.message_type is MessageType.OBSERVATION
            and session.chunk_settings["chunk_merge"] == "aligned"
        ):
            identity.update({key: body[key] for key in ("observation_cursor", "cursor")})
        if message.message_type is MessageType.LANGUAGE_REQUEST:
            answer = self.runner.query(observation, kind=body["kind"], text=body["text"])
            if (
                not isinstance(answer, str)
                or not answer.strip()
                or len(answer) > self.max_output_chars
                or len(answer.encode("utf-8")) > CodecLimits().max_string_bytes
            ):
                raise ValueError("Policy returned an invalid or oversized text answer")
            elapsed = time.monotonic() - started
            if elapsed > self.language_deadline_s:
                session.faulted = True
                return message.error(ErrorCode.TIMEOUT, "Language execution deadline exceeded")
            return message.reply(
                MessageType.LANGUAGE_RESULT,
                {
                    **identity,
                    "answer": answer,
                    "kind": body["kind"],
                    "intent_generation": body["intent_generation"],
                    "duration_s": elapsed,
                },
            )

        def tensor(name: str) -> torch.Tensor | None:
            value = body.get(name)
            return None if value is None else torch.from_numpy(value.copy())

        chunk = self.runner.predict(
            observation,
            mode=session.mode,
            inference_delay=body["delay"],
            model_continuation=tensor("model_continuation"),
            canonical_continuation=tensor("canonical_continuation"),
        )
        if time.monotonic() - started > self.action_deadline_s:
            session.faulted = True
            return message.error(ErrorCode.TIMEOUT, "Action execution deadline exceeded")
        return message.reply(
            MessageType.ACTION,
            {
                **identity,
                "canonical_actions": chunk.canonical_actions,
                "model_actions": chunk.model_actions,
                "execution_steps": chunk.execution_steps,
                "server_durations": chunk.server_durations,
            },
        )

    def _run(self) -> None:
        while not self._stopping.is_set():
            try:
                message, future, queued_at = self._commands.get(timeout=0.05)
            except Empty:
                continue
            started = time.monotonic()
            operation = (
                message.body["operation"]
                if message.message_type is MessageType.CONTROL
                else str(message.message_type)
            )
            with self._lock:
                self._active_operation = operation
            inference_operation = message.message_type in {
                MessageType.OBSERVATION,
                MessageType.LANGUAGE_REQUEST,
            }
            deadline = (
                self.language_deadline_s
                if message.message_type is MessageType.LANGUAGE_REQUEST
                else self.action_deadline_s
            )
            try:
                if inference_operation and started - queued_at > deadline:
                    response = message.error(ErrorCode.TIMEOUT, "Policy worker queue deadline exceeded")
                    if self._session is not None:
                        self._session.faulted = True
                else:
                    response = self._execute(message)
            except Exception as exc:
                logger.exception(
                    "Policy worker failure instance=%s session=%s request=%s",
                    self.instance_id,
                    message.session_id,
                    message.request_id,
                )
                timed_out = time.monotonic() - queued_at > deadline
                response = message.error(ErrorCode.TIMEOUT if timed_out else ErrorCode.EXECUTION, str(exc))
                if (
                    timed_out or message.message_type is not MessageType.LANGUAGE_REQUEST
                ) and self._session is not None:
                    self._session.faulted = True
            finished = time.monotonic()
            if response.message_type is MessageType.ACK and operation in {"invalidate", "reset"}:
                if operation == "invalidate":
                    self._invalidations_since_summary += 1
                else:
                    self._resets_since_summary += 1
                self._control_wait_max = max(self._control_wait_max, finished - queued_at)
            if inference_operation:
                if finished - queued_at > deadline:
                    response = message.error(ErrorCode.TIMEOUT, "Policy operation deadline exceeded")
                    if self._session is not None:
                        self._session.faulted = True
                elif response.message_type in {MessageType.ACTION, MessageType.LANGUAGE_RESULT}:
                    response.body["server_durations"] = {
                        **response.body.get("server_durations", {}),
                        "queue": started - queued_at,
                        "worker": finished - started,
                    }
                logger.debug(
                    "Policy operation deployment=%s instance=%s session=%s generation=%s request=%s type=%s outcome=%s queue_s=%.6f worker_s=%.6f",
                    self.deployment,
                    self.instance_id,
                    message.session_id,
                    message.generation,
                    message.request_id,
                    message.message_type,
                    response.body.get("code", str(response.message_type)),
                    started - queued_at,
                    finished - started,
                )
                self._recent_operations.append((started - queued_at, finished - started))
                if message.message_type is MessageType.OBSERVATION:
                    self._actions_since_summary += 1
                else:
                    self._language_since_summary += 1
                    logger.info(
                        "Language request %s: deployment=%s kind=%s elapsed=%.3fs outcome=%s",
                        "completed" if response.message_type is MessageType.LANGUAGE_RESULT else "failed",
                        self.deployment,
                        message.body["kind"],
                        finished - queued_at,
                        response.body.get("code", str(response.message_type)),
                    )
                if response.message_type is MessageType.ERROR:
                    if response.body.get("code") == ErrorCode.STALE:
                        self._stale_since_summary += 1
                    else:
                        self._errors_since_summary += 1
                if finished - self._last_summary_at >= 5:
                    count = len(self._recent_operations)
                    logger.info(
                        "Policy progress: deployment=%s actions=%d text_queries=%d errors=%d stale=%d "
                        "invalidations=%d resets=%d control wait max=%.3fs over %.1fs; "
                        "queue mean/max=%.3f/%.3fs worker mean/max=%.3f/%.3fs (last %d operations, all outcomes); "
                        "client logs include transport delay and playback headroom",
                        self.deployment,
                        self._actions_since_summary,
                        self._language_since_summary,
                        self._errors_since_summary,
                        self._stale_since_summary,
                        self._invalidations_since_summary,
                        self._resets_since_summary,
                        self._control_wait_max,
                        finished - self._last_summary_at,
                        sum(sample[0] for sample in self._recent_operations) / count,
                        max(sample[0] for sample in self._recent_operations),
                        sum(sample[1] for sample in self._recent_operations) / count,
                        max(sample[1] for sample in self._recent_operations),
                        count,
                    )
                    self._last_summary_at = finished
                    self._actions_since_summary = self._language_since_summary = 0
                    self._errors_since_summary = self._stale_since_summary = 0
                    self._invalidations_since_summary = self._resets_since_summary = 0
                    self._control_wait_max = 0.0
                if (
                    response.message_type is MessageType.ERROR
                    and response.body.get("code") != ErrorCode.STALE
                ):
                    logger.warning(
                        "Policy operation rejected: deployment=%s code=%s reason=%s request=%s; "
                        "check client logs for the local motion outcome",
                        self.deployment,
                        response.body.get("code"),
                        response.body.get("message"),
                        message.request_id,
                    )
            with self._lock:
                self._active_operation = None
                if self._session is not None and message.message_type in {
                    MessageType.OBSERVATION,
                    MessageType.LANGUAGE_REQUEST,
                }:
                    self._session.busy = False
            future.set_result(response)

    def expire(self, *, present: bool) -> None:
        """Bound initial presence and later absence; release only through the worker."""
        with self._lock:
            session = self._session
            if session is None or session.closing or self._failure is not None:
                return
            now = time.monotonic()
            if present:
                if not session.presence_established:
                    logger.info(
                        "Session initial presence established instance=%s session=%s",
                        self.instance_id,
                        session.identity,
                    )
                elif session.present is False:
                    logger.info(
                        "Session presence restored instance=%s session=%s; absence cleanup cancelled",
                        self.instance_id,
                        session.identity,
                    )
                session.present = True
                session.presence_established = True
                session.absent_since = None
                session.cleanup_queue_blocked = False
                return
            session.present = False
            if session.absent_since is None:
                session.absent_since = now
                logger.info(
                    (
                        "Session client absent instance=%s session=%s grace_s=%.3f inference_pending=%s"
                        if session.presence_established
                        else "Session awaiting initial presence instance=%s session=%s grace_s=%.3f inference_pending=%s"
                    ),
                    self.instance_id,
                    session.identity,
                    self.idle_timeout_s,
                    session.busy,
                )
            if now - session.absent_since < self.idle_timeout_s:
                return
            if self._commands.full():
                if not session.cleanup_queue_blocked:
                    logger.info(
                        "Session presence grace expired instance=%s session=%s; cleanup waiting for worker queue capacity",
                        self.instance_id,
                        session.identity,
                    )
                session.cleanup_queue_blocked = True
                return
            close = Envelope(
                MessageType.CONTROL,
                self.instance_id,
                session.identity,
                session.generation,
                uuid4().hex,
                {"operation": "close"},
            )
            self._mark_closing_locked(
                session,
                reason="client_absence" if session.presence_established else "initial_presence_timeout",
            )
            self._commands.put_nowait((close, Future(), now))

    def close(self) -> bool:
        """Stop after bounded waiting, retaining a hung worker instead of replacing it."""
        self._stopping.set()
        self._thread.join(timeout=1)
        with self._lock:
            stopped = not self._thread.is_alive()
            logger.log(
                logging.INFO if stopped else logging.WARNING,
                "Policy worker %s: deployment=%s active_operation=%s queued_operations=%d "
                "session=%s cleanup_pending=%s; queued operations are not drained at process shutdown",
                "stopped" if stopped else "stop wait timed out after 1.0s (worker still running)",
                self.deployment,
                self._active_operation,
                self._commands.qsize(),
                None if self._session is None else self._session.identity,
                self._session is not None and self._session.closing,
            )
        return stopped


class PolicyServer:
    """Zenoh IO pump around SessionWorker; data publications use DROP congestion."""

    def __init__(self, worker: SessionWorker, transport: ZenohTransport) -> None:
        """Attach the preloaded worker to its deployment transport."""
        self.worker = worker
        self.transport = transport
        self._stop = Event()
        self._stop_reason = "serving loop exited"

    @staticmethod
    def _encode_reply(response: Envelope) -> bytes:
        """Return a correlated failure if a completed policy reply cannot be encoded."""
        try:
            return encode_message(response)
        except ProtocolError:
            logger.exception("Policy reply could not be encoded request=%s", response.request_id)
            return encode_message(response.error(ErrorCode.EXECUTION, "Policy reply could not be encoded"))

    def _declare_session(self, session_id: str) -> _SessionChannels:
        key = session_prefix(self.worker.deployment, self.worker.instance_id, session_id)
        return _SessionChannels(
            session_id=session_id,
            key=key,
            control=self.transport.declare_queryable(
                key + "/control",
                capacity=4,
                reply_timeout=self.worker.language_deadline_s + self.worker.action_deadline_s + 30,
            ),
            observations=self.transport.subscribe(key + "/obs", capacity=1),
            language=self.transport.subscribe(key + "/language/request", capacity=1),
            presence=self.transport.subscribe_liveliness(key + "/alive"),
        )

    def _pump_queryables(
        self,
        queryables: list[tuple[BoundedQueryable, MessageType]],
        outbound: list[_PendingReply],
    ) -> None:
        for queryable, expected in queryables:
            try:
                query = queryable.get(timeout=0)
            except Empty:
                continue
            try:
                message = decode_message(query.payload)
                if message.message_type is not expected:
                    query.reply(encode_message(message.error(ErrorCode.MALFORMED, "Wrong control surface")))
                    continue
                if len(outbound) >= 16:
                    query.reply(encode_message(message.error(ErrorCode.BUSY, "Reply capacity exhausted")))
                    continue
                bound = message.session_id if message.message_type is MessageType.CONTROL else None
                outbound.append(_PendingReply(self.worker.submit(message), query, bound))
            except ProtocolError:
                query.drop()
                logger.warning("Rejected malformed control envelope")
            except Exception:
                query.drop()
                logger.exception("Control request or reply failed")

    def _pump_subscribers(self, channels: _SessionChannels, outbound: list[_PendingReply]) -> None:
        for subscriber, expected, suffix in (
            (channels.observations, MessageType.OBSERVATION, "/act"),
            (channels.language, MessageType.LANGUAGE_REQUEST, "/language/result"),
        ):
            try:
                payload = subscriber.get(timeout=0)
            except Empty:
                continue
            try:
                message = decode_message(payload)
                if message.message_type is not expected or len(outbound) >= 16:
                    self.transport.publish(
                        channels.key + suffix,
                        encode_message(
                            message.error(ErrorCode.BUSY, "Invalid topic or reply capacity exhausted")
                        ),
                    )
                    continue
                outbound.append(_PendingReply(self.worker.submit(message), channels.key + suffix))
            except ProtocolError:
                logger.warning("Rejected malformed inference envelope")
        while True:
            try:
                channels.present = channels.presence.get(timeout=0).alive
            except Empty:
                break
        self.worker.expire(present=channels.present and not channels.presence.dropped)

    def _flush_replies(
        self, outbound: list[_PendingReply], channels: _SessionChannels | None
    ) -> list[_PendingReply]:
        pending = []
        session_id = None if channels is None else channels.session_id
        for reply in outbound:
            if not reply.future.done():
                pending.append(reply)
                continue
            try:
                response = reply.future.result()
                if response.message_type is MessageType.ACCEPTED and response.session_id != session_id:
                    if response.session_id == self.worker.session_id:
                        # Install the admitted session channels before acknowledging OPEN.
                        pending.append(reply)
                        continue
                    response = response.error(ErrorCode.STALE, "Open operation belongs to a closed session")
                if (
                    response.message_type is MessageType.ACK
                    and response.body.get("operation") == "reset"
                    and response.generation == 0
                ):
                    # Only admission establishes these subscriptions. A later
                    # reset must not stall the IO pump if the client disappears.
                    ready_key = session_prefix(
                        self.worker.deployment, self.worker.instance_id, response.session_id
                    )
                    self.transport.wait_for_subscriber(ready_key + "/act", 5.0)
                    self.transport.wait_for_subscriber(ready_key + "/language/result", 5.0)
                payload = self._encode_reply(response)
                if isinstance(reply.target, PendingQuery):
                    reply.target.reply(payload)
                else:
                    self.transport.publish(reply.target, payload)
            except Exception:
                if isinstance(reply.target, PendingQuery):
                    reply.target.drop()
                logger.exception("Reply failed or query deadline expired")
        return pending

    def serve(self) -> None:
        """Serve bounded control, action and language channels until explicitly stopped."""
        transport, worker = self.transport, self.worker
        channels: _SessionChannels | None = None
        outbound: list[_PendingReply] = []
        try:
            transport.open()
            prefix = instance_prefix(worker.deployment, worker.instance_id)
            describe = transport.declare_queryable(
                deployment_prefix(worker.deployment) + "/describe", capacity=4
            )
            opening = transport.declare_queryable(prefix + "/open", capacity=4)
            transport.declare_token(prefix + "/alive")
            logger.info(
                "Policy server ready: deployment=%s instance=%s; awaiting one client",
                worker.deployment,
                worker.instance_id,
            )
            while not self._stop.is_set():
                session_id = worker.session_id
                channel_session_id = None if channels is None else channels.session_id
                # Retain old session controls until their replies have been flushed.
                # OPEN replies are unbound so they cannot prevent their own setup.
                if session_id != channel_session_id and (
                    channels is None or not any(reply.session_id == channels.session_id for reply in outbound)
                ):
                    if channels is not None:
                        channels.close()
                    channels = None if session_id is None else self._declare_session(session_id)
                queryables = [(describe, MessageType.DESCRIBE), (opening, MessageType.OPEN)]
                if channels is not None:
                    queryables.append((channels.control, MessageType.CONTROL))
                self._pump_queryables(queryables, outbound)
                if channels is not None:
                    self._pump_subscribers(channels, outbound)
                outbound = self._flush_replies(outbound, channels)
                self._stop.wait(0.002)
        except KeyboardInterrupt:
            self.stop(reason="SIGINT")
        finally:
            with worker._lock:
                logger.info(
                    "Policy server stopping: deployment=%s reason=%s session=%s "
                    "active_operation=%s queued_operations=%d pending_replies=%d cleanup_pending=%s",
                    worker.deployment,
                    self._stop_reason,
                    None if worker._session is None else worker._session.identity,
                    worker._active_operation,
                    worker._commands.qsize(),
                    len(outbound),
                    worker._session is not None and worker._session.closing,
                )
            try:
                # The transport owns every declared channel, including partial startup.
                transport.close()
                logger.info("Policy server transport closed: deployment=%s", worker.deployment)
            finally:
                worker.close()

    def stop(self, reason: str = "stop requested") -> None:
        """Request bounded endpoint teardown from the serving thread."""
        if not self._stop.is_set():
            self._stop_reason = reason
            self._stop.set()
            logger.info(
                "Policy server shutdown requested: deployment=%s reason=%s", self.worker.deployment, reason
            )
