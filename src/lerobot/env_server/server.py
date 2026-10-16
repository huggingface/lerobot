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

"""Bounded simulator serving. All physics and world mutations run on the serving worker."""

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from queue import Empty
from threading import Event
from typing import Any, cast

import numpy as np

from lerobot.sims.adapters import hold_action
from lerobot.sims.backend import Backend, BackendConfig
from lerobot.transport.wire.codec import CodecLimits, decode_message, encode_message
from lerobot.transport.wire.features import validate_array
from lerobot.transport.wire.protocol import Envelope, ErrorCode, MessageType, ProtocolError
from lerobot.transport.zenoh import (
    BoundedSubscriber,
    PendingQuery,
    PresenceEvent,
    PresenceToken,
    ZenohConfig,
    ZenohTransport,
)

from .contracts import EnvDescriptor, ExecutionFeedback, StepResult, validate_result
from .protocol import deployment_prefix, instance_prefix, session_prefix

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Bound simulator resources and select explicit Zenoh endpoints."""

    sim: BackendConfig = field(default_factory=BackendConfig)
    deployment: str = "default"
    zenoh: ZenohConfig = field(default_factory=lambda: ZenohConfig(listen_endpoints=["tcp/0.0.0.0:7448"]))
    max_envs: int = 32
    max_sessions: int = 1
    request_timeout_s: float = 120
    presence_grace_s: float = 10
    command_timeout_s: float = 1


@dataclass
class _Command:
    """One accepted realtime command, acknowledged after the tick that executes it."""

    message: Envelope
    actions: np.ndarray
    pending: PendingQuery


@dataclass
class _Session:
    backend: Backend
    clock: str
    prefix: str
    presence: BoundedSubscriber[PresenceEvent]
    token: PresenceToken
    target: np.ndarray
    generation: int = 0
    next_tick: float = 0
    last_command: float = 0
    opened_at: float = field(default_factory=time.monotonic)
    seen_client: bool = False
    # Realtime worlds stay frozen until the first command; the open/reset target is not a policy command.
    commanded: bool = False
    fault: str | None = None
    command: _Command | None = None
    execution: ExecutionFeedback | None = None
    sequence: int = 0


class EnvServer:
    """Serve world operations and physics ticks from one owner thread."""

    def __init__(self, config: ServerConfig):
        """Initialize owned resources without advancing simulation time."""
        self.config = config
        if config.max_envs < 1 or config.max_sessions < 1 or config.command_timeout_s <= 0:
            raise ValueError("Server resource bounds must be positive")
        self.instance = uuid.uuid4().hex
        self.transport = ZenohTransport(config.zenoh)
        self.sessions: dict[str, _Session] = {}
        self.ready = Event()
        self.stop = Event()
        self.descriptor: EnvDescriptor

    def serve(self, should_stop: Callable[[], bool] | None = None) -> None:
        """Pump bounded requests and realtime worlds until stopped.

        ``should_stop`` lets the owner check a signal flag without acquiring a lock
        inside the signal handler. Other threads can continue to use ``stop.set()``.
        """
        try:
            # Listen before probing: slow simulators must not exceed the clients' connect timeout.
            # Describe queries wait for the queryable declared below, within their own deadline.
            self.transport.open()
            # Includes descriptor probing; native renderer setup stays on the owner thread.
            probe = Backend(self.config.sim)
            try:
                self.descriptor = probe.descriptor()
            finally:
                probe.close()
            root = deployment_prefix(self.config.deployment)
            instance = instance_prefix(self.config.deployment, self.instance)
            describe = self.transport.declare_queryable(
                root + "/describe", reply_timeout=self.config.request_timeout_s
            )
            requests = self.transport.declare_queryable(
                instance + "/**", reply_timeout=self.config.request_timeout_s
            )
            self.transport.declare_token(instance + "/alive")
            self.ready.set()
            while not self.stop.is_set() and not (should_stop is not None and should_stop()):
                for channel in (describe, requests):
                    for _ in range(8):
                        try:
                            pending = channel.get()
                        except Empty:
                            break
                        self._reply(pending)
                self._tick()
                self.stop.wait(0.002)
        finally:
            for session_id in list(self.sessions):
                try:
                    self._close_session(session_id)
                except Exception:
                    logger.exception("Failed to close simulator session %s", session_id)
            self.transport.close()

    @staticmethod
    def _send(pending: PendingQuery, reply: Envelope) -> None:
        """Reply once; a lost reply leaves the client waiting for its full timeout, so log it."""
        if not pending.reply(encode_message(reply)):
            logger.warning(
                "Reply %s for %s was not sent: the query expired %.1fs ago",
                reply.message_type,
                pending.key,
                time.monotonic() - pending.deadline,
            )

    def _reply(self, pending: PendingQuery) -> None:
        message = None
        try:
            message = decode_message(pending.payload)
            body = self._dispatch(pending.key, message, pending)
            if body is None:
                return
            reply = message.reply(MessageType.ACK if message.session_id else MessageType.ACCEPTED, body)
        except Exception as exc:
            code = exc.code if isinstance(exc, ProtocolError) else ErrorCode.EXECUTION
            reply = (message or Envelope(MessageType.DESCRIBE)).error(code, str(exc))
            logger.debug("Simulator request failed", exc_info=True)
        self._send(pending, reply)

    def _dispatch(
        self, key: str, message: Envelope, pending: PendingQuery | None = None
    ) -> dict[str, Any] | None:
        if key == deployment_prefix(self.config.deployment) + "/describe":
            return {"instance": self.instance, "descriptor": self.descriptor.to_dict()}
        if message.instance_id != self.instance:
            raise ProtocolError(ErrorCode.STALE, "Simulator boot changed")
        if key == instance_prefix(self.config.deployment, self.instance) + "/open":
            return self._open(message.body)
        session = self.sessions.get(message.session_id)
        if session is None or not key.startswith(session.prefix + "/"):
            raise ProtocolError(ErrorCode.STALE, "Unknown simulator session")
        if message.generation != session.generation:
            raise ProtocolError(ErrorCode.STALE, "World generation changed")
        operation = key.rsplit("/", 1)[-1]
        body = message.body
        if operation == "control":
            operation = body.get("operation", "")
        if operation == "close":
            self._close_session(message.session_id)
            return {}
        if session.fault is not None:
            raise ProtocolError(ErrorCode.EXECUTION, session.fault)
        if operation == "reset":
            self._cancel_command(session, ErrorCode.STALE, "World reset before command execution")
            session.execution = None
            try:
                result = session.backend.reset(body.get("seeds"))
            except Exception as exc:
                session.fault = f"Simulator reset failed: {exc}"
                raise
            session.generation += 1
            session.commanded = False
            session.target = self._initial_target(session.backend, result)
            session.last_command = time.monotonic()
            return {"generation": session.generation, "result": result.to_dict()}
        if operation in {"step", "apply"}:
            value = body.get("actions")
            try:
                validate_array(
                    value, self.descriptor.action_feature, leading_shape=(session.backend.num_envs,)
                )
            except ValueError as exc:
                raise ProtocolError(ErrorCode.INCOMPATIBLE, "Command shape, dtype or values invalid") from exc
            actions = cast(np.ndarray, value)
        if operation == "step":
            if session.clock != "lockstep":
                raise ProtocolError(ErrorCode.UNSUPPORTED, "Step requires a lockstep session")
            try:
                # Lockstep replies are the transition itself; execution feedback serves realtime only.
                return {"result": session.backend.step(actions).to_dict()}
            except Exception as exc:
                session.fault = f"Simulator step failed: {exc}"
                raise
        if operation in {"apply", "hold"}:
            if session.clock != "realtime":
                raise ProtocolError(ErrorCode.UNSUPPORTED, "Commands require realtime mode")
            if operation == "apply":
                # Clients send one blocking command at a time; the query's own deadline bounds it.
                if pending is None or session.command is not None:
                    raise ProtocolError(ErrorCode.BUSY, "A command is already pending execution")
                session.command = _Command(message, actions.copy(), pending)
                session.commanded = True
                return None
            else:
                self._cancel_command(session, ErrorCode.STALE, "Hold superseded the pending command")
                session.target = hold_action(
                    session.target,
                    self.descriptor.control,
                    self.descriptor.gripper_indices + self.descriptor.command_retention_indices,
                )
                session.commanded = True
            session.last_command = time.monotonic()
            return {"applied": session.target}
        if operation == "snapshot":
            body = {"result": session.backend.snapshot().to_dict()}
            if session.execution is not None:
                body["execution"] = session.execution.to_dict()
            return body
        if operation == "render":
            return {"frames": session.backend.render()}
        raise ProtocolError(ErrorCode.UNSUPPORTED, f"Unsupported operation: {operation}")

    def _initial_target(self, backend: Backend, result: StepResult) -> np.ndarray:
        dim = self.descriptor.action_feature.shape[0]
        target: np.ndarray = np.zeros((backend.num_envs, dim), dtype=np.float32)
        if self.descriptor.control in {"position", "eef_pose"}:
            state = next((f for f in self.descriptor.features if f.name == "observation.state"), None)
            if state is None or not set(self.descriptor.action_feature.names).issubset(state.names):
                raise ValueError("Position control requires observed initial positions")
            target = result.obs[state.name][
                :, [state.names.index(n) for n in self.descriptor.action_feature.names]
            ].copy()
        return target

    def _open(self, body: dict) -> dict:
        count = body.get("num_envs", 1)
        clock = body.get("clock", "lockstep")
        if (
            type(count) is not int
            or not 1 <= count <= self.config.max_envs
            or clock not in self.descriptor.clocks
        ):
            raise ProtocolError(ErrorCode.INCOMPATIBLE, "Unsupported batch size or clock")
        batch_bytes = count * sum(
            int(np.prod(f.shape)) * np.dtype(f.dtype).itemsize for f in self.descriptor.features
        )
        if batch_bytes + 65536 > min(CodecLimits().max_payload_bytes, self.config.zenoh.max_payload_bytes):
            raise ProtocolError(ErrorCode.INCOMPATIBLE, "Observation batch exceeds transport payload budget")
        if len(self.sessions) >= self.config.max_sessions:
            raise ProtocolError(ErrorCode.BUSY, "Simulator capacity is owned by another session")
        backend = Backend(self.config.sim, count, body.get("task_group"), body.get("task_id", 0))
        try:
            result = backend.reset(body.get("seeds"))
            validate_result(result, backend.descriptor(), count)
            target = self._initial_target(backend, result)
            session_id = uuid.uuid4().hex
            prefix = session_prefix(self.config.deployment, self.instance, session_id)
            session = _Session(
                backend,
                clock,
                prefix,
                self.transport.subscribe_liveliness(prefix + "/client_alive"),
                self.transport.declare_token(prefix + "/alive"),
                target,
            )
            session.next_tick = time.monotonic() + 1 / self.config.sim.fps
            session.last_command = time.monotonic()
            self.sessions[session_id] = session
            return {
                "session": session_id,
                "result": result.to_dict(),
                "generation": 0,
                "descriptor": backend.descriptor().to_dict(),
            }
        except BaseException:
            backend.close()
            raise

    def _tick(self) -> None:
        now = time.monotonic()
        for session_id, session in list(self.sessions.items()):
            lost = session.presence.dropped > 0
            while True:
                try:
                    event = session.presence.get()
                    if event.alive:
                        session.seen_client = True
                    elif session.seen_client:
                        lost = True
                except Empty:
                    break
            if lost or (not session.seen_client and now - session.opened_at > self.config.presence_grace_s):
                self._close_session(session_id)
                continue
            if (
                session.clock != "realtime"
                or not session.commanded
                or session.fault is not None
                or now < session.next_tick
            ):
                continue
            command, session.command = session.command, None
            try:
                if command is not None and command.pending.expired:
                    # The client already gave up; never execute a command it can no longer see.
                    command.pending.drop()
                    command = None
                if command is not None:
                    session.target = command.actions
                    session.last_command = now
                elif now - session.last_command > self.config.command_timeout_s:
                    session.target = hold_action(
                        session.target,
                        self.descriptor.control,
                        self.descriptor.gripper_indices + self.descriptor.command_retention_indices,
                    )
                before = session.backend.snapshot()
                applied = session.target.copy()
                started = time.monotonic()
                result = session.backend.step(applied)
                request_id = command.message.request_id if command else uuid.uuid4().hex
                body = self._executed(session, request_id, applied, before, result, started)
                if self.descriptor.control == "delta":
                    session.target = hold_action(
                        session.target,
                        "delta",
                        self.descriptor.gripper_indices + self.descriptor.command_retention_indices,
                    )
                if command is not None:
                    self._send(command.pending, command.message.reply(MessageType.ACK, body))
                # Schedule from the previous deadline so poll latency does not accumulate.
                # When execution overruns, resume at once without replaying missed ticks.
                session.next_tick = max(session.next_tick + 1 / self.config.sim.fps, time.monotonic())
            except Exception as exc:
                session.fault = f"Environment execution failed: {exc}"
                if command is not None:
                    self._send(command.pending, command.message.error(ErrorCode.EXECUTION, session.fault))
                logger.exception("Environment session failed")

    def _executed(
        self,
        session: _Session,
        request_id: str,
        actions: np.ndarray,
        before: StepResult,
        result: StepResult,
        started: float,
    ) -> dict[str, Any]:
        advanced = ~(before.terminated | before.truncated)
        applied = actions.copy()
        # No action reaches an absorbing world. Zero is a placeholder qualified by advanced=False.
        applied[~advanced] = 0
        session.sequence += 1
        feedback = ExecutionFeedback(
            request_id,
            session.generation,
            session.sequence,
            applied,
            advanced,
            started,
            time.monotonic(),
            result,
        )
        feedback.validate(self.descriptor, session.backend.num_envs)
        session.execution = feedback
        return {"result": result.to_dict(), "execution": feedback.to_dict()}

    @staticmethod
    def _cancel_command(session: _Session, code: ErrorCode, reason: str) -> None:
        command, session.command = session.command, None
        if command is not None:
            EnvServer._send(command.pending, command.message.error(code, reason))

    def _close_session(self, session_id: str) -> None:
        session = self.sessions.pop(session_id)
        try:
            self._cancel_command(session, ErrorCode.STALE, "Environment session closed")
            session.backend.close()
        finally:
            session.presence.close()
            session.token.undeclare()
