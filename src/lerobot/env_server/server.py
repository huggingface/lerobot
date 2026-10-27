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

"""Serve simulator environments; one thread owns and steps them."""

import contextlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from queue import Empty
from threading import Event
from typing import Any, cast

import numpy as np

from lerobot.sim.config import SimConfig
from lerobot.sim.types import EnvDescriptor
from lerobot.sim.vector_env import VectorEnv
from lerobot.transport.wire.codec import CodecLimits, RGBImage, decode_message, encode_message
from lerobot.transport.wire.protocol import Envelope, ErrorCode, MessageType, ProtocolError
from lerobot.transport.zenoh import (
    BoundedSubscriber,
    PendingQuery,
    PresenceEvent,
    PresenceToken,
    ZenohConfig,
    ZenohTransport,
)
from lerobot.utils.feature_spec import validate_array

from .protocol import (
    ANY_DEPLOYMENT_DESCRIBE,
    deployment_prefix,
    encode_descriptor,
    encode_step_result,
    instance_prefix,
    session_prefix,
)

logger = logging.getLogger(__name__)

# Room in a reply for what is not observations: rewards, flags, task text and the envelope.
_REPLY_HEADROOM_BYTES = 64 * 1024


@dataclass
class ServerConfig:
    """The simulator a server runs, its resource limits and its Zenoh endpoints."""

    sim: SimConfig
    deployment: str = "default"
    zenoh: ZenohConfig = field(default_factory=lambda: ZenohConfig(listen_endpoints=["tcp/0.0.0.0:7448"]))
    max_envs: int = 32
    max_sessions: int = 1
    request_timeout_s: float = 120
    presence_grace_s: float = 10


class _Session:
    """One client's batch of environments; closed when the client's presence is lost."""

    def __init__(
        self,
        server: "EnvServer",
        prefix: str,
        env: VectorEnv,
        seeds: list[int | None] | None,
        encoding: str,
        jpeg_quality: int,
    ):
        self.env = env
        self.prefix = prefix
        self.encoding = encoding
        self.jpeg_quality = jpeg_quality
        self.rgb = {f.name for f in env.descriptor.features if f.kind == "rgb"}
        # A reset counts a generation: requests sent before it are rejected as stale.
        self.generation = 0
        # Set when the simulator fails: the session then refuses every operation.
        self.fault: str | None = None
        self.opened_at = time.monotonic()
        self.seen_client = False
        with contextlib.ExitStack() as cleanup:
            cleanup.callback(env.close)
            env.reset(seed=seeds)
            self.presence: BoundedSubscriber[PresenceEvent] = server.transport.subscribe_liveliness(
                prefix + "/client_alive"
            )
            cleanup.callback(self.presence.close)
            self.token: PresenceToken = server.transport.declare_token(prefix + "/alive")
            cleanup.pop_all()

    def handle(self, operation: str, body: dict[str, Any]) -> dict[str, Any]:
        """Reset or step the owned environments."""
        if self.fault is not None:
            raise ProtocolError(ErrorCode.EXECUTION, self.fault)
        if operation == "reset":
            try:
                self.env.reset(seed=body.get("seeds"))
            except Exception as exc:
                self.fault = f"Simulator reset failed: {exc}"
                logger.exception(self.fault)
                raise
            self.generation += 1
            return {"generation": self.generation, "result": encode_step_result(self.env.latest)}
        if operation == "step":
            actions = self.actions(body.get("actions"))
            try:
                self.env.step(actions)
            except Exception as exc:
                self.fault = f"Simulator step failed: {exc}"
                logger.exception(self.fault)
                raise
            return {"result": encode_step_result(self.env.latest)}
        raise ProtocolError(ErrorCode.UNSUPPORTED, f"Unsupported operation: {operation}")

    def actions(self, value: Any) -> np.ndarray:
        """Validate a batch of actions against the environment contract."""
        try:
            validate_array(value, self.env.descriptor.action_feature, leading_shape=(self.env.num_envs,))
        except ValueError as exc:
            raise ProtocolError(ErrorCode.INCOMPATIBLE, "Action shape, dtype or values invalid") from exc
        return cast(np.ndarray, value)

    def close(self) -> None:
        """Release the environments."""
        self.env.close()

    def encode(self, reply: dict[str, Any]) -> dict[str, Any]:
        """Send each environment's camera frames as JPEG when requested at open."""
        result = reply.get("result")
        if self.encoding == "raw" or result is None:
            return reply
        obs = {
            name: [RGBImage(image, "jpeg", self.jpeg_quality) for image in value]
            if name in self.rgb
            else value
            for name, value in result["obs"].items()
        }
        return {**reply, "result": {**result, "obs": obs}}


class EnvServer:
    """Serve one simulator's environments to clients; a single thread owns and steps them."""

    def __init__(self, config: ServerConfig):
        """Create the transport; simulators are built when serving starts."""
        self.config = config
        if config.max_envs < 1 or config.max_sessions < 1:
            raise ValueError("Server resource bounds must be positive")
        self.instance = uuid.uuid4().hex
        self.transport = ZenohTransport(config.zenoh)
        self.sessions: dict[str, _Session] = {}
        # Built at startup to describe the simulator, then handed to the first matching session.
        self._startup_env: VectorEnv | None = None
        self.ready = Event()
        self.stop = Event()
        self.descriptor: EnvDescriptor

    def serve(self) -> None:
        """Answer requests until stopped; every simulator call happens on this thread."""
        try:
            # Listen first: clients connecting while a slow simulator starts wait instead of failing.
            self.transport.open()
            # Built on the thread that later steps it, and kept for the first session: some simulators
            # (Isaac Sim) start only once per process.
            self._startup_env = VectorEnv(self.config.sim)
            self.descriptor = self._startup_env.descriptor
            root = deployment_prefix(self.config.deployment)
            instance = instance_prefix(self.config.deployment, self.instance)
            describe = self.transport.declare_queryable(
                root + "/describe", reply_timeout=self.config.request_timeout_s
            )
            # Answered too for clients that take whichever deployment this endpoint serves.
            any_describe = self.transport.declare_queryable(
                ANY_DEPLOYMENT_DESCRIBE, reply_timeout=self.config.request_timeout_s
            )
            requests = self.transport.declare_queryable(
                instance + "/**", reply_timeout=self.config.request_timeout_s
            )
            self.transport.declare_token(instance + "/alive")
            zenoh = self.config.zenoh
            logger.info(
                "Serving %s task %r as deployment %r on %s",
                self.config.sim.type,
                self.config.sim.task,
                self.config.deployment,
                ", ".join(zenoh.listen_endpoints or zenoh.connect_endpoints),
            )
            self.ready.set()
            while not self.stop.is_set():
                for channel in (describe, any_describe, requests):
                    for _ in range(8):
                        try:
                            pending = channel.get()
                        except Empty:
                            break
                        self._reply(pending)
                self._reap_sessions()
                self.stop.wait(0.002)
        finally:
            for session_id in list(self.sessions):
                try:
                    self._close_session(session_id)
                except Exception:
                    logger.exception("Failed to close simulator session %s", session_id)
            if self._startup_env is not None:
                self._startup_env.close()
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
            body = self._dispatch(pending.key, message)
            reply = message.reply(MessageType.ACK if message.session_id else MessageType.ACCEPTED, body)
        except Exception as exc:
            code = exc.code if isinstance(exc, ProtocolError) else ErrorCode.EXECUTION
            # A request that cannot be decoded has no envelope to answer; reply with an empty one.
            reply = (message or Envelope(MessageType.DESCRIBE)).error(code, str(exc))
            logger.debug("Simulator request failed", exc_info=True)
        self._send(pending, reply)

    def _dispatch(self, key: str, message: Envelope) -> dict[str, Any]:
        if key in (deployment_prefix(self.config.deployment) + "/describe", ANY_DEPLOYMENT_DESCRIBE):
            return {
                "deployment": self.config.deployment,
                "instance": self.instance,
                "descriptor": encode_descriptor(self.descriptor),
            }
        if message.instance_id != self.instance:
            raise ProtocolError(ErrorCode.STALE, "Simulator boot changed")
        if key == instance_prefix(self.config.deployment, self.instance) + "/open":
            return self._open(message.body)
        session = self.sessions.get(message.session_id)
        if session is None or not key.startswith(session.prefix + "/"):
            raise ProtocolError(ErrorCode.STALE, "Unknown simulator session")
        if message.generation != session.generation:
            raise ProtocolError(ErrorCode.STALE, "Environment generation changed")
        operation = key.rsplit("/", 1)[-1]
        if operation == "close":
            self._close_session(message.session_id)
            return {}
        return session.encode(session.handle(operation, message.body))

    def _open(self, body: dict[str, Any]) -> dict[str, Any]:
        count = body.get("num_envs", 1)
        use_async_envs = bool(body.get("use_async_envs", False))
        encoding = body.get("encoding", "raw")
        jpeg_quality = body.get("jpeg_quality", 90)
        if type(count) is not int or not 1 <= count <= self.config.max_envs:
            raise ProtocolError(
                ErrorCode.INCOMPATIBLE, f"Batch size {count!r} is not between 1 and {self.config.max_envs}"
            )
        if encoding not in {"raw", "jpeg"} or type(jpeg_quality) is not int or not 1 <= jpeg_quality <= 100:
            raise ProtocolError(
                ErrorCode.UNSUPPORTED, "Image encoding must be raw or jpeg, quality in [1, 100]"
            )
        batch_bytes = count * sum(
            int(np.prod(f.shape)) * np.dtype(f.dtype).itemsize for f in self.descriptor.features
        )
        # JPEG frames are far smaller on the wire, but the client still decodes them to full size.
        budget = (
            min(CodecLimits().max_payload_bytes, self.config.zenoh.max_payload_bytes)
            if encoding == "raw"
            else CodecLimits().max_decoded_bytes
        )
        if batch_bytes + _REPLY_HEADROOM_BYTES > budget:
            raise ProtocolError(ErrorCode.INCOMPATIBLE, "Observation batch exceeds transport payload budget")
        if len(self.sessions) >= self.config.max_sessions:
            raise ProtocolError(ErrorCode.BUSY, "Simulator capacity is owned by another session")
        env = self._session_env(count, body.get("task_group"), body.get("task_id", 0), use_async_envs)
        session_id = uuid.uuid4().hex
        session = _Session(
            self,
            session_prefix(self.config.deployment, self.instance, session_id),
            env,
            body.get("seeds"),
            encoding,
            jpeg_quality,
        )
        self.sessions[session_id] = session
        return session.encode(
            {
                "session": session_id,
                "result": encode_step_result(env.latest),
                "generation": 0,
                "descriptor": encode_descriptor(env.descriptor),
            }
        )

    def _session_env(
        self, count: int, task_group: str | None, task_id: int, use_async_envs: bool
    ) -> VectorEnv:
        """The startup environment when the session asks for it, else a new one."""
        env, self._startup_env = self._startup_env, None
        if env is not None:
            asked = (count, task_group or env.task_group, task_id, use_async_envs)
            if asked == (env.num_envs, env.task_group, env.task_id, env.use_async_envs):
                return env
            env.close()
        return VectorEnv(self.config.sim, count, task_group, task_id, use_async_envs)

    def _reap_sessions(self) -> None:
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

    def _close_session(self, session_id: str) -> None:
        session = self.sessions.pop(session_id)
        try:
            session.close()
        finally:
            session.presence.close()
            session.token.undeclare()
