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

"""Bounded simulator serving. All environment mutations run on the serving worker."""

import logging
import time
import uuid
from dataclasses import dataclass, field
from queue import Empty
from threading import Event
from typing import Any

import numpy as np

from lerobot.sims.backend import Backend, BackendConfig
from lerobot.sims.metadata import describe_backend
from lerobot.sims.types import EnvDescriptor
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

from .contracts import encode_descriptor, encode_step_result
from .protocol import any_deployment_describe, deployment_prefix, instance_prefix, session_prefix
from .session import EnvSession, check_open

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


class _Session(EnvSession):
    """An environment session served while the client's presence token is alive."""

    def __init__(
        self,
        server: "EnvServer",
        prefix: str,
        *args: Any,
        encoding: str = "raw",
        jpeg_quality: int = 90,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.server = server
        self.prefix = prefix
        self.encoding = encoding
        self.jpeg_quality = jpeg_quality
        self.rgb = {f.name for f in self.descriptor.features if f.kind == "rgb"}
        self.opened_at = time.monotonic()
        self.seen_client = False
        self.presence: BoundedSubscriber[PresenceEvent] = server.transport.subscribe_liveliness(
            prefix + "/client_alive"
        )
        self.token: PresenceToken = server.transport.declare_token(prefix + "/alive")

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
    """Serve explicit environment operations from one owner thread."""

    def __init__(self, config: ServerConfig):
        """Initialize owned resources without advancing simulation time."""
        self.config = config
        if config.max_envs < 1 or config.max_sessions < 1:
            raise ValueError("Server resource bounds must be positive")
        self.instance = uuid.uuid4().hex
        self.transport = ZenohTransport(config.zenoh)
        self.sessions: dict[str, _Session] = {}
        self.ready = Event()
        self.stop = Event()
        self.descriptor: EnvDescriptor

    def serve(self) -> None:
        """Pump bounded requests until stopped."""
        try:
            # Listen before probing: slow simulators must not exceed the clients' connect timeout.
            # Describe queries wait for the queryable declared below, within their own deadline.
            self.transport.open()
            # Includes descriptor probing; native renderer setup stays on the owner thread.
            probe = Backend(self.config.sim)
            try:
                self.descriptor = describe_backend(probe)
            finally:
                probe.close()
            root = deployment_prefix(self.config.deployment)
            instance = instance_prefix(self.config.deployment, self.instance)
            describe = self.transport.declare_queryable(
                root + "/describe", reply_timeout=self.config.request_timeout_s
            )
            # Answered too for clients that take whichever deployment this endpoint serves.
            any_describe = self.transport.declare_queryable(
                any_deployment_describe(), reply_timeout=self.config.request_timeout_s
            )
            requests = self.transport.declare_queryable(
                instance + "/**", reply_timeout=self.config.request_timeout_s
            )
            self.transport.declare_token(instance + "/alive")
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
            reply = (message or Envelope(MessageType.DESCRIBE)).error(code, str(exc))
            logger.debug("Simulator request failed", exc_info=True)
        self._send(pending, reply)

    def _dispatch(self, key: str, message: Envelope) -> dict[str, Any]:
        if key in (deployment_prefix(self.config.deployment) + "/describe", any_deployment_describe()):
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
        body = message.body
        if operation == "control":
            operation = body.get("operation", "")
        if operation == "close":
            self._close_session(message.session_id)
            return {}
        return session.encode(session.handle(operation, body))

    def _open(self, body: dict) -> dict:
        count = body.get("num_envs", 1)
        use_async_envs = body.get("use_async_envs", False)
        encoding = body.get("encoding", "raw")
        jpeg_quality = body.get("jpeg_quality", 90)
        check_open(count, self.config.max_envs)
        if type(use_async_envs) is not bool:
            raise ProtocolError(ErrorCode.MALFORMED, "use_async_envs must be a boolean")
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
        if batch_bytes + 65536 > budget:
            raise ProtocolError(ErrorCode.INCOMPATIBLE, "Observation batch exceeds transport payload budget")
        if len(self.sessions) >= self.config.max_sessions:
            raise ProtocolError(ErrorCode.BUSY, "Simulator capacity is owned by another session")
        session_id = uuid.uuid4().hex
        session = _Session(
            self,
            session_prefix(self.config.deployment, self.instance, session_id),
            self.config.sim,
            self.descriptor,
            count,
            body.get("task_group"),
            body.get("task_id", 0),
            body.get("seeds"),
            encoding=encoding,
            jpeg_quality=jpeg_quality,
            use_async_envs=use_async_envs,
        )
        self.sessions[session_id] = session
        return session.encode(
            {
                "session": session_id,
                "result": encode_step_result(session.opened),
                "generation": 0,
                "descriptor": encode_descriptor(describe_backend(session.backend)),
            }
        )

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
