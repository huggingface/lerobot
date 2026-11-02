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

"""Serve simulator envs to clients. One thread owns and steps every env."""

import contextlib
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from queue import Empty
from threading import Event
from typing import Any, cast

import numpy as np

from lerobot.sim.config import SimConfig, SimSettings
from lerobot.sim.local_vector_env import LocalVectorEnv
from lerobot.sim.types import EnvDescriptor
from lerobot.transport.protocol.codec import CodecLimits, RGBImage, decode_message, encode_message
from lerobot.transport.protocol.envelope import (
    Envelope,
    ErrorCode,
    MessageType,
    ProtocolError,
    validate_segment,
)
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

# Reply space kept for everything except observations: rewards, flags, task text and the envelope.
_REPLY_HEADROOM_BYTES = 64 * 1024


def _check_request_type(message: Envelope, expected: MessageType) -> None:
    """Reject a request whose type is not the one its key takes."""
    if message.message_type is not expected:
        raise ProtocolError(
            ErrorCode.MALFORMED, f"This key takes {expected} requests, not {message.message_type}"
        )


def _checked_seeds(value: Any, num_envs: int) -> list[int | None] | None:
    """Return reset seeds that are absent or one int (or None) per env; reject anything else."""
    if value is not None and (
        not isinstance(value, list)
        or len(value) != num_envs
        or not all(seed is None or type(seed) is int for seed in value)
    ):
        raise ProtocolError(
            ErrorCode.INCOMPATIBLE, f"Expected one seed per env ({num_envs}), each an int or None"
        )
    return value


def _checked_actions(value: Any, env: LocalVectorEnv) -> np.ndarray:
    """Return the action batch if it matches the env's action feature and batch size."""
    try:
        validate_array(value, env.descriptor.action_feature, leading_shape=(env.num_envs,))
    except ValueError as exc:
        raise ProtocolError(ErrorCode.INCOMPATIBLE, f"Invalid actions: {exc}") from exc
    return cast(np.ndarray, value)


@dataclass
class EnvServerConfig:
    """Simulator to serve, resource limits and Zenoh endpoints.

    Args:
        sim (`SimSettings`, *optional*):
            The simulator to serve. Settings left unset come from its integration's conventions when the server
            starts. `sim.type` is required.
        deployment (`str`, *optional*, defaults to `""`):
            Name clients find this server under, e.g. `libero`. Required.
        zenoh (`ZenohConfig`, *optional*):
            Network settings. Listens on `tcp/127.0.0.1:7448` by default.
        max_envs (`int`, *optional*, defaults to `32`):
            Most envs one session may run.
        max_sessions (`int`, *optional*, defaults to `1`):
            Most sessions open at once.
        request_timeout_s (`float`, *optional*, defaults to `120`):
            How long the server may take to reply to one request.
        initial_presence_timeout_s (`float`, *optional*, defaults to `10`):
            How long a new session waits for its client to announce itself before it is closed.
    """

    sim: SimSettings = field(default_factory=SimSettings)
    deployment: str = ""
    zenoh: ZenohConfig = field(default_factory=lambda: ZenohConfig(listen_endpoints=["tcp/127.0.0.1:7448"]))
    max_envs: int = 32
    max_sessions: int = 1
    request_timeout_s: float = 120
    initial_presence_timeout_s: float = 10

    def __post_init__(self):
        """Reject a missing simulator or deployment, bad resource limits and bad endpoints."""
        if not self.sim.type or not self.deployment:
            raise ValueError("sim.type and deployment are required")
        validate_segment(self.deployment, "deployment")
        if self.max_envs < 1 or self.max_sessions < 1:
            raise ValueError("Server resource bounds must be positive")
        # Checked here so a bad endpoint fails at parse time, before the simulator builds.
        self.zenoh.validate()


class _Session:
    """The envs one client opened. It stays open until the client closes it or goes away."""

    def __init__(
        self,
        transport: ZenohTransport,
        prefix: str,
        env: LocalVectorEnv,
        seeds: list[int | None] | None,
        encoding: str,
        jpeg_quality: int,
    ):
        """Reset the envs and start watching for the client. Closes the envs if this fails.

        Args:
            transport (`ZenohTransport`):
                The server's transport, used to watch for the client and to announce the session.
            prefix (`str`):
                Key prefix of this session.
            env (`LocalVectorEnv`):
                The envs this session runs. The session closes them.
            seeds (`list[int | None] | None`):
                One seed per env for the first reset. `None` leaves the seeds to the simulator.
            encoding (`str`):
                How camera frames are sent: `"raw"` or `"jpeg"`.
            jpeg_quality (`int`):
                JPEG quality, from 1 to 100.
        """
        self.env = env
        self.prefix = prefix
        self.encoding = encoding
        self.jpeg_quality = jpeg_quality
        self.camera_keys = {f.name for f in env.descriptor.features if f.kind == "rgb"}
        # Each reset bumps the generation, so requests sent before it are rejected as stale.
        self.generation = 0
        # Set when the simulator fails; after that the session refuses every operation.
        self.failure: str | None = None
        self.presence_established = False
        with contextlib.ExitStack() as cleanup:
            cleanup.callback(env.close)
            env.reset(seed=seeds)
            self.presence: BoundedSubscriber[PresenceEvent] = transport.subscribe_liveliness(
                prefix + "/client_alive"
            )
            cleanup.callback(self.presence.close)
            self.token: PresenceToken = transport.declare_token(prefix + "/alive")
            cleanup.pop_all()
        # Start the presence timeout only now: the client announces itself after it gets the open reply,
        # and a slow first reset must not use up its time.
        self.opened_at = time.monotonic()

    def handle(self, operation: str, body: dict[str, Any]) -> dict[str, Any]:
        """Run a reset or step and return the reply body. A simulator error fails the session."""
        if self.failure is not None:
            raise ProtocolError(ErrorCode.EXECUTION, self.failure)
        if operation == "reset":
            seeds = _checked_seeds(body.get("seeds"), self.env.num_envs)
            try:
                self.env.reset(seed=seeds)
            except Exception as exc:
                self.failure = f"Simulator reset failed: {exc}"
                raise
            self.generation += 1
            return {"generation": self.generation, "result": encode_step_result(self.env.latest)}
        if operation == "step":
            actions = _checked_actions(body.get("actions"), self.env)
            try:
                self.env.step(actions)
            except Exception as exc:
                self.failure = f"Simulator step failed: {exc}"
                raise
            return {"result": encode_step_result(self.env.latest)}
        raise ProtocolError(ErrorCode.UNSUPPORTED, f"Unsupported operation: {operation}")

    def close(self) -> None:
        """Close the envs and stop the presence signals."""
        try:
            self.env.close()
        finally:
            self.presence.close()
            self.token.undeclare()

    def with_jpeg_frames(self, reply: dict[str, Any]) -> dict[str, Any]:
        """Mark camera frames for JPEG encoding if the client asked for it at open."""
        result = reply.get("result")
        if self.encoding == "raw" or result is None:
            return reply
        obs = {
            name: [RGBImage(image, "jpeg", self.jpeg_quality) for image in value]
            if name in self.camera_keys
            else value
            for name, value in result["obs"].items()
        }
        return {**reply, "result": {**result, "obs": obs}}


class EnvServer:
    """Serve one simulator's envs to clients. Only the thread running `serve` builds and steps them."""

    def __init__(self, config: EnvServerConfig):
        """Create the transport. Envs are built in `serve`."""
        self.config = config
        self.instance_id = uuid.uuid4().hex
        payload_bytes = config.zenoh.max_payload_bytes
        self.limits = CodecLimits(
            max_payload_bytes=payload_bytes,
            max_tensor_bytes=payload_bytes,
            max_decoded_bytes=2 * payload_bytes,
        )
        self.transport = ZenohTransport(config.zenoh)
        self.sessions: dict[str, _Session] = {}
        # Built at startup to read the descriptor, then reused by the first session asking for the same batch.
        self._startup_env: LocalVectorEnv | None = None
        # The simulator's full config, with its integration's conventions filled in at startup.
        self.sim: SimConfig
        self.descriptor: EnvDescriptor
        self.ready = Event()
        self._stop = Event()

    def serve(self) -> None:
        """Answer requests until `stop` is called. Every simulator call happens on this thread."""
        try:
            # Open the transport first, so a port already in use fails before the slow simulator build.
            self.transport.open()
            self.sim = SimConfig.resolve(asdict(self.config.sim))
            self._startup_env = LocalVectorEnv(self.sim)
            self.descriptor = self._startup_env.descriptor
            root = deployment_prefix(self.config.deployment)
            instance = instance_prefix(self.config.deployment, self.instance_id)
            describe = self.transport.declare_queryable(
                root + "/describe", reply_timeout=self.config.request_timeout_s
            )
            # Also answer clients that do not name a deployment.
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
                self.sim.type,
                self.sim.task,
                self.config.deployment,
                ", ".join(zenoh.listen_endpoints or zenoh.connect_endpoints),
            )
            self.ready.set()
            while not self._stop.is_set():
                for channel in (describe, any_describe, requests):
                    for _ in range(8):
                        try:
                            pending = channel.get()
                        except Empty:
                            break
                        self._reply(pending)
                self._reap_sessions()
                self._stop.wait(0.002)
        finally:
            for session_id in list(self.sessions):
                self._close_session(session_id)
            if self._startup_env is not None:
                self._startup_env.close()
            self.transport.close()

    def stop(self) -> None:
        """Ask `serve` to close every session and return. Safe to call from another thread."""
        self._stop.set()

    def _reply(self, pending: PendingQuery) -> None:
        """Decode one request, pass it to `_dispatch`, and send back its answer or an error."""
        message = None
        try:
            message = decode_message(pending.payload, self.limits)
            reply = self._dispatch(pending.key, message)
        except Exception as exc:
            if isinstance(exc, ProtocolError):
                code = exc.code
                logger.debug("Request %s rejected: %s", pending.key, exc)
            else:
                code = ErrorCode.EXECUTION
                logger.exception("Request %s failed", pending.key)
            # An undecodable request has no envelope to reply to, so reply with an empty one.
            reply = (message or Envelope(MessageType.DESCRIBE)).error(code, str(exc))
        self._send(pending, reply)

    def _send(self, pending: PendingQuery, reply: Envelope) -> None:
        """Send the reply, and warn if it was not sent: the client then waits for its full timeout."""
        if not pending.reply(encode_message(reply, self.limits)):
            logger.warning(
                "Reply %s for %s was not sent: the query expired %.1fs ago",
                reply.message_type,
                pending.key,
                time.monotonic() - pending.deadline,
            )

    def _dispatch(self, key: str, message: Envelope) -> Envelope:
        """Answer a request by its key: describe, open, or a session's reset, step or close."""
        if key in (deployment_prefix(self.config.deployment) + "/describe", ANY_DEPLOYMENT_DESCRIBE):
            _check_request_type(message, MessageType.DESCRIBE)
            return message.reply(
                MessageType.DESCRIPTOR,
                {
                    "deployment": self.config.deployment,
                    "instance_id": self.instance_id,
                    "descriptor": encode_descriptor(self.descriptor),
                },
            )
        if message.instance_id != self.instance_id:
            raise ProtocolError(ErrorCode.STALE, "Server instance does not match")
        if key == instance_prefix(self.config.deployment, self.instance_id) + "/open":
            _check_request_type(message, MessageType.OPEN)
            return message.reply(MessageType.ACCEPTED, self._open(message.body))
        _check_request_type(message, MessageType.CONTROL)
        session = self.sessions.get(message.session_id)
        if session is None or not key.startswith(session.prefix + "/"):
            raise ProtocolError(ErrorCode.STALE, "Unknown simulator session")
        if message.generation != session.generation:
            raise ProtocolError(ErrorCode.STALE, "Environment generation changed")
        operation = key.rsplit("/", 1)[-1]
        if operation == "close":
            self._close_session(message.session_id)
            return message.reply(MessageType.ACK, {})
        return message.reply(
            MessageType.ACK, session.with_jpeg_frames(session.handle(operation, message.body))
        )

    def _open(self, body: dict[str, Any]) -> dict[str, Any]:
        """Check an open request and start a session for it."""
        num_envs = body.get("num_envs", 1)
        use_async_envs = bool(body.get("use_async_envs", False))
        encoding = body.get("encoding", "raw")
        jpeg_quality = body.get("jpeg_quality", 90)
        client_payload_bytes = body.get("max_payload_bytes", self.limits.max_payload_bytes)
        if type(num_envs) is not int or not 1 <= num_envs <= self.config.max_envs:
            raise ProtocolError(
                ErrorCode.INCOMPATIBLE, f"Batch size {num_envs!r} is not between 1 and {self.config.max_envs}"
            )
        if encoding not in {"raw", "jpeg"} or type(jpeg_quality) is not int or not 1 <= jpeg_quality <= 100:
            raise ProtocolError(
                ErrorCode.UNSUPPORTED, "Image encoding must be raw or jpeg, quality in [1, 100]"
            )
        if type(client_payload_bytes) is not int or client_payload_bytes < 1:
            raise ProtocolError(ErrorCode.INCOMPATIBLE, "max_payload_bytes must be a positive integer")
        seeds = _checked_seeds(body.get("seeds"), num_envs)
        batch_bytes = num_envs * sum(
            int(np.prod(f.shape)) * np.dtype(f.dtype).itemsize for f in self.descriptor.features
        )
        # JPEG frames are far smaller on the wire, but the client still decodes them to full size.
        payload_bytes = min(self.limits.max_payload_bytes, client_payload_bytes)
        limits = CodecLimits(
            max_payload_bytes=payload_bytes,
            max_tensor_bytes=payload_bytes,
            max_decoded_bytes=2 * payload_bytes,
        )
        budget = limits.max_payload_bytes if encoding == "raw" else limits.max_decoded_bytes
        if batch_bytes + _REPLY_HEADROOM_BYTES > budget:
            raise ProtocolError(
                ErrorCode.INCOMPATIBLE,
                f"Observation batch of {batch_bytes} bytes exceeds the payload budget of {budget} bytes.",
            )
        if len(self.sessions) >= self.config.max_sessions:
            raise ProtocolError(ErrorCode.BUSY, "Server is busy with another session")
        env = self._session_env(num_envs, body.get("task_group"), body.get("task_id"), use_async_envs)
        session_id = uuid.uuid4().hex
        session = _Session(
            self.transport,
            session_prefix(self.config.deployment, self.instance_id, session_id),
            env,
            seeds,
            encoding,
            jpeg_quality,
        )
        self.sessions[session_id] = session
        return session.with_jpeg_frames(
            {
                "session_id": session_id,
                "result": encode_step_result(env.latest),
                "generation": session.generation,
                "descriptor": encode_descriptor(env.descriptor),
            }
        )

    def _session_env(
        self, num_envs: int, task_group: str | None, task_id: int | None, use_async_envs: bool
    ) -> LocalVectorEnv:
        """Return the startup env if it matches the request; otherwise close it and build a new one."""
        env, self._startup_env = self._startup_env, None
        if env is not None:
            # No task_id means the group's first task, which the startup env runs.
            task = env.task_id if task_id is None else task_id
            asked = (num_envs, task_group or env.task_group, task, use_async_envs)
            if asked == (env.num_envs, env.task_group, env.task_id, env.use_async_envs):
                return env
            env.close()
        return LocalVectorEnv(self.sim, num_envs, task_group, task_id, use_async_envs)

    def _reap_sessions(self) -> None:
        """Close sessions whose client left or never showed up."""
        now = time.monotonic()
        for session_id, session in list(self.sessions.items()):
            lost = session.presence.dropped > 0
            while True:
                try:
                    event = session.presence.get()
                except Empty:
                    break
                if event.alive:
                    session.presence_established = True
                elif session.presence_established:
                    lost = True
            elapsed = now - session.opened_at
            if lost or (
                not session.presence_established and elapsed > self.config.initial_presence_timeout_s
            ):
                self._close_session(session_id)

    def _close_session(self, session_id: str) -> None:
        """Remove a session and close it. A failed close is logged, so it never stops the server."""
        session = self.sessions.pop(session_id)
        try:
            session.close()
        except Exception:
            logger.exception("Failed to close session %s", session_id)
