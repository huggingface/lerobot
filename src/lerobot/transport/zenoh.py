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

"""Small explicit-endpoint Zenoh channels with bounded, nonblocking handoff.

No policy or robot semantics belong here. Call open/query/close from setup or a
background worker. Pub/sub congestion drops instead of blocking. Applications must
use their own request deadlines and must inspect channel overflow counters.
"""

import json
import math
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from typing import TYPE_CHECKING, Any, Literal, cast

from lerobot.utils.import_utils import _zenoh_available, require_package

if TYPE_CHECKING or _zenoh_available:
    import zenoh


class TransportError(RuntimeError):
    """A transport operation failed validation or could not complete successfully."""


class QueryCancelled(TransportError):  # noqa: N818
    """A local caller stopped waiting; remote execution may still be running."""


@dataclass
class ZenohConfig:
    """Explicit peer/router topology and resource bounds.

    ``config_file`` supplies security settings; explicit endpoints override its topology.
    ``max_payload_bytes`` bounds each message and the aggregate replies to one query.
    ``open_timeout_s`` bounds connection establishment; router clients cannot listen."""

    mode: Literal["peer", "client"] = "peer"
    connect_endpoints: list[str] = field(default_factory=list)
    listen_endpoints: list[str] = field(default_factory=list)
    config_file: str | None = None
    max_payload_bytes: int = 32 * 1024 * 1024
    open_timeout_s: float = 10.0

    def validate(self) -> None:
        """Reject invalid explicit topology without importing or opening Zenoh."""
        if self.mode not in ("peer", "client"):
            raise ValueError("Zenoh mode must be peer (direct) or client (router)")
        if not (self.connect_endpoints or self.listen_endpoints):
            raise ValueError("Explicit Zenoh connect or listen endpoints are required")
        if self.mode == "client" and (not self.connect_endpoints or self.listen_endpoints):
            raise ValueError("Router clients require connect endpoints and cannot listen")
        if type(self.max_payload_bytes) is not int or self.max_payload_bytes <= 0:
            raise ValueError("max_payload_bytes must be positive")
        _timeout(self.open_timeout_s)

    def build(self) -> "zenoh.Config":
        """Build explicit routing, disabling discovery and shared memory.

        Validate topology and bounds before loading Zenoh. Preserve security settings
        from the base JSON5 file; this method does not open a session."""
        self.validate()
        require_package("eclipse-zenoh", "remote", import_name="zenoh")
        config = zenoh.Config.from_file(self.config_file) if self.config_file else zenoh.Config()
        # TLS, certificate and ACL settings in the supplied JSON5 are preserved.
        for key, value in {
            "mode": self.mode,
            "connect/endpoints": self.connect_endpoints,
            "listen/endpoints": self.listen_endpoints,
            "scouting/multicast/enabled": False,
            "scouting/gossip/enabled": False,
            "transport/shared_memory/enabled": False,
            "connect/timeout_ms": int(self.open_timeout_s * 1000),
            "connect/exit_on_failure": True,
        }.items():
            config.insert_json5(key, json.dumps(value))
        return config


def _timeout(value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Timeout must be positive and finite")


def _payload(value: bytes, maximum: int) -> None:
    if not isinstance(value, bytes) or len(value) > maximum:
        raise TransportError("Payload is not bytes or exceeds transport limit")


class BoundedSubscriber[T]:
    """Bounded callback-to-consumer FIFO; full queues drop incoming values.

    Consumers must inspect ``dropped`` and ``oversized``. The transport owns declared
    channels, but consumers may close them earlier on their owning thread."""

    def __init__(self, capacity: int):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("Channel capacity must be positive")
        self._queue: Queue[T] = Queue(capacity)
        self._handle: Any = None
        self.dropped = 0
        self.oversized = 0
        self._closed = False
        self._on_close: Callable[[], None] | None = None

    def _offer(self, value: T) -> bool:
        if self._closed:
            return False
        try:
            self._queue.put_nowait(value)
            return True
        except Full:
            self.dropped += 1
            return False

    def get(self, timeout: float | None = 0) -> T:
        """Return the oldest item or raise ``queue.Empty`` after ``timeout`` seconds.

        Zero polls immediately; ``None`` waits indefinitely on the caller's thread."""
        return self._queue.get(timeout=timeout)

    def close(self) -> None:
        """Undeclare the subscription, discard queued items and release its ownership.

        Repeated calls are harmless. Closing does not wake a consumer already
        blocked in ``get(None)``; applications should use bounded waits.
        """
        self._closed = True
        if self._handle is not None:
            self._handle.undeclare()
            self._handle = None
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                break
        if self._on_close is not None:
            self._on_close()
            self._on_close = None


@dataclass(frozen=True)
class PresenceEvent:
    """Liveliness token key and whether that token was declared or removed."""

    key: str
    alive: bool


class PresenceToken:
    """Transport-owned liveliness token, normally created by ``declare_token``.

    Undeclare on the owning setup/IO thread, coordinated with transport teardown.
    Successful removal calls ``on_close`` exactly once to release ownership."""

    def __init__(self, handle: Any, on_close: Callable[["PresenceToken"], None]):
        self._handle = handle
        self._on_close = on_close

    def undeclare(self) -> None:
        """Remove the token once and release it from its owning transport."""
        if self._handle is not None:
            self._handle.undeclare()
            self._handle = None
            self._on_close(self)


class PendingQuery:
    """Retain a query beyond its callback until reply, drop or server-local expiry.

    The owner's capacity includes dequeued queries until released."""

    def __init__(self, query: "zenoh.Query", payload: bytes, owner: "BoundedQueryable"):
        self.payload = payload
        self.key = str(query.key_expr)
        self.deadline = time.monotonic() + owner.reply_timeout
        self._query: zenoh.Query | None = query
        self._owner = owner
        self._lock = threading.Lock()

    def reply(self, payload: bytes) -> bool:
        """Claim one bounded reply and release the query even on failure.

        Return whether it was submitted before expiry, not acknowledged by the peer;
        DROP congestion may discard it. Invalid payloads raise ``TransportError``."""
        with self._lock:
            query, self._query = self._query, None
        if query is None:
            return False
        try:
            _payload(payload, self._owner.max_payload_bytes)
            if time.monotonic() > self.deadline:
                return False
            # Zenoh 1.9 inherits the query's DROP congestion policy for replies.
            query.reply(self.key, payload)
            return True
        finally:
            try:
                query.drop()
            finally:
                self._owner._release(self)

    def drop(self) -> None:
        """Release the query without replying; repeated calls are harmless."""
        with self._lock:
            query, self._query = self._query, None
        if query is not None:
            try:
                query.drop()
            finally:
                self._owner._release(self)

    @property
    def expired(self) -> bool:
        """Whether the query was released or its server-local deadline has elapsed."""
        return self._query is None or time.monotonic() > self.deadline


class BoundedQueryable(BoundedSubscriber[PendingQuery]):
    """Bound queued and in-flight queries until reply, drop or server-local expiry.

    Callbacks only enqueue. The serving thread pumps ``get`` and must reply to or
    drop each result. Dequeuing does not free capacity; closing releases all queries.
    ``maximum`` bounds request/reply bytes; ``reply_timeout`` bounds handle lifetime."""

    def __init__(self, capacity: int, maximum: int, reply_timeout: float):
        super().__init__(capacity)
        _timeout(reply_timeout)
        self.reply_timeout = reply_timeout
        self.max_payload_bytes = maximum
        self._slots = threading.BoundedSemaphore(capacity)
        self._pending: set[PendingQuery] = set()

    def _receive(self, query: "zenoh.Query") -> None:
        # Direct callbacks make no network calls and never wait for capacity or model work.
        value = query.payload
        if value is not None and len(value) > self.max_payload_bytes:
            self.oversized += 1
            return
        if self._closed or not self._slots.acquire(blocking=False):
            self.dropped += 1
            return
        pending = PendingQuery(query, b"" if value is None else value.to_bytes(), self)
        self._pending.add(pending)
        if not self._offer(pending):
            # Dropping the last Python reference releases the query without a blocking reply.
            self._release(pending)

    def _release(self, pending: PendingQuery) -> None:
        if pending in self._pending:
            self._pending.remove(pending)
            self._slots.release()

    def get(self, timeout: float | None = 0) -> PendingQuery:
        """Reclaim expired handles and return the next live query to reply to or drop.

        Uses the same timeout semantics as ``BoundedSubscriber.get`` and raises
        ``queue.Empty`` if no live query arrives within that wait."""
        # The server worker polls even while idle, reclaiming expired reply handles.
        for pending in tuple(self._pending):
            if time.monotonic() > pending.deadline:
                pending.drop()
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            pending = super().get(remaining)
            if not pending.expired:
                return pending
            pending.drop()

    def close(self) -> None:
        """Undeclare the queryable and release queued and in-flight query handles."""
        super().close()
        for pending in tuple(self._pending):
            pending.drop()


class ZenohTransport:
    """Own a Zenoh session, channels and liveliness tokens.

    Open, declare and close on setup/IO threads, coordinated with active operations.
    Callbacks hand off bounded payloads. Blocking setup and query methods must never
    run on the robot control thread."""

    def __init__(self, config: ZenohConfig):
        self.config = config
        self._session: zenoh.Session | None = None
        self._channels: set[BoundedSubscriber[Any]] = set()
        self._tokens: set[PresenceToken] = set()

    def open(self) -> "ZenohTransport":
        """Open the session within the connection deadline if needed and return this transport."""
        if self._session is None:
            config = self.config.build()
            self._session = zenoh.open(config)
        return self

    @property
    def session(self) -> "zenoh.Session":
        """Return the owned session, or raise ``TransportError`` if it is not open."""
        if self._session is None:
            raise TransportError("Zenoh session is not open")
        return self._session

    def publish(self, key: str, payload: bytes) -> None:
        """Publish bounded bytes with DROP congestion and no delivery acknowledgment.

        Invalid payloads or a closed session raise ``TransportError``."""
        _payload(payload, self.config.max_payload_bytes)
        self.session.put(key, payload, congestion_control=zenoh.CongestionControl.DROP)

    def wait_for_subscriber(self, key: str, timeout: float) -> None:
        """Wait at most ``timeout`` seconds for a subscriber, or raise ``TimeoutError``.

        Setup/worker only. Matching proves routing declarations, not application
        readiness or delivery of subsequent publications."""
        _timeout(timeout)
        deadline = time.monotonic() + timeout
        with self.session.declare_publisher(
            key, congestion_control=zenoh.CongestionControl.DROP
        ) as publisher:
            # 1.9.0's stub says bool; the runtime returns MatchingStatus.
            while not cast(zenoh.MatchingStatus, publisher.matching_status).matching:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"No subscriber for {key} before setup deadline")
                time.sleep(min(remaining, 0.005))

    def subscribe(self, key: str, capacity: int = 2) -> BoundedSubscriber[bytes]:
        """Return a transport-owned channel with bounded callback handoff.

        New arrivals are dropped above ``capacity``; consumers must inspect overflow
        counters and may close the channel before transport teardown."""
        channel: BoundedSubscriber[bytes] = BoundedSubscriber(capacity)

        def receive(sample: zenoh.Sample) -> None:
            if len(sample.payload) > self.config.max_payload_bytes:
                channel.oversized += 1
            else:
                channel._offer(sample.payload.to_bytes())

        # indirect=False avoids the binding's additional blocking default FIFO.
        channel._handle = self.session.declare_subscriber(
            key, zenoh.handlers.Callback(receive, indirect=False)
        )
        self._channels.add(channel)
        channel._on_close = lambda: self._channels.discard(channel)
        return channel

    def declare_queryable(self, key: str, capacity: int = 8, reply_timeout: float = 30.0) -> BoundedQueryable:
        """Declare a transport-owned endpoint bounded across queued and in-flight queries.

        Each handle expires after ``reply_timeout`` seconds of server-local time.
        The consumer must reply to or drop each query returned by ``get``."""
        channel = BoundedQueryable(capacity, self.config.max_payload_bytes, reply_timeout)
        channel._handle = self.session.declare_queryable(
            key, zenoh.handlers.Callback(channel._receive, indirect=False), complete=True
        )
        self._channels.add(channel)
        channel._on_close = lambda: self._channels.discard(channel)
        return channel

    def query(
        self,
        key: str,
        payload: bytes,
        timeout: float,
        max_replies: int = 16,
        *,
        cancelled: Callable[[], bool] | None = None,
    ) -> list[bytes]:
        """Collect bounded replies to one publication within a caller-local deadline.

        Blocks through routing and reply completion without retrying. All matching
        queryables may reply, up to ``max_replies`` and the aggregate payload bound;
        an empty result is possible. ``cancelled`` is polled on the caller thread.
        Cancellation stops local waiting, not already-admitted remote work.

        Raises ``QueryCancelled`` on cancellation, ``TimeoutError`` on expiry,
        ``TransportError`` for peer errors or excess replies, and ``ValueError``
        for invalid timeout or reply-count bounds."""
        _payload(payload, self.config.max_payload_bytes)
        _timeout(timeout)
        if type(max_replies) is not int or not 1 <= max_replies <= 128:
            raise ValueError("max_replies must be between 1 and 128")
        if cancelled is not None and cancelled():
            raise QueryCancelled("Query wait cancelled before publication")
        deadline = time.monotonic() + timeout
        # A freshly opened connection may precede routing declaration propagation.
        # Wait before issuing the operation; never replay stateful queries.
        with self.session.declare_querier(key) as querier:
            while not cast(zenoh.MatchingStatus, querier.matching_status).matching:
                if cancelled is not None and cancelled():
                    raise QueryCancelled("Query wait cancelled before publication")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"No queryable for {key} before setup deadline")
                time.sleep(min(remaining, 0.005))
        timeout = max(0.001, deadline - time.monotonic())
        replies: list[bytes] = []
        errors: list[str] = []
        finished = threading.Event()
        cancellation_token = zenoh.CancellationToken()

        def receive(reply: zenoh.Reply) -> None:
            if errors:
                return
            if reply.ok is None:
                if reply.err is None or len(reply.err.payload) > 2048:
                    errors.append("Invalid or oversized Zenoh query error")
                else:
                    errors.append(reply.err.payload.to_bytes().decode("utf-8", errors="replace"))
                return
            value = reply.ok.payload
            if (
                len(value) + sum(map(len, replies)) > self.config.max_payload_bytes
                or len(replies) >= max_replies
            ):
                errors.append("Zenoh query reply count or payload exceeds limit")
                return
            replies.append(value.to_bytes())

        if cancelled is not None and cancelled():
            raise QueryCancelled("Query wait cancelled before publication")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Zenoh query deadline expired before publication for {key}")
        self.session.get(
            key,
            zenoh.handlers.Callback(receive, drop=finished.set, indirect=False),
            payload=payload,
            timeout=timeout,
            target=zenoh.QueryTarget.ALL,
            consolidation=zenoh.ConsolidationMode.NONE,
            congestion_control=zenoh.CongestionControl.DROP,
            cancellation_token=cancellation_token,
        )
        try:
            while not finished.is_set():
                if cancelled is not None and cancelled():
                    raise QueryCancelled("Query wait cancelled; remote operation may still be running")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Zenoh query deadline expired for {key}")
                finished.wait(min(remaining, 0.02))
        finally:
            cancellation_token.cancel()
        if errors:
            if "timeout" in errors[0].lower():
                raise TimeoutError(f"Zenoh query deadline expired for {key}")
            raise TransportError(errors[0])
        return replies

    def declare_token(self, key: str) -> PresenceToken:
        """Declare an owned liveliness token that can be undeclared before transport teardown."""
        token = PresenceToken(self.session.liveliness().declare_token(key), self._tokens.discard)
        self._tokens.add(token)
        return token

    def subscribe_liveliness(self, key: str, capacity: int = 8) -> BoundedSubscriber[PresenceEvent]:
        """Return a bounded channel of existing and future liveliness changes.

        New arrivals are dropped above ``capacity``. Overflow means lost presence
        information, not confirmed liveness. The transport owns the channel."""
        channel: BoundedSubscriber[PresenceEvent] = BoundedSubscriber(capacity)

        def receive(sample: zenoh.Sample) -> None:
            channel._offer(PresenceEvent(str(sample.key_expr), sample.kind == zenoh.SampleKind.PUT))

        channel._handle = self.session.liveliness().declare_subscriber(
            key, zenoh.handlers.Callback(receive, indirect=False), history=True
        )
        self._channels.add(channel)
        channel._on_close = lambda: self._channels.discard(channel)
        return channel

    def close(self) -> None:
        """Close owned channels and tokens before closing the network session.

        Coordinate this call with application workers so none continue using the
        transport. Repeated calls are harmless; no outstanding policy work is cancelled.
        """
        for channel in tuple(self._channels):
            channel.close()
        self._channels.clear()
        for token in tuple(self._tokens):
            # A caller may have explicitly ended its token earlier.
            with suppress(zenoh.ZError):
                token.undeclare()
        self._tokens.clear()
        if self._session is not None:
            self._session.close()
            self._session = None

    def __enter__(self) -> "ZenohTransport":
        return self.open()

    def __exit__(self, *_: Any) -> None:
        self.close()
