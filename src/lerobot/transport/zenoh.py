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
    """Explicit topology and payload bounds for one transport session.

    Args:
        mode (`Literal["peer", "client"]`, *optional*, defaults to `"peer"`):
            Direct peer mode or router client mode.
        connect_endpoints (`list[str]`, *optional*):
            Explicit peers or routers to connect to.
        listen_endpoints (`list[str]`, *optional*):
            Local listening endpoints. Router clients cannot listen.
        config_file (`str | None`, *optional*):
            Base JSON5 configuration whose security settings are preserved.
        max_payload_bytes (`int`, *optional*, defaults to `33554432`):
            Maximum message size and combined query-reply payload size.
        open_timeout_s (`float`, *optional*, defaults to `10.0`):
            Positive connection establishment deadline in seconds.
    """

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
        """Build a Zenoh configuration without opening a network session.

        Explicit topology overrides the base file. Discovery and shared memory are
        disabled so routing follows the supplied endpoints.

        Returns:
            `zenoh.Config`: Validated configuration for ``zenoh.open``.

        Raises:
            ValueError: If topology, payload bounds or timeout are invalid.
        """
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
    """Hand callback payloads to a consumer through a bounded FIFO.

    Full queues drop the incoming value instead of blocking the callback. The
    consumer must inspect ``dropped`` and ``oversized`` to detect lost input.
    The transport owns declared channels and closes them at session teardown;
    consumers may close a channel earlier on their owning thread.

    Args:
        capacity (`int`):
            Positive number of queued items allowed.
    """

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
        """Remove the next item, optionally waiting on the caller's thread.

        Args:
            timeout (`float | None`, *optional*, defaults to `0`):
                Seconds to wait; zero polls immediately and ``None`` waits indefinitely.

        Returns:
            `T`: Oldest queued item.

        Raises:
            queue.Empty: If no item arrives within the requested wait.
        """
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
    """A liveliness declaration or removal delivered by a presence subscription.

    Args:
        key (`str`):
            Key expression of the token whose state changed.
        alive (`bool`):
            Whether the token is currently declared according to this event.
    """

    key: str
    alive: bool


class PresenceToken:
    """A token whose explicit removal also releases its transport's ownership.

    Instances are normally created by ``ZenohTransport.declare_token``. Remove them
    on the owning setup or IO thread, coordinated with transport teardown.

    Args:
        handle (`Any`):
            Live Zenoh liveliness-token handle.
        on_close (`Callable[[PresenceToken], None]`):
            Ownership-release callback invoked once after successful removal.
    """

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
    """Retain a Zenoh query beyond its callback; reply/drop releases it exactly once.

    Expiry uses only server-local time. The owner's bounded capacity includes queries
    that have left the handoff queue but whose worker has not yet replied.

    Args:
        query (`zenoh.Query`):
            Incoming query retained beyond its callback lifetime.
        payload (`bytes`):
            Bounded request bytes copied by the queryable callback.
        owner (`BoundedQueryable`):
            Queryable whose retained-query capacity this handle occupies.
    """

    def __init__(self, query: "zenoh.Query", payload: bytes, owner: "BoundedQueryable"):
        self.payload = payload
        self.key = str(query.key_expr)
        self.deadline = time.monotonic() + owner.reply_timeout
        self._query: zenoh.Query | None = query
        self._owner = owner
        self._lock = threading.Lock()

    def reply(self, payload: bytes) -> bool:
        """Attempt one reply and release the retained query, including on failure.

        Reply and drop claim the query under a lock, so only one caller uses it.

        Args:
            payload (`bytes`):
                Encoded reply within the owning queryable's payload bound.

        Returns:
            `bool`: Whether the reply was submitted before expiry. This does not
            acknowledge remote receipt; DROP congestion may discard traffic.

        Raises:
            TransportError: If the payload is invalid or exceeds the size bound.
        """
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
    """Bound queued and in-flight queries until each is replied to, dropped or expired.

    Direct Zenoh callbacks only enqueue work. The application pumps ``get`` on its
    serving thread and owns each returned ``PendingQuery`` until reply or drop.
    Dequeuing does not free capacity. Closing releases all remaining queries.

    Args:
        capacity (`int`):
            Positive bound across queued and in-flight queries.
        maximum (`int`):
            Maximum request and reply payload size in bytes.
        reply_timeout (`float`):
            Positive server-local lifetime of each retained query in seconds.
    """

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
        """Reclaim expired handles and return the next live query.

        Args:
            timeout (`float | None`, *optional*, defaults to `0`):
                Seconds to wait; zero polls immediately and ``None`` waits indefinitely.

        Returns:
            `PendingQuery`: Query the consumer must eventually reply to or drop.

        Raises:
            queue.Empty: If no live query arrives before the wait expires.
        """
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
    """Own one Zenoh session and its channel and liveliness-token handles.

    Open, declare and close resources from the application's setup or IO worker,
    with teardown coordinated against active operations. Callbacks only hand off
    bounded payloads; policy execution belongs to the application. Blocking setup
    and query methods must not run on the robot control thread.

    Args:
        config (`ZenohConfig`):
            Explicit endpoint configuration and resource bounds.
    """

    def __init__(self, config: ZenohConfig):
        self.config = config
        self._session: zenoh.Session | None = None
        self._channels: set[BoundedSubscriber[Any]] = set()
        self._tokens: set[PresenceToken] = set()

    def open(self) -> "ZenohTransport":
        """Open the session if needed, waiting for configured connection setup.

        Returns:
            `ZenohTransport`: This transport with an open session.
        """
        if self._session is None:
            config = self.config.build()
            self._session = zenoh.open(config)
        return self

    @property
    def session(self) -> "zenoh.Session":
        """Return the open session without transferring resource ownership.

        Returns:
            `zenoh.Session`: Session owned by this transport.

        Raises:
            TransportError: If the transport has not been opened or was closed.
        """
        if self._session is None:
            raise TransportError("Zenoh session is not open")
        return self._session

    def publish(self, key: str, payload: bytes) -> None:
        """Publish bytes with DROP congestion, without acknowledging delivery.

        Args:
            key (`str`):
                Publication key expression.
            payload (`bytes`):
                Encoded message within the configured payload bound.

        Raises:
            TransportError: If payload validation fails or the session is not open.
        """
        _payload(payload, self.config.max_payload_bytes)
        self.session.put(key, payload, congestion_control=zenoh.CongestionControl.DROP)

    def wait_for_subscriber(self, key: str, timeout: float) -> None:
        """Wait during setup for routing declarations; never call on a control thread.

        Matching confirms a subscriber declaration, not application readiness or
        delivery of subsequent publications.

        Args:
            key (`str`):
                Publication key expression that needs a matching subscriber.
            timeout (`float`):
                Positive caller-local wait bound in seconds.

        Raises:
            TimeoutError: If no matching subscription appears before the deadline.
        """
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
        """Declare a subscription with bounded callback-to-consumer handoff.

        Args:
            key (`str`):
                Subscription key expression.
            capacity (`int`, *optional*, defaults to `2`):
                Maximum queued payloads before new arrivals are dropped.

        Returns:
            `BoundedSubscriber[bytes]`: Channel owned by this transport. The consumer
            must inspect overflow counters and may close it before session teardown.
        """
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
        """Declare a query endpoint whose worker handoff and retained replies are bounded.

        Args:
            key (`str`):
                Queryable key expression.
            capacity (`int`, *optional*, defaults to `8`):
                Combined bound on queued and in-flight queries.
            reply_timeout (`float`, *optional*, defaults to `30.0`):
                Server-local lifetime of each query handle in seconds.

        Returns:
            `BoundedQueryable`: Endpoint owned by this transport; its consumer must
            eventually reply to or drop each query returned by ``get``.
        """
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
        """Collect replies to one publication within a caller-local deadline.

        This blocks the caller while waiting for routing and replies. It does not
        retry the operation. Cancellation stops the local wait and cannot undo work
        already admitted by a remote queryable.

        Args:
            key (`str`):
                Query key expression. All matching queryables may reply.
            payload (`bytes`):
                Encoded request within the configured payload bound.
            timeout (`float`):
                Positive total routing-and-reply wait bound in seconds.
            max_replies (`int`, *optional*, defaults to `16`):
                Maximum retained replies, between one and 128 inclusive.
            cancelled (`Callable[[], bool] | None`, *optional*):
                Predicate polled on the caller's thread to cancel local waiting.

        Returns:
            `list[bytes]`: Replies received before the query completes, within the
            configured aggregate payload bound. An empty result is possible.

        Raises:
            QueryCancelled: If the cancellation predicate returns true.
            TimeoutError: If routing or query completion exceeds the deadline.
            TransportError: If a reply reports an error or resource bounds are exceeded.
            ValueError: If timeout or reply-count bounds are invalid.
        """
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
        """Declare a liveliness token owned by this transport until removal or close.

        Args:
            key (`str`):
                Token key expression.

        Returns:
            `PresenceToken`: Handle that may be explicitly undeclared before teardown.
        """
        token = PresenceToken(self.session.liveliness().declare_token(key), self._tokens.discard)
        self._tokens.add(token)
        return token

    def subscribe_liveliness(self, key: str, capacity: int = 8) -> BoundedSubscriber[PresenceEvent]:
        """Subscribe to existing and future liveliness declarations and removals.

        Args:
            key (`str`):
                Liveliness key expression to observe, including existing tokens.
            capacity (`int`, *optional*, defaults to `8`):
                Maximum queued presence events before new events are dropped.

        Returns:
            `BoundedSubscriber[PresenceEvent]`: Transport-owned channel. The consumer
            must treat overflow as lost presence information, not confirmed liveness.
        """
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
