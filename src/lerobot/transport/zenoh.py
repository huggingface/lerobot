# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
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
    pass


class QueryCancelled(TransportError):  # noqa: N818
    """A local caller stopped waiting; remote execution may still be running."""


@dataclass
class ZenohConfig:
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
    """FIFO with drop-new overflow; get(0) raises queue.Empty when there is no item."""

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
        return self._queue.get(timeout=timeout)

    def close(self) -> None:
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
    key: str
    alive: bool


class PresenceToken:
    """A token whose explicit removal also releases its transport's ownership."""

    def __init__(self, handle: Any, on_close: Callable[["PresenceToken"], None]):
        self._handle = handle
        self._on_close = on_close

    def undeclare(self) -> None:
        if self._handle is not None:
            self._handle.undeclare()
            self._handle = None
            self._on_close(self)


class PendingQuery:
    """Retain a Zenoh query beyond its callback; reply/drop releases it exactly once.

    Expiry uses only server-local time. The owner's bounded capacity includes queries
    that have left the handoff queue but whose worker has not yet replied.
    """

    def __init__(self, query: "zenoh.Query", payload: bytes, owner: "BoundedQueryable"):
        self.payload = payload
        self.key = str(query.key_expr)
        self.deadline = time.monotonic() + owner.reply_timeout
        self._query: zenoh.Query | None = query
        self._owner = owner
        self._lock = threading.Lock()

    def reply(self, payload: bytes) -> bool:
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
        with self._lock:
            query, self._query = self._query, None
        if query is not None:
            try:
                query.drop()
            finally:
                self._owner._release(self)

    @property
    def expired(self) -> bool:
        return self._query is None or time.monotonic() > self.deadline


class BoundedQueryable(BoundedSubscriber[PendingQuery]):
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
        super().close()
        for pending in tuple(self._pending):
            pending.drop()


class ZenohTransport:
    """Own one Zenoh session; channel and token handles close with it."""

    def __init__(self, config: ZenohConfig):
        self.config = config
        self._session: zenoh.Session | None = None
        self._channels: set[BoundedSubscriber[Any]] = set()
        self._tokens: set[PresenceToken] = set()

    def open(self) -> "ZenohTransport":
        if self._session is None:
            config = self.config.build()
            self._session = zenoh.open(config)
        return self

    @property
    def session(self) -> "zenoh.Session":
        if self._session is None:
            raise TransportError("Zenoh session is not open")
        return self._session

    def publish(self, key: str, payload: bytes) -> None:
        _payload(payload, self.config.max_payload_bytes)
        self.session.put(key, payload, congestion_control=zenoh.CongestionControl.DROP)

    def wait_for_subscriber(self, key: str, timeout: float) -> None:
        """Wait during setup for routing declarations; never call on a control thread."""
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
        """Collect replies within a deadline, retaining at most max_replies bounded payloads."""
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
        token = PresenceToken(self.session.liveliness().declare_token(key), self._tokens.discard)
        self._tokens.add(token)
        return token

    def subscribe_liveliness(self, key: str, capacity: int = 8) -> BoundedSubscriber[PresenceEvent]:
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
