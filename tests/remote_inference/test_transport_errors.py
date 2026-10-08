# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Reply capacity is released after failure; overflow cannot become valid motion."""

import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("zenoh")
pytest.importorskip("msgpack")
pytest.importorskip("datasets")

from lerobot.remote_inference.client import RemoteClient
from lerobot.remote_inference.codec import encode_message
from lerobot.remote_inference.protocol import Envelope, ErrorCode, MessageType, ProtocolError
from lerobot.remote_inference.server import PolicyServer
from lerobot.transport.zenoh import (
    BoundedQueryable,
    PendingQuery,
    TransportError,
)
from tests.inference.test_policy_runner import observation
from tests.remote_inference import test_remote_path as helpers

remote_server = helpers.remote_server


@pytest.mark.parametrize("failure", ["oversized", "reply", "drop"])
def test_failed_reply_immediately_releases_query_capacity(failure):
    channel = BoundedQueryable(capacity=1, maximum=4, reply_timeout=30)
    query = SimpleNamespace(key_expr="control", payload=None, reply=Mock(), drop=Mock())
    if failure == "reply":
        query.reply.side_effect = RuntimeError("network reply failed")
    elif failure == "drop":
        query.drop.side_effect = RuntimeError("network drop failed")
    channel._receive(query)
    pending = channel.get()
    payload = b"oversized" if failure == "oversized" else b"ok"
    with pytest.raises(TransportError if failure == "oversized" else RuntimeError):
        pending.reply(payload)
    assert pending.expired
    assert not channel._pending
    query.drop.assert_called_once()
    assert not pending.reply(b"ok"), "failed replies cannot be replayed"
    pending.drop()
    query.drop.assert_called_once()

    next_query = SimpleNamespace(key_expr="control", payload=None, reply=Mock(), drop=Mock())
    channel._receive(next_query)
    assert channel.get().reply(b"ok"), "the failed reply must not hold the sole capacity slot"
    assert channel.dropped == 0


def test_unencodable_control_reply_releases_handle_and_pump_keeps_serving(remote_server, monkeypatch):
    _, config = remote_server
    original_encode = PolicyServer._encode_reply
    original_drop = PendingQuery.drop
    dropped = []
    fail_once = True

    def encode_reply(response):
        nonlocal fail_once
        if response.message_type is MessageType.DESCRIPTOR and fail_once:
            fail_once = False
            raise RuntimeError("unexpected encoder failure")
        return original_encode(response)

    def drop(self):
        dropped.append(self.key)
        original_drop(self)

    monkeypatch.setattr(PolicyServer, "_encode_reply", staticmethod(encode_reply))
    monkeypatch.setattr(PendingQuery, "drop", drop)
    with pytest.raises(ProtocolError, match="No ready instance"):
        RemoteClient.connect(config)
    assert any(key.endswith("/describe") for key in dropped)
    replacement = RemoteClient.connect(config)
    try:
        helpers.admit(replacement)
        assert replacement.session_id
    finally:
        replacement.close()


def test_action_reply_overflow_is_malformed_before_acceptance(remote_server):
    _, config = remote_server
    client = RemoteClient.connect(config)
    try:
        helpers.admit(client)
        stale = encode_message(
            Envelope(MessageType.ACTION, client.instance_id, client.session_id, request_id="obsolete")
        )
        for _ in range(4):
            assert client._actions._offer(stale)
        assert not client._actions._offer(stale)
        assert client._actions.dropped == 1
        runtime = helpers.runtime_for(client)
        source = replace(observation(), capture_time=time.monotonic())
        request = runtime.begin(source)
        assert request is not None
        with pytest.raises(ProtocolError, match="reply channel overflow") as failed:
            client.infer(request)
        assert failed.value.code is ErrorCode.MALFORMED
        assert runtime.queue.qsize() == 0
    finally:
        client.close()
