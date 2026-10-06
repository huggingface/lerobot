# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Bounded replies, connection diagnostics and admission errors."""

import json
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("zenoh")
pytest.importorskip("msgpack")
pytest.importorskip("datasets")

from lerobot.inference import ExecutionMode, FeatureSpec, PolicyCapabilities, RemoteInferenceConfig
from lerobot.remote_inference.client import RemoteClient
from lerobot.remote_inference.codec import encode_message
from lerobot.remote_inference.protocol import Envelope, ErrorCode, MessageType, ProtocolError
from lerobot.remote_inference.server import PolicyServer
from lerobot.transport.zenoh import (
    BoundedQueryable,
    PendingQuery,
    TransportError,
    ZenohConfig,
    ZenohTransport,
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


def test_explicit_peer_connection_must_be_established_at_open():
    config = ZenohConfig(connect_endpoints=["tcp/127.0.0.1:7447"]).build()
    assert json.loads(config.get_json("connect/exit_on_failure")) is True


def test_unreachable_endpoint_names_connection_and_deployment():
    with socket.socket() as port:
        port.bind(("127.0.0.1", 0))
        endpoint = f"tcp/127.0.0.1:{port.getsockname()[1]}"
    config = RemoteInferenceConfig(
        endpoint=endpoint,
        deployment="missing-server",
        semantics="joints",
        handshake_timeout_s=0.1,
    )
    with pytest.raises(ConnectionError) as failed:
        RemoteClient.connect(config)
    assert endpoint in str(failed.value)
    assert "missing-server" in str(failed.value)
    assert "server/router" in str(failed.value)
    assert failed.value.__cause__ is not None


def test_connected_endpoint_with_wrong_deployment_has_discovery_guidance(remote_server):
    _, config = remote_server
    config = replace(config, deployment="wrong-deployment", handshake_timeout_s=0.1)
    with pytest.raises(TimeoutError) as failed:
        RemoteClient.connect(config)
    assert config.endpoint in str(failed.value)
    assert "wrong-deployment" in str(failed.value)
    assert "--inference.deployment" in str(failed.value)


@pytest.fixture
def contract_client():
    caps = PolicyCapabilities(
        modes=(ExecutionMode.CHUNK,),
        prediction_steps=2,
        execution_steps=2,
        action_interval=1 / 30,
        features=(FeatureSpec("observation.state", (2,), "float32", names=("a", "b"), semantics="joints"),),
        action_feature=FeatureSpec("action", (2,), "float32", names=("a", "b"), semantics="joints"),
    )
    transport = SimpleNamespace(subscribe_liveliness=lambda _: None, query=Mock())
    client = RemoteClient(
        transport,
        RemoteInferenceConfig(deployment="test", semantics="joints"),
        {
            "capabilities": asdict(caps),
            "instance_id": "server",
            "artifact_identity": "model",
            "semantics": "joints",
        },
    )
    return client


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("names", ("b", "a"), "field 'names'"),
        ("shape", (3,), "field 'shape'"),
        ("dtype", "float64", "field 'dtype'"),
        ("semantics", "degrees", "field 'semantics'"),
        ("name", "observation.other", "Observation feature names/order"),
    ],
)
def test_observation_mismatch_identifies_field_and_both_values(contract_client, field, value, expected):
    client = contract_client
    feature = client.capabilities.features[0]
    overrides = {field: value}
    if field == "shape":
        overrides["names"] = ()
    with pytest.raises(ProtocolError) as failed:
        client.admit(
            features=(replace(feature, **overrides),),
            action_feature=client.capabilities.action_feature,
            semantics="joints",
            action_interval=1 / 30,
            mode="chunk",
        )
    assert failed.value.code is ErrorCode.INCOMPATIBLE
    assert expected in str(failed.value)
    assert "client=" in str(failed.value)
    assert "server=" in str(failed.value)
    client.transport.query.assert_not_called()


@pytest.mark.parametrize(
    "override,expected",
    [
        ({"semantics": "degrees"}, "Semantic convention differs: client='degrees', server='joints'"),
        ({"action_interval": 1 / 25}, "Use --fps=30"),
        ({"mode": "rtc_guided"}, "Execution mode 'rtc_guided' is not supported"),
        ({"mode": "typo"}, "Execution mode 'typo' is not supported"),
    ],
)
def test_non_feature_mismatch_has_actionable_context(contract_client, override, expected):
    client = contract_client
    arguments = {
        "features": client.capabilities.features,
        "action_feature": client.capabilities.action_feature,
        "semantics": "joints",
        "action_interval": 1 / 30,
        "mode": "chunk",
        **override,
    }
    with pytest.raises(ProtocolError, match=expected):
        client.admit(**arguments)
    client.transport.query.assert_not_called()


def test_action_mismatch_names_the_action_field(contract_client):
    client = contract_client
    with pytest.raises(ProtocolError, match="Feature 'action' field 'names'"):
        client.admit(
            features=client.capabilities.features,
            action_feature=replace(client.capabilities.action_feature, names=("b", "a")),
            semantics="joints",
            action_interval=1 / 30,
            mode="chunk",
        )
    client.transport.query.assert_not_called()


def test_later_reset_does_not_wait_for_missing_reply_subscribers(remote_server, monkeypatch):
    _, config = remote_server
    client = RemoteClient.connect(config)
    original_wait = ZenohTransport.wait_for_subscriber
    waits = []

    def wait_for_subscriber(self, key, timeout):
        waits.append(key)
        return original_wait(self, key, timeout)

    monkeypatch.setattr(ZenohTransport, "wait_for_subscriber", wait_for_subscriber)
    cancelled = Event()
    try:
        helpers.admit(client)
        assert any(key.endswith("/act") for key in waits)
        assert any(key.endswith("/language/result") for key in waits)
        waits.clear()
        client._actions.close()
        client._language.close()
        with ThreadPoolExecutor() as pool:
            reset = pool.submit(client.control, "reset", 1, cancelled=cancelled.is_set)
            try:
                reset.result(1)
            finally:
                cancelled.set()
        assert client.generation == 1
        assert waits == [], "matching-status setup waits belong only to the admission reset"
    finally:
        client.close()


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
