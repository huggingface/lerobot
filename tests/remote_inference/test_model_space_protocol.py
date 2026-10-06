# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Exercise separate model/canonical widths through the complete encoded RPC boundary."""

import time
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
import torch

pytest.importorskip("datasets")
pytest.importorskip("msgpack")

from lerobot.inference import ChunkRequest, ExecutionMode, RemoteInferenceConfig
from lerobot.policies.rtc.action_queue import QueueSnapshot
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.remote_inference.chunk_contract import RTC_MODEL_SPACE
from lerobot.remote_inference.client import RemoteClient
from lerobot.remote_inference.codec import decode_message, encode_message
from lerobot.remote_inference.protocol import ErrorCode, MessageType, ProtocolError
from lerobot.remote_inference.server import SessionWorker
from lerobot.transport.zenoh import BoundedSubscriber
from tests.inference.test_policy_contracts import PaddedPolicy, make_runner
from tests.inference.test_policy_runner import ConformingPolicy, observation, tiny_config
from tests.remote_inference.test_session import assert_error, open_request


class EncodedWorkerTransport:
    """Synchronous delivery only; worker, codec, admission and client validation remain real."""

    def __init__(self, worker):
        self.worker = worker
        self.queries = []
        self.channels = {}
        self.change_reply = lambda response: response

    def subscribe_liveliness(self, key):
        return BoundedSubscriber(capacity=4)

    def subscribe(self, key, *, capacity):
        channel = BoundedSubscriber(capacity)
        self.channels[key] = channel
        return channel

    def declare_token(self, key):
        return SimpleNamespace(close=lambda: None)

    def wait_for_subscriber(self, key, timeout):
        pass

    def query(self, key, payload, timeout, *, cancelled=None):
        request = decode_message(payload)
        self.queries.append(request)
        response = self.worker.submit(request).result(timeout=2)
        return [encode_message(response)]

    def publish(self, key, payload):
        request = decode_message(payload)
        response = self.worker.submit(request).result(timeout=2)
        self.channels[key.removesuffix("/obs") + "/act"]._offer(encode_message(self.change_reply(response)))

    def close(self):
        for channel in self.channels.values():
            channel.close()


@pytest.fixture(params=[True, False], ids=["padded", "canonical"])
def worker(request):
    padded = request.param
    config = tiny_config()
    config.rtc_config = RTCConfig(execution_horizon=4)
    policy = PaddedPolicy(config) if padded else ConformingPolicy(config)
    runner = make_runner(
        policy,
        padded=padded,
        language_enabled=False,
        modes=(ExecutionMode.CHUNK, ExecutionMode.RTC_GUIDED),
    )
    runner.predict(observation(), mode=ExecutionMode.RTC_GUIDED)
    runner.reset()
    instance = SessionWorker(runner, deployment="test", artifact_identity="artifact", semantics="radians")
    try:
        yield instance
    finally:
        instance.close()


def client_for(worker, *, descriptor=None, mode="rtc_guided", chunk_merge="auto"):
    transport = EncodedWorkerTransport(worker)
    config = RemoteInferenceConfig(deployment="test", semantics="radians", mode=mode, chunk_merge=chunk_merge)
    return RemoteClient(transport, config, worker.descriptor if descriptor is None else descriptor)


def admit(client):
    caps = client.capabilities
    client.admit(
        features=caps.features,
        action_feature=caps.action_feature,
        semantics="radians",
        action_interval=caps.action_interval,
        mode=client.config.mode,
    )


def action_request(previous=None, *, mode=ExecutionMode.RTC_GUIDED):
    snapshot = QueueSnapshot(
        generation=0,
        cursor=0 if previous is None else 4,
        index=0 if previous is None else 4,
        model_actions=None if previous is None else previous.model_actions[-4:],
        canonical_actions=None if previous is None else previous.canonical_actions[-4:],
        provenance=() if previous is None else (previous.provenance,) * 4,
    )
    return ChunkRequest(
        observation(), snapshot, uuid4().hex, 0, mode, int(previous is not None), time.monotonic()
    )


def test_warmup_descriptor_and_admission_preserve_distinct_widths(worker):
    descriptor = worker.descriptor
    model_width = worker.runner.capabilities.model_action_dim
    padded = model_width != 3
    assert ("model_action_dim" in descriptor["capabilities"]) is padded
    assert (RTC_MODEL_SPACE in descriptor["execution_contracts"]) is padded
    client = client_for(worker)
    admit(client)
    opening = next(
        request for request in client.transport.queries if request.message_type is MessageType.OPEN
    )
    assert opening.body.get("required_capabilities", []) == ([RTC_MODEL_SPACE] if padded else [])
    first = client.infer(action_request())
    assert first.model_actions.shape == (8, model_width)
    assert first.canonical_actions.shape == (8, 3)
    second = client.infer(action_request(first))
    assert second.model_actions.shape == (8, model_width)
    torch.testing.assert_close(
        worker.runner.policy.last_kwargs["prev_chunk_left_over"], first.model_actions[-4:]
    )
    client.close()


@pytest.mark.parametrize("worker", [True], indirect=True)
def test_padded_rtc_requires_client_capability_before_allocating_session(worker):
    request = open_request(worker)
    request.body["mode"] = "rtc_guided"
    assert_error(worker.submit(request).result(2), ErrorCode.INCOMPATIBLE)
    assert worker.session_id is None


@pytest.mark.parametrize("worker", [True], indirect=True)
def test_padded_rtc_requires_server_capability_before_open(worker):
    descriptor = worker.descriptor
    descriptor["execution_contracts"].remove(RTC_MODEL_SPACE)
    client = client_for(worker, descriptor=descriptor)
    with pytest.raises(ProtocolError, match="model-space") as error:
        admit(client)
    assert error.value.code is ErrorCode.UNSUPPORTED
    assert not client.transport.queries
    assert worker.session_id is None


@pytest.mark.parametrize("worker", [True], indirect=True)
def test_admission_cannot_change_the_advertised_model_width(worker):
    descriptor = worker.descriptor
    descriptor["capabilities"]["model_action_dim"] = 7
    client = client_for(worker, descriptor=descriptor)
    with pytest.raises(ProtocolError, match="changed the advertised contract"):
        admit(client)
    assert not client.session_id


def test_plain_chunks_do_not_require_rtc_model_space_or_return_model_values(worker):
    client = client_for(worker, mode="chunk", chunk_merge="append")
    admit(client)
    opening = next(
        request for request in client.transport.queries if request.message_type is MessageType.OPEN
    )
    assert not opening.body.get("required_capabilities")
    chunk = client.infer(action_request(mode=ExecutionMode.CHUNK))
    assert chunk.canonical_actions.shape == (3, 3)
    assert chunk.model_actions is None
    client.close()


@pytest.mark.parametrize("worker", [True], indirect=True)
@pytest.mark.parametrize("space,width", [("model_actions", 3), ("canonical_actions", 6)])
def test_reply_width_mismatch_is_rejected_by_client(worker, space, width):
    client = client_for(worker)
    admit(client)

    def change_reply(response):
        response.body[space] = np.ones((8, width), dtype=np.float32)
        return response

    client.transport.change_reply = change_reply
    with pytest.raises(ProtocolError) as error:
        client.infer(action_request())
    assert error.value.code is ErrorCode.MALFORMED
    client.close()


@pytest.mark.parametrize("worker", [True], indirect=True)
@pytest.mark.parametrize("space,width", [("model_actions", 3), ("canonical_actions", 6)])
def test_continuation_width_mismatch_is_rejected_before_policy_execution(worker, space, width):
    client = client_for(worker)
    admit(client)
    first = client.infer(action_request())
    request = action_request(first)
    request = replace(request, continuation=replace(request.continuation, **{space: torch.ones(4, width)}))
    worker.runner.policy.last_kwargs = {}
    with pytest.raises(ProtocolError, match="Invalid continuation") as error:
        client.infer(request)
    assert error.value.code is ErrorCode.MALFORMED
    assert worker.runner.policy.last_kwargs == {}
    client.close()
