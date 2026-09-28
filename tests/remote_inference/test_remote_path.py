# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""A complete direct-Zenoh path with real canonical processor pipelines."""

import json
import socket
import time
from dataclasses import replace
from threading import Event, Thread
from types import SimpleNamespace

import msgpack
import pytest
import torch

pytest.importorskip("zenoh")

from lerobot.inference.contracts import ExecutionMode, FeatureSpec
from lerobot.inference.execution import ChunkRuntime
from lerobot.inference.policy_runner import PolicyRunner
from lerobot.remote_inference.client import RemoteClient, RequestCancelled
from lerobot.remote_inference.codec import encode_message
from lerobot.remote_inference.protocol import Envelope, ErrorCode, MessageType, ProtocolError
from lerobot.remote_inference.server import PolicyServer, SessionWorker
from lerobot.rollout.configs import RolloutConfig
from lerobot.rollout.context import build_rollout_context
from lerobot.rollout.inference.factory import RemoteInferenceConfig
from lerobot.rollout.strategies.core import send_next_action
from lerobot.transport.zenoh import ZenohConfig, ZenohTransport
from lerobot.utils.action_interpolator import ActionInterpolator
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE
from tests.inference.test_policy_runner import (
    ConformingPolicy,
    observation,
    processors,
    runner_for,
    tiny_config,
)


@pytest.fixture
def remote_server(request):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        endpoint = f"tcp/127.0.0.1:{sock.getsockname()[1]}"
    if getattr(request, "param", None) == "robot":
        config = tiny_config()
        del config.input_features[OBS_ENV_STATE]
        policy = ConformingPolicy(config)
        names = tuple(f"joint_{index}.pos" for index in range(3))
        runner = PolicyRunner(
            policy,
            *processors(config),
            action_interval=1 / 30,
            features=(FeatureSpec(OBS_STATE, (3,), "float32", names=names, semantics="radians-v1"),),
            action_feature=FeatureSpec(ACTION, (3,), "float32", names=names, semantics="radians-v1"),
        )
    else:
        runner = runner_for(ConformingPolicy(tiny_config()))
    worker = SessionWorker(
        runner, deployment="loopback", artifact_identity="processor-test", semantics="radians-v1"
    )
    transport = ZenohTransport(ZenohConfig(listen_endpoints=[endpoint]))
    server = PolicyServer(worker, transport)
    ready = Event()
    original_token = transport.declare_token

    def declare_token(key):
        token = original_token(key)
        ready.set()
        return token

    transport.declare_token = declare_token
    thread = Thread(target=server.serve, daemon=True)
    thread.start()
    assert ready.wait(3), "server did not advertise readiness"
    config = RemoteInferenceConfig(
        endpoint=endpoint,
        deployment="loopback",
        semantics="radians-v1",
        hold_mode="position",
        handshake_timeout_s=2,
        action_timeout_s=2,
    )
    try:
        yield worker, config
    finally:
        server.stop()
        thread.join(3)
        assert not thread.is_alive()


def admit(client):
    caps = client.capabilities
    client.admit(
        features=caps.features,
        action_feature=caps.action_feature,
        semantics="radians-v1",
        action_interval=caps.action_interval,
        mode="chunk",
    )


def runtime_for(client):
    runtime = ChunkRuntime(
        mode=ExecutionMode.CHUNK,
        action_interval=client.capabilities.action_interval,
        refill_seconds=0.2,
        max_observation_age_s=5,
        action_timeout_s=2,
        startup_timeout_s=3,
    )
    runtime.active = True
    return runtime


def test_direct_remote_actions_match_local_canonical_pipeline_and_reset(remote_server):
    _, config = remote_server
    client = RemoteClient.connect(config)
    try:
        admit(client)
        source = replace(observation(), capture_time=time.monotonic())
        expected = runner_for(ConformingPolicy(tiny_config())).predict(source)
        runtime = runtime_for(client)
        request = runtime.begin(source)
        assert request is not None
        result = client.infer(request)
        torch.testing.assert_close(result.canonical_actions, expected.canonical_actions, rtol=0, atol=0)
        assert result.provenance.observation_id == source.observation_id
        assert result.provenance.session_id == client.session_id
        assert runtime.accept(request, result, task_version=source.task_version)
        assert runtime.pop() is not None

        # A drained ActionQueue has empty (0, A) continuation tensors. Those are
        # encoded as absent, not invalid zero-dimension tensor payloads.
        assert runtime.pop() is not None
        assert runtime.pop() is not None
        drained = runtime.begin(replace(source, capture_time=time.monotonic(), observation_id="drained"))
        assert drained is not None
        assert drained.continuation.model_actions.shape == (0, 3)
        assert runtime.accept(drained, client.infer(drained), task_version=source.task_version)

        generation = runtime.invalidate()
        client.control("reset", generation)
        next_request = runtime.begin(
            replace(source, capture_time=time.monotonic(), observation_id="after-reset")
        )
        assert next_request is not None
        next_result = client.infer(next_request)
        assert next_result.provenance.generation == generation
        assert runtime.accept(next_request, next_result, task_version=source.task_version)
        assert not runtime.accept(request, result, task_version=source.task_version)
    finally:
        client.close()


def test_direct_remote_exclusive_admission_and_new_session_after_close(remote_server):
    _, config = remote_server
    first = RemoteClient.connect(config)
    second = RemoteClient.connect(config)
    try:
        admit(first)
        old_session = first.session_id
        with pytest.raises(ProtocolError) as error:
            admit(second)
        assert error.value.code is ErrorCode.BUSY
        first.close()
        admit(second)
        assert second.session_id != old_session
    finally:
        first.close()
        second.close()


def test_direct_remote_language_roundtrip_preserves_context(remote_server):
    _, config = remote_server
    client = RemoteClient.connect(config)
    try:
        admit(client)
        client.control("invalidate", 1)
        source = replace(observation(), capture_time=time.monotonic())
        answer = client.query_language(
            source,
            kind="vqa",
            text="What is visible?",
            intent_generation=3,
            generation=1,
            cancelled=lambda: False,
        )
        assert answer == "a cube"
        request = runtime_for(client).begin(source)
        assert request is not None
        request = replace(request, generation=1)
        assert client.infer(request).canonical_actions.shape == (3, 3)
    finally:
        client.close()


def test_obsolete_malformed_body_is_discarded_before_tensor_decoding(remote_server):
    _, config = remote_server
    client = RemoteClient.connect(config)
    try:
        admit(client)
        request = runtime_for(client).begin(replace(observation(), capture_time=time.monotonic()))
        assert request is not None
        obsolete = Envelope(
            MessageType.ACTION,
            client.instance_id,
            client.session_id,
            request.generation,
            "completed-request",
            {},
        )
        packed = msgpack.unpackb(encode_message(obsolete), raw=False)
        packed["body"] = {
            "canonical_actions": {
                "__lerobot_type__": "tensor",
                "dtype": "object",
                "shape": [999999999],
                "data": b"",
            }
        }
        assert client._actions._offer(msgpack.packb(packed, use_bin_type=True))
        assert client.infer(request).canonical_actions.shape == (3, 3)
    finally:
        client.close()


@pytest.mark.parametrize("cancel_before_encoding", [True, False])
def test_cancelled_request_is_never_published(remote_server, monkeypatch, cancel_before_encoding):
    _, config = remote_server
    client = RemoteClient.connect(config)
    try:
        admit(client)
        request = runtime_for(client).begin(replace(observation(), capture_time=time.monotonic()))
        assert request is not None
        cancelled = [cancel_before_encoding]

        def encode_then_cancel(envelope):
            result = encode_message(envelope)
            cancelled[0] = True
            return result

        monkeypatch.setattr("lerobot.remote_inference.client.encode_message", encode_then_cancel)
        monkeypatch.setattr(
            client.transport, "publish", lambda *args: pytest.fail("published cancelled inference")
        )
        with pytest.raises(RequestCancelled):
            client.infer(request, cancelled=lambda: cancelled[0])
    finally:
        client.close()


@pytest.mark.parametrize("remote_server", ["robot"], indirect=True)
def test_remote_context_without_weights_dispatches_and_holds_fake_robot(remote_server, monkeypatch, tmp_path):
    _, remote_config = remote_server

    class FakeRobot:
        supports_position_hold = True
        action_features = {f"joint_{index}.pos": float for index in range(3)}
        observation_features = action_features
        name = "fake_position_robot"
        robot_type = name
        is_connected = False
        cameras = {}

        def __init__(self):
            self.sent = []

        def connect(self):
            self.is_connected = True

        def disconnect(self):
            self.is_connected = False

        def get_observation(self):
            return dict.fromkeys(self.action_features, 1.0)

        def send_action(self, command):
            self.sent.append(command.copy())
            return command

    robot = FakeRobot()
    monkeypatch.setattr("lerobot.rollout.remote_context.make_robot_from_config", lambda _: robot)
    monkeypatch.setattr("lerobot.rollout.configs.parser.get_path_arg", lambda _: None)
    monkeypatch.setattr(
        "lerobot.rollout.context._load_pretrained_policy", lambda _: pytest.fail("client loaded weights")
    )
    monkeypatch.setattr(
        "lerobot.rollout.context.get_policy_class", lambda _: pytest.fail("client instantiated a policy")
    )
    cfg = RolloutConfig(robot=SimpleNamespace(), inference=remote_config, task="pick up the cube")
    ctx = build_rollout_context(cfg, Event())
    assert ctx.policy.policy is ctx.policy.preprocessor is ctx.policy.postprocessor is None
    engine = ctx.policy.inference
    event_path = tmp_path / "inference-events.jsonl"
    engine.configure_event_log(event_path)
    ctx.processors.robot_action_processor = lambda pair: {key: value + 0.5 for key, value in pair[0].items()}
    interpolator = ActionInterpolator(multiplier=3)
    engine.reset()
    engine.start()
    engine.resume()
    try:
        deadline = time.monotonic() + 3
        canonical = None
        while canonical is None and time.monotonic() < deadline:
            obs = ctx.hardware.robot_wrapper.get_observation()
            engine.notify_observation(obs)
            canonical = send_next_action(obs, obs, ctx, interpolator)
            if canonical is None:
                Event().wait(0.002)
        assert canonical == {"joint_0.pos": 1.0, "joint_1.pos": 4.0, "joint_2.pos": 7.0}
        assert robot.sent[-1] == {key: value + 0.5 for key, value in canonical.items()}

        # Fetch the second endpoint so interpolation still has intermediate motor
        # ticks when the inference fault arrives.
        obs = ctx.hardware.robot_wrapper.get_observation()
        engine.notify_observation(obs)
        assert send_next_action(obs, obs, ctx, interpolator) is not None
        assert not interpolator.needs_new_action()
        engine._fault("test server failure")
        assert send_next_action(obs, obs, ctx, interpolator) is None
        assert robot.sent[-1] == dict.fromkeys(robot.action_features, 1.0)
        assert ctx.runtime.shutdown_event.is_set()
        assert interpolator.needs_new_action()
    finally:
        engine.stop()
        robot.disconnect()
    events = [json.loads(line) for line in event_path.read_text().splitlines()]
    dispatched = next(event for event in events if event["event"] == "dispatch")
    assert dispatched["canonical"]["joint_0.pos"] == 1.0
    assert dispatched["command"]["joint_0.pos"] == 1.5
    assert dispatched["measured"]["joint_0.pos"] == 1.0
    assert dispatched["provenance"]["task"] == "pick up the cube"
    assert dispatched["provenance"]["request_id"]
    assert any(event["event"] == "fault" for event in events)
    assert events[-1] == {"event": "writer_closed", "dropped_events": 0}
