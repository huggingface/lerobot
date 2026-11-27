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

"""Controlled remote clients and protocol requests without network timing."""

import time
from contextlib import contextmanager
from dataclasses import asdict, replace
from threading import Event, Lock
from types import SimpleNamespace
from uuid import uuid4

import pytest
import torch

from lerobot.inference import (
    ActionChunk,
    ActionProvenance,
    ExecutionMode,
    FeatureSpec,
    PolicyCapabilities,
    RemoteInferenceConfig,
)
from lerobot.remote_inference import RemoteInferenceEngine
from lerobot.remote_inference.client import RemoteClient, RequestCancelled
from lerobot.remote_inference.protocol import Envelope, MessageType
from lerobot.remote_inference.server import SessionWorker
from tests.inference.policy_helpers import ConformingPolicy, observation, runner_for, tiny_config


class ControlledClient:
    """Explicit completion gates expose races without a model or network sleep."""

    def __init__(self):
        feature = FeatureSpec(
            "observation.state", (2,), "float32", names=("a.pos", "b.pos"), semantics="radians"
        )
        action = FeatureSpec("action", (2,), "float32", names=("a.pos", "b.pos"), semantics="radians")
        self.capabilities = PolicyCapabilities(
            (ExecutionMode.CHUNK,), 4, 4, 0.1, (feature,), action, language=True
        )
        self.descriptor = {"limits": {"max_input_chars": 4096}}
        self.blend_indices = ()
        self.instance_id = "instance"
        self.session_id = "session"
        self.action_started, self.text_started, self.control_started = Event(), Event(), Event()
        self.action_release, self.text_release = Event(), Event()
        self.action_release.set()
        self.closed = Event()
        self.calls = []
        self.requests = []
        self.query_error = None
        self.text = "an answer"
        self.on_presence = None
        self._lock = Lock()
        self.active_calls = 0
        self.max_active_calls = 0

    @property
    def present(self):
        if self.on_presence is not None:
            callback, self.on_presence = self.on_presence, None
            callback()
        return True

    @contextmanager
    def call(self, name, context):
        with self._lock:
            self.calls.append((name, context))
            self.active_calls += 1
            self.max_active_calls = max(self.active_calls, self.max_active_calls)
        try:
            yield
        finally:
            with self._lock:
                self.active_calls -= 1

    def control(self, operation, generation, *, cancelled=None):
        with self.call(operation, generation):
            self.control_started.set()

    def infer(self, request, *, cancelled):
        with self.call("action", request.generation):
            self.requests.append(request)
            self.action_started.set()
            assert self.action_release.wait(2), "test did not complete the action call"
            if cancelled():
                raise RequestCancelled()
            actions = torch.ones(4, 2)
            return ActionChunk(
                None, actions, ActionProvenance(request.observation.capture_time, request.observation.task), 4
            )

    def query_language(self, observation, *, kind, text, intent_generation, generation, cancelled):
        with self.call("language", generation):
            self.text_started.set()
            assert self.text_release.wait(2), "test did not complete the language call"
            if self.query_error is not None:
                raise self.query_error
            # Deliberately return after cancellation, exercising the engine's
            # context check independently of cooperative transport cancellation.
            return self.text

    def close(self):
        self.closed.set()


@pytest.fixture
def session(request):
    client = ControlledClient()
    # Mode-independent lifecycle checks use the default; scheduling checks choose their variants.
    merge, blend_steps = getattr(request, "param", ("aligned", 0))
    config = RemoteInferenceConfig(
        deployment="test",
        semantics="radians",
        max_observation_age_s=5,
        refill_seconds=0.2,
        chunk_merge=merge,
        blend_steps=blend_steps,
        blend_components=["a.pos"] if blend_steps else [],
    )
    client.blend_indices = tuple(
        client.capabilities.action_feature.names.index(name) for name in config.blend_components
    )
    wrapper = SimpleNamespace(observation_time=None)
    engine = RemoteInferenceEngine(
        client,
        config,
        {"observation.state": {"dtype": "float32", "shape": (2,), "names": ["a.pos", "b.pos"]}},
        {},
        wrapper,
        "initial task",
    )
    try:
        yield engine, client
    finally:
        client.action_release.set()
        client.text_release.set()
        engine.stop()
        assert engine._thread is None or not engine._thread.is_alive()


def capture(engine):
    engine._robot.observation_time = time.monotonic()
    engine.notify_observation({"a.pos": 0.0, "b.pos": 0.0})


def start_query(engine, client):
    engine.resume()
    assert engine.ask("What is visible?")
    engine.start()
    assert not engine.dispatch_allowed()
    engine.acknowledge_hold()
    capture(engine)
    assert client.text_started.wait(2)


def open_request(worker, request_id=None):
    caps = worker.runner.capabilities
    return Envelope(
        MessageType.OPEN,
        worker.instance_id,
        request_id=request_id or uuid4().hex,
        body={
            "expected_artifact": worker.artifact_identity,
            "semantics": worker.semantics,
            "features": [asdict(feature) for feature in caps.features],
            "action_feature": asdict(caps.action_feature),
            "mode": "chunk",
            "action_interval": caps.action_interval,
            "encoding": "raw",
        },
    )


def admit_session(worker):
    accepted = worker.submit(open_request(worker)).result(2)
    assert accepted.message_type is MessageType.ACCEPTED
    return accepted.session_id


def action_request(worker, session, generation=0, request_id=None):
    source = observation()
    sequence = getattr(worker, "_test_sequence", 0) + 1
    worker._test_sequence = sequence
    return Envelope(
        MessageType.OBSERVATION,
        worker.instance_id,
        session,
        generation,
        request_id or uuid4().hex,
        {
            "artifact_identity": worker.artifact_identity,
            "features": dict(source.features),
            "task": source.task,
            "task_version": source.task_version,
            "observation_id": source.observation_id,
            "capture_time": source.capture_time,
            "delay": 0,
            "sequence": sequence,
        },
    )


def control_request(worker, session, generation, operation):
    return Envelope(
        MessageType.CONTROL, worker.instance_id, session, generation, uuid4().hex, {"operation": operation}
    )


def block_predict(worker):
    entered, release = Event(), Event()
    original = worker.runner.policy.predict_action_chunk

    def predict(batch, **kwargs):
        entered.set()
        assert release.wait(5), "test did not release the policy worker"
        return original(batch, **kwargs)

    worker.runner.policy.predict_action_chunk = predict
    return entered, release


def assert_error(response, code):
    assert response.message_type is MessageType.ERROR
    assert response.body["code"] == code


def client_config(**kwargs):
    return RemoteInferenceConfig(deployment="test", semantics="test-radians", **kwargs)


@pytest.fixture
def chunk_worker():
    runner = runner_for(ConformingPolicy(tiny_config()))
    runner.capabilities = replace(
        runner.capabilities,
        action_feature=replace(
            runner.capabilities.action_feature, names=("shoulder.pos", "elbow.pos", "gripper.pos")
        ),
    )
    worker = SessionWorker(
        runner,
        deployment="test",
        artifact_identity="artifact",
        semantics="test-radians",
    )
    yield worker
    worker.close()


def client_for(worker, config, descriptor=None):
    transport = SimpleNamespace(
        subscribe_liveliness=lambda *_: SimpleNamespace(),
        subscribe=lambda *args, **kwargs: SimpleNamespace(),
        declare_token=lambda *_: SimpleNamespace(),
        wait_for_subscriber=lambda *_: None,
    )
    return RemoteClient(transport, config, worker.descriptor if descriptor is None else descriptor)


def admit(client):
    caps = client.capabilities
    client.admit(
        features=caps.features,
        action_feature=caps.action_feature,
        semantics=client.config.semantics,
        action_interval=caps.action_interval,
        mode=client.config.mode,
    )
