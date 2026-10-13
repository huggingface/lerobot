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

"""Loopback transports and separately spawned policy servers."""

import multiprocessing
import socket
from threading import Event, Thread

import pytest

from lerobot.inference import ChunkRuntime, ExecutionMode, FeatureSpec, PolicyRunner, RemoteInferenceConfig
from lerobot.remote_inference.server import PolicyServer, SessionWorker
from lerobot.transport.zenoh import ZenohConfig, ZenohTransport
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE
from tests.inference.policy_helpers import (
    ConformingPolicy,
    processors,
    runner_for,
    tiny_config,
)


@pytest.fixture
def remote_server(request):
    pytest.importorskip("zenoh")
    pytest.importorskip("msgpack")
    endpoint = f"tcp/127.0.0.1:{free_port()}"
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
        runner,
        deployment="loopback",
        artifact_identity="processor-test",
        semantics="radians-v1",
    )
    transport = ZenohTransport(ZenohConfig(listen_endpoints=[endpoint]))
    server = PolicyServer(worker, transport)
    ready = Event()
    signal_server_ready(transport, ready)
    thread = Thread(target=server.serve, daemon=True)
    try:
        thread.start()
        assert ready.wait(10), "server did not advertise readiness"
        config = RemoteInferenceConfig(
            endpoint=endpoint,
            deployment="loopback",
            semantics="radians-v1",
            chunk_merge="aligned" if getattr(request, "param", None) == "robot" else "append",
            handshake_timeout_s=2,
            action_timeout_s=2,
        )
        yield worker, config
    finally:
        server.stop()
        if thread.ident is not None:
            thread.join(10)
        assert not thread.is_alive()
        # serve() may fail before entering its resource-management block.
        transport.close()
        worker.close()


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


ACTION_NAMES = ("shoulder.pos", "elbow.pos", "gripper.pos")


class GatedConformingPolicy(ConformingPolicy):
    """Keep the second real inference outstanding while the client commits an endpoint."""

    def __init__(self, config, entered, release):
        super().__init__(config)
        self.calls = 0
        self.entered = entered
        self.release = release

    def predict_action_chunk(self, batch, **kwargs):
        self.calls += 1
        if self.calls == 2:
            self.entered.set()
            assert self.release.wait(5), "client did not release the second policy call"
        return super().predict_action_chunk(batch, **kwargs)


def serve_in_child(endpoint, ready, stopped, entered, release):
    """Build the model and both processor pipelines entirely inside the spawned server."""
    config = tiny_config()
    runner = PolicyRunner(
        GatedConformingPolicy(config, entered, release),
        *processors(config, relative=True),
        action_interval=1 / 30,
        features=tuple(
            FeatureSpec(key, (3,), "float32", names=ACTION_NAMES, semantics="radians-v1")
            for key in (OBS_STATE, OBS_ENV_STATE)
        ),
        action_feature=FeatureSpec(ACTION, (3,), "float32", names=ACTION_NAMES, semantics="radians-v1"),
    )
    worker = SessionWorker(
        runner,
        deployment="aligned-process",
        artifact_identity="relative-conformance",
        semantics="radians-v1",
        action_deadline_s=5,
        language_deadline_s=5,
    )
    transport = ZenohTransport(ZenohConfig(listen_endpoints=[endpoint]))
    server = PolicyServer(worker, transport)
    signal_server_ready(transport, ready)

    def stop_server():
        stopped.wait(30)
        server.stop()

    Thread(target=stop_server, daemon=True).start()
    server.serve()


@pytest.fixture
def process_server():
    pytest.importorskip("zenoh")
    pytest.importorskip("msgpack")
    endpoint = f"tcp/127.0.0.1:{free_port()}"
    # Spawn creates an independent model/runtime and avoids inheriting Zenoh threads.
    context = multiprocessing.get_context("spawn")
    ready, stopped, entered, release = (context.Event() for _ in range(4))
    process = context.Process(target=serve_in_child, args=(endpoint, ready, stopped, entered, release))
    process.start()
    try:
        # Spawn imports torch and the policy stack afresh; allow cold/loaded CI runners.
        assert ready.wait(60), f"server did not start; exitcode={process.exitcode}"
        assert process.is_alive()
        yield process, endpoint, entered, release
    finally:
        # A killed child may hold an Event lock; only touch its events while alive.
        graceful = process.is_alive()
        if graceful:
            release.set()
            stopped.set()
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(2)
        if process.is_alive():
            process.kill()
            process.join(2)
        assert not process.is_alive(), "server process could not be stopped"
        if graceful:
            assert process.exitcode == 0, f"server exited abnormally: {process.exitcode}"
        process.close()


@pytest.fixture
def transports():
    pytest.importorskip("zenoh")
    endpoint = f"tcp/127.0.0.1:{free_port()}"
    with (
        ZenohTransport(ZenohConfig(listen_endpoints=[endpoint], max_payload_bytes=1024)) as server,
        ZenohTransport(ZenohConfig(connect_endpoints=[endpoint], max_payload_bytes=1024)) as client,
    ):
        yield server, client


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def signal_server_ready(transport, ready):
    """Signal only after the server advertises its instance presence."""
    original_token = transport.declare_token

    def declare_token(key):
        token = original_token(key)
        ready.set()
        return token

    transport.declare_token = declare_token
