# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Aligned plain chunks across real, separately spawned client/server processes."""

import multiprocessing
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Thread

import pytest
import torch

pytest.importorskip("zenoh")
pytest.importorskip("msgpack")
pytest.importorskip("datasets")

from lerobot.inference import ChunkRuntime, ExecutionMode, FeatureSpec, PolicyRunner, RemoteInferenceConfig
from lerobot.remote_inference.client import RemoteClient
from lerobot.remote_inference.server import PolicyServer, SessionWorker
from lerobot.transport.zenoh import ZenohConfig, ZenohTransport
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE
from tests.inference.test_policy_runner import ConformingPolicy, observation, processors, tiny_config

ACTION_NAMES = ("shoulder.pos", "elbow.pos", "gripper.pos")
BLEND_COMPONENTS = ACTION_NAMES[:2]


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
    original_token = transport.declare_token

    def declare_token(key):
        token = original_token(key)
        ready.set()
        return token

    transport.declare_token = declare_token

    def stop_server():
        stopped.wait(30)
        server.stop()

    Thread(target=stop_server, daemon=True).start()
    server.serve()


@pytest.fixture
def process_server():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        endpoint = f"tcp/127.0.0.1:{sock.getsockname()[1]}"
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


def test_aligned_relative_chunks_between_processes(process_server):
    _, endpoint, entered, release = process_server
    config = RemoteInferenceConfig(
        endpoint=endpoint,
        deployment="aligned-process",
        semantics="radians-v1",
        blend_steps=2,
        blend_weight=0.25,
        blend_components=list(BLEND_COMPONENTS),
        handshake_timeout_s=3,
        action_timeout_s=5,
    )
    client = RemoteClient.connect(config)
    try:
        caps = client.capabilities
        client.admit(
            features=caps.features,
            action_feature=caps.action_feature,
            semantics=config.semantics,
            action_interval=caps.action_interval,
            mode="chunk",
        )
        assert client.chunk_settings["chunk_merge"] == "aligned"
        assert client.blend_indices == (0, 1)
        runtime = ChunkRuntime(
            mode=ExecutionMode.CHUNK,
            action_interval=caps.action_interval,
            refill_seconds=2 * caps.action_interval,
            max_observation_age_s=10,
            action_timeout_s=5,
            startup_timeout_s=5,
            chunk_merge=config.chunk_merge,
            blend_steps=config.blend_steps,
            blend_weight=config.blend_weight,
            blend_indices=client.blend_indices,
        )
        runtime.active = True
        first_source = runtime.anchor_observation(
            replace(observation(1.0), capture_time=time.monotonic(), observation_id="first")
        )
        first = runtime.begin(first_source)
        assert first is not None
        first_result = client.infer(first)
        # Real normalization and relative-to-absolute conversion happen on the server.
        old_actions = torch.tensor([[2.0, 5.0, 8.0], [8.0, 11.0, 14.0], [14.0, 17.0, 20.0]])
        torch.testing.assert_close(first_result.canonical_actions, old_actions)
        assert first_result.execution_steps == 3  # prediction horizon is eight
        assert runtime.accept(first, first_result, task_version=first_source.task_version)

        torch.testing.assert_close(runtime.pop()[0], old_actions[0])

        # Capture after the gate opens; the next chunk must use this latest state.
        next_source = runtime.anchor_observation(
            replace(observation(10.0), capture_time=time.monotonic(), observation_id="next")
        )
        assert next_source.action_cursor == 1
        request = runtime.begin(next_source)
        assert request is not None
        with ThreadPoolExecutor(max_workers=1) as pool:
            inference = pool.submit(client.infer, request)
            try:
                assert entered.wait(3), "server did not enter the second inference"
                assert not inference.done()
                committed = runtime.pop()
                torch.testing.assert_close(committed[0], old_actions[1])
                assert runtime.current.request_id == first.request_id
            finally:
                release.set()
            incoming = inference.result(5)
        new_actions = torch.tensor([[11.0, 14.0, 17.0], [17.0, 20.0, 23.0], [23.0, 26.0, 29.0]])
        torch.testing.assert_close(incoming.canonical_actions, new_actions)
        assert incoming.provenance.observation_id == next_source.observation_id
        assert runtime.accept(request, incoming, task_version=next_source.task_version)
        assert runtime.last_accept["trimmed_actions"] == 1
        assert runtime.last_accept["blended_steps"] == 1
        assert runtime.queue.qsize() == 2
        # Replacing future motion cannot revise the endpoint already given to interpolation.
        torch.testing.assert_close(committed[0], old_actions[1])
        assert runtime.current.request_id == first.request_id

        action, source = runtime.pop()
        expected = new_actions[1].clone()
        expected[:2] = old_actions[2, :2] * 0.75 + new_actions[1, :2] * 0.25
        assert source.capture_time == first_source.capture_time
        assert source.oldest_contributor.observation_id == first_source.observation_id
        assert source.contributor_count == 2
        torch.testing.assert_close(action, expected)
        assert action[2] == new_actions[1, 2]  # gripper is always the incoming target
        assert source.request_id == request.request_id
        assert source.observation_id == next_source.observation_id
        action, source = runtime.pop()
        torch.testing.assert_close(action, new_actions[2])
        assert source.capture_time == next_source.capture_time
        assert source.contributor_count == 1
        assert runtime.failure is None
    finally:
        release.set()
        client.close()
