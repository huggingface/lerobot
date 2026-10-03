# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Actual loopback cancellation and abrupt server-process loss, without hardware."""

from __future__ import annotations

import multiprocessing
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Empty
from threading import Event
from types import SimpleNamespace

import pytest

pytest.importorskip("zenoh")
pytest.importorskip("msgpack")
pytest.importorskip("datasets")

from lerobot.remote_inference.client import RemoteClient, RequestCancelled
from lerobot.remote_inference.protocol import MessageType
from lerobot.rollout.inference.factory import RemoteInferenceConfig
from lerobot.rollout.inference.remote import RemoteInferenceEngine
from lerobot.transport.zenoh import QueryCancelled
from tests.remote_inference.test_aligned_process import ACTION_NAMES, serve_in_child
from tests.remote_inference.test_remote_path import admit, remote_server as _remote_server
from tests.remote_inference.test_zenoh import transports as _transports

remote_server = _remote_server
transports = _transports


def wait_for(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        Event().wait(0.005)
    return False


def test_cancelled_matching_query_is_not_published(transports):
    server, client = transports
    incoming = server.declare_queryable("test/cancelled-control")
    # Establish routing with a completed round trip first: cancellation must be
    # checked even when no matching-status wait is needed.
    with ThreadPoolExecutor() as pool:
        probe = pool.submit(client.query, "test/cancelled-control", b"probe", 2)
        assert incoming.get(2).reply(b"ready")
        assert probe.result(2) == [b"ready"]
    with pytest.raises(QueryCancelled):
        client.query("test/cancelled-control", b"must-not-execute", 2, cancelled=lambda: True)
    with pytest.raises(Empty):
        incoming.get(0.1)


def test_cancelled_control_wait_returns_while_server_operation_still_runs(remote_server, monkeypatch):
    worker, config = remote_server
    client = RemoteClient.connect(config)
    entered, release = Event(), Event()
    original_execute = worker._execute
    calls = []

    def execute(message):
        if (
            message.message_type is MessageType.CONTROL
            and message.body["operation"] == "reset"
            and message.generation == 1
        ):
            calls.append(message.request_id)
            entered.set()
            assert release.wait(5), "test must release server-side control work"
        return original_execute(message)

    monkeypatch.setattr(worker, "_execute", execute)
    cancelled = Event()
    try:
        admit(client)
        baseline_resets = worker.runner.policy.resets
        with ThreadPoolExecutor() as pool:
            future = pool.submit(client.control, "reset", 1, cancelled=cancelled.is_set)
            assert entered.wait(3)
            assert not future.done()
            cancelled.set()
            with pytest.raises(RequestCancelled):
                future.result(1)
            assert client.generation == 0, "cancelling a wait cannot acknowledge remote completion"
            assert worker.runner.policy.resets == baseline_resets
            release.set()
        assert wait_for(lambda: worker.runner.policy.resets == baseline_resets + 1)
        assert len(calls) == 1, "cancellation must not replay the stateful reset"
        # An explicit later operation can acknowledge a newer generation. The
        # cancelled operation may complete, but its late reply cannot acknowledge this one.
        client.control("invalidate", 2)
        assert client.generation == 2
        assert len(calls) == 1
    finally:
        cancelled.set()
        release.set()
        client.close()


@pytest.fixture
def crashable_server():
    with socket.socket() as port:
        port.bind(("127.0.0.1", 0))
        endpoint = f"tcp/127.0.0.1:{port.getsockname()[1]}"
    context = multiprocessing.get_context("spawn")
    ready, stopped, entered, release = (context.Event() for _ in range(4))
    process = context.Process(target=serve_in_child, args=(endpoint, ready, stopped, entered, release))
    process.start()
    try:
        assert ready.wait(15), f"server did not start; exitcode={process.exitcode}"
        yield process, endpoint, entered, release
    finally:
        # A killed process may have died while owning an Event's condition lock.
        # Do not touch those shared primitives after abrupt process death.
        if process.is_alive():
            release.set()
            stopped.set()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(3)
        assert not process.is_alive()
        process.close()


def test_server_process_death_latches_presence_exhausts_motion_and_skips_close_ack(
    crashable_server, monkeypatch
):
    process, endpoint, entered, _release = crashable_server
    config = RemoteInferenceConfig(
        endpoint=endpoint,
        deployment="aligned-process",
        semantics="radians-v1",
        hold_mode="position",
        refill_seconds=0.2,
        max_observation_age_s=10,
        handshake_timeout_s=3,
        action_timeout_s=5,
    )
    client = RemoteClient.connect(config)
    shutdown = Event()
    engine = None
    try:
        admit(client)
        engine = RemoteInferenceEngine(
            client=client,
            config=config,
            dataset_features={
                feature.name: {"dtype": feature.dtype, "shape": feature.shape, "names": feature.names}
                for feature in client.capabilities.features
            },
            rename_map={},
            robot_wrapper=SimpleNamespace(observation_time=None, supports_hold=True),
            task="pick",
            shutdown_event=shutdown,
        )
        engine.start()
        engine.resume()
        engine.notify_observation(dict.fromkeys(ACTION_NAMES, 1.0))
        assert wait_for(lambda: engine.runtime.queue.qsize() == 3)
        assert entered.wait(3), "the next real inference should be outstanding on the server"
        assert client.present
        process.kill()
        process.join(3)
        assert not process.is_alive()
        assert wait_for(lambda: not client.present), "actual Zenoh liveliness must detect process death"

        close_attempts = []
        original_control = client.control

        def control(operation, generation, **kwargs):
            close_attempts.append(operation)
            return original_control(operation, generation, **kwargs)

        monkeypatch.setattr(client, "control", control)
        for _ in range(3):
            assert engine.dispatch_allowed()
            assert engine.get_action(None) is not None
        assert engine.get_action(None) is None
        assert engine.failed
        assert "Active motion buffer exhausted" in engine.failure_traceback
        assert not engine.dispatch_allowed()
        assert not shutdown.is_set()
        engine.acknowledge_hold()
        assert shutdown.is_set()
        assert wait_for(lambda: client._closed)
        assert close_attempts == [], "a dead server must not delay shutdown with a control acknowledgement"
        assert not client.present
        with pytest.raises(RuntimeError, match="new rollout"):
            engine.resume()
    finally:
        if engine is not None:
            engine.stop()
        else:
            client.close()
