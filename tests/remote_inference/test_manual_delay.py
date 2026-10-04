# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Prove the physical H2 delay procedure hits starvation while server IO remains live."""

import time
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import pytest

pytest.importorskip("zenoh")
pytest.importorskip("datasets")
pytest.importorskip("msgpack")

from lerobot.inference import RemoteInferenceEngine
from lerobot.remote_inference import RemoteClient
from tests.manual import delayed_policy_server
from tests.remote_inference.test_remote_path import admit, remote_server as _remote_server
from tests.remote_inference.test_transport_cancellation import wait_for

remote_server = _remote_server


@pytest.mark.parametrize("remote_server", ["robot"], indirect=True)
@pytest.mark.parametrize("expires", [False, True], ids=["recovers", "expires"])
def test_one_shot_worker_delay_preserves_presence_and_enforces_grace(
    remote_server, expires, caplog, monkeypatch
):
    worker, config = remote_server
    entered, release = Event(), Event()
    delays = []

    def gated_sleep(seconds):
        delays.append(seconds)
        entered.set()
        assert release.wait(10), "Test did not release the delayed worker"

    monkeypatch.setattr(delayed_policy_server, "time", SimpleNamespace(sleep=gated_sleep))
    delayed_policy_server.delay_action_once(worker, delay_s=0.35, on_action=2)
    worker.action_deadline_s = 15
    config = replace(
        config,
        refill_seconds=0.04,
        action_timeout_s=15,
        handshake_timeout_s=10,
        max_observation_age_s=30,
        action_starvation_grace_s=5,
    )
    client = RemoteClient.connect(config)
    admit(client)
    names = client.capabilities.action_feature.names
    robot = SimpleNamespace(observation_time=None)
    engine = RemoteInferenceEngine(
        client,
        config,
        {"observation.state": {"dtype": "float32", "shape": (3,), "names": list(names)}},
        {},
        robot,
        "pick up the cube",
    )
    now = [time.monotonic()]
    engine.runtime.clock = lambda: now[0]

    def capture():
        now[0] += 0.001
        robot.observation_time = now[0]
        engine.notify_observation(dict.fromkeys(names, 1.0))

    engine.start()
    engine.resume()
    try:
        capture()
        assert wait_for(lambda: not engine.runtime.queue.empty(), timeout=10)
        while engine.runtime.queue.qsize() > 1:
            assert engine.get_action(None) is not None
        capture()
        assert entered.wait(10)
        assert engine.get_action(None) is not None
        assert engine.get_action(None) is None
        held_at = now[0]
        assert engine.runtime.starvation_deadline is not None
        assert "MANUAL TEST: delaying action 2" in caplog.text
        assert client.present
        assert worker.descriptor["ready"]
        engine.acknowledge_hold()
        if expires:
            now[0] = engine.runtime.starvation_deadline + 0.01
            assert not engine.dispatch_allowed()
            assert engine.failed
            assert "grace expired" in engine.failure_traceback
        else:
            release.set()
            assert wait_for(lambda: not engine._hold_requested, timeout=10)
            capture()
            assert wait_for(lambda: not engine.runtime.queue.empty(), timeout=10)
            assert engine.get_action(None) is not None
            assert not engine.failed, engine.failure_traceback
            assert engine.runtime.current.capture_time > held_at
        assert delays == [0.35]
    finally:
        release.set()
        engine.stop()
