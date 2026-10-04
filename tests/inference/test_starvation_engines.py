# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Backend worker/hold races around bounded action-starvation recovery."""

import time
from collections.abc import Iterator
from threading import Thread
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("datasets")

from lerobot.inference import RemoteInferenceConfig, RemoteInferenceEngine
from tests.inference.test_local_rtc_regressions import make_engine, wait_for
from tests.remote_inference.test_engine import ControlledClient


@pytest.fixture(params=["local_rtc", "append", "aligned", "blended"])
def backend(request: pytest.FixtureRequest) -> Iterator[SimpleNamespace]:
    """Reuse the existing gated workers; only deadlines use a controlled clock."""
    now = [time.monotonic()]
    local = request.param == "local_rtc"
    if local:
        engine = make_engine(rtc_queue_threshold=0, action_starvation_grace_s=1.0)
        runtime = engine._runtime
        worker = engine._policy
        release = worker.release.release

        def count() -> int:
            return len(worker.calls)

        names = ("j1.pos", "j2.pos")
    else:
        worker = ControlledClient()
        worker.action_release.clear()
        config = RemoteInferenceConfig(
            deployment="test",
            semantics="radians",
            hold_mode="position",
            max_observation_age_s=5,
            refill_seconds=0.1,
            action_starvation_grace_s=1.0,
            chunk_merge="append" if request.param == "append" else "aligned",
            blend_steps=2 if request.param == "blended" else 0,
            blend_components=["a.pos"] if request.param == "blended" else [],
        )
        worker.blend_indices = (0,) if config.blend_steps else ()
        names = ("a.pos", "b.pos")
        engine = RemoteInferenceEngine(
            worker,
            config,
            {"observation.state": {"dtype": "float32", "shape": (2,), "names": list(names)}},
            {},
            SimpleNamespace(observation_time=None),
            "pick",
        )
        runtime = engine.runtime
        release = worker.action_release.set

        def count() -> int:
            return len(worker.requests)

    runtime.clock = lambda: now[0]

    def capture(value: float) -> None:
        now[0] += 0.001
        engine._robot.observation_time = now[0]
        engine.notify_observation(dict.fromkeys(names, value))

    state = SimpleNamespace(
        engine=engine,
        runtime=runtime,
        worker=worker,
        release=release,
        count=count,
        capture=capture,
        now=now,
        local=local,
    )
    engine.start()
    engine.resume()
    try:
        yield state
    finally:
        if local:
            worker.release.release(20)
        else:
            worker.action_release.set()
            worker.text_release.set()
        engine.stop()


def exhaust_with_old_inference_pending(backend: SimpleNamespace) -> None:
    """Execute a real first result, then exhaust its tail while call two is blocked."""
    engine, runtime = backend.engine, backend.runtime
    backend.capture(1.0)
    assert wait_for(lambda: backend.count() == 1)
    backend.release()
    assert wait_for(lambda: not runtime.queue.empty())
    if not backend.local:
        backend.worker.action_release.clear()
    while runtime.queue.qsize() > 1:
        assert engine.get_action(None) is not None
    backend.capture(2.0)
    assert wait_for(lambda: backend.count() == 2)
    assert engine.get_action(None) is not None
    assert runtime.queue.empty()
    previous_generation = runtime.generation
    assert engine.get_action(None) is None
    assert runtime.generation > previous_generation
    assert runtime.starvation_deadline is not None
    assert runtime._starvation_request_deadline is not None
    assert not engine.dispatch_allowed()
    assert not engine.failed


def test_starvation_discards_old_result_and_pre_ack_capture_then_resumes_fresh(
    backend: SimpleNamespace,
) -> None:
    engine, runtime = backend.engine, backend.runtime
    exhaust_with_old_inference_pending(backend)
    deadline = runtime.starvation_deadline
    backend.capture(3.0)  # Captured before the physical hold acknowledgment: unusable.
    engine.acknowledge_hold()
    backend.release()
    assert wait_for(lambda: runtime._starvation_request_deadline is None)
    assert wait_for(lambda: not engine._hold_requested)
    assert backend.count() == 2
    assert runtime.queue.empty()
    assert engine.get_action(None) is None
    assert not engine.dispatch_allowed()
    assert runtime.starvation_deadline == deadline

    # The old call finishing and the hold acknowledgment are insufficient; only
    # a new control-thread capture can condition the replacement trajectory.
    backend.capture(9.0)
    assert wait_for(lambda: backend.count() == 3)
    backend.release()
    assert wait_for(lambda: not runtime.queue.empty())
    assert runtime.starvation_deadline is None
    assert engine.dispatch_allowed()
    assert engine.get_action(None) is not None
    assert not engine.failed
    if backend.local:
        assert torch.equal(backend.worker.action_observations[2], torch.tensor([[9.0, 9.0]]))
    else:
        request = backend.worker.requests[2]
        assert request.generation == runtime.generation
        assert request.observation.features["observation.state"].tolist() == [9.0, 9.0]
    assert runtime.queue.snapshot().provenance[0].generation == runtime.generation


@pytest.mark.parametrize("transition", ["expiry", "reset", "stop"])
def test_starvation_late_result_cannot_restore_motion_after_transition(
    backend: SimpleNamespace, transition: str
) -> None:
    engine, runtime = backend.engine, backend.runtime
    exhaust_with_old_inference_pending(backend)
    engine.acknowledge_hold()
    stopping = None
    if transition == "expiry":
        backend.now[0] = runtime.starvation_deadline + 0.01
        assert not engine.dispatch_allowed()
        assert engine.failed
    elif transition == "reset":
        engine.reset()
    else:
        stopping = Thread(target=engine.stop)
        stopping.start()
        assert wait_for(lambda: not runtime.active)
    backend.release()
    if stopping is not None:
        stopping.join(timeout=2)
        assert not stopping.is_alive()
    elif transition == "expiry":
        thread = engine._rtc_thread if backend.local else engine._thread
        assert wait_for(lambda: not thread.is_alive())
    elif backend.local:
        assert wait_for(lambda: not engine._reset_pending)
    else:
        assert wait_for(lambda: engine._control is None)
    assert runtime.queue.empty()
    assert not engine.dispatch_allowed()
    assert engine.get_action(None) is None
    if transition == "expiry":
        assert "grace expired" in engine.failure_traceback
        engine.reset()
        assert engine.failed
        assert not engine.dispatch_allowed()


@pytest.mark.parametrize("awaiting_capture", [False, True])
def test_language_request_during_starvation_cannot_renew_grace(
    backend: SimpleNamespace, awaiting_capture: bool
) -> None:
    engine, runtime = backend.engine, backend.runtime
    exhaust_with_old_inference_pending(backend)
    deadline = runtime.starvation_deadline
    engine.acknowledge_hold()
    if awaiting_capture:
        backend.release()
        assert wait_for(lambda: not engine._hold_requested)
    backend.now[0] += 0.2
    assert engine.ask("What is visible?")
    for _ in range(3):
        assert not engine.dispatch_allowed()
        engine.acknowledge_hold()
        assert runtime.starvation_deadline == deadline
    backend.now[0] = deadline + 0.01
    assert not engine.dispatch_allowed()
    assert engine.failed
    assert "grace expired" in engine.failure_traceback
    assert runtime.queue.empty()


def test_fresh_recovery_result_arriving_after_grace_cannot_restore_motion(backend: SimpleNamespace) -> None:
    engine, runtime = backend.engine, backend.runtime
    exhaust_with_old_inference_pending(backend)
    engine.acknowledge_hold()
    backend.release()
    assert wait_for(lambda: not engine._hold_requested)
    if not backend.local:
        backend.worker.action_release.clear()
    backend.capture(9.0)
    assert wait_for(lambda: backend.count() == 3)
    backend.now[0] = runtime.starvation_deadline + 0.01
    assert not engine.dispatch_allowed()
    backend.release()
    thread = engine._rtc_thread if backend.local else engine._thread
    assert wait_for(lambda: not thread.is_alive())
    assert engine.failed
    assert "grace expired" in engine.failure_traceback
    assert runtime.queue.empty()
    assert engine.get_action(None) is None
