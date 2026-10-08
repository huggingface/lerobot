"""Operator transitions and control cancellation do not consume motion startup time."""

from threading import Event

import pytest

pytest.importorskip("datasets")
pytest.importorskip("msgpack")

from tests.remote_inference import test_engine as helpers  # noqa: E402

session = helpers.session
capture = helpers.capture
wait_for = helpers.wait_for


def test_inflight_retarget_holds_and_resumes_only_from_fresh_capture(session):
    engine, client = session
    engine.resume()
    capture(engine)
    engine.start()
    assert wait_for(lambda: engine.runtime.queue.qsize() == 4)
    engine.get_action(None)
    engine.get_action(None)
    client.action_release.clear()
    client.action_started.clear()
    capture(engine)
    assert client.action_started.wait(2)
    obsolete = client.requests[-1]
    assert engine.set_task("new task")
    assert engine.runtime.held
    assert engine.runtime.queue.empty()
    assert not engine.dispatch_allowed()
    engine.acknowledge_hold()
    capture(engine)
    client.action_release.set()
    assert wait_for(lambda: not engine.runtime.held)
    assert not engine.failed
    assert engine.runtime.queue.empty()
    assert engine.runtime.generation != obsolete.generation
    capture(engine)
    assert wait_for(lambda: not engine.runtime.queue.empty())
    assert engine.get_action(None) is not None
    assert engine.dispatched_task == "new task"
    assert client.max_active_calls == 1


def test_control_ack_has_its_own_bound_before_fresh_startup_budget(session, monkeypatch):
    engine, client = session
    entered, release = Event(), Event()
    now = [100.0]
    engine.runtime.clock = lambda: now[0]

    def control(operation, generation, *, cancelled=None):
        entered.set()
        while not release.wait(0.002):
            if cancelled and cancelled():
                return

    monkeypatch.setattr(client, "control", control)
    engine.reset()
    engine.resume()
    engine.start()
    try:
        assert entered.wait(2)
        now[0] += engine.config.startup_timeout_s + 1
        assert not engine.dispatch_allowed()
        assert not engine.failed
        release.set()
        assert wait_for(lambda: engine._control is None)
        assert engine.runtime.started_at == now[0]
        assert not engine.failed
    finally:
        release.set()


def test_stop_cancels_outstanding_control_wait(session, monkeypatch):
    engine, client = session
    entered = Event()

    def control(operation, generation, *, cancelled=None):
        entered.set()
        assert wait_for(cancelled)

    monkeypatch.setattr(client, "control", control)
    engine.reset()
    engine.resume()
    engine.start()
    assert entered.wait(2)
    engine.stop()
    assert not engine._thread.is_alive()
    assert client.closed.is_set()


def test_pause_invalidates_inflight_request_before_resuming_acknowledged_generation(session, monkeypatch):
    engine, client = session
    control_entered, acknowledge = Event(), Event()

    def control(operation, generation, *, cancelled=None):
        with client.call(operation, generation):
            control_entered.set()
            assert acknowledge.wait(2), "test must acknowledge the generation transition"

    monkeypatch.setattr(client, "control", control)
    client.action_release.clear()
    engine.resume()
    capture(engine)
    engine.start()
    try:
        assert client.action_started.wait(2)
        old_request = client.requests[0]
        engine.pause()
        generation = engine.runtime.generation
        assert generation != old_request.generation
        client.action_release.set()
        assert control_entered.wait(2)
        assert client.calls[-1] == ("invalidate", generation)
        assert engine.runtime.pending is None
        assert engine.runtime.queue.empty()
        assert not engine.failed

        # Even an immediate resume/capture must wait for the server's ACK.
        engine.resume()
        capture(engine)
        client.action_started.clear()
        assert not client.action_started.wait(0.02)
        assert len(client.requests) == 1
        assert not engine.dispatch_allowed()
        acknowledge.set()
        assert wait_for(lambda: engine._control is None)
        assert engine._observation is None, "the pre-ACK capture cannot condition resumed motion"
        capture(engine)
        assert wait_for(lambda: len(client.requests) == 2)
        assert client.requests[-1].generation == generation
        assert wait_for(lambda: not engine.runtime.queue.empty())
        assert client.calls[:3] == [
            ("action", old_request.generation),
            ("invalidate", generation),
            ("action", generation),
        ]
        assert not engine.failed
    finally:
        acknowledge.set()


def test_planned_hold_deadline_faults_and_cannot_release_motion(session):
    engine, _ = session
    shutdown = Event()
    engine._global_shutdown = shutdown
    engine.resume()
    assert engine.ask("A query that cannot start")
    engine._hold_started -= (
        engine.config.language_timeout_s
        + engine.config.action_timeout_s
        + engine.config.handshake_timeout_s
        + 1
    )
    assert not engine.dispatch_allowed()
    assert engine.failed
    assert "hold deadline" in engine.failure_traceback
    assert not shutdown.is_set()
    engine.acknowledge_hold()
    assert shutdown.is_set()
    assert not engine.runtime.release_hold(engine.runtime.generation)
