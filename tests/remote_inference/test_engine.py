# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Worker/control-thread boundaries independent of network and model latency."""

import logging
import time
from contextlib import contextmanager
from threading import Event, Lock, Thread
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("datasets")

from lerobot.inference import (
    ActionChunk,
    ActionProvenance,
    ExecutionMode,
    FeatureSpec,
    PolicyCapabilities,
    RemoteInferenceConfig,
    RemoteInferenceEngine,
)
from lerobot.remote_inference.client import RequestCancelled
from lerobot.remote_inference.protocol import ErrorCode, ProtocolError


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


@pytest.fixture(params=[("append", 0), ("aligned", 0), ("aligned", 2)], ids=["append", "aligned", "blended"])
def session(request):
    client = ControlledClient()
    config = RemoteInferenceConfig(
        deployment="test",
        semantics="radians",
        hold_mode="position",
        max_observation_age_s=5,
        refill_seconds=0.2,
        chunk_merge=request.param[0],
        blend_steps=request.param[1],
        blend_components=["a.pos"] if request.param[1] else [],
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


def wait_for(condition):
    until = time.monotonic() + 2
    while time.monotonic() < until:
        if condition():
            return True
        Event().wait(0.002)
    return False


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


def test_remote_engine_refuses_restart_while_running_and_after_stop(session):
    engine, client = session
    engine.start()
    worker = engine._thread
    try:
        with pytest.raises(RuntimeError, match="already started"):
            engine.start()
        assert engine._thread is worker
        assert worker.is_alive()
    finally:
        engine.stop()

    assert client.closed.is_set()
    assert not worker.is_alive()
    with pytest.raises(RuntimeError, match="already started"):
        engine.start()


def test_query_waits_for_hold_acknowledgment_and_new_capture(session):
    engine, client = session
    engine.resume()
    capture(engine)
    assert engine.ask("What is visible?")
    engine.start()
    assert not client.control_started.wait(0.02)
    assert not client.text_started.is_set()
    assert not engine.dispatch_allowed()
    engine.acknowledge_hold()
    assert client.control_started.wait(2)
    assert not client.text_started.wait(0.02), "the pre-hold capture cannot condition text"
    capture(engine)
    assert client.text_started.wait(2)
    client.text_release.set()
    assert wait_for(lambda: bool(engine._ready_answers))
    delivered = []
    engine.set_answer_observer(delivered.append)
    assert delivered == []
    engine.pump_query()
    assert delivered[0].answer == "an answer"
    assert not client.action_started.is_set(), "action resumption requires another fresh capture"
    capture(engine)
    assert client.action_started.wait(2)
    assert client.max_active_calls == 1


def test_slow_action_finishes_before_language_begins_and_cannot_restore_motion(session):
    engine, client = session
    client.action_release.clear()
    engine.resume()
    capture(engine)
    engine.start()
    assert client.action_started.wait(2)
    assert engine.ask("What is visible?")
    assert not engine.dispatch_allowed()
    engine.acknowledge_hold()
    capture(engine)
    assert not client.text_started.wait(0.02)
    client.action_release.set()
    assert client.text_started.wait(2)
    assert engine.runtime.queue.empty()
    assert not engine.dispatch_allowed()
    assert client.max_active_calls == 1


@pytest.mark.parametrize("acknowledge_first", [False, True])
@pytest.mark.parametrize("autosteer", [False, True])
def test_instruction_hold_language_upgrade_preserves_ack_control_and_deadline(
    session, acknowledge_first, autosteer
):
    engine, _ = session
    now = [time.monotonic()]
    engine.runtime.clock = lambda: now[0]
    engine.resume()
    engine._robot.observation_time = now[0]
    engine.notify_observation({"a.pos": 0.0, "b.pos": 0.0})
    assert engine.runtime.begin(engine._observation) is not None
    assert engine.set_task("new task")
    generation, control, started = engine.runtime.generation, engine._control, engine._hold_started
    if acknowledge_first:
        engine.acknowledge_hold()
    now[0] += 0.5
    if autosteer:
        engine.start_autosteer("pick", 1)
        engine.pump_query({})
    else:
        assert engine.ask("what?")
    assert engine._hold_reason == "language"
    assert engine._hold_acknowledged == acknowledge_first
    assert engine.runtime.generation == generation
    assert engine._control == control
    assert engine._hold_started == started
    engine.acknowledge_hold()
    started = engine._hold_started
    instruction_budget = engine.config.action_timeout_s + engine.config.handshake_timeout_s
    now[0] = started + instruction_budget + 1
    for _ in range(3):
        assert not engine.dispatch_allowed()
        engine.acknowledge_hold()
        assert engine._hold_started == started
    assert not engine.failed, "language work must receive its longer budget"
    now[0] = started + instruction_budget + engine.config.language_timeout_s + 0.1
    assert not engine.dispatch_allowed()
    assert engine.failed
    assert "Planned language hold deadline exceeded" in engine.failure_traceback


def test_upgraded_language_hold_requires_fresh_query_and_action_observations(session):
    engine, client = session
    client.action_release.clear()
    engine.resume()
    capture(engine)
    engine.start()
    assert client.action_started.wait(2)
    assert engine.set_task("new task")
    generation = engine.runtime.generation
    assert engine.ask("what?")
    client.action_release.set()
    assert not client.control_started.wait(0.02), "generation control must wait for the motor hold"
    engine.acknowledge_hold()
    assert client.control_started.wait(2)
    assert not client.text_started.wait(0.02), "language cannot reuse the pre-hold capture"
    capture(engine)
    assert client.text_started.wait(2)
    assert engine.runtime.generation == generation
    assert engine.runtime.queue.empty(), "old-task inference cannot restore motion"
    assert engine._control is None
    engine.resume()
    assert engine.runtime.held, "idempotent resume must preserve an acknowledged language hold"
    client.text_release.set()
    assert wait_for(lambda: not engine._hold_requested)
    assert len(client.requests) == 1, "actions need a post-query capture"
    client.action_started.clear()
    capture(engine)
    assert client.action_started.wait(2)
    assert client.requests[1].observation.capture_time > engine._query_observation.capture_time
    assert client.requests[1].observation.task == "new task"
    assert wait_for(lambda: engine.runtime.queue.qsize() > 0)
    assert engine.get_action(None) is not None
    assert engine.dispatched_task == "new task"
    assert not engine.failed
    assert client.max_active_calls == 1


def test_vqa_after_autosteering_uses_its_own_query_context(session):
    engine, client = session
    engine.start_autosteer("goal", 10)
    engine.stop_autosteer()
    start_query(engine, client)
    client.text_release.set()
    assert wait_for(lambda: bool(engine._ready_answers))
    assert engine._ready_answers[0].answer == "an answer"


@pytest.mark.parametrize("operation", ["pause", "reset"])
def test_invalidation_reports_an_unclaimed_question_once(session, operation):
    engine, client = session
    delivered = []
    engine.set_answer_observer(delivered.append)
    assert engine.ask("What is visible?")
    getattr(engine, operation)()
    engine.drop_pending_query()
    engine.pump_query()
    engine.pump_query()
    assert not client.text_started.is_set()
    assert not engine.has_pending_query
    assert len(delivered) == 1
    assert delivered[0].answer is None
    assert "cancelled" in delivered[0].error


@pytest.mark.parametrize("operation", ["pause", "reset", "task"])
def test_invalidation_during_text_reports_cancellation_and_prevents_stale_resumption(session, operation):
    engine, client = session
    start_query(engine, client)
    if operation == "task":
        engine.set_task("new instruction")
    else:
        getattr(engine, operation)()
    client.text_release.set()
    assert wait_for(lambda: not engine._query_in_flight)
    delivered = []
    engine.set_answer_observer(delivered.append)
    engine.pump_query()
    engine.pump_query()
    assert len(delivered) == 1
    assert delivered[0].answer is None
    assert "cancelled" in delivered[0].error
    assert engine.task == ("new instruction" if operation == "task" else "initial task")
    assert engine.runtime.queue.empty()
    assert not client.action_started.is_set()
    assert not engine.dispatch_allowed()


def test_cancelled_same_text_autosteer_intent_cannot_apply_old_subtask(session):
    engine, client = session
    engine.resume()
    engine.start_autosteer("goal", 10)
    engine.pump_query({})
    engine.start()
    assert not engine.dispatch_allowed()
    engine.acknowledge_hold()
    capture(engine)
    assert client.text_started.wait(2)
    engine.stop_autosteer()
    engine.start_autosteer("goal", 10)
    client.text_release.set()
    assert wait_for(lambda: not engine._query_in_flight)
    assert not engine._ready_answers
    assert engine.task == "initial task"
    assert engine.autosteer_goal == "goal"


@pytest.mark.parametrize("error_kind", ["execution", "timeout", "presence"])
def test_language_errors_reach_control_thread_and_uncertain_execution_faults(session, error_kind):
    engine, client = session
    terminal = error_kind != "execution"
    client.query_error = {
        "execution": ProtocolError(ErrorCode.EXECUTION, "empty answer"),
        "timeout": TimeoutError("hung language"),
        "presence": ConnectionError("Server presence lost during language query; session cannot recover"),
    }[error_kind]
    start_query(engine, client)
    client.text_release.set()
    assert wait_for(lambda: bool(engine._ready_answers))
    delivered = []
    engine.set_answer_observer(delivered.append)
    engine.pump_query()
    assert len(delivered) == 1, "a terminal query error must not gain a duplicate cancellation notice"
    assert delivered[0].error
    assert engine.failed is terminal
    if terminal:
        with pytest.raises(RuntimeError, match="new rollout"):
            engine.resume()
        assert not engine.dispatch_allowed()
    else:
        capture(engine)
        assert client.action_started.wait(2)


def test_planned_hold_preserves_pending_full_reset(session):
    engine, client = session
    engine.resume()
    engine.reset()
    assert engine.ask("What is visible?")
    generation = engine.runtime.generation
    engine.start()
    engine.acknowledge_hold()
    capture(engine)
    assert client.text_started.wait(2)
    assert client.calls[:2] == [("reset", generation), ("language", generation)]


def test_reset_after_loop_snapshot_cannot_submit_before_control_ack(session):
    engine, client = session
    engine.resume()
    capture(engine)

    def reset_during_presence_check():
        engine.reset()
        capture(engine)

    client.on_presence = reset_during_presence_check
    engine.start()
    assert client.control_started.wait(2)
    assert wait_for(lambda: engine._control is None)
    capture(engine)
    assert client.action_started.wait(2)
    assert client.calls[0] == ("reset", 1)
    assert client.calls[1] == ("action", 1)


def test_request_binds_latest_task_without_refreshing_capture_time(session):
    engine, client = session
    engine.resume()
    capture(engine)
    original_capture = engine._observation.capture_time
    engine.set_task("updated task")
    engine.start()
    assert client.action_started.wait(2)
    assert client.requests[0].observation.task == "updated task"
    assert client.requests[0].observation.task_version == 1
    assert client.requests[0].observation.capture_time == original_capture


def test_retarget_and_result_acceptance_have_one_order(session, monkeypatch):
    """A task change cannot slip between reading its version and merging a result."""
    engine, client = session
    accepting, release, changed = Event(), Event(), Event()
    original_accept = engine.runtime.accept

    def gated_accept(*args, **kwargs):
        accepting.set()
        assert release.wait(2)
        return original_accept(*args, **kwargs)

    monkeypatch.setattr(engine.runtime, "accept", gated_accept)
    engine.resume()
    capture(engine)
    engine.start()
    assert accepting.wait(2)

    def retarget():
        engine.set_task("new task")
        changed.set()

    thread = Thread(target=retarget)
    thread.start()
    try:
        assert not changed.wait(0.03), "retarget interleaved with result acceptance"
    finally:
        release.set()
        thread.join(2)
    assert changed.is_set()
    assert not engine.failed
    # The result was accepted before the task change. Its original label remains
    # valid for buffered continuity; subsequent requests use the new instruction.
    assert engine.runtime.queue.snapshot().provenance[0].task == "initial task"


def test_refill_wait_selects_latest_capture_and_reports_request_progress(session, caplog):
    engine, client = session
    caplog.set_level(logging.DEBUG, logger="lerobot.inference.remote")
    engine.resume()
    capture(engine)
    engine.start()
    assert wait_for(lambda: engine.runtime.queue.qsize() == 4)
    assert engine.runtime.pop() is not None
    capture(engine)
    waiting_source = engine._observation
    client.action_started.clear()
    assert not client.action_started.wait(0.03), "a fresh advanced sample cannot bypass playback need"
    # Publish another observation while gated; selecting a request must use this
    # newer capture, with its original anchor even if consumption follows capture.
    capture(engine)
    latest = engine._observation
    assert latest.capture_time > waiting_source.capture_time
    client.action_release.clear()
    assert engine.runtime.pop() is not None  # two endpoints left, exactly at refill
    assert client.action_started.wait(2)
    request = client.requests[1]
    assert request.observation.capture_time == latest.capture_time
    assert request.observation.observation_id == latest.observation_id
    assert request.playback_at_submission == pytest.approx(0.2)
    assert request.continuation.cursor == 2
    if engine.config.chunk_merge == "aligned":
        assert request.observation.action_cursor == 1
    events = [record.args for record in caplog.records if record.msg == "Remote inference %s"]
    assert events[0]["event"] == "scheduling"
    requests = [event for event in events if event["event"] == "request"]
    assert requests[-1]["scheduling"] == "playback_threshold"
    assert requests[-1]["committed_actions_since_request"] == 2
    assert requests[-1]["request_spacing_s"] > 0
    assert not requests[-1]["task_changed_since_request"]


def test_only_aligned_retarget_bypasses_refill_without_reusing_capture(session):
    engine, client = session
    engine.resume()
    capture(engine)
    engine.start()
    assert wait_for(lambda: engine.runtime.queue.qsize() == 4)
    client.action_started.clear()
    engine.set_task("new goal")
    assert not client.action_started.wait(0.03), "retarget must still wait for a fresh capture"
    assert engine.runtime.queue.qsize() * engine.runtime.interval > engine.runtime.effective_refill
    capture(engine)
    if engine.config.chunk_merge == "append":
        assert not client.action_started.wait(0.03), "append retarget retains the playback gate"
        return
    assert client.action_started.wait(2)
    request = client.requests[1]
    assert request.observation.task_version == 1
    assert request.observation.action_cursor == 0
    assert request.playback_at_submission == pytest.approx(0.4)
    # Acceptance clears pending before replacing the queue. Wait for the
    # observable replacement, not an unsynchronized intermediate runtime field.
    assert wait_for(lambda: engine.runtime.queue.snapshot().provenance[0].request_id == request.request_id)
    assert [source.task for source in engine.runtime.queue.snapshot().provenance] == ["new goal"] * 4


@pytest.mark.parametrize("measured", [False, True], ids=["configured", "latency-floor"])
def test_aligned_full_horizon_warning_is_bounded_and_does_not_change_settings(session, caplog, measured):
    engine, _ = session
    caplog.set_level(logging.WARNING, logger="lerobot.inference.remote")
    if measured:
        engine.start()
        assert not caplog.records
        engine.runtime.turnarounds.append(0.3)
    else:
        engine.runtime.refill_seconds = 0.4
        engine.start()
    before = engine.runtime.refill_seconds
    engine._warn_refill_horizon()
    engine._warn_refill_horizon()
    warnings = [record for record in caplog.records if "full execution horizon" in record.message]
    assert len(warnings) == (1 if engine.config.chunk_merge == "aligned" else 0)
    assert engine.runtime.refill_seconds == before
    assert engine.runtime.effective_refill == pytest.approx(0.4)


def test_default_logs_are_concise_and_debug_keeps_request_identity(session, caplog):
    engine, client = session
    caplog.set_level(logging.INFO, logger="lerobot.inference.remote")
    engine.resume()
    capture(engine)
    engine.start()
    assert wait_for(lambda: engine.runtime.queue.qsize() == 4)
    assert "Remote execution: mode=chunk" in caplog.text
    assert "horizon=0.400s" in caplog.text
    assert "Remote inference {" not in caplog.text
    caplog.set_level(logging.DEBUG, logger="lerobot.inference.remote")
    client.action_started.clear()
    assert engine.runtime.pop() is not None
    assert engine.runtime.pop() is not None
    capture(engine)
    assert client.action_started.wait(2)
    records = [record for record in caplog.records if record.msg == "Remote inference %s"]
    request = next(record.args for record in records if record.args["event"] == "request")
    assert request["deployment"] == "test"
    assert request["session"] == "session"
    assert request["request_id"] == client.requests[-1].request_id


def test_progress_is_rate_limited_and_labels_estimated_headroom(session, caplog):
    engine, _ = session
    caplog.set_level(logging.INFO, logger="lerobot.inference.remote")
    engine._event("result", turnaround_s=0.3, timing_margin_s=None, accepted=True)
    engine._event("result", turnaround_s=0.2, timing_margin_s=-0.1, accepted=False)
    engine._drain_log_events()
    assert "Remote progress" not in caplog.text
    engine._last_summary_at -= 5
    engine._drain_log_events()
    assert "accepted=1 rejected=1" in caplog.text
    assert "mean/max=0.250/0.300s" in caplog.text
    assert "estimated submission headroom min=-0.100s" in caplog.text
    engine._drain_log_events()
    assert caplog.text.count("Remote progress") == 1


def test_fault_diagnostics_are_bounded_and_do_not_log_on_control_thread(session, caplog):
    engine, _ = session
    caplog.set_level(logging.INFO, logger="lerobot.inference.remote")
    # A full diagnostics handoff must not hide a fault or block its producer.
    for _ in range(130):
        engine._event("first_dispatch")
    engine._fault("Active motion buffer exhausted")
    assert engine.failed
    assert not caplog.records
    assert engine._log_events.qsize() == 128
    engine._drain_log_events()
    assert "Active motion buffer exhausted" in caplog.text
    assert "configured shutdown/return procedure" in caplog.text
    assert "Last submission playback=unavailable" in caplog.text
    assert "turnaround max=unavailable" in caplog.text
    assert "usable suffix after trimming" in caplog.text
    engine._drain_log_events()
    assert caplog.text.count("Remote inference stopped") == 1


def test_slow_console_does_not_block_local_motion_permission(session, monkeypatch):
    engine, client = session
    entered, release, control_completed = Event(), Event(), Event()

    def slow_console(*args, **kwargs):
        entered.set()
        assert release.wait(2)

    monkeypatch.setattr("lerobot.inference.remote.logger.info", slow_console)
    engine.resume()
    capture(engine)
    engine.start()
    try:
        assert entered.wait(2)

        def control_tick():
            engine.dispatch_allowed()
            engine.runtime.pop()
            control_completed.set()

        control = Thread(target=control_tick)
        control.start()
        assert control_completed.wait(0.2), "console output acquired a motor-thread lock"
        control.join(2)
    finally:
        release.set()
    assert client.action_started.wait(2)
