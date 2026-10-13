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

"""Worker/control-thread boundaries independent of network and model latency."""

import time
from threading import Event, Thread

import numpy as np
import pytest

from lerobot.remote_inference.protocol import ErrorCode, ProtocolError
from tests.remote_inference.helpers import capture, start_query
from tests.utils import wait_for


@pytest.mark.parametrize("session", [("append", 0), ("aligned", 2)], indirect=True)
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
    assert wait_for(lambda: not engine._hold.requested)
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


@pytest.mark.parametrize("session", [("append", 0), ("aligned", 0)], indirect=True)
def test_refill_wait_selects_latest_capture(session):
    engine, client = session
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


@pytest.mark.parametrize("session", [("append", 0), ("aligned", 0)], indirect=True)
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


def test_slow_console_does_not_block_local_motion_permission(session, monkeypatch):
    engine, client = session
    entered, release, control_completed = Event(), Event(), Event()

    def slow_console(*args, **kwargs):
        entered.set()
        assert release.wait(2)

    monkeypatch.setattr("lerobot.remote_inference.engine.logger.info", slow_console)
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


@pytest.mark.parametrize("session", [("aligned", 0), ("append", 0)], indirect=True)
def test_submit_observation_publishes_canonical_features_with_the_current_instruction(session):
    """Control loops outside rollout publish checkpoint-named arrays directly."""
    engine, _ = session
    engine.submit_observation({"observation.state": np.zeros(2, dtype=np.float32)}, 12.5)
    snapshot = engine._observation
    assert (snapshot.task, snapshot.task_version, snapshot.capture_time) == ("initial task", 0, 12.5)
    assert snapshot.action_cursor == (0 if engine.config.chunk_merge == "aligned" else None)
    assert engine.set_task("new task")
    before = time.monotonic()
    engine.submit_observation({"observation.state": np.ones(2, dtype=np.float32)})
    snapshot = engine._observation
    assert (snapshot.task, snapshot.task_version) == ("new task", 1)
    assert snapshot.capture_time >= before
