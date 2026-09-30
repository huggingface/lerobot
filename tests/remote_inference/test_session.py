# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Exclusive admission and ordered policy-worker lifecycle, without network timing."""

from dataclasses import asdict
from threading import Event
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from lerobot.remote_inference.protocol import Envelope, ErrorCode, MessageType
from lerobot.remote_inference.server import SessionWorker
from tests.inference.test_policy_runner import ConformingPolicy, observation, runner_for, tiny_config


@pytest.fixture
def worker():
    instance = SessionWorker(
        runner_for(ConformingPolicy(tiny_config())),
        deployment="test",
        artifact_identity="artifact",
        semantics="test-radians",
        action_deadline_s=5,
        language_deadline_s=5,
    )
    yield instance
    instance.close()


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


def admit(worker):
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


def test_unknown_required_capability_rejects_before_session_admission(worker):
    request = open_request(worker)
    request.body["required_capabilities"] = ["future-observation-history"]
    assert_error(worker.submit(request).result(2), ErrorCode.UNSUPPORTED)
    assert worker.session_id is None


def test_invalid_continuation_is_rejected_before_stateful_work(worker):
    session = admit(worker)
    request = action_request(worker, session)
    request.body["model_continuation"] = np.ones((2, 3), dtype=np.int64)
    assert_error(worker.submit(request).result(2), ErrorCode.MALFORMED)
    request.body["model_continuation"] = np.ones((2, 3), dtype=np.float32)
    request.body["canonical_continuation"] = np.ones((1, 3), dtype=np.float32)
    assert_error(worker.submit(request).result(2), ErrorCode.MALFORMED)
    assert not worker.runner.policy.last_kwargs


def test_open_is_idempotent_and_a_second_client_cannot_replace_session(worker):
    request = open_request(worker)
    first = worker.submit(request)
    accepted = first.result(2)
    assert worker.submit(request) is first
    assert_error(worker.submit(open_request(worker)).result(2), ErrorCode.BUSY)
    assert worker.session_id == accepted.session_id
    assert worker.runner.policy.resets == 2  # constructor, admitted reset


def test_duplicate_data_request_never_reexecutes_stateful_policy(worker):
    session = admit(worker)
    request = action_request(worker, session)
    assert worker.submit(request).result(2).message_type is MessageType.ACTION
    assert_error(worker.submit(request).result(2), ErrorCode.STALE)


def test_evicted_request_identity_cannot_be_replayed(worker):
    session = admit(worker)
    first = action_request(worker, session)
    assert worker.submit(first).result(2).message_type is MessageType.ACTION
    for _ in range(128):
        assert worker.submit(action_request(worker, session)).result(2).message_type is MessageType.ACTION
    assert first.request_id not in worker._seen
    assert_error(worker.submit(first).result(2), ErrorCode.STALE)


def test_equal_generation_reset_remains_idempotent_after_control_cache_eviction(worker):
    session = admit(worker)
    reset = control_request(worker, session, 1, "reset")
    assert worker.submit(reset).result(2).message_type is MessageType.ACK
    resets = worker.runner.policy.resets
    for _ in range(32):
        assert (
            worker.submit(control_request(worker, session, 1, "status")).result(2).message_type
            is MessageType.ACK
        )
    assert reset.request_id not in worker._controls
    assert worker.submit(reset).result(2).message_type is MessageType.ACK
    assert worker.runner.policy.resets == resets


def test_close_operation_remains_idempotent_after_session_release(worker):
    session = admit(worker)
    close = control_request(worker, session, 0, "close")
    first = worker.submit(close)
    assert first.result(2).message_type is MessageType.ACK
    assert worker.submit(close) is first
    new_session = admit(worker)
    assert worker.submit(close) is first
    assert worker.session_id == new_session


def test_reset_acknowledges_only_after_current_model_call_and_reset(worker):
    session = admit(worker)
    entered, release = block_predict(worker)
    try:
        action = worker.submit(action_request(worker, session))
        assert entered.wait(2)
        reset_request = control_request(worker, session, 1, "reset")
        reset = worker.submit(reset_request)
        assert worker.submit(reset_request) is reset
        assert not reset.done()
        assert worker.runner.policy.resets == 2
        assert_error(worker.submit(action_request(worker, session, generation=1)).result(2), ErrorCode.STALE)
        release.set()
        assert action.result(2).message_type is MessageType.ACTION
        assert reset.result(2).body["applied_generation"] == 1
        assert worker.runner.policy.resets == 3
        assert_error(worker.submit(action_request(worker, session)).result(2), ErrorCode.STALE)
        assert (
            worker.submit(action_request(worker, session, generation=1)).result(2).message_type
            is MessageType.ACTION
        )
    finally:
        release.set()


def test_queued_controls_cannot_roll_back_a_newer_generation(worker):
    session = admit(worker)
    entered, release = block_predict(worker)
    try:
        action = worker.submit(action_request(worker, session))
        assert entered.wait(2)
        newer = worker.submit(control_request(worker, session, 2, "reset"))
        older = worker.submit(control_request(worker, session, 1, "reset"))
        release.set()
        action.result(2)
        assert newer.result(2).message_type is MessageType.ACK
        assert_error(older.result(2), ErrorCode.STALE)
        assert (
            worker.submit(action_request(worker, session, generation=2)).result(2).message_type
            is MessageType.ACTION
        )
    finally:
        release.set()


def test_close_during_blocked_call_keeps_ownership_until_worker_finishes(worker):
    session = admit(worker)
    entered, release = block_predict(worker)
    try:
        action = worker.submit(action_request(worker, session))
        assert entered.wait(2)
        close = worker.submit(control_request(worker, session, 0, "close"))
        assert not close.done()
        rejected = worker.submit(open_request(worker)).result(2)
        assert_error(rejected, ErrorCode.BUSY)
        assert "admission_blocker=unfinished_inference" in rejected.body["message"]
        state = worker.descriptor["session"]
        assert state["owner"] == session
        assert state["cleanup_pending"]
        assert state["cleanup_reason"] == "client_close"
        assert state["inference_pending"]
        release.set()
        action.result(2)
        assert close.result(2).message_type is MessageType.ACK
        assert admit(worker) != session
    finally:
        release.set()


@pytest.fixture
def cleanup_clock(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("lerobot.remote_inference.server.time", SimpleNamespace(monotonic=lambda: clock[0]))
    return clock


@pytest.mark.parametrize("initially_present", [False, True])
def test_first_presence_never_logs_a_disconnect_or_recovery(worker, cleanup_clock, caplog, initially_present):
    caplog.set_level("INFO", logger="lerobot.remote_inference.server")
    session = admit(worker)
    if not initially_present:
        worker.expire(present=False)
        cleanup_clock[0] += 1
        worker.expire(present=False)
        assert caplog.text.count("Session awaiting initial presence") == 1
        assert worker.descriptor["session"]["admission_blocker"] == "awaiting_initial_presence"
    worker.expire(present=True)
    worker.expire(present=True)
    assert worker.session_id == session
    assert worker.descriptor["session"]["client_present"] is True
    assert worker.descriptor["session"]["absence_grace_remaining_s"] is None
    assert caplog.text.count("Session initial presence established") == 1
    assert "Session client absent" not in caplog.text
    assert "Session presence restored" not in caplog.text


def test_incomplete_handshake_expires_without_retries_extending_grace(worker, cleanup_clock, caplog):
    caplog.set_level("INFO", logger="lerobot.remote_inference.server")
    session = admit(worker)
    worker.expire(present=False)
    for advance, remaining in [(4, 6.0), (5, 1.0)]:
        cleanup_clock[0] += advance
        worker.expire(present=False)
        rejected = worker.submit(open_request(worker)).result(2)
        assert_error(rejected, ErrorCode.BUSY)
        assert rejected.body["details"] == {
            "admission_blocker": "awaiting_initial_presence",
            "absence_grace_remaining_s": remaining,
        }
        assert worker.session_id == session
    cleanup_clock[0] += 1
    worker.expire(present=False)
    assert_error(worker.submit(control_request(worker, session, 0, "status")).result(2), ErrorCode.STALE)
    assert worker.descriptor["available"]
    assert admit(worker) != session
    assert "reason=initial_presence_timeout" in caplog.text
    assert "Session client absent" not in caplog.text
    assert "Session presence restored" not in caplog.text


def test_absent_client_retains_ownership_during_grace_then_allows_fresh_session(
    worker, cleanup_clock, caplog
):
    caplog.set_level("INFO", logger="lerobot.remote_inference.server")
    session = admit(worker)
    assert worker.descriptor["limits"]["idle_timeout_s"] == 10.0
    worker.expire(present=True)
    worker.expire(present=False)
    cleanup_clock[0] += 9
    worker.expire(present=False)
    state = worker.descriptor["session"]
    assert state["owner"] == session
    assert state["client_present"] is False
    assert state["absence_grace_remaining_s"] == 1.0
    assert not state["cleanup_pending"]
    rejected = worker.submit(open_request(worker)).result(2)
    assert_error(rejected, ErrorCode.BUSY)
    assert "admission_blocker=absence_grace" in rejected.body["message"]
    assert "absence_grace_remaining_s=1.0" in rejected.body["message"]
    assert rejected.body["details"] == {
        "admission_blocker": "absence_grace",
        "absence_grace_remaining_s": 1.0,
    }
    cleanup_clock[0] += 1
    worker.expire(present=False)
    # This control either queues after the worker close or observes its completion.
    assert_error(worker.submit(control_request(worker, session, 0, "status")).result(2), ErrorCode.STALE)
    assert worker.descriptor["available"]
    assert worker.descriptor["session"]["owner"] is None
    assert admit(worker) != session
    assert worker.runner.policy.resets == 4  # constructor, first open, close, new open
    assert "Session client absent" in caplog.text
    assert "Session cleanup queued" in caplog.text
    assert "Session released" in caplog.text


def test_present_paused_client_never_expires_and_restored_presence_restarts_grace(
    worker, cleanup_clock, caplog
):
    caplog.set_level("INFO", logger="lerobot.remote_inference.server")
    session = admit(worker)
    worker.expire(present=True)
    cleanup_clock[0] += 300
    worker.expire(present=True)
    assert worker.session_id == session
    assert worker.descriptor["session"]["absence_grace_remaining_s"] is None
    worker.expire(present=False)
    cleanup_clock[0] += 9
    worker.expire(present=True)
    state = worker.descriptor["session"]
    assert state["client_present"]
    assert state["admission_blocker"] == "active_session"
    assert state["absence_grace_remaining_s"] is None
    cleanup_clock[0] += 100
    worker.expire(present=False)
    assert worker.descriptor["session"]["absence_grace_remaining_s"] == 10.0
    assert worker.session_id == session
    assert caplog.text.count("Session presence restored") == 1


def test_absence_cleanup_never_reuses_a_blocked_model_after_grace(worker, cleanup_clock):
    session = admit(worker)
    worker.expire(present=True)
    entered, release = block_predict(worker)
    try:
        action = worker.submit(action_request(worker, session))
        assert entered.wait(2)
        worker.expire(present=False)
        cleanup_clock[0] += 10
        worker.expire(present=False)
        state = worker.descriptor["session"]
        assert state["owner"] == session
        assert state["absence_grace_remaining_s"] == 0.0
        assert state["cleanup_reason"] == "client_absence"
        assert state["admission_blocker"] == "unfinished_inference"
        # Neither additional absence nor a late presence token revokes an ordered close.
        cleanup_clock[0] += 300
        worker.expire(present=False)
        worker.expire(present=True)
        assert_error(worker.submit(open_request(worker)).result(2), ErrorCode.BUSY)
        assert worker.runner.policy.resets == 2
        after_cleanup = worker.submit(control_request(worker, session, 0, "status"))
        assert not after_cleanup.done()
        release.set()
        assert_error(action.result(2), ErrorCode.TIMEOUT)
        assert_error(after_cleanup.result(2), ErrorCode.STALE)
        assert admit(worker) != session
    finally:
        release.set()


def test_absence_cleanup_retries_a_full_worker_queue_without_extending_grace(worker, cleanup_clock):
    session = admit(worker)
    worker.expire(present=True)
    entered, release = block_predict(worker)
    try:
        action = worker.submit(action_request(worker, session))
        assert entered.wait(2)
        queued = [worker.submit(control_request(worker, session, 0, "status")) for _ in range(8)]
        worker.expire(present=False)
        cleanup_clock[0] += 10
        worker.expire(present=False)
        state = worker.descriptor["session"]
        assert state["admission_blocker"] == "cleanup_queue_full"
        assert not state["cleanup_pending"]
        assert state["absence_grace_remaining_s"] == 0.0
        release.set()
        action.result(2)
        for future in queued:
            assert future.result(2).message_type is MessageType.ACK
        worker.expire(present=False)
        assert_error(worker.submit(control_request(worker, session, 0, "status")).result(2), ErrorCode.STALE)
        assert admit(worker) != session
    finally:
        release.set()


def test_old_queued_control_cannot_mutate_new_session(worker):
    session = admit(worker)
    entered, release = block_predict(worker)
    new_open = []
    try:
        action = worker.submit(action_request(worker, session))
        assert entered.wait(2)
        closing = worker.submit(control_request(worker, session, 0, "close"))
        old_reset = worker.submit(control_request(worker, session, 99, "reset"))
        closing.add_done_callback(lambda _: new_open.append(worker.submit(open_request(worker))))
        release.set()
        action.result(2)
        assert_error(old_reset.result(2), ErrorCode.STALE)
        accepted = new_open[0].result(2)
        assert accepted.message_type is MessageType.ACCEPTED
        assert (
            worker.submit(action_request(worker, accepted.session_id)).result(2).message_type
            is MessageType.ACTION
        )
    finally:
        release.set()


def test_full_control_queue_does_not_apply_rejected_close(worker):
    session = admit(worker)
    entered, release = block_predict(worker)
    try:
        action = worker.submit(action_request(worker, session))
        assert entered.wait(2)
        queued = [worker.submit(control_request(worker, session, 0, "status")) for _ in range(8)]
        assert_error(worker.submit(control_request(worker, session, 0, "close")).result(2), ErrorCode.BUSY)
        release.set()
        action.result(2)
        for future in queued:
            assert future.result(2).message_type is MessageType.ACK
        assert worker.submit(action_request(worker, session)).result(2).message_type is MessageType.ACTION
    finally:
        release.set()


def test_language_exception_after_deadline_faults_session(worker, monkeypatch):
    session = admit(worker)
    clock = [100.0]
    monkeypatch.setattr("lerobot.remote_inference.server.time", SimpleNamespace(monotonic=lambda: clock[0]))

    def fail_after_deadline(batch):
        clock[0] += 6
        raise ValueError("generation failed late")

    worker.runner.policy.generate_text = fail_after_deadline
    action = action_request(worker, session)
    body = {**action.body, "kind": "vqa", "text": "What is visible?", "intent_generation": 1}
    query = Envelope(MessageType.LANGUAGE_REQUEST, worker.instance_id, session, 0, uuid4().hex, body)
    assert_error(worker.submit(query).result(2), ErrorCode.TIMEOUT)
    assert_error(worker.submit(action_request(worker, session)).result(2), ErrorCode.STALE)
