# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Malformed peers and failed resets must not silently kill or wedge a deployment."""

from threading import Event
from uuid import uuid4

import pytest

from lerobot.remote_inference.codec import CodecLimits, decode_message
from lerobot.remote_inference.protocol import Envelope, ErrorCode, MessageType
from lerobot.remote_inference.server import PolicyServer, SessionWorker
from tests.inference.test_policy_runner import ConformingPolicy, runner_for, tiny_config
from tests.remote_inference.test_session import (
    action_request,
    admit,
    assert_error,
    control_request,
    open_request,
)


@pytest.fixture
def worker():
    instance = SessionWorker(
        runner_for(ConformingPolicy(tiny_config())),
        deployment="review",
        artifact_identity="artifact",
        semantics="radians",
    )
    yield instance
    instance.close()


@pytest.mark.parametrize("value", [None, True, 12, [], {"invalid": "semantics"}])
@pytest.mark.parametrize("feature", ["features", "action_feature"])
def test_invalid_feature_metadata_is_rejected_without_owning_session(worker, feature, value):
    request = open_request(worker)
    spec = request.body[feature][0] if feature == "features" else request.body[feature]
    spec["semantics"] = value
    assert_error(worker.submit(request).result(2), ErrorCode.MALFORMED)
    assert worker.session_id is None
    assert worker._thread.is_alive()
    assert admit(worker)


@pytest.mark.parametrize("operation", ["open", "close", "reset", "invalidate"])
def test_reset_failure_requires_restart_without_running_more_policy_work(worker, operation, monkeypatch):
    session = None if operation == "open" else admit(worker)
    reset_calls = []

    def failed_reset(*, full):
        reset_calls.append(full)
        raise RuntimeError("custom processor reset failed")

    monkeypatch.setattr(worker.runner, "reset", failed_reset)
    request = open_request(worker) if operation == "open" else control_request(worker, session, 1, operation)
    response = worker.submit(request).result(2)
    assert_error(response, ErrorCode.EXECUTION)
    assert "server restarts" in response.body["message"]
    assert worker.session_id is not None
    assert not worker.descriptor["ready"]
    assert not worker.descriptor["available"]
    for request in (
        open_request(worker),
        control_request(worker, worker.session_id, 2, "reset"),
        action_request(worker, worker.session_id),
    ):
        denied = worker.submit(request).result(2)
        assert_error(denied, ErrorCode.EXECUTION)
        assert "server restarts" in denied.body["message"]
    worker.expire(present=False)
    assert len(reset_calls) == 1
    described = worker.submit(Envelope(MessageType.DESCRIBE, request_id=uuid4().hex)).result(2)
    assert described.message_type is MessageType.DESCRIPTOR
    assert not described.body["ready"]


def test_oversized_utf8_answer_is_query_error_and_next_action_still_works(worker, monkeypatch):
    session = admit(worker)
    answer = "界" * (CodecLimits().max_string_bytes // 3 + 1)
    assert len(answer) < worker.max_output_chars
    monkeypatch.setattr(worker.runner, "query", lambda *args, **kwargs: answer)
    source = action_request(worker, session)
    request = source.reply(
        MessageType.LANGUAGE_REQUEST,
        {**source.body, "kind": "vqa", "text": "What is visible?", "intent_generation": 1},
    )
    response = worker.submit(request).result(2)
    assert_error(response, ErrorCode.EXECUTION)
    assert "oversized text answer" in response.body["message"]
    assert worker.submit(action_request(worker, session)).result(2).message_type is MessageType.ACTION


def test_commands_queued_behind_a_failed_reset_do_not_reenter_policy(worker, monkeypatch):
    session = admit(worker)
    entered, release = Event(), Event()
    resets = []

    def failed_reset(*, full):
        resets.append(full)
        entered.set()
        assert release.wait(2)
        raise RuntimeError("reset failed")

    monkeypatch.setattr(worker.runner, "reset", failed_reset)
    first = worker.submit(control_request(worker, session, 1, "reset"))
    try:
        assert entered.wait(2)
        queued = worker.submit(control_request(worker, session, 2, "reset"))
        closing = worker.submit(control_request(worker, session, 2, "close"))
    finally:
        release.set()
    for future in (first, queued, closing):
        assert_error(future.result(2), ErrorCode.EXECUTION)
    assert resets == [True]
    assert worker.session_id == session


def test_unencodable_response_returns_correlated_execution_error():
    pytest.importorskip("msgpack")
    response = Envelope(MessageType.ACTION, "instance", "session", 3, "request", {"invalid": object()})
    failure = decode_message(PolicyServer._encode_reply(response))
    assert_error(failure, ErrorCode.EXECUTION)
    assert failure.instance_id == response.instance_id
    assert failure.session_id == response.session_id
    assert failure.generation == response.generation
    assert failure.request_id == response.request_id
