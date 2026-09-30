# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Expected admission contention is concise; other failures remain debuggable."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest

from lerobot.inference.contracts import ExecutionMode, FeatureSpec, PolicyCapabilities
from lerobot.remote_inference.client import RemoteClient
from lerobot.remote_inference.codec import decode_message, encode_message
from lerobot.remote_inference.protocol import (
    AdmissionDeniedError,
    Envelope,
    ErrorCode,
    MessageType,
    ProtocolError,
)
from lerobot.rollout.inference import RemoteInferenceConfig
from lerobot.scripts import lerobot_rollout


def test_busy_open_is_distinct_from_busy_during_an_admitted_session():
    details = {"admission_blocker": "absence_grace", "absence_grace_remaining_s": 7.25}

    def query(_key, payload, _timeout):
        request = decode_message(payload)
        return [encode_message(request.error(ErrorCode.BUSY, "existing owner", details=details))]

    caps = PolicyCapabilities(
        modes=(ExecutionMode.CHUNK,),
        prediction_steps=2,
        execution_steps=2,
        action_interval=1 / 30,
        features=(FeatureSpec("observation.state", (1,), "float32", semantics="joints"),),
        action_feature=FeatureSpec("action", (1,), "float32", semantics="joints"),
    )
    transport = SimpleNamespace(subscribe_liveliness=lambda _: None, query=query)
    config = RemoteInferenceConfig(deployment="test", semantics="joints", hold_mode="position")
    client = RemoteClient(
        transport,
        config,
        {
            "capabilities": asdict(caps),
            "instance_id": "server",
            "artifact_identity": "model",
            "semantics": "joints",
        },
    )
    with pytest.raises(AdmissionDeniedError) as denied:
        client.admit(
            features=caps.features,
            action_feature=caps.action_feature,
            semantics="joints",
            action_interval=caps.action_interval,
            mode="chunk",
        )
    assert denied.value.deployment == "test"
    assert denied.value.details == details
    assert not client.session_id

    request = Envelope(MessageType.CONTROL, "server", "session", request_id="control")
    with pytest.raises(ProtocolError) as busy_control:
        client._query("control", request, 1, MessageType.ACK)
    assert type(busy_control.value) is ProtocolError
    assert busy_control.value.code is ErrorCode.BUSY


@pytest.mark.parametrize("body", [{"details": []}, {"message": 5}, {"code": []}])
def test_malformed_error_is_not_presented_as_expected_contention(body):
    response = Envelope(MessageType.ERROR, body={"code": "busy", "message": "busy", **body})
    with pytest.raises(ProtocolError) as malformed:
        RemoteClient._raise_error(response)
    assert malformed.value.code is ErrorCode.MALFORMED


@pytest.mark.parametrize("blocker", ["absence_grace", "awaiting_initial_presence"])
def test_cli_reports_approximate_grace_without_promising_admission(blocker, monkeypatch, caplog):
    error = AdmissionDeniedError(
        "smolvla",
        "server diagnostic",
        details={"admission_blocker": blocker, "absence_grace_remaining_s": 7.25},
    )

    def denied():
        raise error

    monkeypatch.setattr(lerobot_rollout, "register_third_party_plugins", lambda: None)
    monkeypatch.setattr(lerobot_rollout, "rollout", denied)
    with pytest.raises(SystemExit) as stopped:
        lerobot_rollout.main()
    assert stopped.value.code == 1
    assert "Remote admission denied for deployment 'smolvla'" in caplog.text
    assert "about 7.2 s" in caplog.text
    assert "worker cleanup may take longer" in caplog.text
    assert not any(record.exc_info for record in caplog.records)


@pytest.mark.parametrize("blocker", ["unfinished_inference", "worker_cleanup_pending", "cleanup_queue_full"])
def test_pending_worker_cleanup_has_no_retry_eta(blocker):
    error = AdmissionDeniedError(
        "smolvla",
        "server diagnostic",
        details={"admission_blocker": blocker, "absence_grace_remaining_s": 0.0},
    )
    message = lerobot_rollout._admission_denial_message(error)
    assert "completion time is unknown" in message
    assert "0.0" not in message
    assert (
        "model call" in message if blocker == "unfinished_inference" else "worker session cleanup" in message
    )


def test_older_error_without_details_remains_readable():
    message = lerobot_rollout._admission_denial_message(AdmissionDeniedError("smolvla", "busy"))
    assert "deployment is busy" in message
    assert "server logs" in message


@pytest.mark.parametrize(
    "error",
    [
        ProtocolError(ErrorCode.INCOMPATIBLE, "schema differs"),
        ProtocolError(ErrorCode.BUSY, "inference pending"),
        RuntimeError("unexpected failure"),
    ],
)
def test_cli_does_not_swallow_other_failures(error, monkeypatch):
    def failed():
        raise error

    monkeypatch.setattr(lerobot_rollout, "register_third_party_plugins", lambda: None)
    monkeypatch.setattr(lerobot_rollout, "rollout", failed)
    with pytest.raises(type(error)) as raised:
        lerobot_rollout.main()
    assert raised.value is error
