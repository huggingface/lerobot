# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Expected admission contention is concise; other failures remain debuggable."""

import pytest

pytest.importorskip("datasets")
pytest.importorskip("msgpack")

from lerobot.remote_inference.codec import decode_message, encode_message
from lerobot.remote_inference.protocol import (
    AdmissionDeniedError,
    Envelope,
    ErrorCode,
    MessageType,
    ProtocolError,
)
from lerobot.scripts import lerobot_rollout
from tests.remote_inference import test_chunk_contract as helpers

worker = helpers.worker


def test_busy_open_is_distinct_from_busy_during_an_admitted_session(worker):
    details = {"admission_blocker": "absence_grace", "absence_grace_remaining_s": 7.25}

    def query(_key, payload, _timeout):
        request = decode_message(payload)
        return [encode_message(request.error(ErrorCode.BUSY, "existing owner", details=details))]

    client = helpers.client_for(worker, helpers.client_config(chunk_merge="append"))
    client.transport.query = query
    with pytest.raises(AdmissionDeniedError) as denied:
        helpers.admit(client)
    assert denied.value.deployment == "test"
    assert denied.value.details == details
    assert not client.session_id

    request = Envelope(MessageType.CONTROL, "server", "session", request_id="control")
    with pytest.raises(ProtocolError) as busy_control:
        client._query("control", request, 1, MessageType.ACK)
    assert type(busy_control.value) is ProtocolError
    assert busy_control.value.code is ErrorCode.BUSY


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


@pytest.mark.parametrize(
    "error",
    [
        ProtocolError(ErrorCode.INCOMPATIBLE, "schema differs"),
        RuntimeError("unexpected failure"),
    ],
)
def test_cli_does_not_swallow_other_failures(error, monkeypatch, caplog):
    def failed():
        raise error

    monkeypatch.setattr(lerobot_rollout, "register_third_party_plugins", lambda: None)
    monkeypatch.setattr(lerobot_rollout, "rollout", failed)
    with pytest.raises(type(error)) as raised:
        lerobot_rollout.main()
    assert raised.value is error
    if isinstance(error, ProtocolError) and error.code is ErrorCode.INCOMPATIBLE:
        assert "Remote compatibility check failed" in caplog.text
        assert "schema differs" in caplog.text
        assert "loaded client/server builds" in caplog.text
