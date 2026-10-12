# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""A refused open explains itself to the operator; later BUSY replies stay ordinary protocol errors."""

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
from tests.remote_inference import test_chunk_contract as helpers

worker = helpers.worker


@pytest.mark.parametrize("blocker", ["absence_grace", "awaiting_initial_presence"])
def test_busy_open_is_distinct_from_busy_during_an_admitted_session(worker, blocker):
    details = {"admission_blocker": blocker, "absence_grace_remaining_s": 7.25}

    def query(_key, payload, _timeout):
        request = decode_message(payload)
        return [encode_message(request.error(ErrorCode.BUSY, "existing owner", details=details))]

    client = helpers.client_for(worker, helpers.client_config(chunk_merge="append"))
    client.transport.query = query
    with pytest.raises(AdmissionDeniedError) as denied:
        helpers.admit(client)
    assert denied.value.deployment == "test"
    assert denied.value.details == details
    assert denied.value.diagnostic == "existing owner"
    assert str(denied.value).startswith("Remote admission denied for deployment 'test'")
    assert "about 7.2 s" in str(denied.value)
    assert "worker cleanup may take longer" in str(denied.value)
    assert not client.session_id

    request = Envelope(MessageType.CONTROL, "server", "session", request_id="control")
    with pytest.raises(ProtocolError) as busy_control:
        client._query("control", request, 1, MessageType.ACK)
    assert type(busy_control.value) is ProtocolError
    assert busy_control.value.code is ErrorCode.BUSY
