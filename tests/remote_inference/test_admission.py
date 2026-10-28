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

"""A refused open explains itself to the operator; later BUSY replies stay ordinary protocol errors."""

from dataclasses import replace

import pytest

pytest.importorskip("msgpack")

from lerobot.remote_inference.client import _compare_feature
from lerobot.remote_inference.codec import decode_message, encode_message
from lerobot.remote_inference.protocol import (
    AdmissionDeniedError,
    Envelope,
    ErrorCode,
    MessageType,
    ProtocolError,
)
from lerobot.utils.feature_spec import FeatureSpec
from tests.remote_inference import helpers


@pytest.mark.parametrize("blocker", ["absence_grace", "awaiting_initial_presence"])
def test_busy_open_is_distinct_from_busy_during_an_admitted_session(chunk_worker, blocker):
    details = {"admission_blocker": blocker, "absence_grace_remaining_s": 7.25}

    def query(_key, payload, _timeout, *, cancelled=None):
        request = decode_message(payload)
        return [encode_message(request.error(ErrorCode.BUSY, "existing owner", details=details))]

    client = helpers.client_for(chunk_worker, helpers.client_config(chunk_merge="append"))
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


# A server whose schema differs is refused with the first field that differs, not a schema dump.
@pytest.mark.parametrize(
    "changes",
    [
        {"name": "other"},
        {"shape": (3,), "names": ()},
        {"dtype": "float64"},
        {"names": ("y", "x")},
        {"semantics": "position-v1"},
    ],
)
def test_feature_comparison_identifies_incompatible_fields(changes):
    expected = FeatureSpec("action", (2,), "float32", names=("x", "y"), semantics="delta-v1")
    _compare_feature(expected, expected)
    with pytest.raises(ProtocolError, match=f"field '{next(iter(changes))}' differs") as error:
        _compare_feature(replace(expected, **changes), expected)
    assert error.value.code is ErrorCode.INCOMPATIBLE


def test_feature_comparison_checks_modality():
    expected = FeatureSpec("image", (2, 2, 3), "uint8", kind="rgb", semantics="rgb-v1")
    with pytest.raises(ProtocolError, match="field 'kind' differs"):
        _compare_feature(replace(expected, kind="tensor"), expected)
