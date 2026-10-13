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

"""Negotiate alignment while validating blending entirely on the client."""

from dataclasses import replace

import pytest

from lerobot.inference import ExecutionMode
from lerobot.remote_inference.chunk_contract import (
    CHUNK_ALIGNMENT,
    chunk_settings,
    default_chunk_settings,
    required_chunk_capabilities,
    validate_blend_settings,
)
from lerobot.remote_inference.protocol import ErrorCode, MessageType, ProtocolError
from tests.remote_inference.helpers import (
    action_request,
    admit,
    assert_error,
    client_config,
    client_for,
    control_request,
    open_request,
)


@pytest.mark.parametrize("mode", ["chunk", "rtc_guided", "rtc_trained"])
def test_client_defaults_align_plain_chunks_without_changing_rtc_or_legacy_wire(mode):
    config = client_config(mode=mode)
    expected = "aligned" if mode == "chunk" else "append"
    assert (config.chunk_merge, config.blend_steps, config.blend_components) == (expected, 0, [])
    assert client_config(mode=mode, chunk_merge="append").chunk_merge == "append"
    # Omitted wire settings keep their original meaning; alignment is negotiated explicitly.
    assert default_chunk_settings() == {"chunk_merge": "append"}


def aligned_open(chunk_worker):
    request = open_request(chunk_worker)
    request.body["chunk_settings"] = {"chunk_merge": "aligned"}
    request.body["required_capabilities"] = required_chunk_capabilities(request.body["chunk_settings"])
    return request


def test_admission_echoes_only_merge_settings(chunk_worker):
    descriptor = chunk_worker.descriptor
    assert descriptor["execution_contracts"] == [CHUNK_ALIGNMENT]
    assert "blendable_components" not in descriptor
    request = aligned_open(chunk_worker)
    response = chunk_worker.submit(request).result(2)
    assert response.message_type is MessageType.ACCEPTED
    assert response.body["chunk_settings"] == request.body["chunk_settings"]


@pytest.mark.parametrize(
    "updates",
    [
        {"blend_steps": 2},
        {"chunk_merge": "unknown"},
        {"chunk_merge": []},
        {"chunk_merge": {}},
        {"chunk_merge": "append"},
    ],
)
def test_server_rejects_invalid_merge_contract_before_allocating_session(chunk_worker, updates):
    request = aligned_open(chunk_worker)
    request.body["chunk_settings"].update(updates)
    assert_error(chunk_worker.submit(request).result(2), ErrorCode.INCOMPATIBLE)
    assert chunk_worker.session_id is None


def test_alignment_cannot_omit_its_required_capability(chunk_worker):
    request = aligned_open(chunk_worker)
    del request.body["required_capabilities"]
    assert_error(chunk_worker.submit(request).result(2), ErrorCode.INCOMPATIBLE)
    assert chunk_worker.session_id is None


@pytest.mark.parametrize("alteration", ["noncanonical", "unnamed", "integer"])
def test_blending_requires_named_canonical_float_coordinates(chunk_worker, alteration):
    caps = chunk_worker.runner.capabilities
    if alteration == "noncanonical":
        caps = replace(caps, action_representation="relative")
    elif alteration == "unnamed":
        caps = replace(caps, action_feature=replace(caps.action_feature, names=()))
    else:
        caps = replace(caps, action_feature=replace(caps.action_feature, dtype="int32"))
    with pytest.raises(ValueError):
        validate_blend_settings(chunk_settings("aligned", 2, 0.5, ["elbow.pos"]), caps)


@pytest.mark.parametrize("advertised", [[], ["chunk_alignment_v1", "chunk_blending_v1"]])
def test_old_server_is_rejected_before_open_for_alignment(chunk_worker, advertised):
    descriptor = chunk_worker.descriptor
    descriptor["execution_contracts"] = advertised
    client = client_for(chunk_worker, client_config(chunk_merge="aligned"), descriptor)
    with pytest.raises(ProtocolError, match="update the server") as error:
        admit(client)
    assert error.value.code is ErrorCode.UNSUPPORTED
    assert not client.session_id
    assert chunk_worker.session_id is None


def test_clients_can_change_blending_without_server_reconfiguration(chunk_worker, monkeypatch):
    for components in ([], ["elbow.pos", "shoulder.pos"], ["gripper.pos"]):
        client = client_for(
            chunk_worker,
            client_config(blend_steps=2 if components else 0, blend_components=components),
        )

        def query(key, request, timeout, expected):
            assert request.body["chunk_settings"] == {"chunk_merge": "aligned"}
            assert request.body["required_capabilities"] == [CHUNK_ALIGNMENT]
            return chunk_worker.submit(request).result(2)

        monkeypatch.setattr(client, "_query", query)
        monkeypatch.setattr(client, "control", lambda *_: None)
        admit(client)
        names = chunk_worker.runner.capabilities.action_feature.names
        assert client.blend_indices == tuple(names.index(name) for name in components)
        closed = chunk_worker.submit(control_request(chunk_worker, client.session_id, 0, "close")).result(2)
        assert closed.message_type is MessageType.ACK


@pytest.mark.parametrize("steps,components", [(2, ["unknown.pos"]), (4, ["elbow.pos"])])
def test_invalid_blend_settings_fail_client_side_before_open(chunk_worker, monkeypatch, steps, components):
    client = client_for(chunk_worker, client_config(blend_steps=steps, blend_components=components))
    monkeypatch.setattr(client, "_query", lambda *args: pytest.fail("invalid blending must fail before OPEN"))
    with pytest.raises(ProtocolError) as error:
        admit(client)
    assert error.value.code is ErrorCode.INCOMPATIBLE
    assert chunk_worker.session_id is None


def test_server_cannot_silently_change_accepted_merge_settings(chunk_worker, monkeypatch):
    client = client_for(chunk_worker, client_config(chunk_merge="aligned"))

    def query(key, request, timeout, expected):
        accepted = chunk_worker.submit(request).result(2)
        accepted.body["chunk_settings"] = {**accepted.body["chunk_settings"], "chunk_merge": "append"}
        return accepted

    monkeypatch.setattr(client, "_query", query)
    with pytest.raises(ProtocolError, match="changed the advertised contract"):
        admit(client)
    assert not client.session_id


@pytest.mark.parametrize(
    "observation_cursor,cursor", [(None, 2), (2, None), (-1, 2), (3, 2), (True, 2), (2, 2**63)]
)
def test_aligned_request_rejects_invalid_cursors_before_policy_call(chunk_worker, observation_cursor, cursor):
    accepted = chunk_worker.submit(aligned_open(chunk_worker)).result(2)
    request = action_request(chunk_worker, accepted.session_id)
    request.body.update(observation_cursor=observation_cursor, cursor=cursor)
    assert_error(chunk_worker.submit(request).result(2), ErrorCode.MALFORMED)
    assert not chunk_worker.runner.policy.last_kwargs


def test_action_echoes_both_aligned_cursors_and_client_rejects_rebinding(chunk_worker):
    accepted = chunk_worker.submit(aligned_open(chunk_worker)).result(2)
    request = action_request(chunk_worker, accepted.session_id)
    request.body.update(observation_cursor=2, cursor=4)
    result = chunk_worker.submit(request).result(2)
    assert result.message_type is MessageType.ACTION
    assert result.body["observation_cursor"] == 2
    assert result.body["cursor"] == 4
    client = client_for(chunk_worker, client_config(chunk_merge="aligned"))
    client._validate_context(result, request.body, MessageType.ACTION)
    result.body["observation_cursor"] = 3
    with pytest.raises(ProtocolError, match="context differs"):
        client._validate_context(result, request.body, MessageType.ACTION)


def test_server_rejects_alignment_for_rtc_session(chunk_worker):
    chunk_worker.runner.capabilities = replace(
        chunk_worker.runner.capabilities, modes=(ExecutionMode.CHUNK, ExecutionMode.RTC_GUIDED)
    )
    request = aligned_open(chunk_worker)
    request.body["mode"] = "rtc_guided"
    assert_error(chunk_worker.submit(request).result(2), ErrorCode.INCOMPATIBLE)


def test_client_rejects_action_schema_mismatch_before_open(chunk_worker, monkeypatch):
    client = client_for(chunk_worker, client_config())
    monkeypatch.setattr(client, "_query", lambda *args: pytest.fail("schema must be checked before OPEN"))
    caps = client.capabilities
    with pytest.raises(ProtocolError, match="names") as failed:
        client.admit(
            features=caps.features,
            action_feature=replace(caps.action_feature, names=tuple(reversed(caps.action_feature.names))),
            semantics=chunk_worker.semantics,
            action_interval=caps.action_interval,
            mode="chunk",
        )
    assert failed.value.code is ErrorCode.INCOMPATIBLE
    assert chunk_worker.session_id is None
