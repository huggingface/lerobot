# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Plain alignment and blending must be negotiated before accepting motion."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

pytest.importorskip("datasets")

from lerobot.inference import ExecutionMode, RemoteInferenceConfig
from lerobot.remote_inference.chunk_contract import (
    CHUNK_ALIGNMENT,
    CHUNK_BLENDING,
    chunk_settings,
    default_chunk_settings,
    required_chunk_capabilities,
    validate_chunk_contract,
)
from lerobot.remote_inference.client import RemoteClient
from lerobot.remote_inference.configs import ExecutionConfig, ModelConfig, ServerConfig
from lerobot.remote_inference.protocol import ErrorCode, MessageType, ProtocolError
from lerobot.remote_inference.server import SessionWorker
from tests.inference.test_policy_runner import ConformingPolicy, runner_for, tiny_config
from tests.remote_inference.test_session import action_request, assert_error, open_request


def client_config(**kwargs):
    return RemoteInferenceConfig(deployment="test", semantics="test-radians", hold_mode="position", **kwargs)


def test_client_defaults_match_implicit_legacy_chunk_contract():
    config = client_config()
    assert (config.chunk_merge, config.blend_steps, config.blend_components) == ("append", 0, [])
    assert (
        chunk_settings(config.chunk_merge, config.blend_steps, config.blend_weight, config.blend_components)
        == default_chunk_settings()
    )
    altered = default_chunk_settings()
    altered["blend_components"].append("shoulder.pos")
    assert not default_chunk_settings()["blend_components"]


@pytest.fixture
def worker():
    runner = runner_for(ConformingPolicy(tiny_config()))
    runner.capabilities = replace(
        runner.capabilities,
        action_feature=replace(
            runner.capabilities.action_feature, names=("shoulder.pos", "elbow.pos", "gripper.pos")
        ),
    )
    worker = SessionWorker(
        runner,
        deployment="test",
        artifact_identity="artifact",
        semantics="test-radians",
        blendable_components=("shoulder.pos", "elbow.pos"),
    )
    yield worker
    worker.close()


def aligned_open(worker, *, blend=False):
    request = open_request(worker)
    request.body["chunk_settings"] = chunk_settings(
        "aligned", 2 if blend else 0, 0.5, ["elbow.pos"] if blend else []
    )
    request.body["required_capabilities"] = required_chunk_capabilities(request.body["chunk_settings"])
    return request


def client_for(worker, config, descriptor=None):
    transport = SimpleNamespace(
        subscribe_liveliness=lambda *_: SimpleNamespace(),
        subscribe=lambda *args, **kwargs: SimpleNamespace(),
        declare_token=lambda *_: SimpleNamespace(),
        wait_for_subscriber=lambda *_: None,
    )
    return RemoteClient(transport, config, worker.descriptor if descriptor is None else descriptor)


def admit(client):
    caps = client.capabilities
    client.admit(
        features=caps.features,
        action_feature=caps.action_feature,
        semantics="test-radians",
        action_interval=caps.action_interval,
        mode=client.config.mode,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_merge": "unknown"},
        {"blend_steps": 2},
        {"chunk_merge": "aligned", "blend_steps": 2},
        {"chunk_merge": "aligned", "blend_components": ["shoulder.pos"]},
        {"chunk_merge": "aligned", "blend_steps": -1},
        {"chunk_merge": "aligned", "blend_steps": True},
        {"chunk_merge": "aligned", "blend_weight": float("nan")},
        {"chunk_merge": "aligned", "blend_weight": 0},
        {"chunk_merge": "aligned", "blend_weight": 1.1},
        {"chunk_merge": "aligned", "blend_steps": 2, "blend_components": ["shoulder.pos", "shoulder.pos"]},
        {"chunk_merge": "aligned", "mode": "rtc_guided"},
        {"chunk_merge": "aligned", "mode": "rtc_trained"},
    ],
)
def test_invalid_client_settings_fail_before_connect(kwargs):
    with pytest.raises(ValueError):
        client_config(**kwargs)


@pytest.mark.parametrize("components", [["missing"], ["shoulder.pos", "shoulder.pos"], [""]])
def test_server_configuration_rejects_ambiguous_blend_components(worker, components):
    with pytest.raises(ValueError, match="blendable_components"):
        ServerConfig(
            deployment="test",
            model=ModelConfig(repo_or_path="local"),
            semantics="test-radians",
            features=list(worker.runner.capabilities.features),
            action_feature=worker.runner.capabilities.action_feature,
            execution=ExecutionConfig(blendable_components=components),
        )


@pytest.mark.parametrize("blend", [False, True])
def test_admission_echoes_exact_merge_settings_and_explicit_components(worker, blend):
    descriptor = worker.descriptor
    assert descriptor["execution_contracts"] == [CHUNK_ALIGNMENT, CHUNK_BLENDING]
    assert descriptor["blendable_components"] == ["shoulder.pos", "elbow.pos"]
    request = aligned_open(worker, blend=blend)
    response = worker.submit(request).result(2)
    assert response.message_type is MessageType.ACCEPTED
    assert response.body["chunk_settings"] == request.body["chunk_settings"]


def test_deployment_without_components_does_not_advertise_blending():
    worker = SessionWorker(
        runner_for(ConformingPolicy(tiny_config())), deployment="test", artifact_identity="a", semantics="s"
    )
    try:
        assert worker.descriptor["execution_contracts"] == [CHUNK_ALIGNMENT]
        assert worker.descriptor["blendable_components"] == []
        assert_error(worker.submit(aligned_open(worker, blend=True)).result(2), ErrorCode.UNSUPPORTED)
    finally:
        worker.close()


@pytest.mark.parametrize(
    "updates",
    [
        {"blend_components": ["gripper.pos"]},
        {"blend_steps": 4},
        {"blend_weight": float("inf")},
        {"blend_components": "elbow.pos"},
        {"extra_option": True},
        {"chunk_merge": "append"},
    ],
)
def test_server_rejects_invalid_merge_contract_before_allocating_session(worker, updates):
    request = aligned_open(worker, blend=True)
    request.body["chunk_settings"].update(updates)
    assert_error(worker.submit(request).result(2), ErrorCode.INCOMPATIBLE)
    assert worker.session_id is None


def test_alignment_cannot_omit_its_required_capability(worker):
    request = aligned_open(worker)
    del request.body["required_capabilities"]
    assert_error(worker.submit(request).result(2), ErrorCode.INCOMPATIBLE)
    assert worker.session_id is None


@pytest.mark.parametrize("alteration", ["noncanonical", "unnamed", "integer"])
def test_blending_requires_named_canonical_float_coordinates(worker, alteration):
    caps = worker.runner.capabilities
    if alteration == "noncanonical":
        caps = replace(caps, action_representation="relative")
    elif alteration == "unnamed":
        caps = replace(caps, action_feature=replace(caps.action_feature, names=()))
    else:
        caps = replace(caps, action_feature=replace(caps.action_feature, dtype="int32"))
    with pytest.raises(ValueError):
        validate_chunk_contract(chunk_settings("aligned", 2, 0.5, ["elbow.pos"]), caps, ["elbow.pos"])


def test_old_server_is_rejected_before_open_for_alignment(worker):
    descriptor = worker.descriptor
    del descriptor["execution_contracts"]
    del descriptor["blendable_components"]
    client = client_for(worker, client_config(chunk_merge="aligned"), descriptor)
    with pytest.raises(ProtocolError, match="update the server") as error:
        admit(client)
    assert error.value.code is ErrorCode.UNSUPPORTED
    assert not client.session_id
    assert worker.session_id is None


def test_append_open_remains_compatible_with_descriptor_without_new_fields(worker, monkeypatch):
    descriptor = worker.descriptor
    del descriptor["execution_contracts"]
    del descriptor["blendable_components"]
    client = client_for(worker, client_config(), descriptor)

    def query(key, request, timeout, expected):
        assert "chunk_settings" not in request.body
        assert "required_capabilities" not in request.body
        accepted = worker.submit(request).result(2)
        del accepted.body["chunk_settings"]
        return accepted

    monkeypatch.setattr(client, "_query", query)
    monkeypatch.setattr(client, "control", lambda *_: None)
    admit(client)
    assert client.session_id


def test_client_resolves_components_and_checks_exact_acceptance(worker, monkeypatch):
    client = client_for(
        worker,
        client_config(chunk_merge="aligned", blend_steps=2, blend_components=["elbow.pos", "shoulder.pos"]),
    )
    monkeypatch.setattr(
        client, "_query", lambda key, request, timeout, expected: worker.submit(request).result(2)
    )
    monkeypatch.setattr(client, "control", lambda *_: None)
    admit(client)
    assert client.blend_indices == (1, 0)


@pytest.mark.parametrize("change", ["omit", "downgrade", "weight"])
def test_server_cannot_silently_change_accepted_merge_settings(worker, monkeypatch, change):
    client = client_for(worker, client_config(chunk_merge="aligned"))

    def query(key, request, timeout, expected):
        accepted = worker.submit(request).result(2)
        if change == "omit":
            del accepted.body["chunk_settings"]
        else:
            accepted.body["chunk_settings"] = {
                **accepted.body["chunk_settings"],
                **({"chunk_merge": "append"} if change == "downgrade" else {"blend_weight": 0.9}),
            }
        return accepted

    monkeypatch.setattr(client, "_query", query)
    with pytest.raises(ProtocolError, match="changed the advertised contract"):
        admit(client)
    assert not client.session_id


@pytest.mark.parametrize(
    "observation_cursor,cursor", [(None, 2), (2, None), (-1, 2), (3, 2), (True, 2), (2, 2**63)]
)
def test_aligned_request_rejects_invalid_cursors_before_policy_call(worker, observation_cursor, cursor):
    accepted = worker.submit(aligned_open(worker)).result(2)
    request = action_request(worker, accepted.session_id)
    request.body.update(observation_cursor=observation_cursor, cursor=cursor)
    assert_error(worker.submit(request).result(2), ErrorCode.MALFORMED)
    assert not worker.runner.policy.last_kwargs


def test_action_echoes_both_aligned_cursors_and_client_rejects_rebinding(worker):
    accepted = worker.submit(aligned_open(worker)).result(2)
    request = action_request(worker, accepted.session_id)
    request.body.update(observation_cursor=2, cursor=4)
    result = worker.submit(request).result(2)
    assert result.message_type is MessageType.ACTION
    assert result.body["observation_cursor"] == 2
    assert result.body["cursor"] == 4
    client = client_for(worker, client_config(chunk_merge="aligned"))
    client._validate_context(result, request.body, MessageType.ACTION)
    result.body["observation_cursor"] = 3
    with pytest.raises(ProtocolError, match="context differs"):
        client._validate_context(result, request.body, MessageType.ACTION)


def test_server_rejects_alignment_for_rtc_session(worker):
    worker.runner.capabilities = replace(
        worker.runner.capabilities, modes=(ExecutionMode.CHUNK, ExecutionMode.RTC_GUIDED)
    )
    request = aligned_open(worker)
    request.body["mode"] = "rtc_guided"
    assert_error(worker.submit(request).result(2), ErrorCode.INCOMPATIBLE)
