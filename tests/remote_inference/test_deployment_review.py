# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Deployment mode selection and portable artifact pins, without remote services."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from shutil import copytree
from types import SimpleNamespace

import pytest

from lerobot.inference.contracts import ExecutionMode, FeatureSpec
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.remote_inference.configs import ExecutionConfig, ModelConfig, ServerConfig
from lerobot.scripts import lerobot_policy_server as serving
from lerobot.transport.zenoh import ZenohConfig
from tests.inference.test_policy_runner import ConformingPolicy, observation, processors, tiny_config


@pytest.fixture
def deployment(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    config = tiny_config()
    ACTPolicy(config).save_pretrained(checkpoint)
    pre, post = processors(config)
    pre.save_pretrained(checkpoint)
    post.save_pretrained(checkpoint)
    return ServerConfig(
        deployment="test",
        model=ModelConfig(str(checkpoint)),
        execution=ExecutionConfig(warmup_calls=1),
        zenoh=ZenohConfig(listen_endpoints=["tcp/127.0.0.1:7447"]),
        semantics="radians",
        features=[FeatureSpec(key, (3,), "float32", semantics="radians") for key in config.input_features],
        action_feature=FeatureSpec("action", (3,), "float32", semantics="radians"),
    )


def test_identical_checkpoint_pin_survives_host_path_and_transport_changes(deployment, tmp_path):
    _, original = serving.load_deployment(deployment)
    destination = tmp_path / "other-host-cache" / "checkpoint"
    copytree(deployment.model.repo_or_path, destination)
    relocated = replace(
        deployment,
        deployment="other-robot",
        model=replace(deployment.model, repo_or_path=str(destination)),
        execution=replace(deployment.execution, idle_timeout_s=20, action_deadline_s=7, warmup_calls=2),
        language=replace(deployment.language, deadline_s=80),
        zenoh=ZenohConfig(mode="client", connect_endpoints=["tcp/router.example:7447"]),
        log_level="DEBUG",
    )
    _, moved = serving.load_deployment(relocated)
    assert moved == original


def test_identity_excludes_compute_placement_but_preserves_inference_contract(deployment):
    config = tiny_config()
    first = serving._inference_identity(deployment, config)
    elsewhere = deepcopy(config)
    elsewhere.device = "cuda:1"
    elsewhere.pretrained_path = "/another/cache/path"
    assert serving._inference_identity(deployment, elsewhere) == first
    assert serving._inference_identity(replace(deployment, semantics="degrees"), config) != first
    assert (
        serving._inference_identity(
            replace(deployment, execution=replace(deployment.execution, action_fps=20)), config
        )
        != first
    )
    changed = deepcopy(config)
    changed.n_action_steps = 2
    assert serving._inference_identity(deployment, changed) != first


def test_requested_rtc_is_attached_without_a_saved_dataclass_field(deployment, monkeypatch):
    def load_policy(path, config):
        assert config.rtc_config.enabled
        return ConformingPolicy(config)

    monkeypatch.setattr(
        serving, "get_policy_class", lambda name: SimpleNamespace(from_pretrained=load_policy)
    )
    requested = replace(
        deployment,
        execution=replace(
            deployment.execution,
            supported_modes=["chunk", "rtc_guided"],
            rtc=RTCConfig(execution_horizon=4),
            warmup_calls=2,
        ),
    )
    runner, _ = serving.load_deployment(requested)
    assert runner.capabilities.modes == (ExecutionMode.CHUNK, ExecutionMode.RTC_GUIDED)
    assert runner.policy.config.rtc_config is not requested.execution.rtc


def test_chunk_only_ignores_unrelated_deployment_rtc_settings(deployment):
    requested = replace(
        deployment,
        execution=replace(deployment.execution, rtc=RTCConfig(mode="trained", execution_horizon=4)),
    )
    runner, identity = serving.load_deployment(requested)
    assert runner.capabilities.modes == (ExecutionMode.CHUNK,)
    _, default_identity = serving.load_deployment(deployment)
    assert identity == default_identity


def test_requesting_rtc_does_not_invent_policy_capability(deployment):
    requested = replace(
        deployment,
        execution=replace(
            deployment.execution,
            supported_modes=["rtc_guided"],
            rtc=RTCConfig(execution_horizon=4),
        ),
    )
    with pytest.raises(ValueError, match="Unsupported execution mode"):
        serving.load_deployment(requested)


@pytest.mark.parametrize("deployment_name", ["lab/arm", "*", "", "has space"])
def test_invalid_deployment_name_fails_in_config(deployment, deployment_name):
    with pytest.raises(ValueError):
        replace(deployment, deployment=deployment_name)


@pytest.mark.parametrize("modes", [[], ["chunks"], ["chunk", "chunk"], ["rtc_trained"]])
def test_invalid_or_mismatched_serving_modes_fail_in_config(deployment, modes):
    with pytest.raises(ValueError):
        replace(deployment, execution=replace(deployment.execution, supported_modes=modes))


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_text_limits_and_warmup_counts_are_positive_integers(deployment, value):
    with pytest.raises(ValueError, match="positive integer"):
        replace(deployment, language=replace(deployment.language, max_output_chars=value))
    with pytest.raises(ValueError, match="positive integer"):
        replace(deployment, language=replace(deployment.language, max_input_chars=value))
    with pytest.raises(ValueError, match="warmup"):
        replace(deployment, execution=replace(deployment.execution, warmup_calls=value))


@pytest.mark.parametrize(
    "transport",
    [ZenohConfig(), ZenohConfig(mode="client", listen_endpoints=["tcp/127.0.0.1:7447"])],
)
def test_transport_configuration_fails_before_resolving_checkpoint(deployment, monkeypatch, transport):
    def unexpected_download(*args, **kwargs):
        pytest.fail("transport validation must precede checkpoint resolution")

    monkeypatch.setattr(serving, "resolve_artifact", unexpected_download)
    with pytest.raises(ValueError):
        serving.load_deployment(replace(deployment, zenoh=transport))


def test_server_owned_execution_slice_does_not_modify_checkpoint(deployment):
    checkpoint_config = Path(deployment.model.repo_or_path) / "config.json"
    original = checkpoint_config.read_bytes()
    runner, default_identity = serving.load_deployment(deployment)
    requested = replace(deployment, execution=replace(deployment.execution, n_action_steps=2))
    shorter, shorter_identity = serving.load_deployment(requested)
    assert runner.capabilities.execution_steps == 3
    assert shorter.capabilities.prediction_steps == 8
    assert shorter.capabilities.execution_steps == 2
    assert shorter.predict(observation()).canonical_actions.shape == (2, 3)
    assert shorter_identity != default_identity
    assert checkpoint_config.read_bytes() == original


@pytest.mark.parametrize("steps", [0, -1, True, 2.5])
def test_invalid_execution_slice_is_rejected_in_config(deployment, steps):
    with pytest.raises(ValueError, match="positive integer"):
        replace(deployment, execution=replace(deployment.execution, n_action_steps=steps))


def test_slice_override_is_restricted_to_plain_chunks(deployment):
    with pytest.raises(ValueError, match="chunk-only"):
        replace(
            deployment,
            execution=replace(deployment.execution, n_action_steps=2, supported_modes=["rtc_guided"]),
        )


def test_oversized_slice_fails_before_loading_weights(deployment, monkeypatch):
    monkeypatch.setattr(serving, "get_policy_class", lambda name: pytest.fail("must not load policy"))
    with pytest.raises(ValueError, match="prediction horizon"):
        serving.load_deployment(
            replace(deployment, execution=replace(deployment.execution, n_action_steps=9))
        )


def test_native_loader_cannot_silently_replace_requested_slice(deployment, monkeypatch):
    def load_policy(path, config):
        config.n_action_steps = 3
        return ConformingPolicy(config)

    monkeypatch.setattr(
        serving, "get_policy_class", lambda name: SimpleNamespace(from_pretrained=load_policy)
    )
    with pytest.raises(ValueError, match="does not honor"):
        serving.load_deployment(
            replace(deployment, execution=replace(deployment.execution, n_action_steps=2))
        )
