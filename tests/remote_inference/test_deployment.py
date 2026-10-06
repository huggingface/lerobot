# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Deployment mode selection and portable artifact pins, without remote services."""

from dataclasses import replace
from pathlib import Path
from shutil import copytree
from types import SimpleNamespace

import pytest
import torch

from lerobot.inference import ExecutionMode, FeatureSpec
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.remote_inference.configs import ExecutionConfig, ModelConfig, ServerConfig
from lerobot.scripts import lerobot_policy_server as serving
from lerobot.transport.zenoh import ZenohConfig
from tests.inference.test_policy_runner import ConformingPolicy, observation, processors, tiny_config


@pytest.fixture(scope="module")
def deployment(tmp_path_factory):
    checkpoint = tmp_path_factory.mktemp("deployment") / "checkpoint"
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


def test_server_loads_saved_act_and_processors_warms_and_resets(deployment):
    policy = ACTPolicy.from_pretrained(deployment.model.repo_or_path)
    pre, post = processors(policy.config)
    runner, identity = serving.load_deployment(deployment)
    assert identity.startswith("sha256:") and len(identity) == 71
    assert not runner.capabilities.language
    assert not runner.policy._action_queue
    with torch.inference_mode():
        expected = post(policy.predict_action_chunk(pre(runner._batch(observation()))))[0, :3]
    torch.testing.assert_close(runner.predict(observation()).canonical_actions, expected)
    changed, changed_identity = serving.load_deployment(replace(deployment, semantics="degrees-v1"))
    assert changed_identity != identity
    assert changed.capabilities.execution_steps == 3
    _, debug_identity = serving.load_deployment(replace(deployment, log_level="DEBUG"))
    assert debug_identity == identity, "console verbosity must not invalidate a pinned artifact"


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
