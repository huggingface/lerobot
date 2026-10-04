# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Regression checks at the checkpoint/model/canonical action boundary."""

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lerobot.inference import ExecutionMode, FeatureSpec, PolicyRunner
from lerobot.policies.evo1.processor_evo1 import Evo1ActionProcessorStep
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE
from tests.inference.test_policy_runner import ConformingPolicy, observation, processors, tiny_config

NAMES = ("shoulder.pos", "elbow.pos", "gripper.pos")


def make_runner(policy, *, state_names=NAMES, action_names=NAMES, padded=False, relative=False, **kwargs):
    pre, post = processors(policy.config, relative=relative)
    if padded:
        post.steps.insert(0, Evo1ActionProcessorStep(action_dim=3))
    return PolicyRunner(
        policy,
        pre,
        post,
        action_interval=1 / 30,
        features=(
            FeatureSpec(OBS_STATE, (3,), "float32", names=state_names, semantics="radians"),
            FeatureSpec(OBS_ENV_STATE, (3,), "float32", semantics="radians"),
        ),
        action_feature=FeatureSpec(ACTION, (3,), "float32", names=action_names, semantics="radians"),
        **kwargs,
    )


@pytest.mark.parametrize("names", [("elbow.pos", "shoulder.pos", "gripper.pos"), ()])
def test_checkpoint_order_is_enforced_without_relative_actions(names):
    config = tiny_config()
    config.action_feature_names = list(NAMES)
    with pytest.raises(ValueError, match="checkpoint action_feature_names"):
        make_runner(ConformingPolicy(config), action_names=names)


def test_state_permutation_is_rejected_even_without_checkpoint_names_or_relative_actions():
    with pytest.raises(ValueError, match="State and action component order"):
        make_runner(ConformingPolicy(tiny_config()), state_names=NAMES[::-1])


def test_checkpoint_order_accepts_matching_canonical_metadata():
    config = tiny_config()
    config.action_feature_names = list(NAMES)
    assert make_runner(ConformingPolicy(config)).predict(observation()).canonical_actions.shape == (3, 3)


class PaddedPolicy(ConformingPolicy):
    name = "padded_conformance"

    def __init__(self, config):
        super().__init__(config)
        self.width = 6

    def predict_action_chunk(self, batch, **kwargs):
        self.last_kwargs = kwargs
        return torch.arange(8 * self.width, dtype=torch.float32).reshape(1, 8, self.width)


def test_padded_prediction_uses_real_evo1_crop_before_canonical_validation():
    runner = make_runner(PaddedPolicy(tiny_config()), padded=True)
    chunk = runner.predict(observation())
    expected = torch.arange(48, dtype=torch.float32).reshape(8, 6)[:3, :3]
    expected = expected * 2 + torch.tensor([1.0, 2.0, 3.0])
    torch.testing.assert_close(chunk.canonical_actions, expected)
    assert chunk.model_actions is None
    assert runner.capabilities.model_action_dim == 6


def test_padded_rtc_keeps_all_model_columns_in_successor_prefix():
    config = tiny_config()
    config.rtc_config = RTCConfig(execution_horizon=4)
    policy = PaddedPolicy(config)
    runner = make_runner(policy, padded=True, modes=(ExecutionMode.RTC_GUIDED,))
    first = runner.predict(observation(), mode=ExecutionMode.RTC_GUIDED)
    assert first.model_actions.shape == (8, 6)
    assert first.canonical_actions.shape == (8, 3)
    second = runner.predict(
        replace(observation(), capture_time=11),
        mode=ExecutionMode.RTC_GUIDED,
        inference_delay=1,
        model_continuation=first.model_actions[-4:],
        canonical_continuation=first.canonical_actions[-4:],
    )
    torch.testing.assert_close(policy.last_kwargs["prev_chunk_left_over"], first.model_actions[-4:])
    assert second.model_actions.shape == (8, 6)
    policy.width = 7
    with pytest.raises(ValueError, match="width changed"):
        runner.predict(observation(), mode=ExecutionMode.RTC_GUIDED)


def test_width_changing_relative_rtc_requires_explicit_adapter():
    config = tiny_config()
    config.rtc_config = RTCConfig(execution_horizon=4)
    runner = make_runner(PaddedPolicy(config), padded=True, relative=True, modes=(ExecutionMode.RTC_GUIDED,))
    with pytest.raises(ValueError, match="different model/canonical widths"):
        runner.predict(observation(), mode=ExecutionMode.RTC_GUIDED)


def test_molmoact2_declaration_matches_its_actual_returned_execution_slice(monkeypatch):
    # The native wrapper is exercised without loading a transformer or weights.
    module = pytest.importorskip("lerobot.policies.molmoact2.modeling_molmoact2")
    config_module = pytest.importorskip("lerobot.policies.molmoact2.configuration_molmoact2")
    config = config_module.MolmoAct2Config(device="cpu", chunk_size=12, n_action_steps=5)
    policy = object.__new__(module.MolmoAct2Policy)
    torch.nn.Module.__init__(policy)
    policy.config = config
    policy.anchor = torch.nn.Parameter(torch.zeros(1))
    policy._checkpoint_action_mode = None
    monkeypatch.setattr(policy, "_model_inputs", lambda batch: {"state": torch.zeros(1, 3)})
    monkeypatch.setattr(policy, "_resolve_inference_action_mode", lambda mode: "continuous")
    monkeypatch.setattr(policy, "_output_action_dim", lambda batch: 3)
    monkeypatch.setattr(policy, "_autocast_context", nullcontext)
    monkeypatch.setattr(policy, "_rtc_enabled", lambda: False)
    monkeypatch.setattr(policy, "_generation_action_horizon", lambda: 12)
    monkeypatch.setattr(
        policy,
        "_backbone",
        lambda: SimpleNamespace(generate_actions_from_inputs=lambda **kwargs: torch.ones(1, 12, 32)),
    )
    spec = policy.chunk_inference_spec()
    actual = policy.predict_action_chunk({}, generator=torch.Generator())
    assert actual.shape == (1, spec.prediction_steps, 3)
    assert spec.prediction_steps == spec.execution_steps == 5
