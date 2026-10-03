# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""Reusable current-observation runner conformance and real processor coverage."""

import time
from collections import deque
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.inference.contracts import ExecutionMode, FeatureSpec, ObservationSnapshot, QueryKind
from lerobot.inference.policy_runner import PolicyRunner
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.processor import AbsoluteActionsProcessorStep, RelativeActionsProcessorStep
from lerobot.rollout.inference.base import QueryKind as LegacyQueryKind
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE, QUERY_KIND, QUERY_TEXT


def tiny_config(**kwargs):
    return ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(3,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
        normalization_mapping={
            FeatureType.STATE: NormalizationMode.MEAN_STD,
            FeatureType.ENV: NormalizationMode.IDENTITY,
            FeatureType.ACTION: NormalizationMode.MEAN_STD,
        },
        device="cpu",
        chunk_size=8,
        n_action_steps=3,
        use_vae=False,
        dim_model=32,
        n_heads=2,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        **kwargs,
    )


def processors(config, relative=False):
    stats = {
        OBS_STATE: {"mean": torch.zeros(3), "std": torch.ones(3)},
        ACTION: {"mean": torch.tensor([1.0, 2.0, 3.0]), "std": torch.ones(3) * 2},
    }
    pre, post = make_act_pre_post_processors(config, stats)
    if relative:
        step = RelativeActionsProcessorStep(enabled=True)
        pre.steps.insert(3, step)
        post.steps.insert(1, AbsoluteActionsProcessorStep(enabled=True, relative_step=step))
    return pre, post


def runner_for(policy, *, relative=False, modes=(ExecutionMode.CHUNK,), **kwargs):
    return PolicyRunner(
        policy,
        *processors(policy.config, relative=relative),
        action_interval=1 / 30,
        features=tuple(
            FeatureSpec(key, (3,), "float32", semantics="joint positions in radians")
            for key in (OBS_STATE, OBS_ENV_STATE)
        ),
        action_feature=FeatureSpec(ACTION, (3,), "float32", semantics="joint positions in radians"),
        modes=modes,
        **kwargs,
    )


def observation(value=1.0):
    return ObservationSnapshot(
        {OBS_STATE: np.full(3, value, dtype=np.float32), OBS_ENV_STATE: np.zeros(3, dtype=np.float32)},
        capture_time=10.0,
        task="pick up the cube",
        task_version=4,
        observation_id="obs-1",
    )


class ConformingPolicy(PreTrainedPolicy):
    """A new family exercises the public contract without a serving allowlist."""

    config_class = ACTConfig
    name = "conformance"

    def __init__(self, config):
        super().__init__(config)
        self.resets = 0
        self.last_kwargs = {}
        self.reset()

    def get_optim_params(self):
        return {}

    def reset(self):
        self.resets += 1
        self._action_queue = deque()

    def forward(self, batch):
        raise NotImplementedError

    def select_action(self, batch, **kwargs):
        raise AssertionError("The runner must not call select_action")

    def predict_action_chunk(self, batch, **kwargs):
        self.last_kwargs = kwargs
        return torch.arange(24, dtype=torch.float32).reshape(1, 8, 3)

    def supports_rtc(self):
        return True

    def supports_text_generation(self):
        return True

    def generate_text(self, batch):
        assert batch[QUERY_KIND] == "vqa"
        assert batch[QUERY_TEXT] == "What is visible?"
        return "a cube"


def test_real_act_matches_canonical_processor_pipeline_and_execution_slice():
    policy = ACTPolicy(tiny_config())
    runner = runner_for(policy)
    source = observation()
    with torch.inference_mode():
        prepared = runner.preprocessor(
            {key: torch.from_numpy(value.copy()).unsqueeze(0) for key, value in source.features.items()}
        )
        expected = runner.postprocessor(policy.predict_action_chunk(prepared))[0, :3]
    actual = runner.predict(source)
    torch.testing.assert_close(actual.canonical_actions, expected)
    assert actual.canonical_actions.shape == (3, 3)
    assert actual.execution_steps == 3
    assert actual.model_actions is None
    assert actual.provenance.task == source.task
    assert actual.provenance.capture_time == 10.0
    assert set(actual.server_durations) == {"preprocessing", "policy", "postprocessing"}


class GuidedConformingPolicy(ConformingPolicy):
    """Exercise the real autograd guidance rather than only RTC keyword forwarding."""

    name = "guided_conformance"

    def __init__(self, config):
        super().__init__(config)
        self.guided_calls = 0
        self.rtc = RTCProcessor(config.rtc_config)

    @torch.no_grad()
    def predict_action_chunk(self, batch, **kwargs):
        prefix = kwargs.get("prev_chunk_left_over")
        result = self.rtc.denoise_step(
            x_t=torch.ones(1, 8, 3),
            prev_chunk_left_over=prefix,
            inference_delay=kwargs.get("inference_delay", 0),
            time=0.5,
            original_denoise_step_partial=lambda x: x * 0.25,
        )
        if prefix is not None:
            self.guided_calls += 1
        return result


def test_runner_guided_rtc_allows_real_autograd_on_successor_chunk():
    config = tiny_config()
    config.rtc_config = RTCConfig(execution_horizon=4)
    policy = GuidedConformingPolicy(config)
    runner = runner_for(policy, modes=(ExecutionMode.RTC_GUIDED,))
    first = runner.predict(observation(), mode=ExecutionMode.RTC_GUIDED)
    second = runner.predict(
        observation(),
        mode=ExecutionMode.RTC_GUIDED,
        inference_delay=1,
        model_continuation=first.model_actions,
        canonical_continuation=first.canonical_actions,
    )
    assert policy.guided_calls == 1
    assert torch.isfinite(second.canonical_actions).all()
    assert not second.canonical_actions.requires_grad
    assert not torch.equal(first.canonical_actions, second.canonical_actions)


def test_local_guided_rtc_allows_real_autograd_on_successor_chunk():
    pytest.importorskip("datasets")
    from lerobot.rollout.inference.rtc import RTCInferenceEngine

    config = tiny_config()
    config.rtc_config = RTCConfig(execution_horizon=4)
    policy = GuidedConformingPolicy(config)
    state_names = ("a.pos", "b.pos", "c.pos")
    env_names = ("env_a", "env_b", "env_c")
    engine = RTCInferenceEngine(
        policy,
        *processors(config),
        robot_wrapper=SimpleNamespace(robot_type="test", action_features=dict.fromkeys(state_names, float)),
        rtc_config=config.rtc_config,
        dataset_features={
            OBS_STATE: {"dtype": "float32", "shape": (3,), "names": state_names},
            OBS_ENV_STATE: {"dtype": "float32", "shape": (3,), "names": env_names},
        },
        task="pick up the cube",
        fps=30,
        device="cpu",
        rtc_queue_threshold=4,
    )
    engine.start()
    try:
        engine.resume()
        engine.notify_observation({**dict.fromkeys(state_names, 1.0), **dict.fromkeys(env_names, 0.0)})
        deadline = time.monotonic() + 2
        while engine.action_queue.empty() and not engine.failed and time.monotonic() < deadline:
            time.sleep(0.002)
        assert engine.action_queue.qsize() == 8, engine.failure_traceback
        for _ in range(4):
            assert engine.get_action(None) is not None
        while not policy.guided_calls and not engine.failed and time.monotonic() < deadline:
            time.sleep(0.002)
        assert not engine.failed, engine.failure_traceback
        assert policy.guided_calls == 1
    finally:
        engine.stop()


def test_generic_family_uses_real_canonical_processors_and_honors_n_action_steps():
    runner = runner_for(ConformingPolicy(tiny_config()))
    result = runner.predict(observation())
    torch.testing.assert_close(
        result.canonical_actions, torch.arange(9).reshape(3, 3) * 2.0 + torch.tensor([1, 2, 3])
    )


def test_snapshot_owns_recycled_camera_and_state_buffers():
    array = np.zeros(3, dtype=np.float32)
    snap = ObservationSnapshot({OBS_STATE: array}, 1.0, "task")
    array[:] = 99
    assert not snap.features[OBS_STATE].any()
    with pytest.raises(ValueError):
        snap.features[OBS_STATE][:] = 1
    with pytest.raises(TypeError):
        snap.features[OBS_STATE] = array


@pytest.mark.parametrize(
    "attribute,value", [("n_obs_steps", 2), ("temporal_ensemble_coeff", 0.01), ("use_visual_memory", True)]
)
def test_rejects_configuration_requiring_another_prediction_cadence(attribute, value):
    policy = ConformingPolicy(tiny_config())
    setattr(policy.config, attribute, value)
    with pytest.raises(ValueError, match="history|ensembling|memory"):
        runner_for(policy)


def test_relative_rtc_reanchors_from_canonical_continuation_at_new_state():
    policy = ConformingPolicy(tiny_config())
    policy.config.rtc_config = RTCConfig(execution_horizon=4)
    runner = runner_for(policy, relative=True, modes=(ExecutionMode.RTC_GUIDED,))
    first = runner.predict(observation(10), mode=ExecutionMode.RTC_GUIDED)
    second = runner.predict(
        observation(20),
        mode=ExecutionMode.RTC_GUIDED,
        model_continuation=first.model_actions[3:],
        canonical_continuation=first.canonical_actions[3:],
        inference_delay=1,
    )
    expected = (first.canonical_actions[3:7] - 20 - torch.tensor([1, 2, 3])) / 2
    torch.testing.assert_close(policy.last_kwargs["prev_chunk_left_over"], expected)
    assert second.canonical_actions.shape == (8, 3), "RTC must retain the continuation horizon"


def test_trained_rtc_rejects_prefix_beyond_checkpoint_or_available_actions():
    policy = ConformingPolicy(tiny_config())
    policy.config.rtc_config = RTCConfig(mode="trained", execution_horizon=4)
    policy.config.rtc_training_max_delay = 3
    runner = runner_for(policy, modes=(ExecutionMode.RTC_TRAINED,))
    for delay, available in [(4, 5), (3, 2)]:
        with pytest.raises(ValueError, match="limits"):
            runner.predict(
                observation(),
                mode=ExecutionMode.RTC_TRAINED,
                inference_delay=delay,
                model_continuation=torch.zeros(available, 3),
            )


def test_language_processor_isolation_and_motion_invalidation_preserve_action_anchor():
    policy = ConformingPolicy(tiny_config())
    runner = runner_for(policy, relative=True)
    runner.predict(observation(10))
    state_before = runner._relative_step.get_cached_state().clone()
    assert runner.query(observation(20), kind="vqa", text="What is visible?") == "a cube"
    torch.testing.assert_close(runner._relative_step.get_cached_state(), state_before)
    policy._action_queue.append(torch.ones(3))
    runner.reset(full=False)
    assert policy.resets == 1
    assert not policy._action_queue
    torch.testing.assert_close(runner._relative_step.get_cached_state(), state_before)
    runner.reset()
    assert policy.resets == 2
    assert runner._relative_step.get_cached_state() is None


@pytest.mark.parametrize("kind", list(QueryKind))
def test_language_query_kind_contract_preserves_public_enum_and_policy_strings(kind, monkeypatch):
    assert LegacyQueryKind is QueryKind
    policy = ConformingPolicy(tiny_config())
    runner = runner_for(policy)

    def generate_text(batch):
        assert batch[QUERY_KIND] == kind.value
        return "a cube"

    monkeypatch.setattr(policy, "generate_text", generate_text)
    assert runner.query(observation(), kind=kind.value, text="What is visible?") == "a cube"


def test_mapping_is_not_applied_twice():
    policy = ConformingPolicy(tiny_config())
    pre, post = processors(policy.config)
    pre.steps[0].rename_map = {"observation.robot": OBS_STATE}
    with pytest.raises(ValueError, match="already mapped"):
        PolicyRunner(
            policy,
            pre,
            post,
            action_interval=0.03,
            features=runner_for(policy).capabilities.features,
            action_feature=runner_for(policy).capabilities.action_feature,
        )


@pytest.mark.parametrize("shape,dtype", [((2,), np.float32), ((3,), np.float64)])
def test_observation_must_match_negotiated_shape_and_dtype(shape, dtype):
    runner = runner_for(ConformingPolicy(tiny_config()))
    source = observation()
    features = dict(source.features)
    features[OBS_STATE] = np.zeros(shape, dtype=dtype)
    with pytest.raises(ValueError, match="shape/dtype"):
        runner.predict(ObservationSnapshot(features, 10, source.task))


def test_feature_metadata_rejects_implicit_semantics_and_wrong_rgb_order():
    with pytest.raises(ValueError, match="semantic"):
        FeatureSpec(OBS_STATE, (3,), "float32")
    with pytest.raises(ValueError, match="HWC"):
        FeatureSpec("observation.images.front", (3, 10, 10), "uint8", "rgb", semantics="RGB pixels")


def test_relative_exclusion_uses_explicit_action_names_and_rejects_permuted_state():
    policy = ConformingPolicy(tiny_config())
    pre, post = processors(policy.config, relative=True)
    relative = next(step for step in pre.steps if isinstance(step, RelativeActionsProcessorStep))
    relative.exclude_joints = ["gripper.pos"]
    action_names = ("joint_a.pos", "joint_b.pos", "gripper.pos")
    contract = runner_for(policy).capabilities
    features = tuple(
        replace(feature, names=action_names) if feature.name == OBS_STATE else feature
        for feature in contract.features
    )
    action_feature = replace(contract.action_feature, names=action_names)
    runner = PolicyRunner(
        policy, pre, post, action_interval=0.03, features=features, action_feature=action_feature
    )
    assert relative.action_names == list(action_names)
    torch.testing.assert_close(
        runner.predict(observation(10)).canonical_actions[0], torch.tensor([11.0, 14.0, 7.0])
    )

    reversed_features = tuple(
        replace(feature, names=tuple(reversed(action_names))) if feature.name == OBS_STATE else feature
        for feature in features
    )
    with pytest.raises(ValueError, match="must be aligned"):
        PolicyRunner(
            policy, pre, post, action_interval=0.03, features=reversed_features, action_feature=action_feature
        )


def test_trained_rtc_horizon_covers_declared_checkpoint_delay():
    policy = ConformingPolicy(tiny_config())
    policy.config.rtc_config = RTCConfig(mode="trained", execution_horizon=2)
    policy.config.rtc_training_max_delay = 3
    with pytest.raises(ValueError, match="maximum conditioned delay"):
        runner_for(policy, modes=(ExecutionMode.RTC_TRAINED,))


def test_deployment_can_disable_text_and_cannot_enable_an_unsupported_head():
    policy = ConformingPolicy(tiny_config())
    runner = runner_for(policy, language_enabled=False)
    assert not runner.capabilities.language
    with pytest.raises(ValueError, match="does not support"):
        runner.query(observation(), kind="vqa", text="What is visible?")
    policy.supports_text_generation = lambda: False
    with pytest.raises(ValueError, match="has no text capability"):
        runner_for(policy, language_enabled=True)


def test_real_act_plain_async_matches_runner_and_honors_execution_slice():
    pytest.importorskip("datasets")
    from lerobot.rollout.inference.rtc import RTCInferenceEngine

    policy = ACTPolicy(tiny_config())
    expected = runner_for(policy).predict(observation()).canonical_actions
    state_names = ("a.pos", "b.pos", "c.pos")
    env_names = ("env_a", "env_b", "env_c")
    engine = RTCInferenceEngine(
        policy,
        *processors(policy.config),
        robot_wrapper=SimpleNamespace(robot_type="test", action_features=dict.fromkeys(state_names, float)),
        rtc_config=RTCConfig(enabled=False),
        dataset_features={
            OBS_STATE: {"dtype": "float32", "shape": (3,), "names": state_names},
            OBS_ENV_STATE: {"dtype": "float32", "shape": (3,), "names": env_names},
        },
        task="pick up the cube",
        fps=30,
        device="cpu",
        rtc_queue_threshold=0,
    )
    engine.start()
    try:
        engine.resume()
        engine.notify_observation({**dict.fromkeys(state_names, 1.0), **dict.fromkeys(env_names, 0.0)})
        deadline = time.monotonic() + 2
        while engine.action_queue.empty() and not engine.failed and time.monotonic() < deadline:
            time.sleep(0.002)
        assert not engine.failed, engine.failure_traceback
        assert engine.action_queue.qsize() == 3
        actual = torch.stack([engine.get_action(None) for _ in range(3)])
        torch.testing.assert_close(actual, expected)
    finally:
        engine.stop()
