#!/usr/bin/env python

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

from __future__ import annotations

import pytest
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.lerobot_types import TransitionKey
from lerobot.policies.lingbot_va.configuration_lingbot_va import LingBotVAConfig
from lerobot.policies.lingbot_va.processor_lingbot_va import (
    LingBotEpisodeAnchorStep,
    make_lingbot_va_pre_post_processors,
)
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    PolicyProcessorPipeline,
    RelativeActionsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


def _make_config(action_norm: NormalizationMode | None = None, **overrides) -> LingBotVAConfig:
    cfg = LingBotVAConfig(device="cpu", **overrides)
    if action_norm is not None:
        cfg.normalization_mapping = {**cfg.normalization_mapping, FeatureType.ACTION.value: action_norm}
    cfg.input_features = {f"{OBS_IMAGES}.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128))}
    cfg.output_features = {}
    cfg.validate_features()
    return cfg


def test_make_pre_post_processors_names_and_steps() -> None:
    cfg = _make_config()
    pre, post = make_lingbot_va_pre_post_processors(cfg, dataset_stats=None)
    assert pre.name == POLICY_PREPROCESSOR_DEFAULT_NAME
    assert post.name == POLICY_POSTPROCESSOR_DEFAULT_NAME
    # Actions are unnormalized by the standard built-in quantile unnormalizer.
    assert any(isinstance(s, UnnormalizerProcessorStep) for s in post.steps)


def test_freshly_built_postprocessor_is_identity() -> None:
    # Without action stats the quantile unnormalizer is a no-op (identity passthrough): the real
    # per-benchmark q01/q99 are restored from the saved checkpoint on load, not hardcoded here.
    cfg = _make_config()
    _, post = make_lingbot_va_pre_post_processors(cfg, dataset_stats=None)
    normed = torch.tensor([[0.3, -0.5, 1.0, -1.0, 0.0, 0.7, -0.2]])
    assert torch.allclose(post(normed), normed, atol=1e-6)


def test_postprocessor_quantile_unnormalization() -> None:
    # QUANTILES unnormalize maps [-1, 1] -> [q01, q99]: -1 -> q01, +1 -> q99.
    # The unnormalizer's norm_map is config-driven (kept symmetric with the preprocessor's
    # normalizer), so ACTION has to actually be QUANTILES here -- the class default is IDENTITY.
    cfg = _make_config(action_norm=NormalizationMode.QUANTILES)
    q01 = [-1.0, -0.5, 0.0, -1.0, -1.0, -1.0, -1.0]
    q99 = [1.0, 0.5, 2.0, 1.0, 1.0, 1.0, 1.0]
    stats = {ACTION: {"q01": q01, "q99": q99}}
    _, post = make_lingbot_va_pre_post_processors(cfg, dataset_stats=stats)
    out_lo = post(torch.full((1, 7), -1.0))
    out_hi = post(torch.full((1, 7), 1.0))
    assert torch.allclose(out_lo, torch.tensor(q01).unsqueeze(0), atol=1e-4)
    assert torch.allclose(out_hi, torch.tensor(q99).unsqueeze(0), atol=1e-4)


def test_postprocessor_stats_survive_save_load(tmp_path) -> None:
    # Regression guard for the Hub mechanism: the q01/q99 stats live in the saved post-processor
    # state and must round-trip through save_pretrained / from_pretrained.
    cfg = _make_config(action_norm=NormalizationMode.QUANTILES)
    q01 = [-0.6, -0.8, -0.9, -0.1, -0.15, -0.25, -1.0]
    q99 = [0.9, 0.85, 0.9, 0.17, 0.18, 0.34, 1.0]
    _, post = make_lingbot_va_pre_post_processors(cfg, dataset_stats={ACTION: {"q01": q01, "q99": q99}})
    post.save_pretrained(tmp_path)
    loaded = PolicyProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    out = loaded(torch.full((1, 7), -1.0))
    assert torch.allclose(out, torch.tensor(q01).unsqueeze(0), atol=1e-4)


# --- episode action anchoring -------------------------------------------------------------------


def _anchor_names() -> list[str]:
    return ["right_joint_1.pos", "right_gripper.pos", "left_joint_1.pos", "left_gripper.pos"]


def _anchor_step(**overrides) -> LingBotEpisodeAnchorStep:
    kwargs = {"enabled": True, "exclude_joints": ["gripper"], "action_names": _anchor_names()}
    kwargs.update(overrides)
    return LingBotEpisodeAnchorStep(**kwargs)


def test_anchor_steps_present_but_disabled_when_anchor_is_none() -> None:
    # The steps are always in the pipeline so they get serialised into the checkpoint's processor
    # config; a later run can then flip them on by override instead of relying on injection.
    pre, post = make_lingbot_va_pre_post_processors(_make_config(), dataset_stats=None)
    anchor = next(s for s in pre.steps if isinstance(s, LingBotEpisodeAnchorStep))
    unanchor = next(s for s in post.steps if isinstance(s, AbsoluteActionsProcessorStep))
    assert not anchor.enabled
    assert not unanchor.enabled
    # ...and disabled means genuinely inert, including on the action tensor.
    act = torch.randn(1, 5, 4)
    assert torch.allclose(anchor({TransitionKey.ACTION: act})[TransitionKey.ACTION], act)


def test_anchor_step_runs_before_the_normalizer() -> None:
    # The anchor row must be consumed in physical units: if the normalizer ran first it would be
    # quantile-mapped with anchored-space stats, which is meaningless for an absolute pose.
    cfg = _make_config(action_anchor="episode")
    cfg.action_feature_names = _anchor_names()
    pre, post = make_lingbot_va_pre_post_processors(cfg, dataset_stats=None)
    names = [type(s).__name__ for s in pre.steps]
    assert names.index("LingBotEpisodeAnchorStep") < names.index("NormalizerProcessorStep")
    post_names = [type(s).__name__ for s in post.steps]
    assert post_names.index("UnnormalizerProcessorStep") < post_names.index("AbsoluteActionsProcessorStep")


def test_anchor_step_training_strips_row_zero_and_subtracts() -> None:
    step = _anchor_step()
    # (B=1, T=1+4, D=4); row 0 is the episode's first action.
    act = torch.tensor([[[10.0, 50.0, 20.0, 60.0]] + [[10.0 + i, 50.0, 20.0 + i, 60.0] for i in range(1, 5)]])
    pad = torch.zeros(1, 5, dtype=torch.bool)
    pad[0, 0] = True
    out = step(
        {
            TransitionKey.ACTION: act,
            TransitionKey.OBSERVATION: {},
            TransitionKey.COMPLEMENTARY_DATA: {f"{ACTION}_is_pad": pad},
        }
    )
    anchored = out[TransitionKey.ACTION]
    assert anchored.shape == (1, 4, 4)
    # Joint dims become displacement from the anchor; gripper dims stay absolute.
    assert torch.allclose(anchored[0, :, 0], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.allclose(anchored[0, :, 1], torch.full((4,), 50.0))
    # Padding flags are stripped in lockstep so they still describe the actions they accompany.
    assert out[TransitionKey.COMPLEMENTARY_DATA][f"{ACTION}_is_pad"].shape == (1, 4)


def test_anchor_unanchor_round_trip() -> None:
    step = _anchor_step()
    unstep = AbsoluteActionsProcessorStep(enabled=True, relative_step=step)
    act = torch.randn(1, 5, 4) * 10
    out = step({TransitionKey.ACTION: act, TransitionKey.OBSERVATION: {}})
    back = unstep({TransitionKey.ACTION: out[TransitionKey.ACTION]})[TransitionKey.ACTION]
    assert torch.allclose(back, act[:, 1:], atol=1e-4)


def test_anchor_is_latched_for_the_whole_episode_at_inference() -> None:
    # At inference there is no action to derive the anchor from, so it is latched from the first
    # observation and held: a per-tick anchor is exactly the incoherence this replaces.
    step = _anchor_step()
    first = torch.tensor([[100.0, 7.0, 200.0, 8.0]])
    step({TransitionKey.ACTION: None, TransitionKey.OBSERVATION: {OBS_STATE: first}})
    step({TransitionKey.ACTION: None, TransitionKey.OBSERVATION: {OBS_STATE: first + 25.0}})
    assert torch.allclose(step.get_cached_state(), first)
    # ...and dropped on reset so the next episode latches its own.
    step.reset()
    assert step.get_cached_state() is None


def test_inference_anchor_handles_stacked_state() -> None:
    # LingBot loads several observation steps, so state arrives as (B, T_obs, state_dim); the anchor
    # is the current frame (index 0), matching the relative-action helpers.
    step = _anchor_step()
    stacked = torch.stack([torch.full((1, 4), 1.0), torch.full((1, 4), 9.0)], dim=1)
    step({TransitionKey.ACTION: None, TransitionKey.OBSERVATION: {OBS_STATE: stacked}})
    assert torch.allclose(step.get_cached_state(), torch.full((1, 4), 1.0))


def test_unanchor_without_anchor_raises() -> None:
    unstep = AbsoluteActionsProcessorStep(enabled=True, relative_step=_anchor_step())
    with pytest.raises(RuntimeError, match="no state has been cached"):
        unstep({TransitionKey.ACTION: torch.zeros(1, 4)})


def test_inference_anchor_without_state_raises() -> None:
    step = _anchor_step()
    with pytest.raises(RuntimeError, match="observation.state"):
        step({TransitionKey.ACTION: None, TransitionKey.OBSERVATION: {}})


def test_anchor_step_config_round_trips(tmp_path) -> None:
    cfg = _make_config(action_anchor="episode")
    cfg.action_feature_names = _anchor_names()
    pre, _ = make_lingbot_va_pre_post_processors(cfg, dataset_stats=None)
    pre.save_pretrained(tmp_path)
    loaded = PolicyProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    step = next(s for s in loaded.steps if isinstance(s, LingBotEpisodeAnchorStep))
    assert step.enabled
    assert step.exclude_joints == ["gripper"]
    assert step.action_names == _anchor_names()


def test_pair_is_rebound_after_load(tmp_path) -> None:
    """The anchor/unanchor pair must survive a save/load round trip through the policy factory.

    The two pipelines are deserialized independently, so the postprocessor's `relative_step` comes
    back as None and unanchoring would raise. `factory._reconnect_relative_absolute_steps` fixes
    that up on every pretrained load; it is isinstance-based, which is why the anchor step subclasses
    RelativeActionsProcessorStep instead of being a standalone step needing its own wiring.
    """
    from lerobot.policies.factory import _reconnect_relative_absolute_steps

    cfg = _make_config(action_anchor="episode")
    cfg.action_feature_names = _anchor_names()
    pre, post = make_lingbot_va_pre_post_processors(cfg, dataset_stats=None)
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)

    loaded_pre = PolicyProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    loaded_post = PolicyProcessorPipeline.from_pretrained(
        tmp_path,
        config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    anchor = next(s for s in loaded_pre.steps if isinstance(s, LingBotEpisodeAnchorStep))
    unanchor = next(s for s in loaded_post.steps if isinstance(s, AbsoluteActionsProcessorStep))
    assert anchor.enabled and unanchor.enabled
    assert unanchor.relative_step is None  # independent deserialization

    _reconnect_relative_absolute_steps(loaded_pre, loaded_post)
    assert unanchor.relative_step is anchor

    # ...and the reconnected pair round-trips a real anchor end to end.
    state = torch.tensor([[100.0, 7.0, 200.0, 8.0]])
    loaded_pre_step_out = anchor({TransitionKey.ACTION: None, TransitionKey.OBSERVATION: {OBS_STATE: state}})
    assert loaded_pre_step_out is not None
    offsets = torch.tensor([[[1.0, 0.5, 2.0, 0.9]]])
    out = unanchor({TransitionKey.ACTION: offsets})[TransitionKey.ACTION]
    # Joints get the anchor added; grippers were excluded so they pass through.
    assert torch.allclose(out[0, 0, [0, 2]], torch.tensor([101.0, 202.0]))
    assert torch.allclose(out[0, 0, [1, 3]], torch.tensor([0.5, 0.9]))


def test_anchor_step_is_a_relative_step() -> None:
    # Load-path rebinding, the exclude mask and the cached-reference accessors all come from the
    # base class; this guards the inheritance the wiring depends on.
    step = _anchor_step()
    assert isinstance(step, RelativeActionsProcessorStep)
    assert step._build_mask(4) == [True, False, True, False]
