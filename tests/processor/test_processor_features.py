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

"""Cross-cutting pipeline features, and relative actions as the first of them.

The headline property: a behaviour like relative actions works on a policy that contains no code
referring to it. ACT is the test subject precisely because it knows nothing about relative actions.
"""

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import make_policy_config, make_pre_post_processors
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    AnchoredStep,
    IdentityProcessorStep,
    NormalizerProcessorStep,
    ProcessorBuildContext,
    ProcessorFeature,
    RelativeActionsProcessorStep,
    UnnormalizerProcessorStep,
    splice_anchored_steps,
)
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE

STATE_DIM = 4
ACTION_DIM = 4


def _act_config(**overrides):
    config = make_policy_config("act", push_to_hub=False)
    config.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        f"{OBS_IMAGE}.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
    }
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))}
    config.device = "cpu"
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _stats():
    def block(shape):
        return {"mean": torch.zeros(shape), "std": torch.ones(shape)}

    return {
        OBS_STATE: block((STATE_DIM,)),
        ACTION: block((ACTION_DIM,)),
        f"{OBS_IMAGE}.cam": block((3, 1, 1)),
    }


def _step_names(pipeline):
    return [type(step).__name__ for step in pipeline.steps]


def test_act_gets_relative_actions_without_any_act_specific_code():
    """The acceptance test for 'policy-independent': ACT has no relative-action code at all."""
    pre, post = make_pre_post_processors(
        _act_config(use_relative_actions=True),
        context=ProcessorBuildContext(dataset_stats=_stats()),
    )

    assert "RelativeActionsProcessorStep" in _step_names(pre)
    assert "AbsoluteActionsProcessorStep" in _step_names(post)


def test_relative_actions_are_absent_when_not_enabled():
    """The default must leave every existing pipeline byte-identical."""
    pre, post = make_pre_post_processors(_act_config(), context=ProcessorBuildContext(dataset_stats=_stats()))

    assert "RelativeActionsProcessorStep" not in _step_names(pre)
    assert "AbsoluteActionsProcessorStep" not in _step_names(post)


def test_relative_step_is_anchored_before_normalization():
    """Relative offsets must be computed on raw values, so ordering here is load-bearing."""
    pre, post = make_pre_post_processors(
        _act_config(use_relative_actions=True),
        context=ProcessorBuildContext(dataset_stats=_stats()),
    )

    pre_names = _step_names(pre)
    assert pre_names.index("RelativeActionsProcessorStep") < pre_names.index("NormalizerProcessorStep")

    post_names = _step_names(post)
    assert post_names.index("AbsoluteActionsProcessorStep") > post_names.index("UnnormalizerProcessorStep")


def test_absolute_step_reference_is_live_without_any_relinking():
    """The cross-pipeline reference is not serializable; building both together avoids re-linking."""
    pre, post = make_pre_post_processors(
        _act_config(use_relative_actions=True),
        context=ProcessorBuildContext(dataset_stats=_stats()),
    )

    relative = next(step for step in pre.steps if isinstance(step, RelativeActionsProcessorStep))
    absolute = next(step for step in post.steps if isinstance(step, AbsoluteActionsProcessorStep))
    assert absolute.relative_step is relative


def test_relative_actions_survive_a_checkpoint_round_trip(tmp_path):
    """An old checkpoint has no such step; rebuilding from the config restores it."""
    config = _act_config(use_relative_actions=True)
    pre, post = make_pre_post_processors(config, context=ProcessorBuildContext(dataset_stats=_stats()))
    pre.save_pretrained(tmp_path, config_filename="policy_preprocessor.json")
    post.save_pretrained(tmp_path, config_filename="policy_postprocessor.json")

    reloaded_pre, reloaded_post = make_pre_post_processors(
        config, pretrained_path=str(tmp_path), context=ProcessorBuildContext()
    )

    assert "RelativeActionsProcessorStep" in _step_names(reloaded_pre)
    relative = next(step for step in reloaded_pre.steps if isinstance(step, RelativeActionsProcessorStep))
    absolute = next(step for step in reloaded_post.steps if isinstance(step, AbsoluteActionsProcessorStep))
    assert absolute.relative_step is relative


def test_finetuning_a_checkpoint_that_predates_the_step_does_not_raise(tmp_path):
    """The finetune failure this refactor removes: the step is absent from the saved pipeline.

    Previously `lerobot-train` emitted a `relative_actions_processor` override and
    `_validate_overrides_used` raised `KeyError` because the checkpoint had no such step.
    """
    # A checkpoint written before relative actions existed for this policy.
    plain = _act_config()
    pre, post = make_pre_post_processors(plain, context=ProcessorBuildContext(dataset_stats=_stats()))
    pre.save_pretrained(tmp_path, config_filename="policy_preprocessor.json")
    post.save_pretrained(tmp_path, config_filename="policy_postprocessor.json")
    assert "RelativeActionsProcessorStep" not in _step_names(pre)

    # Finetuning it with relative actions on must now work.
    reloaded_pre, _ = make_pre_post_processors(
        _act_config(use_relative_actions=True),
        pretrained_path=str(tmp_path),
        context=ProcessorBuildContext(dataset_stats=_stats()),
    )

    assert "RelativeActionsProcessorStep" in _step_names(reloaded_pre)


def test_relative_actions_round_trip_numerically():
    """relative then absolute must return the original action."""
    config = _act_config(use_relative_actions=True, relative_exclude_joints=[])
    pre, post = make_pre_post_processors(config, context=ProcessorBuildContext(dataset_stats=_stats()))

    state = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    action = torch.tensor([[1.5, 2.5, 3.5, 4.5]])
    processed = pre(
        {
            OBS_STATE: state,
            f"{OBS_IMAGE}.cam": torch.rand(1, 3, 32, 32),
            ACTION: action,
        }
    )
    # The preprocessed action is a delta relative to the state.
    assert torch.allclose(processed[ACTION], action - state, atol=1e-5)
    # And the postprocessor puts it back.
    assert torch.allclose(post(processed[ACTION]), action, atol=1e-5)


def test_excluded_joints_stay_absolute():
    config = _act_config(use_relative_actions=True, relative_exclude_joints=["gripper"])
    config.action_feature_names = ["j0", "j1", "j2", "gripper"]
    pre, _post = make_pre_post_processors(config, context=ProcessorBuildContext(dataset_stats=_stats()))

    state = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    action = torch.tensor([[1.5, 2.5, 3.5, 4.5]])
    processed = pre({OBS_STATE: state, f"{OBS_IMAGE}.cam": torch.rand(1, 3, 32, 32), ACTION: action})

    assert torch.allclose(processed[ACTION][:, :3], (action - state)[:, :3], atol=1e-5)
    # The gripper dimension is untouched.
    assert torch.allclose(processed[ACTION][:, 3], action[:, 3], atol=1e-5)


def test_a_policy_that_places_the_steps_itself_is_left_alone():
    """Bespoke placement wins, without the feature having to know which policy it is.

    This is how GR00T keeps its own relative/absolute placement instead of being special-cased.
    """
    from lerobot.processor import apply_policy_features

    # Build with the feature off, then hand-place the step where this "policy" wants it.
    pre, post = make_pre_post_processors(_act_config(), context=ProcessorBuildContext(dataset_stats=_stats()))
    hand_placed = RelativeActionsProcessorStep(enabled=True)
    pre.steps = [hand_placed, *pre.steps]

    # Now compose features with the flag on: the feature must notice and stand down.
    apply_policy_features(_act_config(use_relative_actions=True), ProcessorBuildContext(), pre, post)

    assert sum(isinstance(s, RelativeActionsProcessorStep) for s in pre.steps) == 1
    assert pre.steps[0] is hand_placed
    assert not any(isinstance(s, AbsoluteActionsProcessorStep) for s in post.steps)


# Anchor resolution


def test_splice_raises_when_the_anchor_is_missing():
    steps = [IdentityProcessorStep()]
    anchored = [AnchoredStep(IdentityProcessorStep(), NormalizerProcessorStep, "before")]

    with pytest.raises(ValueError, match="matched 0 steps"):
        splice_anchored_steps(steps, anchored)


def test_splice_raises_when_the_anchor_is_ambiguous():
    """Two candidate anchors means the position would be a guess, so it must fail."""
    normalizer_args = {"features": {}, "norm_map": {}}
    steps = [
        NormalizerProcessorStep(**normalizer_args),
        NormalizerProcessorStep(**normalizer_args),
    ]
    anchored = [AnchoredStep(IdentityProcessorStep(), NormalizerProcessorStep, "before")]

    with pytest.raises(ValueError, match="matched 2 steps"):
        splice_anchored_steps(steps, anchored)


def test_splice_places_before_and_after():
    normalizer = NormalizerProcessorStep(features={}, norm_map={})
    marker = IdentityProcessorStep()

    before = splice_anchored_steps([normalizer], [AnchoredStep(marker, NormalizerProcessorStep, "before")])
    assert before == [marker, normalizer]

    after = splice_anchored_steps([normalizer], [AnchoredStep(marker, NormalizerProcessorStep, "after")])
    assert after == [normalizer, marker]


def test_a_context_owned_feature_can_be_composed():
    """The extension point the end-effector kinematics steps will use.

    Its parameters come from the build context rather than the policy config, so they are not
    persisted with the checkpoint — the same checkpoint may run on different hardware.
    """

    class MarkerFeature(ProcessorFeature):
        name = "marker"
        owner = "context"

        def enabled_for(self, config, context):
            return context.training

        def build(self, config, context):
            return (
                [AnchoredStep(IdentityProcessorStep(), NormalizerProcessorStep, "before")],
                [AnchoredStep(IdentityProcessorStep(), UnnormalizerProcessorStep, "after")],
            )

    from lerobot.processor import apply_policy_features

    config = _act_config()
    pre, post = make_pre_post_processors(config, context=ProcessorBuildContext(dataset_stats=_stats()))

    # Disabled: the context says this is not a training build.
    apply_policy_features(config, ProcessorBuildContext(training=False), pre, post, [MarkerFeature()])
    assert "IdentityProcessorStep" not in _step_names(post)

    apply_policy_features(config, ProcessorBuildContext(training=True), pre, post, [MarkerFeature()])
    assert "IdentityProcessorStep" in _step_names(post)
