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

"""Tests for the PreTrainedPolicy contracts the interactive rollout stack
relies on: the drop_queued_actions() queue-attribute convention and the
generate_text()/supports_text_generation() tandem-override rule."""

from __future__ import annotations

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE


def _tiny_act_policy() -> ACTPolicy:
    """A real chunking policy, small enough to instantiate per test."""
    config = ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(4,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
        chunk_size=8,
        n_action_steps=8,
        use_vae=False,
        dim_model=32,
        n_heads=2,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
    )
    return ACTPolicy(config)


# ---------------------------------------------------------------------------
# drop_queued_actions: the queue-attribute convention against a real policy
# ---------------------------------------------------------------------------


def test_drop_queued_actions_forces_fresh_forward_on_real_chunking_policy():
    """After drop_queued_actions(), the next select_action must run a forward.

    This is the conformance check behind the interactive /subtask fast
    switch: if a rename breaks the attribute convention, the policy would
    silently keep serving up to n_action_steps stale actions.
    """
    policy = _tiny_act_policy()
    policy.reset()

    forwards = []
    original_forward = policy.model.forward

    def counting_forward(*args, **kwargs):
        forwards.append(1)
        return original_forward(*args, **kwargs)

    policy.model.forward = counting_forward
    batch = {OBS_STATE: torch.zeros(1, 4), OBS_ENV_STATE: torch.zeros(1, 4)}

    with torch.no_grad():
        policy.select_action(batch)  # fills the queue with one forward
        policy.select_action(batch)  # served from the queue
    assert len(forwards) == 1

    policy.drop_queued_actions()
    with torch.no_grad():
        policy.select_action(batch)
    assert len(forwards) == 2, "drop_queued_actions() did not force a fresh forward"


def test_drop_queued_actions_recognized_attrs_cover_the_real_queue():
    """The real policy's queue attribute is one the base class declares.

    Guards the convention itself: a rename of ACT's `_action_queue` (or a
    new policy inventing a third name) must extend _action_queue_attrs.
    """
    policy = _tiny_act_policy()
    policy.reset()
    assert any(getattr(policy, attr, None) is not None for attr in PreTrainedPolicy._action_queue_attrs)


def test_drop_queued_actions_honours_extended_classvar():
    """A policy declaring a custom queue attribute gets it cleared."""

    class _CustomQueuePolicy:
        _action_queue_attrs = (*PreTrainedPolicy._action_queue_attrs, "_my_cache")

        def __init__(self):
            from collections import deque

            self._my_cache = deque([1, 2, 3])

    policy = _CustomQueuePolicy()
    PreTrainedPolicy.drop_queued_actions(policy)
    assert len(policy._my_cache) == 0


# ---------------------------------------------------------------------------
# generate_text / supports_text_generation tandem override
# ---------------------------------------------------------------------------


def _policy_class_body():
    """Minimal concrete-method stubs so subclassing does not trip other checks."""
    return {
        "config_class": object,
        "get_optim_params": lambda self: {},
        "reset": lambda self: None,
        "forward": lambda self, batch: (torch.tensor(0.0), None),
        "predict_action_chunk": lambda self, batch, **kwargs: torch.zeros(1),
        "select_action": lambda self, batch, **kwargs: torch.zeros(1),
    }


def test_generate_text_override_without_flag_fails_at_class_definition():
    """The mismatch a third-party implementer is most likely to ship must fail loudly.

    Without the guard, the text head exists but supports_text_generation()
    stays False, so /vqa reports "this policy has no text head" — a symptom
    pointing away from the fix.
    """
    with pytest.raises(TypeError, match="supports_text_generation"):
        type(
            "TextHeadWithoutFlag",
            (PreTrainedPolicy,),
            {
                **_policy_class_body(),
                "name": "text_head_without_flag",
                "generate_text": lambda self, batch: "an answer",
            },
        )


def test_generate_text_override_with_flag_is_accepted():
    cls = type(
        "TextHeadWithFlag",
        (PreTrainedPolicy,),
        {
            **_policy_class_body(),
            "name": "text_head_with_flag",
            "supports_text_generation": lambda self: True,
            "generate_text": lambda self, batch: "an answer",
        },
    )
    assert cls is not None


def test_subclass_of_conforming_text_head_parent_is_accepted():
    """Refining generate_text under a parent's inherited flag is conforming.

    The guard resolves through the MRO, so it must not fire on a subclass
    whose supports_text_generation override comes from the parent.
    """
    parent = type(
        "ConformingParent",
        (PreTrainedPolicy,),
        {
            **_policy_class_body(),
            "name": "conforming_parent",
            "supports_text_generation": lambda self: True,
            "generate_text": lambda self, batch: "parent answer",
        },
    )
    child = type(
        "RefinedTextHead",
        (parent,),
        {"name": "refined_text_head", "generate_text": lambda self, batch: "refined answer"},
    )
    assert child is not None


def test_mixin_supplied_generate_text_without_flag_fails():
    """A text head arriving through a mixin must not dodge the guard."""

    class _TextHeadMixin:
        def generate_text(self, batch):
            return "mixin answer"

    with pytest.raises(TypeError, match="supports_text_generation"):
        type(
            "MixinTextHeadWithoutFlag",
            (_TextHeadMixin, PreTrainedPolicy),
            {**_policy_class_body(), "name": "mixin_text_head_without_flag"},
        )


def test_flag_only_override_is_allowed_but_generate_text_fails_loudly():
    """Checkpoint-conditional support may flip the flag without a text head;
    the base generate_text then fails loudly instead of silently no-oping."""
    cls = type(
        "FlagWithoutTextHead",
        (PreTrainedPolicy,),
        {
            **_policy_class_body(),
            "name": "flag_without_text_head",
            "supports_text_generation": lambda self: True,
        },
    )
    instance = cls.__new__(cls)  # skip nn.Module init; generate_text touches no state
    with pytest.raises(NotImplementedError, match="no text head"):
        instance.generate_text({})
