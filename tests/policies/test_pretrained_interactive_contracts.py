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

"""Tests for the PreTrainedPolicy contracts the interactive rollout stack relies on:
drop_queued_actions() and the generate_text()/supports_text_generation() tandem override."""

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


def _policy_class_body():
    """Concrete-method stubs so subclassing trips no other check."""
    return {
        "config_class": object,
        "get_optim_params": lambda self: {},
        "reset": lambda self: None,
        "forward": lambda self, batch: (torch.tensor(0.0), None),
        "predict_action_chunk": lambda self, batch, **kwargs: torch.zeros(1),
        "select_action": lambda self, batch, **kwargs: torch.zeros(1),
    }


def test_drop_queued_actions_forces_fresh_forward_on_real_chunking_policy():
    """Behind the interactive /subtask fast switch: a broken attribute convention would
    silently keep serving stale actions."""
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


@pytest.mark.parametrize("via_mixin", [False, True], ids=["own-method", "mixin-supplied"])
def test_generate_text_override_without_flag_fails_at_class_definition(via_mixin):
    """Without the guard the text head exists but supports_text_generation() stays False, so
    /vqa reports "this policy has no text head" — a symptom pointing away from the fix."""

    class _TextHeadMixin:
        def generate_text(self, batch):
            return "an answer"

    bases = (_TextHeadMixin, PreTrainedPolicy) if via_mixin else (PreTrainedPolicy,)
    body = {**_policy_class_body(), "name": "text_head_without_flag"}
    if not via_mixin:
        body["generate_text"] = lambda self, batch: "an answer"

    with pytest.raises(TypeError, match="supports_text_generation"):
        type("TextHeadWithoutFlag", bases, body)


def test_generate_text_with_flag_is_accepted_including_through_an_inherited_flag():
    """Both overrides on one class is conforming, and so is a subclass refining generate_text
    under the parent's flag (the guard resolves through the MRO)."""
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

    instance = child.__new__(child)  # skip nn.Module init; neither method touches state
    assert instance.supports_text_generation()
    assert instance.generate_text({}) == "refined answer"
