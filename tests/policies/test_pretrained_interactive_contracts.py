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

"""Tests for the PreTrainedPolicy contract: what a policy must implement, and the
generate_text()/supports_text_generation() tandem override the interactive rollout relies on."""

from __future__ import annotations

import pytest
import torch

from lerobot.policies.pretrained import PreTrainedPolicy


def _policy_class_body():
    """Concrete-method stubs so subclassing trips no other check."""
    return {
        "config_class": object,
        "get_optim_params": lambda self: {},
        "reset": lambda self: None,
        "forward": lambda self, batch: (torch.tensor(0.0), None),
        "predict_action_chunk": lambda self, batch, **kwargs: torch.zeros(1),
    }


def test_a_policy_implements_training_and_chunk_prediction_only():
    """``forward`` and ``predict_action_chunk`` are the whole inference/training surface:
    ``reset`` is optional and nothing queues actions per step."""
    body = {**_policy_class_body(), "name": "minimal_policy"}
    del body["reset"]
    minimal = type("MinimalPolicy", (PreTrainedPolicy,), body)
    assert not minimal.__abstractmethods__
    assert not hasattr(minimal, "select_action")
    instance = minimal.__new__(minimal)
    instance.reset()  # the default keeps nothing between calls


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
