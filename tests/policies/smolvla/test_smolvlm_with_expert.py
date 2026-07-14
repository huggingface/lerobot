#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Tests for SmolVLMWithExpertModel parameter freezing."""

import pytest

from tests.utils import skip_if_package_missing


@skip_if_package_missing("transformers")
@pytest.mark.parametrize("num_vlm_layers, num_expert_layers", [(4, 2), (4, 4)])
def test_partial_vlm_freeze_when_training_vlm(num_vlm_layers, num_expert_layers):
    """With train_expert_only=False, the last VLM layer(s), final norm and lm_head must be frozen.

    Regression test for https://github.com/huggingface/lerobot/issues/4018: the freeze
    patterns used a `text_model.model.` prefix that never matched any real parameter name,
    so the partial freeze silently no-oped (only lm_head was frozen).
    """
    from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel

    model = SmolVLMWithExpertModel(
        load_vlm_weights=False,
        train_expert_only=False,
        freeze_vision_encoder=False,
        num_vlm_layers=num_vlm_layers,
        num_expert_layers=num_expert_layers,
    )

    # Layers that must be frozen to avoid unused parameters with DDP.
    expected_frozen_layers = [num_vlm_layers - 1]
    if num_vlm_layers != num_expert_layers and num_vlm_layers % num_expert_layers == 0:
        expected_frozen_layers.append(num_vlm_layers - 2)

    frozen = {name for name, params in model.vlm.named_parameters() if not params.requires_grad}
    trainable = {name for name, params in model.vlm.named_parameters() if params.requires_grad}

    assert any("lm_head" in name for name in frozen)
    assert any("text_model.norm.weight" in name for name in frozen)
    for layer in expected_frozen_layers:
        layer_params = [name for name in frozen | trainable if f"text_model.layers.{layer}." in name]
        assert layer_params, f"no parameters found for VLM layer {layer}"
        assert all(name in frozen for name in layer_params), f"VLM layer {layer} is not fully frozen"

    # Earlier VLM layers must remain trainable in this mode.
    first_trainable_layers = set(range(num_vlm_layers)) - set(expected_frozen_layers)
    for layer in first_trainable_layers:
        assert any(f"text_model.layers.{layer}." in name for name in trainable), (
            f"VLM layer {layer} should be trainable"
        )
