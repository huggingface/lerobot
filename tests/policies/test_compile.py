#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Tests for torch.compile support in policies."""

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


def _make_act_config(compile_model: bool = False) -> ACTConfig:
    """Create an ACT config with small dimensions for fast testing."""
    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        f"{OBS_IMAGES}.laptop": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 84, 84)),
    }
    output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
    }
    return ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=10,
        n_action_steps=10,
        dim_model=64,
        dim_feedforward=256,
        n_heads=2,
        n_encoder_layers=2,
        n_decoder_layers=1,
        n_vae_encoder_layers=2,
        latent_dim=16,
        compile_model=compile_model,
    )


@pytest.mark.timeout(120)
def test_act_compile_correctness():
    """Verify that compiled ACT produces the same output as non-compiled."""
    device = "cpu"

    # Create both policies with identical weights using the same seed
    torch.manual_seed(0)
    baseline = ACTPolicy(_make_act_config(compile_model=False)).to(device).eval()
    torch.manual_seed(0)
    compiled = ACTPolicy(_make_act_config(compile_model=True)).to(device).eval()

    # Create a dummy observation batch
    torch.manual_seed(42)
    batch = {
        OBS_STATE: torch.randn(1, 6, device=device),
        f"{OBS_IMAGES}.laptop": torch.rand(1, 3, 84, 84, device=device),
    }

    with torch.inference_mode():
        baseline.reset()
        compiled.reset()
        action_baseline = baseline.select_action(batch)
        action_compiled = compiled.select_action(batch)

    torch.testing.assert_close(action_baseline, action_compiled, rtol=1e-3, atol=1e-3)


@pytest.mark.timeout(120)
def test_act_compile_training_forward():
    """Verify that compiled ACT works in training mode (forward pass with loss)."""
    device = "cpu"
    policy = ACTPolicy(_make_act_config(compile_model=True)).to(device).train()

    batch = {
        OBS_STATE: torch.randn(2, 6, device=device),
        f"{OBS_IMAGES}.laptop": torch.rand(2, 3, 84, 84, device=device),
        ACTION: torch.randn(2, 10, 6, device=device),
        "action_is_pad": torch.zeros(2, 10, dtype=torch.bool, device=device),
    }

    loss, loss_dict = policy.forward(batch)
    assert loss.requires_grad
    assert "l1_loss" in loss_dict
