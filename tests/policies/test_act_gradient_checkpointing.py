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
"""Verify that enabling gradient checkpointing on ACTPolicy does not change numerics.

Gradient checkpointing recomputes the vision backbone and transformer encoder/decoder forward
passes during the backward pass instead of storing their activations. It must be a pure
memory/compute trade-off: given identical weights and identical inputs, the loss and every
parameter gradient must match a non-checkpointed run.

`dropout=0.0` is used in the test config: the transformer layers use `nn.Dropout` and
`nn.MultiheadAttention`'s internal dropout, and `preserve_rng_state=True` (required precisely
because of this dropout) only guarantees a matching *mask*, not the absence of one — an
`.eval()`-mode comparison would trivially disable dropout, but the checkpoint gate itself
requires `self.training=True`, so dropout must be removed at the config level instead.
"""

import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE


def make_config(*, gradient_checkpointing: bool) -> ACTConfig:
    """A small, CPU-fast, offline-safe ACTConfig for unit testing."""
    return ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
            OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,)),
        },
        normalization_mapping={
            FeatureType.VISUAL: NormalizationMode.IDENTITY,
            FeatureType.STATE: NormalizationMode.IDENTITY,
            FeatureType.ACTION: NormalizationMode.IDENTITY,
        },
        device="cpu",
        chunk_size=8,
        n_action_steps=8,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=2,
        n_decoder_layers=2,
        n_vae_encoder_layers=2,
        latent_dim=8,
        dropout=0.0,  # remove stochasticity; preserve_rng_state=True only matches the *mask*, not zero-dropout
        pretrained_backbone_weights=None,  # avoid downloading ImageNet weights in tests
        gradient_checkpointing=gradient_checkpointing,
    )


def make_batch(config: ACTConfig, batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        OBS_STATE: torch.randn(batch_size, 4),
        OBS_IMAGE: torch.rand(batch_size, 3, 64, 64),
        ACTION: torch.randn(batch_size, config.chunk_size, 3),
        "action_is_pad": torch.zeros(batch_size, config.chunk_size, dtype=torch.bool),
    }


def test_act_gradient_checkpointing_matches_non_checkpointed():
    policy_a = ACTPolicy(make_config(gradient_checkpointing=False))
    policy_b = ACTPolicy(make_config(gradient_checkpointing=True))
    policy_b.load_state_dict(policy_a.state_dict())

    assert policy_a.model.gradient_checkpointing_enabled is False
    assert policy_b.model.gradient_checkpointing_enabled is True
    assert policy_a.model.encoder.gradient_checkpointing_enabled is False
    assert policy_b.model.encoder.gradient_checkpointing_enabled is True
    assert policy_b.model.decoder.gradient_checkpointing_enabled is True
    assert policy_b.model.vae_encoder.gradient_checkpointing_enabled is True

    policy_a.train()
    policy_b.train()  # must be train() mode: the checkpoint gate is `... and self.training`

    batch = make_batch(policy_a.config)

    torch.manual_seed(123)
    loss_a, _ = policy_a(batch)
    loss_a.backward()

    torch.manual_seed(123)
    loss_b, _ = policy_b(batch)
    loss_b.backward()

    torch.testing.assert_close(loss_a, loss_b, rtol=1e-6, atol=1e-7)

    for (name_a, p_a), (name_b, p_b) in zip(
        policy_a.named_parameters(), policy_b.named_parameters(), strict=True
    ):
        assert name_a == name_b
        if p_a.grad is None:
            assert p_b.grad is None
            continue
        torch.testing.assert_close(p_a.grad, p_b.grad, rtol=1e-5, atol=1e-6, msg=f"grad mismatch at {name_a}")


def test_act_gradient_checkpointing_enable_disable_toggle():
    policy = ACTPolicy(make_config(gradient_checkpointing=False))
    policy.train()
    batch = make_batch(policy.config)

    assert policy.model.gradient_checkpointing_enabled is False
    torch.manual_seed(7)
    loss_before, _ = policy(batch)

    policy.model.gradient_checkpointing_enable()
    assert policy.model.gradient_checkpointing_enabled is True
    assert policy.model.encoder.gradient_checkpointing_enabled is True
    torch.manual_seed(7)
    loss_after, _ = policy(batch)

    torch.testing.assert_close(loss_before, loss_after, rtol=1e-6, atol=1e-7)

    policy.model.gradient_checkpointing_disable()
    assert policy.model.gradient_checkpointing_enabled is False
    assert policy.model.encoder.gradient_checkpointing_enabled is False
