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
"""Verify that enabling gradient checkpointing on SmolVLA does not change numerics.

Gradient checkpointing recomputes the joint VLM+expert transformer layer loop
(`SmolVLMWithExpertModel._compute_layer`) during the backward pass instead of storing its activations.
It must be a pure memory/compute trade-off: given identical weights and identical inputs, the loss and
every parameter gradient must match a non-checkpointed run.

No dropout exists anywhere in `SmolVLMWithExpertModel`, so `preserve_rng_state=False` is used and no
special stochasticity handling is needed in the test config (contrast with ACT, which needs
`dropout=0.0`).

Uses `load_vlm_weights=False` (random-init architecture, not the pretrained 500M weights) and small
`num_vlm_layers`/`num_expert_layers` to keep this fast; still exercises the real SmolVLM2 config/tokenizer
path via `AutoConfig.from_pretrained`, so it needs network access (or an HF cache hit) and `transformers`.
"""

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from tests.utils import require_cuda, skip_if_package_missing

OBS_IMAGE = "observation.images.base_0_rgb"


def make_config(*, gradient_checkpointing: bool) -> SmolVLAConfig:
    """A small, fast, offline-safe (given HF cache) SmolVLAConfig for unit testing."""
    config = SmolVLAConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
        },
        output_features={
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
        },
        chunk_size=4,
        n_action_steps=4,
        max_state_dim=6,
        max_action_dim=6,
        resize_imgs_with_padding=(64, 64),
        tokenizer_max_length=8,
        load_vlm_weights=False,  # random-init architecture, not the pretrained 500M weights
        num_vlm_layers=2,
        num_expert_layers=2,
        self_attn_every_n_layers=1,
        expert_width_multiplier=0.5,
        gradient_checkpointing=gradient_checkpointing,
    )
    return config


def make_batch(config: SmolVLAConfig, batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        OBS_STATE: torch.randn(batch_size, 6),
        OBS_IMAGE: torch.rand(batch_size, 3, 64, 64),
        ACTION: torch.randn(batch_size, config.chunk_size, 6),
        "action_is_pad": torch.zeros(batch_size, config.chunk_size, dtype=torch.bool),
        OBS_LANGUAGE_TOKENS: torch.randint(1, 100, (batch_size, config.tokenizer_max_length)),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(batch_size, config.tokenizer_max_length, dtype=torch.bool),
    }


@skip_if_package_missing("transformers")
@require_cuda
def test_smolvla_gradient_checkpointing_matches_non_checkpointed():
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy_a = SmolVLAPolicy(make_config(gradient_checkpointing=False)).to("cuda")
    policy_b = SmolVLAPolicy(make_config(gradient_checkpointing=True)).to("cuda")
    policy_b.load_state_dict(policy_a.state_dict())

    assert policy_a.model.vlm_with_expert.gradient_checkpointing_enabled is False
    assert policy_b.model.vlm_with_expert.gradient_checkpointing_enabled is True

    policy_a.train()
    policy_b.train()  # must be train() mode: the checkpoint gate is `... and self.training`

    batch = {k: v.to("cuda") for k, v in make_batch(policy_a.config).items()}

    torch.manual_seed(123)
    noise = policy_a.model.sample_noise(
        (batch[ACTION].shape[0], policy_a.config.chunk_size, policy_a.config.max_action_dim), "cuda"
    )
    time = policy_a.model.sample_time(batch[ACTION].shape[0], "cuda")

    loss_a, _ = policy_a(dict(batch), noise=noise.clone(), time=time.clone())
    loss_a.backward()

    loss_b, _ = policy_b(dict(batch), noise=noise.clone(), time=time.clone())
    loss_b.backward()

    torch.testing.assert_close(loss_a, loss_b, rtol=1e-4, atol=1e-5)

    for (name_a, p_a), (name_b, p_b) in zip(
        policy_a.named_parameters(), policy_b.named_parameters(), strict=True
    ):
        assert name_a == name_b
        if p_a.grad is None:
            assert p_b.grad is None, f"{name_a}: grad present in checkpointed run but not baseline"
            continue
        if p_b.grad is None:
            pytest.fail(f"{name_a}: grad present in baseline but not checkpointed run")
        torch.testing.assert_close(p_a.grad, p_b.grad, rtol=1e-3, atol=1e-4, msg=f"grad mismatch at {name_a}")


@skip_if_package_missing("transformers")
@require_cuda
def test_smolvla_gradient_checkpointing_enable_disable_toggle():
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy(make_config(gradient_checkpointing=False)).to("cuda")
    policy.train()
    batch = {k: v.to("cuda") for k, v in make_batch(policy.config).items()}

    noise = policy.model.sample_noise(
        (batch[ACTION].shape[0], policy.config.chunk_size, policy.config.max_action_dim), "cuda"
    )
    time = policy.model.sample_time(batch[ACTION].shape[0], "cuda")

    assert policy.model.vlm_with_expert.gradient_checkpointing_enabled is False
    loss_before, _ = policy(dict(batch), noise=noise.clone(), time=time.clone())

    policy.model.vlm_with_expert.gradient_checkpointing_enable()
    assert policy.model.vlm_with_expert.gradient_checkpointing_enabled is True
    loss_after, _ = policy(dict(batch), noise=noise.clone(), time=time.clone())

    torch.testing.assert_close(loss_before, loss_after, rtol=1e-4, atol=1e-5)

    policy.model.vlm_with_expert.gradient_checkpointing_disable()
    assert policy.model.vlm_with_expert.gradient_checkpointing_enabled is False


@skip_if_package_missing("transformers")
@require_cuda
def test_smolvla_gradient_checkpointing_not_used_during_inference():
    """Checkpointing must never engage for KV-cached inference (use_cache=True) even if enabled."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy(make_config(gradient_checkpointing=True)).to("cuda")
    policy.eval()
    assert policy.model.vlm_with_expert.gradient_checkpointing_enabled is True

    batch = {k: v.to("cuda") for k, v in make_batch(policy.config).items()}
    with torch.no_grad():
        actions = policy.predict_action_chunk(dict(batch))
    assert actions.shape[0] == batch[OBS_STATE].shape[0]
