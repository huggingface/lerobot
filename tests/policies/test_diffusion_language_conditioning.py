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
"""Tests for optional language conditioning in Diffusion Policy."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion import modeling_diffusion
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel, DiffusionPolicy
from lerobot.utils.constants import (
    ACTION,
    OBS_ENV_STATE,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


class FakeScheduler:
    config = SimpleNamespace(num_train_timesteps=2)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        del timesteps
        return original_samples + noise


class FakeCLIPTextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)

    @classmethod
    def from_pretrained(cls, model_name: str):
        return cls()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        pooled = (input_ids.float() * attention_mask.float()).sum(dim=1, keepdim=True)
        return SimpleNamespace(pooler_output=pooled.repeat(1, self.config.hidden_size))


@pytest.fixture(autouse=True)
def patch_optional_diffusion_deps(monkeypatch):
    def require_package_stub(pkg_name: str, extra: str, import_name: str | None = None) -> None:
        return None

    monkeypatch.setattr(modeling_diffusion, "require_package", require_package_stub)
    monkeypatch.setattr(modeling_diffusion, "_make_noise_scheduler", lambda *args, **kwargs: FakeScheduler())
    monkeypatch.setattr(modeling_diffusion, "CLIPTextModel", FakeCLIPTextModel)


def make_config(use_language_conditioning: bool = True) -> DiffusionConfig:
    return DiffusionConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
            OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(2,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        n_obs_steps=2,
        horizon=4,
        n_action_steps=2,
        down_dims=(16,),
        device="cpu",
        use_language_conditioning=use_language_conditioning,
        language_condition_dim=5,
        tokenizer_max_length=7,
    )


def make_batch(config: DiffusionConfig, batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        OBS_STATE: torch.randn(batch_size, config.n_obs_steps, 3),
        OBS_ENV_STATE: torch.randn(batch_size, config.n_obs_steps, 2),
        OBS_LANGUAGE_TOKENS: torch.tensor([[1, 2, 3, 4, 0, 0, 0], [5, 6, 7, 8, 0, 0, 0]], dtype=torch.long)[
            :batch_size
        ],
        OBS_LANGUAGE_ATTENTION_MASK: torch.tensor(
            [[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool
        )[:batch_size],
    }


def test_language_conditioning_extends_global_conditioning():
    config = make_config(use_language_conditioning=True)
    model = DiffusionModel(config)
    with torch.no_grad():
        model.language_encoder.projection.weight.fill_(1.0)
        model.language_encoder.projection.bias.zero_()

    batch = make_batch(config)
    global_cond = model._prepare_global_conditioning(batch)

    expected_dim = config.n_obs_steps * (3 + 2 + config.language_condition_dim)
    assert global_cond.shape == (2, expected_dim)

    changed_batch = dict(batch)
    changed_batch[OBS_LANGUAGE_TOKENS] = batch[OBS_LANGUAGE_TOKENS] + 10
    changed_global_cond = model._prepare_global_conditioning(changed_batch)

    assert not torch.allclose(global_cond, changed_global_cond)


def test_language_conditioning_requires_tokenized_language_keys():
    config = make_config(use_language_conditioning=True)
    model = DiffusionModel(config)
    batch = make_batch(config)
    batch.pop(OBS_LANGUAGE_TOKENS)

    with pytest.raises(ValueError, match="missing"):
        model._prepare_global_conditioning(batch)


def test_language_conditioning_forward_computes_training_loss():
    config = make_config(use_language_conditioning=True)
    policy = DiffusionPolicy(config)
    batch = make_batch(config)
    batch[ACTION] = torch.randn(2, config.horizon, config.action_feature.shape[0])
    batch["action_is_pad"] = torch.zeros(2, config.horizon, dtype=torch.bool)

    loss, output_dict = policy.forward(batch)

    assert output_dict is None
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_default_diffusion_does_not_build_language_encoder():
    config = make_config(use_language_conditioning=False)
    model = DiffusionModel(config)
    batch = make_batch(config)
    batch.pop(OBS_LANGUAGE_TOKENS)
    batch.pop(OBS_LANGUAGE_ATTENTION_MASK)

    global_cond = model._prepare_global_conditioning(batch)

    assert model.language_encoder is None
    assert global_cond.shape == (2, config.n_obs_steps * (3 + 2))


def test_language_conditioning_preserves_language_through_action_queue():
    config = make_config(use_language_conditioning=True)
    policy = DiffusionPolicy(config)
    captured_batch = {}

    def fake_generate_actions(batch: dict[str, torch.Tensor], noise=None):
        captured_batch.update(batch)
        return torch.zeros(2, config.n_action_steps, config.action_feature.shape[0])

    policy.diffusion.generate_actions = fake_generate_actions
    batch = {
        OBS_STATE: torch.randn(2, 3),
        OBS_ENV_STATE: torch.randn(2, 2),
        OBS_LANGUAGE_TOKENS: torch.ones(2, config.tokenizer_max_length, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, config.tokenizer_max_length, dtype=torch.bool),
    }

    action = policy.select_action(batch)

    assert action.shape == (2, config.action_feature.shape[0])
    assert captured_batch[OBS_STATE].shape == (2, config.n_obs_steps, 3)
    assert captured_batch[OBS_ENV_STATE].shape == (2, config.n_obs_steps, 2)
    assert captured_batch[OBS_LANGUAGE_TOKENS].shape == (2, config.tokenizer_max_length)
    assert captured_batch[OBS_LANGUAGE_ATTENTION_MASK].shape == (2, config.tokenizer_max_length)


def test_language_conditioning_config_validation():
    with pytest.raises(ValueError, match="language_condition_dim"):
        DiffusionConfig(use_language_conditioning=True, language_condition_dim=0)

    with pytest.raises(ValueError, match="tokenizer_padding_side"):
        DiffusionConfig(use_language_conditioning=True, tokenizer_padding_side="middle")
