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

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.policies.flow_matching import modeling_flow_matching
from lerobot.policies.flow_matching.configuration_flow_matching import FlowMatchingConfig
from lerobot.policies.flow_matching.modeling_flow_matching import (
    FlowMatchingPolicy,
    make_flow_matching_target,
)
from lerobot.processor import TokenizerProcessorStep, tokenizer_processor
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


def make_config(
    *,
    num_cameras: int = 0,
    guidance_scale: float = 1.0,
    text_conditioning: bool = False,
) -> FlowMatchingConfig:
    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(5,)),
    }
    for camera_index in range(num_cameras):
        input_features[f"{OBS_IMAGES}.camera_{camera_index}"] = PolicyFeature(
            type=FeatureType.VISUAL,
            shape=(3, 32, 32),
        )

    return FlowMatchingConfig(
        input_features=input_features,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
        n_obs_steps=2,
        horizon=6,
        n_action_steps=3,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        feed_forward_dim=64,
        dropout=0.0,
        num_inference_steps=2,
        conditioning_dropout_prob=0.0,
        guidance_scale=guidance_scale,
        pretrained_backbone_weights=None,
        text_encoder_name="openai/clip-vit-base-patch16" if text_conditioning else None,
        device="cpu",
    )


def make_training_batch(*, num_cameras: int = 0, batch_size: int = 2) -> dict[str, torch.Tensor]:
    batch = {
        OBS_STATE: torch.randn(batch_size, 2, 5),
        ACTION: torch.randn(batch_size, 6, 3),
        "action_is_pad": torch.tensor(
            [[False, False, False, False, True, True]] * batch_size,
        ),
    }
    for camera_index in range(num_cameras):
        batch[f"{OBS_IMAGES}.camera_{camera_index}"] = torch.rand(batch_size, 2, 3, 32, 32)
    return batch


def test_flow_matching_factory_integration():
    config = make_policy_config(
        "flow_matching",
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(5,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
        horizon=6,
        n_action_steps=3,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        feed_forward_dim=64,
        device="cpu",
    )

    assert isinstance(config, FlowMatchingConfig)
    assert get_policy_class("flow_matching") is FlowMatchingPolicy
    assert config.action_delta_indices == [-1, 0, 1, 2, 3, 4]


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"n_obs_steps": 0}, "n_obs_steps"),
        ({"horizon": 2, "n_obs_steps": 2, "n_action_steps": 2}, "n_action_steps"),
        ({"hidden_dim": 31}, "hidden_dim"),
        ({"hidden_dim": 32, "num_heads": 3}, "num_heads"),
        ({"num_inference_steps": 0}, "num_inference_steps"),
        ({"conditioning_dropout_prob": 1.1}, "conditioning_dropout_prob"),
        ({"guidance_scale": -1.0}, "guidance_scale"),
        ({"text_encoder_name": "bert-base-uncased"}, "text_encoder_name"),
        ({"tokenizer_max_length": 0}, "tokenizer_max_length"),
    ],
)
def test_flow_matching_config_validation(overrides, match):
    with pytest.raises(ValueError, match=match):
        FlowMatchingConfig(**overrides)


def test_make_flow_matching_target_endpoints():
    actions = torch.tensor([[[2.0]], [[4.0]]])
    noise = torch.tensor([[[-1.0]], [[1.0]]])
    time = torch.tensor([0.0, 1.0])

    trajectory, velocity, returned_time = make_flow_matching_target(actions, noise=noise, time=time)

    torch.testing.assert_close(trajectory[0], noise[0])
    torch.testing.assert_close(trajectory[1], actions[1])
    torch.testing.assert_close(velocity, actions - noise)
    torch.testing.assert_close(returned_time, time)


@pytest.mark.parametrize("num_cameras", [0, 1, 2])
def test_flow_matching_forward_backward(num_cameras: int):
    torch.manual_seed(0)
    policy = FlowMatchingPolicy(make_config(num_cameras=num_cameras))
    policy.train()
    batch = make_training_batch(num_cameras=num_cameras)
    original_batch = deepcopy(batch)

    loss, metrics = policy(batch)

    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert set(metrics) == {"mse_loss", "l1_loss"}
    assert all(isinstance(value, float) for value in metrics.values())
    loss.backward()
    assert any(parameter.grad is not None for parameter in policy.parameters())
    assert set(batch) == set(original_batch)
    assert all(torch.equal(batch[key], original_batch[key]) for key in batch)


class FakeCLIPTextModel(nn.Module):
    config = SimpleNamespace(hidden_size=8)

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(257, self.config.hidden_size)

    @classmethod
    def from_pretrained(cls, _model_name: str):
        return cls()

    def forward(self, input_ids, attention_mask):
        embeddings = self.embedding(input_ids)
        weights = attention_mask.unsqueeze(-1).to(embeddings.dtype)
        pooled = (embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return SimpleNamespace(pooler_output=pooled)


def test_flow_matching_language_conditioning(monkeypatch):
    monkeypatch.setattr(modeling_flow_matching, "CLIPTextModel", FakeCLIPTextModel)
    config = make_config(text_conditioning=True)
    policy = FlowMatchingPolicy(config)
    batch = make_training_batch()
    batch[OBS_LANGUAGE_TOKENS] = torch.randint(1, 256, (2, 12))
    batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones(2, 12, dtype=torch.bool)

    loss, _ = policy(batch)
    loss.backward()

    task_encoder = policy.model.observation_encoder.task_encoder
    assert task_encoder is not None
    assert all(parameter.grad is None for parameter in task_encoder.text_encoder.parameters())
    assert task_encoder.projection.weight.grad is not None

    observation = {
        OBS_STATE: torch.randn(2, 5),
        OBS_LANGUAGE_TOKENS: batch[OBS_LANGUAGE_TOKENS],
        OBS_LANGUAGE_ATTENTION_MASK: batch[OBS_LANGUAGE_ATTENTION_MASK],
    }
    assert policy.select_action(observation).shape == (2, 3)


def test_flow_matching_language_processor(monkeypatch):
    class FakeTokenizer:
        def __call__(self, texts, **_kwargs):
            batch_size = len(texts)
            return {
                "input_ids": torch.ones(batch_size, 7, dtype=torch.long),
                "attention_mask": torch.ones(batch_size, 7, dtype=torch.long),
            }

    monkeypatch.setattr(
        tokenizer_processor.AutoTokenizer,
        "from_pretrained",
        lambda _model_name: FakeTokenizer(),
    )
    config = make_config(text_conditioning=True)
    config.normalization_mapping = {
        "STATE": NormalizationMode.IDENTITY,
        "ACTION": NormalizationMode.IDENTITY,
        "VISUAL": NormalizationMode.IDENTITY,
        "ENV": NormalizationMode.IDENTITY,
    }

    preprocessor, _ = make_pre_post_processors(config)
    processed = preprocessor(
        {
            OBS_STATE: torch.zeros(5),
            ACTION: torch.zeros(config.horizon, 3),
            "task": "put the red mug on the plate",
        }
    )

    assert any(isinstance(step, TokenizerProcessorStep) for step in preprocessor.steps)
    assert processed[OBS_LANGUAGE_TOKENS].shape == (1, 7)
    assert processed[OBS_LANGUAGE_ATTENTION_MASK].dtype == torch.bool


def test_flow_matching_per_sample_loss_and_all_padding():
    policy = FlowMatchingPolicy(make_config())
    batch = make_training_batch()
    batch["action_is_pad"] = torch.ones(2, 6, dtype=torch.bool)

    loss, _ = policy(batch, reduction="none")

    assert loss.shape == (2,)
    torch.testing.assert_close(loss, torch.zeros_like(loss))


def test_flow_matching_guidance_and_action_queue():
    torch.manual_seed(0)
    config = make_config(guidance_scale=1.5)
    policy = FlowMatchingPolicy(config)
    policy.eval()
    observation = {OBS_STATE: torch.randn(2, 5)}
    noise = torch.randn(2, config.horizon, 3)

    chunk = policy.predict_action_chunk(observation, noise=noise)
    first_action = policy.select_action(observation, noise=noise)
    second_action = policy.select_action(observation)

    assert chunk.shape == (2, config.n_action_steps, 3)
    torch.testing.assert_close(first_action, chunk[:, 0])
    torch.testing.assert_close(second_action, chunk[:, 1])


def test_flow_matching_rejects_invalid_noise_shape():
    policy = FlowMatchingPolicy(make_config())
    with pytest.raises(ValueError, match="noise"):
        policy.predict_action_chunk(
            {OBS_STATE: torch.randn(2, 5)},
            noise=torch.randn(2, 5, 3),
        )


def test_flow_matching_processor_factory_normalizes_and_restores_actions():
    config = make_config()
    config.normalization_mapping = {
        "STATE": NormalizationMode.MIN_MAX,
        "ACTION": NormalizationMode.MIN_MAX,
        "VISUAL": NormalizationMode.IDENTITY,
        "ENV": NormalizationMode.IDENTITY,
    }
    stats = {
        OBS_STATE: {"min": torch.zeros(5), "max": torch.full((5,), 10.0)},
        ACTION: {"min": torch.full((3,), -2.0), "max": torch.full((3,), 2.0)},
    }
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=stats)
    batch = {
        OBS_STATE: torch.full((5,), 5.0),
        ACTION: torch.tensor([[-2.0, 0.0, 2.0]] * config.horizon),
    }

    processed = preprocessor(batch)
    restored = postprocessor(torch.tensor([[-1.0, 0.0, 1.0]]))

    assert processed[OBS_STATE].shape == (1, 5)
    torch.testing.assert_close(processed[OBS_STATE], torch.zeros(1, 5))
    torch.testing.assert_close(processed[ACTION][0], torch.tensor([-1.0, 0.0, 1.0]))
    torch.testing.assert_close(restored, torch.tensor([[-2.0, 0.0, 2.0]]))


def test_flow_matching_save_and_load(tmp_path):
    config = make_config()
    policy = FlowMatchingPolicy(config)
    save_directory = tmp_path / "flow_matching"

    policy.save_pretrained(save_directory)
    loaded = FlowMatchingPolicy.from_pretrained(save_directory, config=config)

    torch.testing.assert_close(list(policy.parameters()), list(loaded.parameters()), rtol=0, atol=0)


def test_flow_matching_backbone_uses_separate_learning_rate():
    config = make_config(num_cameras=1)
    policy = FlowMatchingPolicy(config)
    parameter_groups = policy.get_optim_params()

    assert len(parameter_groups) == 2
    assert "lr" not in parameter_groups[0]
    assert parameter_groups[1]["lr"] == config.optimizer_lr_backbone
