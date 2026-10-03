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

"""Tests of the ``noise`` and ``video_noise`` inputs of LingBot-VA's ``predict_action_chunk``."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("diffusers")
pytest.importorskip("transformers")

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.lingbot_va.configuration_lingbot_va import LingBotVAConfig
from lerobot.policies.lingbot_va.modeling_lingbot_va import LingBotVAPolicy
from lerobot.utils.constants import ACTION

IMAGE_KEY = "observation.images.image"


def make_config() -> LingBotVAConfig:
    return LingBotVAConfig(
        device="cpu",
        dtype=torch.float32,
        input_features={IMAGE_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        obs_cam_keys=[IMAGE_KEY],
        height=32,
        width=32,
        num_attention_heads=2,
        attention_head_dim=12,
        action_dim=8,
        text_dim=16,
        freq_dim=16,
        ffn_dim=32,
        num_layers=1,
        num_inference_steps=3,
        action_num_inference_steps=3,
    )


def make_batch() -> dict:
    return {IMAGE_KEY: torch.rand(1, 3, 32, 32, generator=torch.Generator().manual_seed(3)), "task": ["pick"]}


@pytest.fixture
def policy(monkeypatch):
    """A tiny random policy; the frozen VAE, text encoder and tokenizer are replaced by fixed tensors."""

    def encode_prompt(self, prompt):
        gen = torch.Generator().manual_seed(0)
        prompt_embeds = torch.randn(1, 8, self.config.text_dim, generator=gen)
        negative_prompt_embeds = (
            torch.randn(1, 8, self.config.text_dim, generator=gen) if self._use_cfg else None
        )
        return prompt_embeds, negative_prompt_embeds

    def encode_frames(self, raw_frames):
        latent_h, latent_w = self._latent_hw
        return torch.randn(
            1, 48, len(raw_frames), latent_h, latent_w, generator=torch.Generator().manual_seed(1)
        )

    monkeypatch.setattr(LingBotVAPolicy, "_ensure_frozen_modules", lambda self: None)
    monkeypatch.setattr(LingBotVAPolicy, "_encode_prompt", encode_prompt)
    monkeypatch.setattr(LingBotVAPolicy, "_encode_frames", encode_frames)
    torch.manual_seed(0)
    return LingBotVAPolicy(make_config())


def draw_like_policy(policy: LingBotVAPolicy) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw one chunk's starting samples exactly as `LingBotVAPolicy._infer` does (video first)."""
    cfg = policy.config
    latent_h, latent_w = policy._latent_hw
    video_noise = torch.randn(
        1, 48, cfg.frame_chunk_size, latent_h, latent_w, device=cfg.device, dtype=policy.dtype
    )
    noise = torch.randn(
        1,
        cfg.action_dim,
        cfg.frame_chunk_size,
        cfg.action_per_frame,
        1,
        device=cfg.device,
        dtype=policy.dtype,
    )
    return noise, video_noise


def test_predict_action_chunk_uses_the_given_noise(policy):
    noise, video_noise = draw_like_policy(policy)
    given = noise.clone(), video_noise.clone()

    first = policy.predict_action_chunk(make_batch(), noise=noise, video_noise=video_noise)
    policy.reset()  # start a new episode, so both calls predict the first chunk
    second = policy.predict_action_chunk(make_batch(), noise=noise, video_noise=video_noise)

    assert first.shape == (1, policy.config.chunk_size - policy.config.action_per_frame, 7)
    assert torch.equal(first, second)
    assert torch.equal(noise, given[0]) and torch.equal(video_noise, given[1])


def test_predict_action_chunk_without_noise_matches_the_seeded_draw_over_two_chunks(policy):
    torch.manual_seed(42)
    default = [policy.predict_action_chunk(make_batch()), policy.predict_action_chunk(make_batch())]

    policy.reset()
    torch.manual_seed(42)
    first_noise, first_video_noise = draw_like_policy(policy)
    second_noise, second_video_noise = draw_like_policy(policy)
    passed = [
        policy.predict_action_chunk(make_batch(), noise=first_noise, video_noise=first_video_noise),
        policy.predict_action_chunk(make_batch(), noise=second_noise, video_noise=second_video_noise),
    ]

    assert torch.equal(default[0], passed[0])
    assert torch.equal(default[1], passed[1])
