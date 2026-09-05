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

from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from lerobot.policies.pi0.modeling_pi0 import PI0Policy, PI0Pytorch  # noqa: E402
from lerobot.utils.constants import (  # noqa: E402
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


class _FakePrefixModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        config = SimpleNamespace(_attn_implementation=None)
        language_model = SimpleNamespace(config=config)
        self.paligemma = SimpleNamespace(model=SimpleNamespace(language_model=language_model))
        self.forward_calls = 0

    def forward(self, *, inputs_embeds, **kwargs):
        del kwargs
        self.forward_calls += 1
        return [inputs_embeds[0] + 3.0, None], ("prefix-kv",)


def _make_core_model() -> tuple[PI0Pytorch, _FakePrefixModel, torch.Tensor, torch.Tensor]:
    model = PI0Pytorch.__new__(PI0Pytorch)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_inference_steps=2,
        chunk_size=4,
        max_action_dim=3,
        rtc_config=None,
    )
    model.rtc_processor = None
    prefix_model = _FakePrefixModel()
    model.paligemma_with_expert = prefix_model

    prefix_embs = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    prefix_mask = torch.tensor([[True, True, False], [True, True, True]])

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        del self, images, img_masks, lang_tokens, lang_masks
        return prefix_embs, prefix_mask, torch.zeros_like(prefix_mask)

    def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        del self, state, prefix_pad_masks, past_key_values, timestep
        return torch.zeros_like(x_t)

    model.embed_prefix = MethodType(embed_prefix, model)
    model.denoise_step = MethodType(denoise_step, model)
    return model, prefix_model, prefix_embs, prefix_mask


def test_sample_actions_with_context_matches_existing_api_and_uses_one_prefill():
    model, prefix_model, prefix_embs, prefix_mask = _make_core_model()
    state = torch.zeros(2, 5)
    noise = torch.randn(2, 4, 3)
    inputs = {
        "images": [],
        "img_masks": [],
        "lang_tokens": torch.zeros(2, 1, dtype=torch.long),
        "lang_masks": torch.ones(2, 1, dtype=torch.bool),
        "state": state,
        "noise": noise,
        "num_steps": 2,
    }

    old_actions = model.sample_actions(**inputs)
    assert prefix_model.forward_calls == 1
    actions, final_tokens, token_mask = model.sample_actions_with_context(**inputs)

    assert prefix_model.forward_calls == 2
    torch.testing.assert_close(actions, old_actions)
    torch.testing.assert_close(final_tokens, prefix_embs + 3.0)
    torch.testing.assert_close(token_mask, prefix_mask)


def test_encode_prefix_returns_tokens_without_action_denoising():
    model, prefix_model, prefix_embs, prefix_mask = _make_core_model()
    final_tokens, token_mask = model.encode_prefix(
        images=[],
        img_masks=[],
        lang_tokens=torch.zeros(2, 1, dtype=torch.long),
        lang_masks=torch.ones(2, 1, dtype=torch.bool),
    )

    assert prefix_model.forward_calls == 1
    torch.testing.assert_close(final_tokens, prefix_embs + 3.0)
    torch.testing.assert_close(token_mask, prefix_mask)


class _FakeCore(nn.Module):
    def sample_actions_with_context(self, images, img_masks, lang_tokens, lang_masks, state, **kwargs):
        del images, img_masks, lang_tokens, lang_masks, kwargs
        batch_size = state.shape[0]
        actions = torch.arange(batch_size * 4 * 3, dtype=torch.float32).reshape(batch_size, 4, 3)
        tokens = torch.ones(batch_size, 5, 8)
        mask = torch.ones(batch_size, 5, dtype=torch.bool)
        return actions, tokens, mask


def test_policy_context_contains_unpadded_normalized_actions_and_proprio():
    policy = PI0Policy.__new__(PI0Policy)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(output_features={ACTION: SimpleNamespace(shape=(2,))})
    policy.model = _FakeCore()
    policy._preprocess_images = MethodType(lambda self, batch: ([], []), policy)
    policy.prepare_state = MethodType(lambda self, batch: torch.zeros(batch[OBS_STATE].shape[0], 6), policy)

    batch = {
        OBS_STATE: torch.randn(2, 3),
        OBS_LANGUAGE_TOKENS: torch.zeros(2, 1, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 1, dtype=torch.bool),
    }
    actions, context = policy.predict_action_chunk_with_context(batch)

    assert actions.shape == (2, 4, 2)
    torch.testing.assert_close(context.reference_actions, actions)
    assert context.final_tokens.shape == (2, 5, 8)
    torch.testing.assert_close(context.proprio, batch[OBS_STATE])
