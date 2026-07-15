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

from lerobot.policies.pi052.modeling_pi052 import PI05Pytorch


class _MockVisionTower:
    def __init__(self):
        self.enable_kwargs = None
        self.disable_calls = 0

    def gradient_checkpointing_enable(self, **kwargs):
        self.enable_kwargs = kwargs

    def gradient_checkpointing_disable(self):
        self.disable_calls += 1


def _checkpoint_model():
    tower = _MockVisionTower()
    language_model = SimpleNamespace(gradient_checkpointing=False)
    expert_model = SimpleNamespace(gradient_checkpointing=False)
    model = PI05Pytorch.__new__(PI05Pytorch)
    nn.Module.__init__(model)
    model.gradient_checkpointing_enabled = False
    model.paligemma_with_expert = SimpleNamespace(
        paligemma=SimpleNamespace(model=SimpleNamespace(language_model=language_model, vision_tower=tower)),
        gemma_expert=SimpleNamespace(model=expert_model),
    )
    return model, tower, language_model, expert_model


def test_gradient_checkpointing_uses_vision_tower_layer_api():
    model, tower, language_model, expert_model = _checkpoint_model()

    PI05Pytorch.gradient_checkpointing_enable(model)

    assert model.gradient_checkpointing_enabled
    assert language_model.gradient_checkpointing
    assert expert_model.gradient_checkpointing
    assert tower.enable_kwargs == {"gradient_checkpointing_kwargs": {"use_reentrant": False}}

    PI05Pytorch.gradient_checkpointing_disable(model)

    assert not model.gradient_checkpointing_enabled
    assert not language_model.gradient_checkpointing
    assert not expert_model.gradient_checkpointing
    assert tower.disable_calls == 1


def test_siglip_layers_recompute_individually():
    from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
    from transformers.models.siglip.modeling_siglip import SiglipVisionModel

    config = SiglipVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_channels=3,
        image_size=16,
        patch_size=8,
    )
    tower = SiglipVisionModel(config).train()
    tower.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    calls = [0] * config.num_hidden_layers

    for index, layer in enumerate(tower.vision_model.encoder.layers):
        original_forward = layer.forward

        def counted_forward(self, *args, _index=index, _forward=original_forward, **kwargs):
            calls[_index] += 1
            return _forward(*args, **kwargs)

        layer.forward = MethodType(counted_forward, layer)

    pixels = torch.randn(2, config.num_channels, config.image_size, config.image_size)
    tower(pixels).last_hidden_state.sum().backward()

    assert calls == [2] * config.num_hidden_layers


def test_embed_prefix_does_not_wrap_the_whole_vision_tower_checkpoint():
    model = PI05Pytorch.__new__(PI05Pytorch)
    nn.Module.__init__(model)
    model.config = SimpleNamespace()
    model.gradient_checkpointing_enabled = True
    model.train()

    image_calls = []

    def embed_image(image):
        image_calls.append(image.shape)
        return image[:, :1, 0, :2]

    def embed_language_tokens(tokens):
        return tokens.to(torch.float32).unsqueeze(-1).expand(*tokens.shape, 2)

    model.paligemma_with_expert = SimpleNamespace(
        embed_image=embed_image,
        embed_language_tokens=embed_language_tokens,
    )
    outer_checkpoint_calls = []

    def apply_checkpoint(func, value):
        outer_checkpoint_calls.append(value.shape)
        return func(value)

    model._apply_checkpoint = apply_checkpoint

    images = [torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 4)]
    image_masks = [torch.ones(2, dtype=torch.bool) for _ in images]
    tokens = torch.ones(2, 3, dtype=torch.long)
    token_masks = torch.ones_like(tokens, dtype=torch.bool)

    embeddings, _, _ = model.embed_prefix(images, image_masks, tokens, token_masks)

    assert image_calls == [image.shape for image in images]
    assert outer_checkpoint_calls == [tokens.shape]
    assert embeddings.shape == (2, 5, 2)
