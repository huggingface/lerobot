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
"""ONNX export support for PI0 policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy


class PI0EncoderWrapper(nn.Module):
    """ONNX-compatible encoder wrapper for PI0."""

    def __init__(self, policy: PI0Policy, num_images: int = 1):
        super().__init__()
        self.model = policy.model
        self.config = policy.config
        self.num_images = num_images

        paligemma = self.model.paligemma_with_expert.paligemma
        self.num_layers = paligemma.config.text_config.num_hidden_layers
        self.num_kv_heads = paligemma.config.text_config.num_key_value_heads
        self.head_dim = paligemma.config.text_config.head_dim

    def forward(self, *args) -> tuple[Tensor, ...]:
        from lerobot.policies.pi0.modeling_pi0 import make_att_2d_masks

        images = list(args[: self.num_images])
        img_masks_float = list(args[self.num_images : 2 * self.num_images])
        lang_tokens = args[2 * self.num_images]
        lang_masks = args[2 * self.num_images + 1]

        img_masks = [m > 0.5 for m in img_masks_float]

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks.long(), dim=1) - 1
        prefix_att_2d_masks_4d = self.model._prepare_attention_masks_4d(prefix_att_2d_masks)

        _, past_key_values = self.model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        outputs = [prefix_pad_masks.float()]
        for layer_idx in range(self.num_layers):
            key = past_key_values.key_cache[layer_idx]
            value = past_key_values.value_cache[layer_idx]
            outputs.append(key)
            outputs.append(value)

        return tuple(outputs)


class PI0DenoiseWrapper(nn.Module):
    """ONNX-compatible denoise step wrapper for PI0."""

    def __init__(self, policy: PI0Policy):
        super().__init__()
        self.model = policy.model
        self.config = policy.config

        paligemma = self.model.paligemma_with_expert.paligemma
        self.num_layers = paligemma.config.text_config.num_hidden_layers

    def forward(
        self,
        state: Tensor,
        x_t: Tensor,
        timestep: Tensor,
        prefix_pad_mask: Tensor,
        *kv_cache_tensors: Tensor,
    ) -> Tensor:
        from transformers import DynamicCache

        past_key_values = DynamicCache()
        for layer_idx in range(self.num_layers):
            key = kv_cache_tensors[layer_idx * 2]
            value = kv_cache_tensors[layer_idx * 2 + 1]
            past_key_values.update(key, value, layer_idx)

        prefix_pad_masks = prefix_pad_mask > 0.5

        v_t = self.model.denoise_step(
            state=state,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=x_t,
            timestep=timestep,
        )

        return v_t
