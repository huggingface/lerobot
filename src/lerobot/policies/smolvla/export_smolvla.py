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
"""ONNX export support for SmolVLA policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


class SmolVLAEncoderWrapper(nn.Module):
    """ONNX-compatible encoder wrapper for SmolVLA.

    SmolVLA embeds state in the prefix (unlike PI0 which passes state to denoise step).
    KV cache format is dict-based: {layer_idx: {"key_states": ..., "value_states": ...}}
    with shape [B, seq, heads, dim]. We transpose to [B, heads, seq, dim] for ONNX consistency.
    """

    def __init__(self, policy: SmolVLAPolicy, num_images: int = 1):
        super().__init__()
        self.model = policy.model
        self.config = policy.config
        self.num_images = num_images

        vlm = self.model.vlm_with_expert.vlm
        self.num_layers = self.config.num_vlm_layers
        self.num_kv_heads = vlm.config.text_config.num_key_value_heads
        self.head_dim = vlm.config.text_config.head_dim

    def forward(self, *args) -> tuple[Tensor, ...]:
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        images = list(args[: self.num_images])
        img_masks_float = list(args[self.num_images : 2 * self.num_images])
        lang_tokens = args[2 * self.num_images]
        lang_masks = args[2 * self.num_images + 1]
        state = args[2 * self.num_images + 2]

        img_masks = [m > 0.5 for m in img_masks_float]

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks.long(), dim=1) - 1

        _, past_key_values = self.model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )

        outputs = [prefix_pad_masks.float()]

        for layer_idx in range(self.num_layers):
            layer_cache = past_key_values[layer_idx]
            key = layer_cache["key_states"].transpose(1, 2)
            value = layer_cache["value_states"].transpose(1, 2)
            outputs.append(key)
            outputs.append(value)

        return tuple(outputs)


class SmolVLADenoiseWrapper(nn.Module):
    """ONNX-compatible denoise wrapper. State is embedded in prefix, not passed here.

    KV cache: dict[layer_idx] = {"key_states": [B, seq, heads, dim], ...}
    ONNX outputs [B, heads, seq, dim], so we transpose back and call .contiguous()
    since SmolVLA's forward_cross_attn_layer uses .view() which requires contiguous memory.
    """

    def __init__(self, policy: SmolVLAPolicy):
        super().__init__()
        self.model = policy.model
        self.config = policy.config

        vlm = self.model.vlm_with_expert.vlm
        self.num_layers = self.config.num_vlm_layers
        self.num_kv_heads = vlm.config.text_config.num_key_value_heads
        self.head_dim = vlm.config.text_config.head_dim

    def forward(
        self,
        x_t: Tensor,
        timestep: Tensor,
        prefix_pad_mask: Tensor,
        *kv_cache_tensors: Tensor,
    ) -> Tensor:
        past_key_values = {}
        for layer_idx in range(self.num_layers):
            key = kv_cache_tensors[layer_idx * 2]
            value = kv_cache_tensors[layer_idx * 2 + 1]
            past_key_values[layer_idx] = {
                "key_states": key.transpose(1, 2).contiguous(),
                "value_states": value.transpose(1, 2).contiguous(),
            }

        prefix_pad_masks = prefix_pad_mask > 0.5

        v_t = self.model.denoise_step(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=x_t,
            timestep=timestep,
        )

        return v_t
