#!/usr/bin/env python

# Copyright 2026 BeingBeyond Ltd. and/or its affiliates.
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

"""Qwen3 mixture-of-transformers layers used by Being-H0.5.

The standard Qwen3 projections, MLPs, normalization, and rotary embeddings come
from Transformers. This module only implements Being-H0.5's packed two-stream
attention: understanding tokens use the Qwen3 weights while state/action tokens
use the checkpoint's smaller ``*_mot_gen`` expert weights.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
)

_compiled_flex_attention = torch.compile(flex_attention)


class BeingH05PackedMoTAttention(Qwen3Attention):
    """Packed Qwen3 attention with a separate action-expert projection stream."""

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        expert_config = config.expert_config
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.q_norm_mot_gen = Qwen3RMSNorm(self.head_dim, eps=expert_config.rms_norm_eps)
        self.k_norm_mot_gen = Qwen3RMSNorm(self.head_dim, eps=expert_config.rms_norm_eps)
        self.q_proj_mot_gen = nn.Linear(
            expert_config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj_mot_gen = nn.Linear(
            expert_config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj_mot_gen = nn.Linear(
            expert_config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj_mot_gen = nn.Linear(
            self.num_heads * self.head_dim,
            expert_config.hidden_size,
            bias=config.attention_bias,
        )

    def forward(
        self,
        packed_sequence_und: Tensor,
        packed_sequence_gen: Tensor,
        sample_lens: Sequence[int],
        attention_mask: BlockMask,
        packed_position_embeddings: tuple[Tensor, Tensor],
        packed_und_token_indexes: Tensor,
        packed_gen_token_indexes: Tensor,
    ) -> tuple[Tensor, Tensor]:
        total_length = packed_sequence_und.shape[0] + packed_sequence_gen.shape[0]
        dtype, device = packed_sequence_und.dtype, packed_sequence_und.device

        queries = torch.zeros((total_length, self.num_heads * self.head_dim), dtype=dtype, device=device)
        keys = torch.zeros(
            (total_length, self.num_key_value_heads * self.head_dim), dtype=dtype, device=device
        )
        values = torch.zeros_like(keys)

        queries[packed_und_token_indexes] = self.q_proj(packed_sequence_und)
        queries[packed_gen_token_indexes] = self.q_proj_mot_gen(packed_sequence_gen)
        keys[packed_und_token_indexes] = self.k_proj(packed_sequence_und)
        keys[packed_gen_token_indexes] = self.k_proj_mot_gen(packed_sequence_gen)
        values[packed_und_token_indexes] = self.v_proj(packed_sequence_und)
        values[packed_gen_token_indexes] = self.v_proj_mot_gen(packed_sequence_gen)

        queries = queries.view(-1, self.num_heads, self.head_dim)
        keys = keys.view(-1, self.num_key_value_heads, self.head_dim)
        values = values.view(-1, self.num_key_value_heads, self.head_dim)

        normalized_queries = torch.zeros_like(queries)
        normalized_keys = torch.zeros_like(keys)
        normalized_queries[packed_und_token_indexes] = self.q_norm(queries[packed_und_token_indexes])
        normalized_queries[packed_gen_token_indexes] = self.q_norm_mot_gen(queries[packed_gen_token_indexes])
        normalized_keys[packed_und_token_indexes] = self.k_norm(keys[packed_und_token_indexes])
        normalized_keys[packed_gen_token_indexes] = self.k_norm_mot_gen(keys[packed_gen_token_indexes])

        cos, sin = packed_position_embeddings
        normalized_queries, normalized_keys = apply_rotary_pos_emb(
            normalized_queries, normalized_keys, cos, sin, unsqueeze_dim=1
        )

        padding = sum(sample_lens) - total_length
        normalized_queries = _pad_sequence(normalized_queries.permute(1, 0, 2), padding)
        normalized_keys = _pad_sequence(normalized_keys.permute(1, 0, 2), padding)
        values = _pad_sequence(values.permute(1, 0, 2), padding)
        attention_output = _compiled_flex_attention(
            normalized_queries.unsqueeze(0),
            normalized_keys.unsqueeze(0),
            values.unsqueeze(0),
            enable_gqa=True,
            block_mask=attention_mask,
        )
        attention_output = attention_output[0, :, :total_length].transpose(0, 1)
        attention_output = attention_output.reshape(total_length, self.num_heads * self.head_dim)

        return (
            self.o_proj(attention_output[packed_und_token_indexes]),
            self.o_proj_mot_gen(attention_output[packed_gen_token_indexes]),
        )


class BeingH05MoTDecoderLayer(nn.Module):
    """One Qwen3 understanding layer paired with one smaller action-expert layer."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        expert_config = config.expert_config
        self.self_attn = BeingH05PackedMoTAttention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.mlp_mot_gen = Qwen3MLP(expert_config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_mot_gen = Qwen3RMSNorm(expert_config.hidden_size, eps=expert_config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_mot_gen = Qwen3RMSNorm(
            expert_config.hidden_size, eps=expert_config.rms_norm_eps
        )

    def forward(
        self,
        packed_sequence_und: Tensor,
        packed_sequence_gen: Tensor,
        sample_lens: Sequence[int],
        attention_mask: BlockMask,
        packed_position_embeddings: tuple[Tensor, Tensor],
        packed_und_token_indexes: Tensor,
        packed_gen_token_indexes: Tensor,
    ) -> tuple[Tensor, Tensor]:
        residual_und, residual_gen = packed_sequence_und, packed_sequence_gen
        hidden_und, hidden_gen = self.self_attn(
            self.input_layernorm(packed_sequence_und),
            self.input_layernorm_mot_gen(packed_sequence_gen),
            sample_lens,
            attention_mask,
            packed_position_embeddings,
            packed_und_token_indexes,
            packed_gen_token_indexes,
        )
        packed_sequence_und = residual_und + hidden_und
        packed_sequence_gen = residual_gen + hidden_gen
        packed_sequence_und = packed_sequence_und + self.mlp(
            self.post_attention_layernorm(packed_sequence_und)
        )
        packed_sequence_gen = packed_sequence_gen + self.mlp_mot_gen(
            self.post_attention_layernorm_mot_gen(packed_sequence_gen)
        )
        return packed_sequence_und, packed_sequence_gen


class BeingH05Qwen3MoTModel(nn.Module):
    """Checkpoint-compatible packed Qwen3-MoT decoder."""

    def __init__(self, config):
        super().__init__()
        expert_config = config.expert_config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            BeingH05MoTDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_mot_gen = Qwen3RMSNorm(expert_config.hidden_size, eps=expert_config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

    def forward(
        self,
        packed_sequence_und: Tensor,
        packed_sequence_gen: Tensor,
        sample_lens: Sequence[int],
        attention_mask: BlockMask,
        packed_position_ids: Tensor,
        packed_und_token_indexes: Tensor,
        packed_gen_token_indexes: Tensor,
    ) -> tuple[Tensor, Tensor]:
        cos, sin = self.rotary_emb(packed_sequence_und, packed_position_ids.unsqueeze(0))
        position_embeddings = (cos.squeeze(0), sin.squeeze(0))
        for layer in self.layers:
            packed_sequence_und, packed_sequence_gen = layer(
                packed_sequence_und,
                packed_sequence_gen,
                sample_lens,
                attention_mask,
                position_embeddings,
                packed_und_token_indexes,
                packed_gen_token_indexes,
            )
        return self.norm(packed_sequence_und), self.norm_mot_gen(packed_sequence_gen)


class BeingH05Qwen3ForCausalLM(nn.Module):
    """Minimal CausalLM owner retaining the released checkpoint's tensor names."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = BeingH05Qwen3MoTModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, **kwargs) -> tuple[Tensor, Tensor]:
        return self.model(**kwargs)


def _pad_sequence(sequence: Tensor, padding: int) -> Tensor:
    if padding == 0:
        return sequence
    return torch.cat([sequence, sequence.new_zeros(sequence.shape[0], padding, sequence.shape[2])], dim=1)
