#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

"""Wall-X Mixture-of-Experts additions on top of the native transformers Qwen2.5-VL model.

It is rebased on the native ``transformers.models.qwen2_5_vl`` classes and only keeps what Wall-X genuinely adds:

- ``BlockSparseMLP`` / ``SparseMoeBlock``: hard-routed (token-type indexed) expert MLPs.
- ``Qwen2_5_VLDecoderLayer_with_MoE``: native decoder layer whose MLP is replaced by the sparse MoE
  block and whose forward casts activations to the parameter dtypes (Wall-X keeps the layernorms in
  float32 while the projections run in bfloat16, see ``to_bfloat16_for_selected_params``).
- ``Qwen2_5_VLMoEModel``: native text model with MoE decoder layers and a ``moe_token_types``-aware
  causal-mask override (tokens of type 1 — the action tokens — attend to each other bidirectionally,
  everything else stays causal).
- ``Qwen2_5_VLACausalLMOutputWithPast``: output dataclass with the extra Wall-X loss fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers.activations import ACT2FN
    from transformers.cache_utils import Cache, DynamicCache
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLDecoderLayer,
        Qwen2_5_VLTextModel,
    )
    from transformers.utils.generic import merge_with_config_defaults
    from transformers.utils.output_capturing import capture_outputs
else:
    ACT2FN = None
    Cache = None
    DynamicCache = None
    create_causal_mask = None
    create_sliding_window_causal_mask = None
    BaseModelOutputWithPast = object
    ModelOutput = object
    Qwen2_5_VLDecoderLayer = nn.Module
    Qwen2_5_VLTextModel = nn.Module

    def merge_with_config_defaults(func):
        return func

    def capture_outputs(func):
        return func


from .configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLTextConfig


@dataclass
class Qwen2_5_VLACausalLMOutputWithPast(ModelOutput):  # noqa: N801
    loss: torch.FloatTensor | None = None
    flow_loss: torch.FloatTensor | None = None
    cross_entropy_loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: list[torch.FloatTensor] | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    rope_deltas: torch.LongTensor | None = None

    channel_loss_dict: dict[torch.FloatTensor] | None = None
    channel_loss_count_dict: dict[torch.FloatTensor] | None = None


class BlockSparseMLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_size = config["hidden_size"]
        self.intermediate_size = config["intermediate_size"]
        self.hidden_act = config["hidden_act"]
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[self.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class SparseMoeBlock(nn.Module):
    def __init__(self, config, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([BlockSparseMLP(config.experts[i]) for i in range(num_experts)])

        if not hasattr(config, "dim_inputs") or not config.dim_inputs:
            raise ValueError("Config must contain valid dim_inputs")

        self.dim_inputs = config.dim_inputs

    def forward(self, hidden_states: torch.Tensor, experts_indices: torch.Tensor) -> torch.Tensor:
        """
        Route different hidden_states to corresponding experts for processing.

        Args:
            hidden_states (torch.Tensor): Tensor of shape (batch_size, seq_length, hidden_dim).
            experts_indices (torch.Tensor): Tensor of shape (batch_size, seq_length),
                indicating the expert index assigned to each token.

        Returns:
            output (torch.Tensor): Tensor of shape (batch_size, seq_length, hidden_dim).
        """
        batch_size, seq_length, hidden_dim = hidden_states.size()
        output = torch.zeros_like(hidden_states)

        for expert_idx, expert in enumerate(self.experts):
            mask = experts_indices == expert_idx
            if mask.sum() == 0:
                continue
            dim_input = self.dim_inputs[expert_idx]

            selected_hidden = hidden_states[mask]
            processed_hidden = expert(selected_hidden[:, :dim_input])

            batch_indices, seq_indices = torch.where(mask)
            output[batch_indices, seq_indices, :dim_input] = processed_hidden

        return output


class Qwen2_5_VLDecoderLayer_with_MoE(Qwen2_5_VLDecoderLayer):  # noqa: N801
    """Native Qwen2.5-VL decoder layer with an optional hard-routed sparse-MoE MLP.

    Differences from the native layer forward:
    - routes the post-attention hidden states through ``SparseMoeBlock`` keyed on ``token_types``
      when ``config.mlp_moe`` is set;
    - casts activations to the parameter dtype before each block, since Wall-X runs with float32
      layernorms and bfloat16 projections in the same module.
    """

    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: int, num_experts: int):
        super().__init__(config, layer_idx)
        if config.mlp_moe:
            del self.mlp
            self.mlp = None
            self.moe = SparseMoeBlock(config, num_experts=num_experts)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        token_types: torch.LongTensor | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = hidden_states.to(self.input_layernorm.weight.dtype)
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = hidden_states.to(self.self_attn.q_proj.weight.dtype)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = hidden_states.to(self.post_attention_layernorm.weight.dtype)
        hidden_states = self.post_attention_layernorm(hidden_states)
        if self.mlp is None:  # using moe mlp
            hidden_states = hidden_states.to(self.moe.experts[0].down_proj.weight.dtype)
            hidden_states = self.moe(hidden_states, token_types)
        else:
            hidden_states = hidden_states.to(self.mlp.down_proj.weight.dtype)
            hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states
        return hidden_states


class Qwen2_5_VLMoEModel(Qwen2_5_VLTextModel):  # noqa: N801
    """Qwen2.5-VL text model with Mixture of Experts (MoE) decoder layers.

    Extends the native ``Qwen2_5_VLTextModel`` with per-token-type expert routing and a causal-mask
    override that gives action-token blocks (``moe_token_types == 1``) bidirectional attention among
    themselves while keeping causal attention everywhere else.
    """

    config_class = Qwen2_5_VLTextConfig
    _no_split_modules = ["Qwen2_5_VLDecoderLayer_with_MoE"]

    def __init__(self, config: Qwen2_5_VLConfig | Qwen2_5_VLTextConfig):
        text_config = config.text_config if isinstance(config, Qwen2_5_VLConfig) else config
        self._require_eager_attention(text_config._attn_implementation)
        # Transformers selects SDPA automatically when no implementation is
        # requested. Wall-X's action-token islands require an explicit 4D
        # mask, so opt into eager before PreTrainedModel performs that choice.
        text_config._attn_implementation = "eager"
        super().__init__(text_config)
        # Free the parent-allocated dense layers before replacing them (pi_gemma.py precedent).
        del self.layers
        self.layers = nn.ModuleList(
            [
                Qwen2_5_VLDecoderLayer_with_MoE(text_config, layer_idx, text_config.num_experts)
                for layer_idx in range(text_config.num_hidden_layers)
            ]
        )
        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def _require_eager_attention(attn_implementation: str | None) -> None:
        if attn_implementation not in {None, "eager"}:
            raise ValueError(
                "Wall-X currently supports only attn_implementation='eager'. "
                "Its bidirectional action-token islands cannot be represented "
                f"correctly by {attn_implementation!r}."
            )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    @merge_with_config_defaults
    # ``capture_outputs`` reads output_hidden_states/output_attentions from kwargs and populates
    # BaseModelOutputWithPast via hooks on the decoder layers and attention modules.
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        moe_token_types: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        self._require_eager_attention(self.config._attn_implementation)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if moe_token_types is None:
            raise ValueError("moe_token_types must be provided for MoE routing")

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if moe_token_types.shape[-1] < inputs_embeds.shape[1]:
            raise ValueError(
                "moe_token_types must cover every input token; got "
                f"{moe_token_types.shape[-1]} types for {inputs_embeds.shape[1]} tokens"
            )
        routing_token_types = moe_token_types[:, -inputs_embeds.shape[1] :]

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if position_ids is None:
            position_ids = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            )
            position_ids = position_ids.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # Native Qwen uses a fourth, text-only position-id row to describe
        # packed sequences. Keep the three multimodal rows for RoPE and pass
        # the text row to both masking and decoder attention.
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        full_token_types = self._prepare_action_token_types(
            moe_token_types,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        action_island_mask = self._action_island_mask(full_token_types)
        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": text_position_ids,
            "or_mask_function": action_island_mask,
        }
        causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
        if self.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                token_types=routing_token_types,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )

    @staticmethod
    def _action_island_mask(token_types: torch.Tensor):
        action_tokens = token_types == 1

        def action_island(batch_idx, head_idx, query_idx, key_idx):
            del head_idx
            return action_tokens[batch_idx, query_idx] & action_tokens[batch_idx, key_idx]

        return action_island

    @staticmethod
    def _prepare_action_token_types(
        token_types: torch.Tensor,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
    ) -> torch.Tensor:
        """Align current-step token types with absolute mask indices.

        Generation passes token types only for the new query tokens, while the
        native mask callbacks receive absolute query/key indices. Cached tokens
        default to type 0; callers may instead pass full-history token types.
        """
        query_length = inputs_embeds.shape[1]
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if isinstance(past_length, torch.Tensor):
            past_length = int(past_length.item())

        token_types = token_types.to(device=inputs_embeds.device)
        if token_types.shape[-1] == query_length and past_length:
            token_types = torch.nn.functional.pad(token_types, (past_length, 0))

        required_length = past_length + query_length
        if attention_mask is not None and attention_mask.ndim == 2:
            required_length = max(required_length, attention_mask.shape[-1])
        if past_key_values is not None:
            kv_length, kv_offset = past_key_values.get_mask_sizes(query_length, 0)
            if isinstance(kv_length, torch.Tensor):
                kv_length = int(kv_length.item())
            if isinstance(kv_offset, torch.Tensor):
                kv_offset = int(kv_offset.item())
            required_length = max(required_length, kv_length + kv_offset)

        if token_types.shape[-1] < required_length:
            token_types = torch.nn.functional.pad(token_types, (0, required_length - token_types.shape[-1]))
        return token_types
