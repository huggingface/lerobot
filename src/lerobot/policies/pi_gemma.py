# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import create_causal_mask
    from transformers.modeling_layers import GradientCheckpointingLayer
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from transformers.models.gemma.modeling_gemma import (
        GemmaAttention,
        GemmaConfig,
        GemmaForCausalLM,
        GemmaMLP,
        GemmaModel,
    )
    from transformers.models.paligemma.modeling_paligemma import (
        PaliGemmaForConditionalGeneration,
        PaliGemmaModel,
    )
else:
    GemmaAttention = None
    GemmaConfig = None
    GemmaForCausalLM = None
    GemmaMLP = None
    GemmaModel = None
    PaliGemmaModel = None
    PaliGemmaForConditionalGeneration = None
    DynamicCache = None
    GradientCheckpointingLayer = None
    BaseModelOutputWithPast = None
    create_causal_mask = None


def _gated_residual(
    x: torch.Tensor | None,
    y: torch.Tensor | None,
    gate: torch.Tensor | None,
) -> torch.Tensor | None:
    """Gated residual: x + y when gate is None, else x + y * gate."""
    if x is None and y is None:
        return None
    if x is None or y is None:
        return x if x is not None else y
    if gate is None:
        return x + y
    return x + y * gate


def layernorm_forward(
    layernorm: nn.Module,
    x: torch.Tensor,
    cond: torch.Tensor | None = None,
):
    """
    call layernorm and return hidden states and gate
    if cond is not None, use conditional norm
    otherwise, use normal gemma norm
    """
    if cond is not None:
        return layernorm(x, cond=cond)
    else:
        return layernorm(x)


class PiGemmaRMSNorm(nn.Module):
    """
    Adaptive RMSNorm for PI Gemma (AdaRMS).
    When cond_dim is set, uses cond to modulate scale/shift/gate; otherwise behaves like standard GemmaRMSNorm.
    forward(x, cond=None) returns (output, gate) for use with _gated_residual.
    """

    def __init__(self, dim: int, eps: float = 1e-6, cond_dim: int | None = None):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.cond_dim = cond_dim
        if cond_dim is not None:
            self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
            nn.init.zeros_(self.dense.weight)
        else:
            self.weight = nn.Parameter(torch.zeros(dim))
            self.dense = None

    def _norm(self, x):
        # Compute variance in float32 (like the source implementation)
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        # Compute normalization in float32
        normed_inputs = x * torch.rsqrt(var + self.eps)
        return normed_inputs

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        dtype = x.dtype
        normed = self._norm(x)
        if cond is None or self.dense is None:
            normed = normed * (1.0 + self.weight.float())
            return normed.type_as(x), None
        if cond.shape[-1] != self.cond_dim:
            raise ValueError(f"Expected cond dim {self.cond_dim}, got {cond.shape[-1]}")
        modulation = self.dense(cond)
        if len(x.shape) == 3:
            modulation = modulation.unsqueeze(1)
        scale, shift, gate = modulation.chunk(3, dim=-1)
        normed = normed * (1 + scale.float()) + shift.float()
        return normed.to(dtype), gate.to(dtype)

    def extra_repr(self) -> str:
        if self.dense is not None:
            return f"dim={self.dim}, eps={self.eps}, adaptive=True, cond_dim={self.cond_dim}"
        return f"dim={self.dim}, eps={self.eps}"


def _get_pi_gemma_decoder_layer_base():
    """base for PiGemmaDecoderLayer"""

    class _PiGemmaDecoderLayerBase(GradientCheckpointingLayer):
        """Decoder layer that uses PiGemmaRMSNorm and _gated_residual, compatible with v5 Gemma."""

        def __init__(self, config: GemmaConfig, layer_idx: int):
            super().__init__()
            self.hidden_size = config.hidden_size
            self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)
            self.mlp = GemmaMLP(config)
            cond_dim = (
                getattr(config, "adarms_cond_dim", None) if getattr(config, "use_adarms", False) else None
            )
            self.input_layernorm = PiGemmaRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim
            )
            self.post_attention_layernorm = PiGemmaRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim
            )

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_key_values=None,
            use_cache: bool = False,
            cache_position: torch.LongTensor | None = None,
            position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
            adarms_cond: torch.Tensor | None = None,
            **kwargs,
        ) -> torch.Tensor:
            residual = hidden_states
            hidden_states, gate = self.input_layernorm(hidden_states, cond=adarms_cond)
            hidden_states, _ = self.self_attn(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            hidden_states = _gated_residual(residual, hidden_states, gate)

            residual = hidden_states
            hidden_states, gate = self.post_attention_layernorm(hidden_states, cond=adarms_cond)
            hidden_states = self.mlp(hidden_states)
            hidden_states = _gated_residual(residual, hidden_states, gate)
            return hidden_states

    return _PiGemmaDecoderLayerBase


class PiGemmaModel(GemmaModel):  # type: ignore[misc]
    """
    GemmaModel extended with AdaRMS (adaptive RMSNorm) and gated residuals when config.use_adarms is True.
    """

    def __init__(self, config: GemmaConfig, **kwargs):
        super().__init__(config, **kwargs)
        # if not getattr(config, "use_adarms", False):
        #     return
        cond_dim = getattr(config, "adarms_cond_dim", None)
        pi_gemma_decoder_layer_base = _get_pi_gemma_decoder_layer_base()
        self.layers = nn.ModuleList(
            [pi_gemma_decoder_layer_base(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = PiGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: DynamicCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        adarms_cond: torch.Tensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        """
        adarms_cond (`torch.Tensor` of shape `(batch_size, cond_dim)`, *optional*):
            Condition for ADARMS.
        """
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            import logging

            logging.warning(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        # embed positions
        hidden_states = inputs_embeds
        # Convert to bfloat16 if the first layer uses bfloat16
        if len(self.layers) > 0 and self.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.bfloat16)

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # normalized
        # Gemma downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
        # See https://github.com/huggingface/transformers/pull/29402

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                adarms_cond=adarms_cond,
                **kwargs,
            )

            hidden_states = layer_outputs

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states, _ = self.norm(hidden_states, adarms_cond)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class PiGemmaForCausalLM(GemmaForCausalLM):  # type: ignore[misc]
    """
    Causal LM wrapper using PiGemmaModel as the backbone, for consistency with GemmaForCausalLM
    and the language model used in pi0_fast. Use this for the action expert in pi0/pi05.
    """

    def __init__(self, config: GemmaConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = PiGemmaModel(config)


class PaliGemmaModelWithPiGemma(PaliGemmaModel):
    """PaliGemmaModel whose language_model is PiGemmaModel (custom decoder with PiGemmaRMSNorm and gated residuals)."""

    def __init__(self, config):
        super().__init__(config)
        self.language_model = PiGemmaModel(config.text_config)


class PaliGemmaForConditionalGenerationWithPiGemma(PaliGemmaForConditionalGeneration):
    """PaliGemmaForConditionalGeneration using PiGemma decoder for the language model."""

    def __init__(self, config):
        super().__init__(config)
        self.model = PaliGemmaModelWithPiGemma(config)

    # Make modules available through conditional class for BC
    @property
    def language_model(self):
        return self.model.language_model


__all__ = [
    "PiGemmaModel",
    "PiGemmaForCausalLM",
    "PiGemmaRMSNorm",
    "_gated_residual",
    "layernorm_forward",
    "PaliGemmaModelWithPiGemma",
    "PaliGemmaForConditionalGenerationWithPiGemma",
]
