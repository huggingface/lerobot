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

from typing import TYPE_CHECKING, Optional

import torch
from torch import nn

from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers.models.gemma.modeling_gemma import (
        GemmaAttention,
        GemmaConfig,
        GemmaDecoderLayer,
        GemmaForCausalLM,
        GemmaMLP,
        GemmaModel,
        GemmaPreTrainedModel,
        GemmaRMSNorm,
    )
else:
    GemmaAttention = None
    GemmaConfig = None
    GemmaDecoderLayer = None
    GemmaForCausalLM = None
    GemmaMLP = None
    GemmaModel = None
    GemmaPreTrainedModel = None
    GemmaRMSNorm = None


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
    cond: Optional[torch.Tensor] = None,
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

    def __init__(self, dim: int, eps: float = 1e-6, cond_dim: Optional[int] = None):
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

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        dtype = x.dtype
        normed = self._norm(x.float()).type_as(x)
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
    from transformers.models.gemma.modeling_gemma import GemmaAttention, GemmaConfig, GemmaMLP

    class _PiGemmaDecoderLayerBase(nn.Module):
        """Decoder layer that uses PiGemmaRMSNorm and _gated_residual, compatible with v5 Gemma."""

        def __init__(self, config: "GemmaConfig", layer_idx: int):
            super().__init__()
            self.hidden_size = config.hidden_size
            self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)
            self.mlp = GemmaMLP(config)
            cond_dim = getattr(config, "adarms_cond_dim", None) if getattr(config, "use_adarms", False) else None
            self.input_layernorm = PiGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim)
            self.post_attention_layernorm = PiGemmaRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim
            )

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values=None,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
            adarms_cond: Optional[torch.Tensor] = None,
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


class PiGemmaModel(GemmaModel if _transformers_available else nn.Module):  # type: ignore[misc]
    """
    GemmaModel extended with AdaRMS (adaptive RMSNorm) and gated residuals when config.use_adarms is True.
    """

    def __init__(self, config: "GemmaConfig", **kwargs):
        super().__init__(config, **kwargs)
        # if not getattr(config, "use_adarms", False):
        #     return
        cond_dim = getattr(config, "adarms_cond_dim", None)
        PiGemmaDecoderLayerBase = _get_pi_gemma_decoder_layer_base()
        self.layers = nn.ModuleList(
            [PiGemmaDecoderLayerBase(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = PiGemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        adarms_cond: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        from transformers.cache_utils import DynamicCache
        from transformers.modeling_outputs import BaseModelOutputWithPast

        # if not getattr(self.config, "use_adarms", False):
        #     kwargs.pop("adarms_cond", None)
        #     return super().forward(
        #         input_ids=input_ids,
        #         attention_mask=attention_mask,
        #         position_ids=position_ids,
        #         past_key_values=past_key_values,
        #         inputs_embeds=inputs_embeds,
        #         use_cache=use_cache,
        #         cache_position=cache_position,
        #         **kwargs,
        #     )

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        try:
            from transformers.masking_utils import create_causal_mask
        except ImportError as e:
            raise ImportError(
                "PiGemmaModel with use_adarms=True requires Transformers v5 (create_causal_mask from masking_utils). "
                "Upgrade with: pip install -U transformers"
            ) from e

        causal_mask = create_causal_mask(
            self.config,
            inputs_embeds,
            attention_mask,
            cache_position,
            past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = hidden_states * normalizer

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                adarms_cond=adarms_cond,
                **kwargs,
            )
        hidden_states, _ = self.norm(hidden_states, cond=adarms_cond)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class PiGemmaForCausalLM(GemmaForCausalLM if _transformers_available else nn.Module):  # type: ignore[misc]
    """
    Causal LM wrapper using PiGemmaModel as the backbone, for consistency with GemmaForCausalLM
    and the language model used in pi0_fast. Use this for the action expert in pi0/pi05.
    """

    def __init__(self, config: "GemmaConfig", **kwargs):
        super().__init__(config, **kwargs)
        self.model = PiGemmaModel(config)


__all__ = [
    "PiGemmaModel",
    "PiGemmaForCausalLM",
    "PiGemmaRMSNorm",
    "_gated_residual",
    "layernorm_forward",
]
