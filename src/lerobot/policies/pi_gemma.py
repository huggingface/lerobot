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

from typing import TYPE_CHECKING, Literal

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from lerobot.utils.import_utils import _transformers_available

# Default PaliGemma SigLIP input resolution. Mirrors
# ``pi05.configuration_pi05.DEFAULT_IMAGE_SIZE``; duplicated as a plain constant
# to avoid importing the pi05 package here (which would create an import cycle:
# pi_gemma -> pi05.__init__ -> modeling_pi05 -> pi_gemma).
DEFAULT_IMAGE_SIZE = 224

if TYPE_CHECKING or _transformers_available:
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import create_causal_mask
    from transformers.modeling_layers import GradientCheckpointingLayer
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.gemma import modeling_gemma
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
    CONFIG_MAPPING = None
    modeling_gemma = None


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
        # Per-sample cond (B, cond_dim) → broadcast over the sequence. A
        # per-token cond (B, T, cond_dim) is already aligned with x and must
        # not be unsqueezed (used by pi052's amortized K_repeat path).
        if len(x.shape) == 3 and modulation.dim() == 2:
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
        # Free parent-allocated layers/norm before replacing to avoid ~2x peak memory.
        del self.layers
        del self.norm
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
            inputs_embeds=inputs_embeds,
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
        if causal_mask is not None and torch.is_floating_point(causal_mask):
            causal_mask = causal_mask.to(dtype=hidden_states.dtype)

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
        del self.model
        self.model = PiGemmaModel(config)


class PaliGemmaModelWithPiGemma(PaliGemmaModel):
    """PaliGemmaModel whose language_model is PiGemmaModel (custom decoder with PiGemmaRMSNorm and gated residuals)."""

    def __init__(self, config):
        super().__init__(config)
        del self.language_model
        self.language_model = PiGemmaModel(config.text_config)


class PaliGemmaForConditionalGenerationWithPiGemma(PaliGemmaForConditionalGeneration):
    """PaliGemmaForConditionalGeneration using PiGemma decoder for the language model."""

    def __init__(self, config):
        super().__init__(config)
        del self.model
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


# PI0.5 / PI052 dual-expert backbone: generic PaliGemma + Gemma action-expert
# transformer machinery used by the pi052 policy. GemmaVariantConfig is openpi's
# width/depth variant config (renamed from GemmaConfig to avoid clashing with
# transformers' GemmaConfig).


def sdpa_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
):
    """Drop-in for ``modeling_gemma.eager_attention_forward`` using
    ``torch.nn.functional.scaled_dot_product_attention``.

    PyTorch SDPA picks the memory-efficient kernel for arbitrary additive
    bias masks (the FA backend only accepts causal/sliding-window). On
    H100 that is ~1.3-1.7x faster and uses ~30-40% less attention memory
    than the eager softmax(QK^T)+matmul path. Mirrors eager's signature
    and output shape (``(B, Lq, H, D)``) so call sites are unchanged.
    """
    n_rep = module.num_key_value_groups
    if n_rep > 1:
        key = key.repeat_interleave(n_rep, dim=1)
        value = value.repeat_interleave(n_rep, dim=1)
    if attention_mask is not None and attention_mask.dtype != query.dtype:
        attention_mask = attention_mask.to(dtype=query.dtype)
    attn_output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout if module.training else 0.0,
        is_causal=False,
        scale=scaling,
    )
    return attn_output.transpose(1, 2).contiguous(), None


# Define the complete layer computation function for gradient checkpointing
def compute_layer_complete(
    layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond, paligemma, gemma_expert
):
    models = [paligemma.model.language_model, gemma_expert.model]
    query_states = []
    key_states = []
    value_states = []
    gates = []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        hidden_states, gate = layernorm_forward(layer.input_layernorm, hidden_states, adarms_cond[i])
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)
    # Concatenate and process attention
    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )
    batch_size = query_states.shape[0]
    scaling = paligemma.model.language_model.layers[layer_idx].self_attn.scaling
    att_output, _ = sdpa_attention_forward(
        paligemma.model.language_model.layers[layer_idx].self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )
    # Get head_dim from the current layer, not from the model
    head_dim = paligemma.model.language_model.layers[layer_idx].self_attn.head_dim
    att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)
    # Process layer outputs
    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        # first residual
        out_emb = _gated_residual(hidden_states, out_emb, gates[i])
        after_first_residual = out_emb.clone()
        out_emb, gate = layernorm_forward(layer.post_attention_layernorm, out_emb, adarms_cond[i])
        # Convert to bfloat16 if the next layer (mlp) uses bfloat16
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        # second residual
        out_emb = _gated_residual(after_first_residual, out_emb, gate)
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds


class GemmaVariantConfig:  # see openpi `gemma.py: Config`
    """Configuration for Gemma model variants."""

    def __init__(self, width, depth, mlp_dim, num_heads, num_kv_heads, head_dim):
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def get_gemma_config(variant: str) -> GemmaVariantConfig:  # see openpi `gemma.py: get_config`
    """Returns config for specified gemma variant."""
    if variant == "gemma_300m":
        return GemmaVariantConfig(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    elif variant == "gemma_2b":
        return GemmaVariantConfig(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


class PaliGemmaWithExpertModel(
    nn.Module
):  # see openpi `gemma_pytorch.py: PaliGemmaWithExpertModel` this class is almost a exact copy of PaliGemmaWithExpertModel in openpi
    """PaliGemma model with action expert for PI05."""

    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        image_size: int = DEFAULT_IMAGE_SIZE,
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.image_size = image_size
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGenerationWithPiGemma(config=vlm_config_hf)
        self.gemma_expert = PiGemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)
        self._set_requires_grad()

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        # Keep full vision path in float32 so we never toggle (toggle causes optimizer
        # "same dtype" error). Saves memory vs full float32; more memory than only 3 params.
        params_to_keep_float32 = [
            "vision_tower",
            "multi_modal_projector",
            "lm_head",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def _set_requires_grad(self):
        if self.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
            for param in self.paligemma.model.vision_tower.parameters():
                param.requires_grad = False
        if self.train_expert_only:
            self.paligemma.eval()
            for param in self.paligemma.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
        if self.train_expert_only:
            self.paligemma.eval()

    def embed_image(self, image: torch.Tensor):
        # Vision tower and multi_modal_projector are kept in float32 (params_to_keep_float32).
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        image_outputs = self.paligemma.model.get_image_features(image)
        # OpenPI / big_vision convention: image (soft) tokens are NOT scaled by the
        # Gemma embedder normalizer (sqrt(hidden_size)) — only text tokens are. lerobot/pi05_base
        # was trained in this regime, so scaling image features here over-scales them ~45x and
        # breaks the pretrained vision-language alignment. Keep image features un-normalized.
        features = image_outputs.pooler_output
        if features.dtype != out_dtype:
            features = features.to(out_dtype)
        return features

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.model.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.model.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            models = [self.paligemma.model.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )

            # final norm
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = layernorm_forward(models[i].norm, hidden_states, adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms,
                    inputs_embeds,
                    adarms_cond,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values
