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

"""
PiStar06: Advantage-Conditioned Pi0.5 Policy (Memory-Efficient)

Standalone implementation that avoids the deep PaliGemma inheritance chain
in pi_gemma.py.  The original pattern (PaliGemmaForConditionalGeneration ->
PaliGemmaModel -> GemmaModel) builds the full parent model at each level
via super().__init__(), then immediately discards and replaces major
submodules, causing ~3x peak memory.

This file builds each component exactly once using flat nn.Module
composition while preserving identical attribute paths so that the
training script (recap_train_pi_star.py) works without modification.

Two-phase workflow:
  1. Train a value network with RECAPTrainSmolVLANetwork
  2. Train PiStar06Policy using pre-computed advantages from the frozen VN
"""

from __future__ import annotations

import builtins
import copy
import csv
import logging
import math
import resource
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers import AutoModel
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import create_causal_mask
    from transformers.modeling_layers import GradientCheckpointingLayer
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.gemma import modeling_gemma
    from transformers.models.gemma.modeling_gemma import (
        GemmaAttention,
        GemmaConfig,
        GemmaMLP,
        GemmaRotaryEmbedding,
        GemmaTextScaledWordEmbedding,
    )
    from transformers.models.paligemma.modeling_paligemma import (
        PaliGemmaMultiModalProjector,
    )
else:
    AutoModel = None
    DynamicCache = None
    create_causal_mask = None
    GradientCheckpointingLayer = None
    BaseModelOutputWithPast = None
    CONFIG_MAPPING = None
    modeling_gemma = None
    GemmaAttention = None
    GemmaConfig = None
    GemmaMLP = None
    GemmaRotaryEmbedding = None
    GemmaTextScaledWordEmbedding = None
    PaliGemmaMultiModalProjector = None

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pistar06.configuration_pistar06 import PiStar06Config
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OPENPI_ATTENTION_MASK_VALUE,
)

DEFAULT_IMAGE_SIZE = 224


def _log_mem(label: str) -> None:
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    parts = [f"[PI_GEMMA_MEM {label}] RSS={rss_mb:.0f}MB"]
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024**2)
        parts.append(f"CUDA_alloc={alloc:.0f}MB")
    logging.info(" ".join(parts))


# ═══════════════════════════════════════════════════════════════════════════════
# Pi-Gemma components (inlined from pi_gemma.py to avoid inheritance chain m)
# ═══════════════════════════════════════════════════════════════════════════════


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
    """Call layernorm; use conditional norm when cond is provided."""
    if cond is not None:
        return layernorm(x, cond=cond)
    else:
        return layernorm(x)


class PiGemmaRMSNorm(nn.Module):
    """Adaptive RMSNorm (AdaRMS) for Pi-Gemma.

    When cond_dim is set, uses cond to modulate scale/shift/gate;
    otherwise behaves like standard GemmaRMSNorm.
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
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
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
        modulation = self.dense(cond.to(dtype=self.dense.weight.dtype))
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
    """Factory for PiGemmaDecoderLayer base class."""

    class _PiGemmaDecoderLayerBase(GradientCheckpointingLayer):
        """Decoder layer with PiGemmaRMSNorm and gated residuals."""

        def __init__(self, config: GemmaConfig, layer_idx: int):
            super().__init__()
            self.hidden_size = config.hidden_size
            self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)
            self.mlp = GemmaMLP(config)
            cond_dim = (
                getattr(config, "adarms_cond_dim", None)
                if getattr(config, "use_adarms", False)
                else None
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


# ═══════════════════════════════════════════════════════════════════════════════
# Flat model components — each component built exactly once (no HF inheritance)
# ═══════════════════════════════════════════════════════════════════════════════


class PiGemmaModelDirect(nn.Module):
    """GemmaModel-equivalent with PiGemma custom layers.

    Built without GemmaModel.__init__() to avoid allocating standard
    GemmaDecoderLayers that would be immediately discarded.
    """

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = GemmaTextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            embed_scale=config.hidden_size**0.5,
        )

        pi_gemma_decoder_layer_base = _get_pi_gemma_decoder_layer_base()
        self.layers = nn.ModuleList(
            [
                pi_gemma_decoder_layer_base(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        cond_dim = getattr(config, "adarms_cond_dim", None)
        self.norm = PiGemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, cond_dim=cond_dim
        )
        self.rotary_emb = GemmaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

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
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logging.warning(
                "`use_cache=True` is incompatible with gradient checkpointing. "
                "Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length()
                if past_key_values is not None
                else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
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

        hidden_states = inputs_embeds
        if (
            len(self.layers) > 0
            and self.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16
        ):
            hidden_states = hidden_states.to(torch.bfloat16)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

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

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class PaliGemmaInnerDirect(nn.Module):
    """PaliGemmaModel-equivalent built without PaliGemmaModel.__init__().

    Constructs vision tower, projector, and language model directly —
    each component is allocated exactly once.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_tower = AutoModel.from_config(config=config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.language_model = PiGemmaModelDirect(config.text_config)

    def get_image_features(self, pixel_values: torch.FloatTensor):
        """Run vision tower and project features (matches HF PaliGemmaModel API)."""
        image_outputs = self.vision_tower(pixel_values)
        selected_image_feature = image_outputs.last_hidden_state
        image_features = self.multi_modal_projector(selected_image_feature)
        image_outputs.pooler_output = image_features
        return image_outputs


class PaliGemmaDirect(nn.Module):
    """PaliGemmaForConditionalGeneration-equivalent without HF model inheritance.

    Preserves .model.X and .config.X attribute paths for compatibility with
    PaliGemmaWithExpertModel and recap_train_pi_star.py.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = PaliGemmaInnerDirect(config)

    @property
    def language_model(self):
        return self.model.language_model


class ActionExpertDirect(nn.Module):
    """PiGemmaForCausalLM-equivalent without GemmaForCausalLM inheritance.

    Preserves .model.X and .config attribute paths.
    """

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.model = PiGemmaModelDirect(config)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions (from modeling_pi05.py)
# ═══════════════════════════════════════════════════════════════════════════════


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device="cpu",
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError(
            "The time tensor is expected to be of shape `(batch_size, )`."
        )
    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.tensor(alpha, dtype=torch.float32)
    beta_t = torch.tensor(beta, dtype=torch.float32)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,)).to(device)


def make_att_2d_masks(pad_masks, att_masks):
    """Build 2D attention masks from padding and autoregressive masks."""
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def pad_vector(vector, new_dim):
    """Pad the last dimension of a vector to new_dim with zeros."""
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def resize_with_pad_torch(
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resize an image to target height/width without distortion by padding."""
    if images.shape[-1] <= 4:
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)
        images = images.permute(0, 3, 1, 2)
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)

    batch_size, channels, cur_height, cur_width = images.shape
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    constant_value = 0 if images.dtype == torch.uint8 else 0.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),
        mode="constant",
        value=constant_value,
    )

    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)

    return padded_images


def compute_layer_complete(
    layer_idx,
    inputs_embeds,
    attention_mask,
    position_ids,
    adarms_cond,
    paligemma,
    gemma_expert,
):
    """Compute a single transformer layer across both VLM and expert models."""
    models = [paligemma.model.language_model, gemma_expert.model]
    query_states = []
    key_states = []
    value_states = []
    gates = []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        hidden_states, gate = layernorm_forward(
            layer.input_layernorm, hidden_states, adarms_cond[i]
        )
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_state = (
            layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        )
        key_state = (
            layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        )
        value_state = (
            layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        )
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)

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
    att_output, _ = modeling_gemma.eager_attention_forward(
        paligemma.model.language_model.layers[layer_idx].self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )
    head_dim = paligemma.model.language_model.layers[layer_idx].self_attn.head_dim
    att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        out_emb = _gated_residual(hidden_states, out_emb, gates[i])
        after_first_residual = out_emb.clone()
        out_emb, gate = layernorm_forward(
            layer.post_attention_layernorm, out_emb, adarms_cond[i]
        )
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        out_emb = _gated_residual(after_first_residual, out_emb, gate)
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds


# ═══════════════════════════════════════════════════════════════════════════════
# Model variant configs
# ═══════════════════════════════════════════════════════════════════════════════


class GemmaVariantConfig:
    """Configuration for Gemma model variants (width, depth, etc.)."""

    def __init__(self, width, depth, mlp_dim, num_heads, num_kv_heads, head_dim):
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def get_gemma_config(variant: str) -> GemmaVariantConfig:
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


# ═══════════════════════════════════════════════════════════════════════════════
# PaliGemma with Expert (flat composition — no init-then-replace)
# ═══════════════════════════════════════════════════════════════════════════════


class PaliGemmaWithExpertModel(nn.Module):
    """PaliGemma VLM + action expert using flat nn.Module composition.

    Replaces the original version that inherited from HuggingFace model classes.
    Each sub-component (vision tower, language model, expert) is built exactly
    once, avoiding the ~3x peak memory of the init-then-replace pattern.
    """

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
        vlm_config_hf.text_config.dtype = precision
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = (
            vlm_config.width if use_adarms[0] else None
        )
        vlm_config_hf.vision_config.image_size = image_size
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = vlm_config.width
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.dtype = precision

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            dtype=precision,
            use_adarms=use_adarms[1],
            adarms_cond_dim=(
                action_expert_config.width if use_adarms[1] else None
            ),
        )

        _log_mem("PaliGemmaWithExpert: before paligemma init")
        self.paligemma = PaliGemmaDirect(config=vlm_config_hf)
        _log_mem("PaliGemmaWithExpert: after paligemma init")
        self.gemma_expert = ActionExpertDirect(config=action_expert_config_hf)
        _log_mem("PaliGemmaWithExpert: after gemma_expert init")
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)
        _log_mem("PaliGemmaWithExpert: after dtype conversion")
        self._set_requires_grad()

    def to_bfloat16_for_selected_params(
        self, precision: Literal["bfloat16", "float32"] = "bfloat16"
    ):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower",
            "multi_modal_projector",
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
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        image_outputs = self.paligemma.model.get_image_features(image)
        features = (
            image_outputs.pooler_output
            * self.paligemma.config.text_config.hidden_size**0.5
        )
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
            models = [
                self.paligemma.model.language_model,
                self.gemma_expert.model,
            ]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (
                hasattr(self, "gradient_checkpointing")
                and self.gradient_checkpointing
                and self.training
            )

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

            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = layernorm_forward(
                        models[i].norm, hidden_states, adarms_cond[i]
                    )
                    outputs_embeds.append(out_emb)
                return outputs_embeds

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


# ═══════════════════════════════════════════════════════════════════════════════
# PiStar06 PyTorch model
# ═══════════════════════════════════════════════════════════════════════════════


class PiStar06Pytorch(nn.Module):
    """Core PiStar06 PyTorch model (adapted from PI05Pytorch)."""

    def __init__(
        self, config: PiStar06Config, rtc_processor: RTCProcessor | None = None
    ):
        super().__init__()
        self.config = config
        self.rtc_processor = rtc_processor

        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)

        if config.image_resolution[0] != config.image_resolution[1]:
            raise ValueError(
                f"PaliGemma expects square image resolution, "
                f"invalid resolution: {config.image_resolution}"
            )

        _log_mem("PiStar06Pytorch: before PaliGemmaWithExpert init")
        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
            image_size=config.image_resolution[0],
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
        )
        _log_mem("PiStar06Pytorch: after PaliGemmaWithExpert init")

        self.action_in_proj = nn.Linear(
            config.max_action_dim, action_expert_config.width
        )
        self.action_out_proj = nn.Linear(
            action_expert_config.width, config.max_action_dim
        )
        self.time_mlp_in = nn.Linear(
            action_expert_config.width, action_expert_config.width
        )
        self.time_mlp_out = nn.Linear(
            action_expert_config.width, action_expert_config.width
        )

        self.gradient_checkpointing_enabled = False

        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(
                self.sample_actions, mode=config.compile_mode
            )
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for PiStar06Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for PiStar06Pytorch model")

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func,
                *args,
                use_reentrant=False,
                preserve_rng_state=False,
                **kwargs,
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha,
            self.config.time_sampling_beta_beta,
            bsize,
            device,
        )
        time = (
            time_beta * self.config.time_sampling_scale
            + self.config.time_sampling_offset
        )
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, tokens, masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer."""
        embs = []
        pad_masks = []
        att_masks = []

        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        def lang_embed_func(tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(
            att_masks, dtype=torch.bool, device=pad_masks.device
        )

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed noisy_actions, timestep to prepare for Expert Gemma processing."""
        model_dtype = self.action_in_proj.weight.dtype
        noisy_actions = noisy_actions.to(dtype=model_dtype)

        embs = []
        pad_masks = []
        att_masks = []

        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.to(dtype=model_dtype)

        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
        action_time_emb = action_emb
        adarms_cond = time_emb

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(
            bsize, action_time_dim, dtype=torch.bool, device=timestep.device
        )
        pad_masks.append(action_time_mask)

        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(
        self, images, img_masks, tokens, masks, actions, noise=None, time=None
    ) -> Tensor:
        """Do a full training forward pass and compute the loss."""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, tokens, masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.embed_suffix(x_t, time)
        )

        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[
                0
            ].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(
            prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        ):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func,
            prefix_embs,
            suffix_embs,
            att_2d_masks_4d,
            position_ids,
            adarms_cond,
        )

        suffix_out = suffix_out[:, -self.config.chunk_size :]

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(
                suffix_out.to(dtype=self.action_out_proj.weight.dtype)
            )

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t.float(), v_t.float(), reduction="none")

    @torch.no_grad()
    def sample_actions(
        self,
        images,
        img_masks,
        tokens,
        masks,
        noise=None,
        num_steps=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = tokens.shape[0]
        device = tokens.device

        if noise is None:
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, tokens, masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(
            prefix_att_2d_masks
        )
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps

        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(
                time, dtype=torch.float32, device=device
            ).expand(bsize)

            def denoise_step_partial_call(
                input_x_t, current_timestep=time_tensor
            ):
                return self.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=input_x_t,
                    timestep=current_timestep,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            x_t = x_t + dt * v_t

            if (
                self.rtc_processor is not None
                and self.rtc_processor.is_debug_enabled()
            ):
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        return x_t

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.embed_suffix(x_t, timestep)
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat(
            [prefix_pad_2d_masks, suffix_att_2d_masks], dim=2
        )

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = (
            prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        )

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(
            full_att_2d_masks
        )
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        past_key_values = copy.deepcopy(past_key_values)
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        return self.action_out_proj(suffix_out).float()


# ═══════════════════════════════════════════════════════════════════════════════
# PiStar06 Policy
# ═══════════════════════════════════════════════════════════════════════════════


class PiStar06Policy(PreTrainedPolicy):
    """Advantage-conditioned Pi0.5 policy using text-token injection.

    Uses flat nn.Module composition to avoid the ~3x peak memory caused by
    the deep PaliGemma inheritance chain in the original PI05 implementation.

    During training, pre-computed advantages (R_t - V(o_t)) are binarized and
    the corresponding text ("Advantage: positive" or "Advantage: negative") is
    appended to the language tokens.  During inference, "Advantage: positive"
    is always appended.
    """

    config: PiStar06Config
    config_class = PiStar06Config
    name = "pistar06"

    def __init__(
        self,
        config: PiStar06Config,
        dataset_meta=None,
        **kwargs,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        _log_mem("PiStar06Policy: before model init")
        self.init_rtc_processor()
        self.model = PiStar06Pytorch(config, rtc_processor=self.rtc_processor)
        _log_mem("PiStar06Policy: after model init")

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)
        _log_mem("PiStar06Policy: after model.to(device)")

        if config.num_expert_layers > 0:
            expert_model = self.model.paligemma_with_expert.gemma_expert.model
            total = len(expert_model.layers)
            if config.num_expert_layers > total:
                raise ValueError(
                    f"num_expert_layers={config.num_expert_layers} exceeds "
                    f"action expert depth {total}"
                )
            expert_model.layers = expert_model.layers[: config.num_expert_layers]
            logging.info(
                f"Truncated action expert to {config.num_expert_layers}/{total} layers"
            )

        self._setup_advantage_tokens()

        self._episode_info: dict[int, dict] | None = None
        self._task_max_len: dict[str, int] | None = None
        if dataset_meta is not None and config.episode_labels_path is not None:
            self._setup_episode_metadata(
                dataset_meta, config.episode_labels_path
            )

        self._train_step_count = 0
        self.reset()

    # ── from_pretrained ──────────────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        """Load PiStar06 from a pretrained Pi0.5 checkpoint."""
        print(
            "The PiStar06 model is based on the PI05 (OpenPI) architecture.\n"
            "Original implementation: https://github.com/Physical-Intelligence/openpi"
        )
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        model = cls(config, **kwargs)
        _log_mem("from_pretrained: after model construction")

        try:
            print(f"Loading model from: {pretrained_name_or_path}")
            try:
                from transformers.utils import cached_file

                resolved_file = cached_file(
                    pretrained_name_or_path,
                    "model.safetensors",
                    cache_dir=kwargs.get("cache_dir"),
                    force_download=kwargs.get("force_download", False),
                    resume_download=kwargs.get("resume_download"),
                    proxies=kwargs.get("proxies"),
                    token=kwargs.get("token"),
                    revision=kwargs.get("revision"),
                    local_files_only=kwargs.get("local_files_only", False),
                )
                from safetensors.torch import load_file

                original_state_dict = load_file(resolved_file)
                _log_mem("from_pretrained: after loading safetensors")
                print("Loaded state dict from model.safetensors")
            except Exception as e:
                print(f"Could not load state dict from remote files: {e}")
                print("Returning model without loading pretrained weights")
                return model

            fixed_state_dict = model._fix_pytorch_state_dict_keys(
                original_state_dict, model.config
            )

            remapped_state_dict = {}
            remap_count = 0
            for key, value in fixed_state_dict.items():
                if not key.startswith("model."):
                    new_key = f"model.{key}"
                    remapped_state_dict[new_key] = value
                    remap_count += 1
                else:
                    remapped_state_dict[key] = value

            if remap_count > 0:
                print(f"Remapped {remap_count} state dict keys")

            missing_keys, unexpected_keys = model.load_state_dict(
                remapped_state_dict, strict=strict
            )
            _log_mem("from_pretrained: after load_state_dict")

            if missing_keys:
                print(
                    f"Missing keys when loading state dict: {len(missing_keys)} keys"
                )
                for key in missing_keys[:5]:
                    print(f"  - {key}")
                if len(missing_keys) > 5:
                    print(f"  ... and {len(missing_keys) - 5} more")

            if unexpected_keys:
                print(
                    f"Unexpected keys when loading state dict: {len(unexpected_keys)} keys"
                )
                for key in unexpected_keys[:5]:
                    print(f"  - {key}")
                if len(unexpected_keys) > 5:
                    print(f"  ... and {len(unexpected_keys) - 5} more")

            if not missing_keys and not unexpected_keys:
                print("All keys loaded successfully!")

        except Exception as e:
            print(f"Warning: Could not load state dict: {e}")

        return model

    def _fix_pytorch_state_dict_keys(self, state_dict, model_config):
        """Fix state dict keys to match current model architecture."""
        import re

        fixed_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\."
                r"(input_layernorm|post_attention_layernorm)\.weight",
                key,
            ):
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config,
                    "use_adarms",
                    False,
                )
                if expert_uses_adarms:
                    logging.warning(
                        f"Skipping layer norm key (adaRMS mismatch): {key}"
                    )
                    continue

            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight",
                key,
            ):
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config,
                    "use_adarms",
                    False,
                )
                if expert_uses_adarms:
                    logging.warning(
                        f"Skipping norm key (adaRMS mismatch): {key}"
                    )
                    continue

            if key.startswith("action_time_mlp_in."):
                new_key = key.replace("action_time_mlp_in.", "time_mlp_in.")
            elif key.startswith("action_time_mlp_out."):
                new_key = key.replace("action_time_mlp_out.", "time_mlp_out.")

            if key.startswith("state_proj."):
                logging.warning(f"Skipping state_proj key: {key}")
                continue

            if "patch_embedding" in key:
                logging.warning(
                    f"Vision embedding key might need handling: {key}"
                )

            if (
                key
                == "model.paligemma_with_expert.paligemma.lm_head.weight"
                or key
                == "paligemma_with_expert.paligemma.lm_head.weight"
            ):
                fixed_state_dict[
                    "model.paligemma_with_expert.paligemma.model."
                    "language_model.embed_tokens.weight"
                ] = value.clone()

            fixed_state_dict[new_key] = value

        return fixed_state_dict

    # ── Optimizer / reset / RTC ──────────────────────────────────────────

    def get_optim_params(self) -> list:
        """Return all trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]

    def reset(self):
        """Reset internal state — called when environment resets."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)
            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def _rtc_enabled(self) -> bool:
        return (
            self.config.rtc_config is not None
            and self.config.rtc_config.enabled
        )

    # ── Image preprocessing ──────────────────────────────────────────────

    def _preprocess_images(
        self, batch: dict[str, Tensor]
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the model."""
        images = []
        img_masks = []
        device = next(self.parameters()).device

        present_img_keys = [
            key for key in self.config.image_features if key in batch
        ]
        missing_img_keys = [
            key for key in self.config.image_features if key not in batch
        ]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. "
                f"At least one expected. "
                f"(batch: {batch.keys()}) "
                f"(image_features: {self.config.image_features})"
            )

        for key in present_img_keys:
            img = batch[key]
            if img.device != device:
                img = img.to(device)
            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            is_channels_first = img.shape[1] == 3
            if is_channels_first:
                img = img.permute(0, 2, 3, 1)

            if img.shape[1:3] != self.config.image_resolution:
                img = resize_with_pad_torch(img, *self.config.image_resolution)

            img = img * 2.0 - 1.0

            if is_channels_first:
                img = img.permute(0, 3, 1, 2)

            images.append(img)
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks.append(mask)

        for _num_empty_cameras in range(len(missing_img_keys)):
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_action(self, batch):
        """Pad action to max_action_dim."""
        return pad_vector(batch[ACTION], self.config.max_action_dim)

    # ── Advantage token setup ────────────────────────────────────────────

    def _setup_advantage_tokens(self) -> None:
        """Pre-tokenize advantage strings with the PaliGemma tokenizer."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.paligemma_tokenizer_name
        )
        pos_ids = tokenizer.encode(
            " Advantage: positive", add_special_tokens=False
        )
        neg_ids = tokenizer.encode(
            " Advantage: negative", add_special_tokens=False
        )

        self.register_buffer(
            "_positive_adv_token_ids",
            torch.tensor(pos_ids, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_negative_adv_token_ids",
            torch.tensor(neg_ids, dtype=torch.long),
            persistent=False,
        )
        logging.info(
            f"Advantage tokens: positive={pos_ids} ({len(pos_ids)} tokens), "
            f"negative={neg_ids} ({len(neg_ids)} tokens)"
        )

    def _inject_advantage_text(
        self,
        tokens: Tensor,
        masks: Tensor,
        advantage_indicator: Tensor,
        dropout_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Append advantage text tokens to each sample's language sequence."""
        tokens = tokens.clone()
        masks = masks.clone()
        seq_len = tokens.shape[1]

        for i in range(tokens.shape[0]):
            if dropout_mask is not None and dropout_mask[i]:
                continue

            content_len = int(masks[i].sum().item())
            adv_ids = (
                self._positive_adv_token_ids
                if advantage_indicator[i]
                else self._negative_adv_token_ids
            )
            n_adv = adv_ids.shape[0]

            if content_len + n_adv > seq_len:
                n_adv = seq_len - content_len
                if n_adv <= 0:
                    continue
                adv_ids = adv_ids[:n_adv]

            tokens[i, content_len : content_len + n_adv] = adv_ids
            masks[i, content_len : content_len + n_adv] = 1

        return tokens, masks

    # ── Episode metadata ─────────────────────────────────────────────────

    def _setup_episode_metadata(
        self, dataset_meta, labels_path: str
    ) -> None:
        """Load episode labels and build per-episode metadata."""
        success_map: dict[int, bool] = {}
        with open(labels_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                success_map[int(row["episode_index"])] = bool(
                    int(row["success"])
                )

        episodes = dataset_meta.episodes
        episode_info: dict[int, dict] = {}
        task_max_len: dict[str, int] = {}

        for i in range(len(episodes)):
            ep = episodes[i]
            ep_idx = ep["episode_index"]
            ep_len = ep["length"]
            ep_tasks = (
                ep["tasks"]
                if isinstance(ep["tasks"], list)
                else [ep["tasks"]]
            )
            task = ep_tasks[0] if ep_tasks else "unknown"

            if ep_idx in success_map:
                episode_info[ep_idx] = {
                    "length": ep_len,
                    "success": success_map[ep_idx],
                    "task": task,
                    "dataset_from_index": ep["dataset_from_index"],
                }
                if task not in task_max_len or ep_len > task_max_len[task]:
                    task_max_len[task] = ep_len

        self._episode_info = episode_info
        self._task_max_len = task_max_len
        logging.info(
            f"Episode metadata loaded: {len(episode_info)} episodes, "
            f"{sum(1 for v in episode_info.values() if v['success'])} successful"
        )

    # ── Advantage computation ────────────────────────────────────────────

    def _compute_advantages(
        self, batch: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, float]]:
        """Read pre-computed advantages from the batch."""
        if "advantage" not in batch:
            raise ValueError(
                "PiStar06 requires pre-computed advantages in "
                "batch['advantage']. The training script should pre-compute "
                "these using the SmolVLA value network."
            )
        adv = batch["advantage"]
        diagnostics: dict[str, float] = {
            "advantage_mean": adv.mean().item(),
            "advantage_std": adv.std().item(),
        }
        if "target_value" in batch:
            diagnostics["R_t_mean"] = batch["target_value"].mean().item()
            diagnostics["R_t_std"] = batch["target_value"].std().item()
        if "predicted_value" in batch:
            diagnostics["V_t_mean"] = batch["predicted_value"].mean().item()
            diagnostics["V_t_std"] = batch["predicted_value"].std().item()
        return adv, diagnostics

    # ── Training forward ─────────────────────────────────────────────────

    def _forward_with_advantage(
        self,
        batch: dict[str, Tensor],
        advantage_indicator: Tensor,
        dropout_mask: Tensor | None = None,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> Tensor:
        """Flow-matching forward with advantage text injected into language tokens."""
        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)

        tokens, masks = self._inject_advantage_text(
            tokens, masks, advantage_indicator, dropout_mask
        )

        if noise is None:
            noise = self.model.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.model.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = (
            self.model.embed_prefix(images, img_masks, tokens, masks)
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.model.embed_suffix(x_t, time)
        )

        if (
            self.model.paligemma_with_expert.paligemma.model.language_model.layers[
                0
            ].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self.model._prepare_attention_masks_4d(att_2d_masks)

        (_, suffix_out), _ = self.model.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        v_t = self.model.action_out_proj(
            suffix_out.to(dtype=self.model.action_out_proj.weight.dtype)
        )
        return F.mse_loss(u_t.float(), v_t.float(), reduction="none")

    def forward(
        self,
        batch: dict[str, Tensor],
        noise=None,
        time=None,
        reduction: str = "mean",
    ) -> tuple[Tensor, dict[str, float]]:
        """Training forward pass with advantage-conditioned flow matching."""
        advantage, adv_diagnostics = self._compute_advantages(batch)

        advantage_indicator = advantage > self.config.advantage_threshold
        n_positive = advantage_indicator.sum().item()
        n_total = advantage_indicator.shape[0]

        dropout_mask = None
        if self.training and self.config.advantage_dropout > 0:
            dropout_mask = (
                torch.rand(advantage.shape[0], device=advantage.device)
                < self.config.advantage_dropout
            )

        losses = self._forward_with_advantage(
            batch, advantage_indicator, dropout_mask, noise=noise, time=time
        )

        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]
        loss_dict: dict[str, float] = {}
        loss_dict["losses_after_forward"] = losses.clone().mean().item()

        actions_is_pad = batch.get("action_is_pad")
        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = (
                losses.clone().mean().item()
            )

        if reduction == "none":
            per_sample_loss = losses.mean(dim=(1, 2))
            loss = per_sample_loss
            loss_dict["loss"] = per_sample_loss.mean().item()
        else:
            loss = losses.mean()
            loss_dict["loss"] = loss.item()

        output_dict = loss_dict
        output_dict.update(adv_diagnostics)
        output_dict["advantage_threshold"] = self.config.advantage_threshold
        output_dict["advantage_pct_positive"] = n_positive / n_total
        n_dropped = (
            int(dropout_mask.sum().item()) if dropout_mask is not None else 0
        )
        output_dict["advantage_pct_dropped"] = n_dropped / n_total

        if self.training:
            self._train_step_count += 1
            if self._train_step_count % 10 == 1:
                parts = [f"[RECAP step {self._train_step_count}]"]
                if "V_t_mean" in adv_diagnostics:
                    parts.append(
                        f"V(o_t)={adv_diagnostics['V_t_mean']:.4f}"
                        f"\u00b1{adv_diagnostics['V_t_std']:.4f}"
                    )
                    parts.append(
                        f"R_t={adv_diagnostics['R_t_mean']:.4f}"
                        f"\u00b1{adv_diagnostics['R_t_std']:.4f}"
                    )
                parts.append(
                    f"adv={adv_diagnostics['advantage_mean']:.4f}"
                    f"\u00b1{adv_diagnostics['advantage_std']:.4f}"
                )
                parts.append(
                    f"text=pos:{n_positive}/{n_total} "
                    f"drop:{n_dropped}/{n_total}"
                )
                parts.append(f"thresh={self.config.advantage_threshold}")
                logging.info("  ".join(parts))

        return loss, output_dict

    # ── Inference ────────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, "
            "use it with predict_action_chunk"
        )

        self.eval()

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[
                :, : self.config.n_action_steps
            ]
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Tensor],
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Inference with 'Advantage: positive' appended to the prompt."""
        self.eval()

        if self.config.cfg_beta > 1.0:
            return self._predict_action_chunk_cfg(batch, **kwargs)

        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        bsize = tokens.shape[0]
        device = tokens.device

        positive_indicator = torch.ones(
            bsize, dtype=torch.bool, device=device
        )
        tokens, masks = self._inject_advantage_text(
            tokens, masks, positive_indicator, None
        )

        actions_shape = (
            bsize,
            self.config.chunk_size,
            self.config.max_action_dim,
        )
        noise = self.model.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = (
            self.model.embed_prefix(images, img_masks, tokens, masks)
        )
        prefix_att_2d_masks = make_att_2d_masks(
            prefix_pad_masks, prefix_att_masks
        )
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self.model._prepare_attention_masks_4d(
            prefix_att_2d_masks
        )
        self.model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        num_steps = self.config.num_inference_steps
        dt = -1.0 / num_steps
        x_t = noise

        for step in range(num_steps):
            time_val = 1.0 + step * dt
            time_tensor = torch.tensor(
                time_val, dtype=torch.float32, device=device
            ).expand(bsize)
            v_t = self.model.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=time_tensor,
            )
            x_t = x_t + dt * v_t

        original_action_dim = self.config.output_features[ACTION].shape[0]
        return x_t[:, :, :original_action_dim]

    def _predict_action_chunk_cfg(
        self,
        batch: dict[str, Tensor],
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Classifier-free guidance inference."""
        images, img_masks = self._preprocess_images(batch)
        tokens = batch[OBS_LANGUAGE_TOKENS]
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        bsize = tokens.shape[0]
        device = tokens.device
        beta = self.config.cfg_beta

        positive_indicator = torch.ones(
            bsize, dtype=torch.bool, device=device
        )
        tokens_cond, masks_cond = self._inject_advantage_text(
            tokens, masks, positive_indicator, None
        )

        actions_shape = (
            bsize,
            self.config.chunk_size,
            self.config.max_action_dim,
        )
        noise = self.model.sample_noise(actions_shape, device)

        self.model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        prefix_embs_c, prefix_pad_masks_c, prefix_att_masks_c = (
            self.model.embed_prefix(
                images, img_masks, tokens_cond, masks_cond
            )
        )
        att_2d_c = make_att_2d_masks(prefix_pad_masks_c, prefix_att_masks_c)
        pos_ids_c = torch.cumsum(prefix_pad_masks_c, dim=1) - 1
        _, past_kv_cond = self.model.paligemma_with_expert.forward(
            attention_mask=self.model._prepare_attention_masks_4d(att_2d_c),
            position_ids=pos_ids_c,
            past_key_values=None,
            inputs_embeds=[prefix_embs_c, None],
            use_cache=True,
        )

        prefix_embs_u, prefix_pad_masks_u, prefix_att_masks_u = (
            self.model.embed_prefix(images, img_masks, tokens, masks)
        )
        att_2d_u = make_att_2d_masks(prefix_pad_masks_u, prefix_att_masks_u)
        pos_ids_u = torch.cumsum(prefix_pad_masks_u, dim=1) - 1
        _, past_kv_uncond = self.model.paligemma_with_expert.forward(
            attention_mask=self.model._prepare_attention_masks_4d(att_2d_u),
            position_ids=pos_ids_u,
            past_key_values=None,
            inputs_embeds=[prefix_embs_u, None],
            use_cache=True,
        )

        num_steps = self.config.num_inference_steps
        dt = -1.0 / num_steps
        x_t = noise

        for step in range(num_steps):
            time_val = 1.0 + step * dt
            time_tensor = torch.tensor(
                time_val, dtype=torch.float32, device=device
            ).expand(bsize)

            v_cond = self.model.denoise_step(
                prefix_pad_masks=prefix_pad_masks_c,
                past_key_values=past_kv_cond,
                x_t=x_t,
                timestep=time_tensor,
            )
            v_uncond = self.model.denoise_step(
                prefix_pad_masks=prefix_pad_masks_u,
                past_key_values=past_kv_uncond,
                x_t=x_t,
                timestep=time_tensor,
            )

            v_t = v_uncond + beta * (v_cond - v_uncond)
            x_t = x_t + dt * v_t

        original_action_dim = self.config.output_features[ACTION].shape[0]
        return x_t[:, :, :original_action_dim]

    # ── PEFT ─────────────────────────────────────────────────────────────

    def _get_default_peft_targets(self) -> dict[str, any]:
        """Return default PEFT target modules for PiStar06 fine-tuning."""
        common_projections = (
            "state_proj|action_in_proj|action_out_proj"
            "|action_time_mlp_in|action_time_mlp_out"
        )
        target_modules = (
            rf"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj"
            rf"|model\.({common_projections}))"
        )
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
        }
