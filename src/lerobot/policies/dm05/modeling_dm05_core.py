#!/usr/bin/env python

# Copyright 2026 Dexmal and HuggingFace Inc. team. All rights reserved.
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

"""DM05 model architecture.

Gemma3 VLM plus action expert VLA model.
"""

from __future__ import annotations

import importlib.util
import logging
import math
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as torch_nn_functional
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.cache_utils import Cache, CacheLayerMixin, DynamicCache
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3CausalLMOutputWithPast,
    Gemma3DecoderLayer,
    Gemma3ForConditionalGeneration,
    Gemma3RotaryEmbedding,
    Gemma3TextModel,
    apply_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,
)

from .constants import IGNORE_INDEX

try:
    from transformers.utils import is_torch_flex_attn_available
except ImportError:

    def is_torch_flex_attn_available() -> bool:
        return importlib.util.find_spec("torch.nn.attention.flex_attention") is not None


logger = logging.getLogger(__name__)


def gemma3_rotary_emb(rotary_emb, x, position_ids, layer_type):
    try:
        return rotary_emb(x, position_ids, layer_type=layer_type)
    except TypeError as exc:
        if "layer_type" not in str(exc):
            raise
        return rotary_emb(x, position_ids)


class SafeCacheDecoderLayer(GradientCheckpointingLayer):
    """Gradient-checkpointing-safe decoder layer that preserves past_key_values.

    DM05 needs prefix forward to write KV cache even during recompute.
    """

    def __call__(self, *args, **kwargs):
        if self.gradient_checkpointing and self.training:
            return self._gradient_checkpointing_func(
                partial(super(GradientCheckpointingLayer, self).__call__, **kwargs),
                *args,
            )
        return super(GradientCheckpointingLayer, self).__call__(*args, **kwargs)


def patch_decoder_layers(model):
    """Patch decoder layers with SafeCacheDecoderLayer."""
    patched_decoder_layer = type(
        "PatchedGemma3DecoderLayer",
        (SafeCacheDecoderLayer,)
        + tuple(b for b in Gemma3DecoderLayer.__bases__ if b is not GradientCheckpointingLayer),
        dict(Gemma3DecoderLayer.__dict__),
    )

    for layer in model.language_model.layers:
        layer.__class__ = patched_decoder_layer


class _OverwriteDynamicLayerBase(CacheLayerMixin):
    """Overwrite-mode cache layer used by DM05 prefix/suffix forwards."""

    is_sliding = False

    def __init__(self):
        super().__init__()
        self.keys = None
        self.values = None
        self.is_initialized = True

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        """No-op; initialized in __init__."""
        pass

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.keys = key_states
        self.values = value_states
        return key_states, value_states

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        kv_offset = 0
        query_length = cache_position if isinstance(cache_position, int) else cache_position.shape[0]
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        if self.keys is None or self.keys.numel() == 0:
            return 0
        return self.keys.shape[-2]

    def get_max_cache_shape(self) -> int:
        return -1


class OverwriteDynamicLayer(_OverwriteDynamicLayerBase):
    """Overwrite cache layer that keeps gradients."""


class _OverwriteDynamicCacheBase(Cache):
    layer_cls: type[CacheLayerMixin]

    def __init__(self, config=None):
        if config is None:
            super().__init__(layer_class_to_replicate=self.layer_cls)
            return
        decoder_config = config.get_text_config(decoder=True)
        layer_types = getattr(decoder_config, "layer_types", None)
        if layer_types is None:
            layer_types = ["full_attention"] * decoder_config.num_hidden_layers
        super().__init__(layers=[self.layer_cls() for _ in layer_types])


class VLADynamicCache(_OverwriteDynamicCacheBase):
    """VLA DynamicCache with overwrite semantics and gradients."""

    layer_cls = OverwriteDynamicLayer


def posemb_sincos(
    time: torch.Tensor,
    dim: int,
    min_period: float = 4e-3,
    max_period: float = 256.0,
) -> torch.Tensor:
    """Compute sinusoidal time embeddings."""
    fraction = torch.linspace(0.0, 1.0, dim // 2, dtype=torch.float32, device=time.device)
    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None].to(torch.float32)
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def make_suffix_attn_mask(
    input_ids: torch.Tensor,
    prefix_len: int,
    suffix_len: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    labels: torch.Tensor | None = None,
    pad_token_id: int = 0,
    invisible_prefix_token_ids: tuple[int, ...] = (),
) -> torch.Tensor:
    """Build the suffix-to-prefix-plus-suffix attention mask.

    Each suffix token can attend to non-action prefix tokens and all suffix tokens.

    Prefix action-label tokens are masked to avoid leaking discrete actions.

    Returns:
        4D attention mask with shape [B, 1, suffix_len, prefix_len + suffix_len].
    """
    neg_inf = -2.3819763e38

    prefix_mask = torch.zeros(batch_size, suffix_len, prefix_len, device=device, dtype=dtype)
    prefix_ids = input_ids[:, :prefix_len]
    pad_mask = prefix_ids == pad_token_id
    for token_id in invisible_prefix_token_ids:
        pad_mask = pad_mask | (prefix_ids == token_id)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, suffix_len, -1)
    prefix_mask = prefix_mask.masked_fill(pad_mask, neg_inf)

    if labels is not None:
        action_positions = labels[:, :prefix_len] != IGNORE_INDEX
        action_positions = action_positions.unsqueeze(1).expand(-1, suffix_len, -1)
        prefix_mask = prefix_mask.masked_fill(action_positions, neg_inf)

    suffix_mask = torch.zeros(batch_size, suffix_len, suffix_len, device=device, dtype=dtype)

    mask = torch.cat([prefix_mask, suffix_mask], dim=2)
    return mask.unsqueeze(1)


def build_action_prefix_mask(
    action_prefill_len: torch.Tensor | None,
    *,
    horizon: int,
    device: torch.device,
) -> torch.Tensor | None:
    if action_prefill_len is None:
        return None

    lengths = action_prefill_len.to(device=device, dtype=torch.long)
    positions = torch.arange(horizon, device=device)
    return positions[None, :] < lengths[:, None]


def build_suffix_position_ids(
    prefix_len: int,
    suffix_len: int,
    batch_size: int,
    device: torch.device,
    *,
    input_ids: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    pad_token_id: int = 0,
    invisible_prefix_token_ids: tuple[int, ...] = (),
) -> torch.Tensor:
    if input_ids is not None:
        valid_prefix = input_ids[:, :prefix_len] != pad_token_id
        for token_id in invisible_prefix_token_ids:
            valid_prefix = valid_prefix & (input_ids[:, :prefix_len] != token_id)
        if labels is not None:
            valid_prefix = valid_prefix & (labels[:, :prefix_len] == IGNORE_INDEX)
        effective_prefix_len = valid_prefix.sum(dim=1)
    else:
        effective_prefix_len = torch.full((batch_size,), prefix_len, device=device)

    suffix_offsets = torch.arange(suffix_len, device=device).unsqueeze(0)
    return effective_prefix_len.unsqueeze(1) + suffix_offsets


def build_prefix_position_ids(
    input_ids: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor | None,
) -> torch.Tensor | None:
    if position_ids is not None or input_ids is None:
        return position_ids
    if attention_mask is None:
        return (
            torch.arange(
                input_ids.shape[1],
                device=input_ids.device,
                dtype=torch.long,
            )
            .unsqueeze(0)
            .expand_as(input_ids)
        )

    valid = attention_mask.to(device=input_ids.device, dtype=torch.bool)
    pos = valid.to(torch.long).cumsum(dim=-1) - 1
    return pos.clamp_min_(0)


def extract_prefix_cache_tensors(
    kv_cache: Cache,
    *,
    num_layers: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    if not hasattr(kv_cache, "layers"):
        raise TypeError(f"Unsupported KV cache type: {type(kv_cache)!r}")
    if len(kv_cache.layers) < num_layers:
        raise ValueError(f"KV cache has {len(kv_cache.layers)} layers, expected at least {num_layers}")

    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    for layer_idx in range(num_layers):
        layer = kv_cache.layers[layer_idx]
        keys.append(layer.keys)
        values.append(layer.values)
    return tuple(keys), tuple(values)


def build_action_adarms_cond(
    *,
    action_in_proj: Any,
    time_mlp_in: Any,
    time_mlp_out: Any,
    time: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    ae_hidden = action_in_proj.out_features
    if time.ndim == 1:
        time_emb = posemb_sincos(time, ae_hidden, max_period=4.0).to(dtype)
    elif time.ndim == 2:
        time_emb = posemb_sincos(
            time.reshape(-1),
            ae_hidden,
            max_period=4.0,
        ).to(dtype)
        time_emb = time_emb.reshape(time.shape[0], time.shape[1], ae_hidden)
    else:
        raise ValueError(f"time must have shape [B] or [B,T], got {tuple(time.shape)}")

    time_emb = time_emb.to(dtype)
    cond = time_mlp_in(time_emb)
    cond = torch_nn_functional.silu(cond)
    cond = time_mlp_out(cond)
    return torch_nn_functional.silu(cond)


def build_inference_suffix_inputs(
    *,
    action_in_proj: Any,
    time_mlp_in: Any,
    time_mlp_out: Any,
    chunk_size: int,
    x_t: torch.Tensor,
    time_tensor: torch.Tensor,
    action_prefix_mask: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    prefix_len: int,
    batch_size: int,
    device: torch.device,
    pad_token_id: int,
    invisible_prefix_token_ids: tuple[int, ...] = (),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if action_prefix_mask is not None:
        time_tokens = time_tensor[:, None].expand(batch_size, chunk_size)
        time_tensor = torch.where(
            action_prefix_mask,
            torch.zeros((), dtype=time_tokens.dtype, device=time_tokens.device),
            time_tokens,
        )

    suffix_embeds = action_in_proj(x_t)
    adarms_cond = build_action_adarms_cond(
        action_in_proj=action_in_proj,
        time_mlp_in=time_mlp_in,
        time_mlp_out=time_mlp_out,
        time=time_tensor,
        dtype=suffix_embeds.dtype,
    )
    suffix_len = int(suffix_embeds.shape[1])
    if input_ids is None:
        visible_token_id = 1 if pad_token_id != 1 else 2
        input_ids = torch.full(
            (batch_size, prefix_len),
            visible_token_id,
            device=suffix_embeds.device,
            dtype=torch.long,
        )
    suffix_attn_mask = make_suffix_attn_mask(
        input_ids=input_ids,
        prefix_len=prefix_len,
        suffix_len=suffix_len,
        batch_size=batch_size,
        device=suffix_embeds.device,
        dtype=suffix_embeds.dtype,
        pad_token_id=pad_token_id,
        invisible_prefix_token_ids=invisible_prefix_token_ids,
    )
    suffix_position_ids = build_suffix_position_ids(
        prefix_len,
        suffix_len,
        batch_size,
        device,
        input_ids=input_ids,
        pad_token_id=pad_token_id,
        invisible_prefix_token_ids=invisible_prefix_token_ids,
    )
    return suffix_embeds, suffix_attn_mask, suffix_position_ids, adarms_cond


def prepare_dm05_config_for_save(
    *,
    runtime_model: Any,
    config: Any,
) -> Any:
    vlm_config = runtime_model.vlm.config
    action_config = runtime_model.action_expert.config

    llm_attn = getattr(
        getattr(vlm_config, "text_config", None),
        "_attn_implementation",
        config.llm_attn_implementation,
    )
    vision_attn = getattr(
        getattr(vlm_config, "vision_config", None),
        "_attn_implementation",
        config.vision_attn_implementation,
    )
    action_attn = getattr(
        runtime_model.action_expert,
        "_suffix_attn_backend",
        config.action_attn_implementation,
    )

    config.vlm_config = vlm_config
    config.action_config = action_config
    config.llm_attn_implementation = llm_attn
    config.vision_attn_implementation = vision_attn
    config.action_attn_implementation = action_attn

    materialize_vlm_attn_config(vlm_config, llm_attn, vision_attn)
    materialize_action_attn_config(action_config, action_attn)
    return config


def apply_dm05_gradient_checkpointing(
    *,
    config: Any,
    model_config: Any,
    vlm_language_model: Any,
    action_expert: Any,
    vlm_enable: bool,
    ae_enable: bool,
    ae_layers: int | None = None,
) -> None:
    if ae_layers is None:
        ae_layers = getattr(config, "ae_gradient_checkpointing_layers", 1)

    vlm_enable = bool(vlm_enable)
    ae_enable = bool(ae_enable)
    ae_layers = int(ae_layers)
    if ae_layers < 1:
        raise ValueError("ae_gradient_checkpointing_layers must be >= 1")

    gradient_checkpointing = bool(vlm_enable or ae_enable)
    config.gradient_checkpointing = gradient_checkpointing
    config.vlm_gradient_checkpointing = vlm_enable
    config.ae_gradient_checkpointing = ae_enable
    config.ae_gradient_checkpointing_layers = ae_layers
    model_config.gradient_checkpointing = gradient_checkpointing
    model_config.vlm_gradient_checkpointing = vlm_enable
    model_config.ae_gradient_checkpointing = ae_enable
    model_config.ae_gradient_checkpointing_layers = ae_layers
    action_expert.set_gradient_checkpointing_layers(ae_layers)

    if vlm_enable:
        vlm_language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        vlm_language_model.gradient_checkpointing_disable()

    if ae_enable:
        action_expert.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        action_expert.gradient_checkpointing_disable()


def compute_dm05_prefix_cache(
    *,
    vlm_model: Any,
    language_model_config: Any,
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    pixel_values: torch.Tensor | None = None,
    token_type_ids: torch.LongTensor | None = None,
    cache_cls: type[Cache],
) -> tuple[Cache, int]:
    kv_cache = cache_cls(config=language_model_config)

    prefix_position_ids = build_prefix_position_ids(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )

    # Use Gemma3Model.forward() for image embeddings, positions, and token types.
    vlm_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=prefix_position_ids,
        past_key_values=kv_cache,
        pixel_values=pixel_values,
        token_type_ids=token_type_ids,
        use_cache=True,
    )

    return kv_cache, kv_cache.get_seq_length()


def is_flash_attention_2_available():
    return importlib.util.find_spec("flash_attn") is not None


def _validate_flash_attention_2(config: Any) -> None:
    if not config.bf16:
        raise ValueError("flash_attention_2 requires bf16=True")
    if not torch.cuda.is_available():
        raise RuntimeError("flash_attention_2 requires CUDA")
    if not is_flash_attention_2_available():
        raise ImportError("flash_attention_2 requires the flash_attn package")


def _validate_flex_attention_available() -> None:
    if not is_torch_flex_attn_available():
        raise ImportError(
            "flex_attention requires PyTorch FlexAttention support "
            "(for example torch>=2.5 with torch.nn.attention.flex_attention)"
        )


def resolve_llm_attn_implementation(
    requested: str,
    config: Any,
) -> str:
    if requested == "auto":
        return "flex_attention" if is_torch_flex_attn_available() else "sdpa"

    if requested == "flash_attention_2":
        _validate_flash_attention_2(config)
        logger.warning(
            "llm_attn_implementation='flash_attention_2' may not preserve "
            "Gemma3 image block bidirectional attention mask semantics."
        )
        return requested

    if requested == "flex_attention":
        _validate_flex_attention_available()
        return requested

    if requested in {"eager", "sdpa"}:
        return requested

    raise ValueError(
        "llm_attn_implementation must be one of "
        "{'auto', 'eager', 'sdpa', 'flash_attention_2', 'flex_attention'}, "
        f"got {requested!r}"
    )


def resolve_vision_attn_implementation(
    requested: str,
    config: Any,
) -> str:
    if requested == "auto":
        if config.bf16 and torch.cuda.is_available() and is_flash_attention_2_available():
            return "flash_attention_2"
        return "sdpa"

    if requested == "flex_attention":
        raise ValueError(
            "vision_attn_implementation='flex_attention' is not supported for "
            "the current Gemma3/SigLIP vision tower because head_dim=72 is not "
            "power-of-2."
        )

    if requested == "flash_attention_2":
        _validate_flash_attention_2(config)
        return requested

    if requested in {"eager", "sdpa"}:
        return requested

    raise ValueError(
        "vision_attn_implementation must be one of "
        "{'auto', 'eager', 'sdpa', 'flash_attention_2'}, "
        f"got {requested!r}"
    )


def resolve_action_attn_implementation(
    requested: str,
    config: Any,
) -> str:
    if requested == "auto":
        return "flex_attention" if is_torch_flex_attn_available() else "sdpa"

    if requested == "flash_attention_2":
        raise ValueError(
            "action_attn_implementation='flash_attention_2' is not supported "
            "for DM05 suffix merged attention because mask semantics differ "
            "from eager/sdpa/flex."
        )

    if requested in {"flex_attention"}:
        _validate_flex_attention_available()
        return requested

    if requested in {"eager", "sdpa"}:
        return requested

    raise ValueError(
        "action_attn_implementation must be one of "
        "{'auto', 'eager', 'sdpa', 'flex_attention'}, "
        f"got {requested!r}"
    )


def hydrate_config(config_ref):
    if isinstance(config_ref, str):
        return AutoConfig.from_pretrained(config_ref)
    if isinstance(config_ref, dict) and "model_type" in config_ref:
        return CONFIG_MAPPING[config_ref["model_type"]](**config_ref)
    return config_ref


def _set_config_attn_implementation(config_ref, attn_implementation: str) -> None:
    config_ref._attn_implementation = attn_implementation
    config_ref.attn_implementation = attn_implementation


def materialize_vlm_attn_config(
    vlm_config,
    llm_attn_implementation: str,
    vision_attn_implementation: str,
) -> None:
    attn_dict = {
        "": llm_attn_implementation,
        "text_config": llm_attn_implementation,
        "vision_config": vision_attn_implementation,
    }
    vlm_config._attn_implementation = attn_dict
    vlm_config.attn_implementation = attn_dict

    if hasattr(vlm_config, "text_config"):
        _set_config_attn_implementation(
            vlm_config.text_config,
            llm_attn_implementation,
        )
    if hasattr(vlm_config, "vision_config"):
        _set_config_attn_implementation(
            vlm_config.vision_config,
            vision_attn_implementation,
        )

    logger.info(
        "Materialized VLM attention implementations: "
        f"LLM={llm_attn_implementation}, Vision={vision_attn_implementation}"
    )


def materialize_action_attn_config(
    action_config,
    action_attn_implementation: str,
) -> str:
    _set_config_attn_implementation(action_config, action_attn_implementation)
    logger.info(f"Materialized Action Expert attention implementation: {action_attn_implementation}")
    return action_attn_implementation


class DM05ActionExpert(Gemma3TextModel):
    """Gemma3 action expert with DM05 suffix-only forward.

    The module takes ownership of the original text model's layers/norm so
    parameter names stay under ``model.action_expert.layers.*`` after wrapping.
    """

    _dm05_fsdp_wrap_action_expert = True

    def __init__(
        self,
        text_model: Gemma3TextModel,
        *,
        rotary_config,
        action_attn_implementation: str,
    ):
        super(Gemma3TextModel, self).__init__(text_model.config)
        self.padding_idx = getattr(text_model, "padding_idx", text_model.config.pad_token_id)
        self.vocab_size = getattr(text_model, "vocab_size", text_model.config.vocab_size)
        self.embed_tokens = None
        self.layers = text_model.layers
        self.norm = text_model.norm
        self.layer_types = list(
            getattr(
                text_model.config,
                "layer_types",
                ["full_attention"] * len(self.layers),
            )
        )
        self.rotary_emb = Gemma3RotaryEmbedding(rotary_config)
        hidden_size = text_model.config.hidden_size
        self.input_time_modulators = nn.ModuleList(
            nn.Linear(hidden_size, 3 * hidden_size) for _ in self.layers
        )
        self.mlp_time_modulators = nn.ModuleList(nn.Linear(hidden_size, 3 * hidden_size) for _ in self.layers)
        self.final_time_modulator = nn.Linear(hidden_size, 3 * hidden_size)
        std = float(getattr(self.config, "initializer_range", 0.02))
        modulators = (
            list(self.input_time_modulators) + list(self.mlp_time_modulators) + [self.final_time_modulator]
        )
        for modulator in modulators:
            nn.init.normal_(modulator.weight, mean=0.0, std=std)
            if modulator.bias is not None:
                nn.init.zeros_(modulator.bias)
        self.gradient_checkpointing = bool(getattr(text_model, "gradient_checkpointing", False))
        self.gradient_checkpointing_layers = int(
            getattr(text_model.config, "ae_gradient_checkpointing_layers", 1)
        )
        if self.gradient_checkpointing_layers < 1:
            raise ValueError("ae_gradient_checkpointing_layers must be >= 1")
        object.__setattr__(self, "_compiled_suffix_layers", None)
        self.set_action_attention_backend(action_attn_implementation)

    def set_action_attention_backend(self, action_attn_implementation: str) -> None:
        if action_attn_implementation == "eager":
            self._suffix_attn_backend = "eager"
            self._attn_fn = self._eager_attention
        elif action_attn_implementation == "sdpa":
            self._suffix_attn_backend = "sdpa"
            self._attn_fn = self._sdpa_attention
        elif action_attn_implementation == "flex_attention":
            self._suffix_attn_backend = "flex_attention"
            self._attn_fn = self._flex_attention
        else:
            raise ValueError(f"Unsupported suffix attention implementation: {action_attn_implementation!r}")
        self.config.action_attn_implementation = action_attn_implementation

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    def set_gradient_checkpointing_layers(self, num_layers: int) -> None:
        num_layers = int(num_layers)
        if num_layers < 1:
            raise ValueError("ae_gradient_checkpointing_layers must be >= 1")
        self.gradient_checkpointing_layers = num_layers
        self.config.ae_gradient_checkpointing_layers = num_layers

    @property
    def has_compiled_suffix_layers(self) -> bool:
        return getattr(self, "_compiled_suffix_layers", None) is not None

    def setup_compiled_suffix_layers(
        self,
        *,
        mode: str = "reduce-overhead",
        dynamic: bool = False,
    ) -> None:
        compiled_layers = torch.compile(
            self._suffix_forward_layers,
            mode=mode,
            dynamic=dynamic,
        )
        object.__setattr__(self, "_compiled_suffix_layers", compiled_layers)

    def clear_compiled_suffix_layers(self) -> None:
        object.__setattr__(self, "_compiled_suffix_layers", None)

    def compiled_suffix_forward(
        self,
        *,
        suffix_embeds: torch.Tensor,
        attention_mask: Any,
        position_ids: torch.LongTensor,
        prefix_cache_keys: tuple[torch.Tensor, ...],
        prefix_cache_values: tuple[torch.Tensor, ...],
        adarms_cond: torch.Tensor,
    ) -> torch.Tensor:
        compiled_layers = getattr(self, "_compiled_suffix_layers", None)
        if compiled_layers is None:
            raise RuntimeError(
                "Compiled suffix layers are not initialized. "
                "Call setup_compiled_suffix_layers() before inference."
            )
        return compiled_layers(
            suffix_embeds=suffix_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            prefix_cache_keys=prefix_cache_keys,
            prefix_cache_values=prefix_cache_values,
            adarms_cond=adarms_cond,
        )

    @staticmethod
    def _sdpa_attention(query_states, key_states, value_states, attention_mask, layer):
        kv_states_k = repeat_kv(key_states, layer.self_attn.num_key_value_groups)
        kv_states_v = repeat_kv(value_states, layer.self_attn.num_key_value_groups)
        bool_mask = (attention_mask == 0) if attention_mask is not None else None
        attn_output = torch_nn_functional.scaled_dot_product_attention(
            query_states,
            kv_states_k,
            kv_states_v,
            attn_mask=bool_mask,
            scale=layer.self_attn.scaling,
        )
        return attn_output.transpose(1, 2).contiguous()

    @staticmethod
    def _eager_attention(query_states, key_states, value_states, attention_mask, layer):
        attn_output, _ = eager_attention_forward(
            layer.self_attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=layer.self_attn.scaling,
        )
        return attn_output

    @staticmethod
    def _flex_attention(
        query_states,
        key_states,
        value_states,
        attention_mask,
        layer,
    ):
        from transformers.integrations.flex_attention import flex_attention_forward

        attn_output, _ = flex_attention_forward(
            layer.self_attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=layer.self_attn.scaling,
        )
        return attn_output

    def forward(
        self,
        *,
        suffix_embeds: torch.Tensor | None = None,
        attention_mask: Any = None,
        position_ids: torch.LongTensor | None = None,
        prefix_cache_keys: tuple[torch.Tensor, ...] | None = None,
        prefix_cache_values: tuple[torch.Tensor, ...] | None = None,
        adarms_cond: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if suffix_embeds is None:
            raise RuntimeError(
                "DM05ActionExpert only supports suffix forward. "
                "Pass suffix_embeds, position_ids, prefix_cache_keys, and "
                "prefix_cache_values."
            )
        if position_ids is None:
            raise ValueError("position_ids is required for DM05ActionExpert suffix forward.")
        if prefix_cache_keys is None or prefix_cache_values is None:
            raise ValueError(
                "prefix_cache_keys and prefix_cache_values are required for DM05ActionExpert suffix forward."
            )
        if adarms_cond is None:
            raise ValueError("adarms_cond is required for DM05ActionExpert suffix forward.")
        return self._suffix_forward_layers(
            suffix_embeds=suffix_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            prefix_cache_keys=prefix_cache_keys,
            prefix_cache_values=prefix_cache_values,
            adarms_cond=adarms_cond,
        )

    def _suffix_forward_layers(
        self,
        suffix_embeds: torch.Tensor,
        attention_mask: Any,
        position_ids: torch.LongTensor,
        prefix_cache_keys: tuple[torch.Tensor, ...],
        prefix_cache_values: tuple[torch.Tensor, ...],
        adarms_cond: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = suffix_embeds
        num_layers = len(self.layers)
        checkpoint_layers = self.gradient_checkpointing_layers

        for layer_idx in range(0, num_layers, checkpoint_layers):
            end_idx = min(layer_idx + checkpoint_layers, num_layers)
            segment_keys = prefix_cache_keys[layer_idx:end_idx]
            segment_values = prefix_cache_values[layer_idx:end_idx]
            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    self._compute_suffix_layer_segment,
                    layer_idx,
                    hidden_states,
                    position_ids,
                    attention_mask,
                    adarms_cond,
                    *segment_keys,
                    *segment_values,
                    use_reentrant=False,
                )
            else:
                hidden_states = self._compute_suffix_layer_segment(
                    layer_idx,
                    hidden_states,
                    position_ids,
                    attention_mask,
                    adarms_cond,
                    *segment_keys,
                    *segment_values,
                )

        hidden_states, _ = self._adaptive_rmsnorm(
            self.norm,
            hidden_states,
            adarms_cond,
            self.final_time_modulator,
        )
        return hidden_states

    def _compute_suffix_layer_segment(
        self,
        start_layer_idx: int,
        suffix_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: Any,
        adarms_cond: torch.Tensor,
        *cache_tensors: torch.Tensor,
    ) -> torch.Tensor:
        if len(cache_tensors) % 2 != 0:
            raise ValueError("cache_tensors must contain matched key/value tensors")

        segment_len = len(cache_tensors) // 2
        cache_keys = cache_tensors[:segment_len]
        cache_values = cache_tensors[segment_len:]

        hidden_states = suffix_embeds
        for offset, (layer_cache_keys, layer_cache_values) in enumerate(
            zip(cache_keys, cache_values, strict=True)
        ):
            hidden_states = self._compute_suffix_layer(
                start_layer_idx + offset,
                hidden_states,
                position_ids,
                attention_mask,
                adarms_cond,
                layer_cache_keys,
                layer_cache_values,
            )
        return hidden_states

    def _adaptive_rmsnorm(
        self,
        norm: nn.Module,
        x: torch.Tensor,
        adarms_cond: torch.Tensor,
        modulator: nn.Linear,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = x.dtype
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        eps = getattr(norm, "eps", None)
        if eps is None:
            eps = norm.variance_epsilon
        normed = x * torch.rsqrt(var + eps)

        modulation = modulator(adarms_cond.to(dtype=modulator.weight.dtype))
        if modulation.ndim == 2:
            modulation = modulation[:, None, :]
        scale, shift, gate = torch.chunk(modulation, 3, dim=-1)
        normed = normed * (1.0 + scale.float()) + shift.float()
        return normed.to(dtype), gate.to(dtype)

    def _compute_suffix_layer(
        self,
        layer_idx: int,
        suffix_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: Any,
        adarms_cond: torch.Tensor,
        cache_keys: torch.Tensor,
        cache_values: torch.Tensor,
    ) -> torch.Tensor:
        ae_layer = self.layers[layer_idx]
        layer_type = getattr(
            ae_layer,
            "attention_type",
            self.layer_types[layer_idx] if layer_idx < len(self.layer_types) else "full_attention",
        )

        prenorm, attn_gate = self._adaptive_rmsnorm(
            ae_layer.input_layernorm,
            suffix_embeds,
            adarms_cond,
            self.input_time_modulators[layer_idx],
        )
        batch_size, seq_len, _ = prenorm.shape
        hidden_shape = (batch_size, seq_len, -1, ae_layer.self_attn.head_dim)

        query_states = ae_layer.self_attn.q_proj(prenorm).view(hidden_shape).transpose(1, 2)
        key_states = ae_layer.self_attn.k_proj(prenorm).view(hidden_shape).transpose(1, 2)
        value_states = ae_layer.self_attn.v_proj(prenorm).view(hidden_shape).transpose(1, 2)

        query_states = ae_layer.self_attn.q_norm(query_states)
        key_states = ae_layer.self_attn.k_norm(key_states)

        try:
            cos, sin = gemma3_rotary_emb(self.rotary_emb, query_states, position_ids, layer_type)
        except TypeError as exc:
            if "layer_type" not in str(exc):
                raise
            cos, sin = self.rotary_emb(query_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = torch.cat([cache_keys, key_states], dim=2)
        value_states = torch.cat([cache_values, value_states], dim=2)
        if attention_mask is not None and attention_mask.shape[-1] != key_states.shape[-2]:
            attention_mask = attention_mask[..., -key_states.shape[-2] :]

        attn_output = self._attn_fn(query_states, key_states, value_states, attention_mask, ae_layer)
        attn_output = attn_output.view(batch_size, seq_len, -1)

        attn_embeds = ae_layer.self_attn.o_proj(attn_output)
        attn_embeds = ae_layer.post_attention_layernorm(attn_embeds)
        residual = suffix_embeds + attn_embeds * attn_gate

        mlp_input, mlp_gate = self._adaptive_rmsnorm(
            ae_layer.pre_feedforward_layernorm,
            residual,
            adarms_cond,
            self.mlp_time_modulators[layer_idx],
        )
        mlp_out = ae_layer.mlp(mlp_input)
        mlp_out = ae_layer.post_feedforward_layernorm(mlp_out)
        return residual + mlp_out * mlp_gate


# ---------------------------------------------------------------------------
# Local base classes and DM05 helpers
# ---------------------------------------------------------------------------


class DM05CoreConfig(PretrainedConfig):
    model_type = "dexbotic_vla"
    vlm_config: dict | str | None = None
    action_config: dict | str | None = None
    processor_config: str | None = None
    action_dim: int = 32
    chunk_size: int = 50
    bf16: bool = True


class DM05CorePreTrainedModel(PreTrainedModel):
    config: DM05CoreConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"

    _supports_flash_attn = True
    _supports_sdpa = True
    _can_compile_fullgraph = True
    _supports_flex_attn = True
    _supports_attention_backend = True


class DM05CoreVLAModel(DM05CorePreTrainedModel):
    def __init__(self, config: DM05CoreConfig):
        super().__init__(config)

    @property
    def language_model(self):
        return self.vlm.model.language_model

    @property
    def lm_head(self):
        return self.vlm.lm_head

    def get_input_embeddings(self):
        return self.vlm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.vlm.set_input_embeddings(value)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class DM05CoreModelConfig(DM05CoreConfig):
    """DM05 model config.

    Extends DM05CoreConfig with training and inference options.
    """

    model_type = "dexbotic_dm05"
    ar_loss_weight: float = 1.0
    gradient_checkpointing: bool = True
    vlm_gradient_checkpointing: bool = True
    ae_gradient_checkpointing: bool = True
    ae_gradient_checkpointing_layers: int = 1
    freeze_vlm_embedding: bool = False
    tie_word_embeddings: bool = True
    llm_attn_implementation: str = "auto"
    vision_attn_implementation: str = "auto"
    action_attn_implementation: str = "auto"

    def __init__(self, *args, **kwargs):
        # Old checkpoints may carry this no-op field; drop it before
        # PretrainedConfig stores unknown kwargs as dynamic attributes.
        kwargs.pop("knowledge_insulation", None)
        tie_word_embeddings = kwargs.pop("tie_word_embeddings", True)
        gradient_checkpointing = bool(kwargs.pop("gradient_checkpointing", True))
        vlm_gradient_checkpointing = kwargs.pop("vlm_gradient_checkpointing", None)
        ae_gradient_checkpointing = kwargs.pop("ae_gradient_checkpointing", None)
        self.vlm_gradient_checkpointing = (
            gradient_checkpointing if vlm_gradient_checkpointing is None else bool(vlm_gradient_checkpointing)
        )
        self.ae_gradient_checkpointing = (
            gradient_checkpointing if ae_gradient_checkpointing is None else bool(ae_gradient_checkpointing)
        )
        self.gradient_checkpointing = bool(self.vlm_gradient_checkpointing or self.ae_gradient_checkpointing)
        self.ae_gradient_checkpointing_layers = int(kwargs.pop("ae_gradient_checkpointing_layers", 1))
        if self.ae_gradient_checkpointing_layers < 1:
            raise ValueError("ae_gradient_checkpointing_layers must be >= 1")
        self.freeze_vlm_embedding = kwargs.pop("freeze_vlm_embedding", False)
        self.llm_attn_implementation = kwargs.pop(
            "llm_attn_implementation",
            "auto",
        )
        self.vision_attn_implementation = kwargs.pop(
            "vision_attn_implementation",
            "auto",
        )
        self.action_attn_implementation = kwargs.pop(
            "action_attn_implementation",
            "auto",
        )
        super().__init__(*args, **kwargs)
        self.tie_word_embeddings = tie_word_embeddings

        # Hydrate nested vlm_config dicts loaded from config.json.
        vlm_config = kwargs.get("vlm_config")
        if isinstance(vlm_config, dict) and "model_type" in vlm_config:
            self.vlm_config = CONFIG_MAPPING[vlm_config["model_type"]](**vlm_config)
            vlm_attn = vlm_config.get("attn_implementation", None)
            if isinstance(vlm_attn, dict):
                llm_attn = vlm_attn.get("text_config") or vlm_attn.get("") or self.llm_attn_implementation
                vision_attn = vlm_attn.get("vision_config") or self.vision_attn_implementation
                materialize_vlm_attn_config(
                    self.vlm_config,
                    llm_attn,
                    vision_attn,
                )
            elif self.llm_attn_implementation != "auto" or self.vision_attn_implementation != "auto":
                materialize_vlm_attn_config(
                    self.vlm_config,
                    self.llm_attn_implementation,
                    self.vision_attn_implementation,
                )

        action_config = kwargs.get("action_config")
        if isinstance(action_config, dict):
            self.action_config = CONFIG_MAPPING[action_config["model_type"]](**action_config)
            action_attn = self.action_attn_implementation
            if action_attn == "auto":
                action_attn = action_config.get("attn_implementation", "auto")
            if action_attn != "auto":
                materialize_action_attn_config(self.action_config, action_attn)


# ---------------------------------------------------------------------------
# Model Outputs
# ---------------------------------------------------------------------------


@dataclass
class DM05OutputWithPast(Gemma3CausalLMOutputWithPast):
    ar_loss: torch.FloatTensor | None = None
    fm_loss: torch.FloatTensor | None = None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class DM05Model(DM05CoreVLAModel):
    """DM05 model body.

    Adds action projections and time conditioning on top of the VLM and action expert.
    """

    def __init__(self, config: DM05CoreModelConfig):
        super().__init__(config)

        torch_dtype = torch.bfloat16 if config.bf16 else torch.float32

        llm_attn = resolve_llm_attn_implementation(
            config.llm_attn_implementation,
            config,
        )
        vision_attn = resolve_vision_attn_implementation(
            config.vision_attn_implementation,
            config,
        )
        action_attn = resolve_action_attn_implementation(
            config.action_attn_implementation,
            config,
        )

        vlm_source = config.vlm_config
        action_source = config.action_config
        vlm_config_ref = hydrate_config(vlm_source)
        action_config_ref = hydrate_config(action_source)

        materialize_vlm_attn_config(vlm_config_ref, llm_attn, vision_attn)
        materialize_action_attn_config(action_config_ref, action_attn)
        config.vlm_config = vlm_config_ref
        config.action_config = action_config_ref
        config.llm_attn_implementation = llm_attn
        config.vision_attn_implementation = vision_attn
        config.action_attn_implementation = action_attn
        logger.info(f"Final DM05CoreModelConfig: {config}")

        # from_pretrained may construct the module under a meta-device context.
        _on_meta = torch.tensor(0.0).is_meta

        # Initialize VLM.
        if isinstance(vlm_source, str) and not _on_meta:
            self.vlm = Gemma3ForConditionalGeneration.from_pretrained(
                vlm_source,
                config=vlm_config_ref,
                torch_dtype=torch_dtype,
            )
        else:
            self.vlm = Gemma3ForConditionalGeneration(vlm_config_ref)

        # Initialize the action expert, then wrap it as a suffix-only module
        # while preserving layer/norm parameter paths.
        if isinstance(action_source, str) and not _on_meta:
            action_expert = Gemma3TextModel.from_pretrained(
                action_source,
                config=action_config_ref,
                torch_dtype=torch_dtype,
            )
        else:
            action_expert = Gemma3TextModel(action_config_ref)
        action_expert.embed_tokens = None
        self.action_expert = DM05ActionExpert(
            action_expert,
            rotary_config=self.language_model.config,
            action_attn_implementation=action_attn,
        )

        ae_hidden_size = action_config_ref.hidden_size

        # Action projection layers.
        self.action_in_proj = nn.Linear(config.action_dim, ae_hidden_size)
        self.action_out_proj = nn.Linear(ae_hidden_size, config.action_dim)

        # Time conditioning layers.
        self.time_mlp_in = nn.Linear(ae_hidden_size, ae_hidden_size)
        self.time_mlp_out = nn.Linear(ae_hidden_size, ae_hidden_size)

    @property
    def visual(self):
        """Return the VLM vision tower."""
        return self.vlm.model.vision_tower


# ---------------------------------------------------------------------------
# ForCausalLM
# ---------------------------------------------------------------------------


class DM05ForCausalLM(DM05CorePreTrainedModel):
    """Full DM05 model for training and action inference.

    forward() currently optimizes only flow-matching loss.
    """

    config_class = DM05CoreModelConfig
    _tied_weights_keys = {
        "model.vlm.lm_head.weight": "model.vlm.model.language_model.embed_tokens.weight",
    }

    def __init__(self, config: DM05CoreModelConfig):
        super().__init__(config)
        config.model_type = self.config_class.model_type
        self.all_tied_weights_keys = dict(getattr(self, "_tied_weights_keys", {}))
        self.model = DM05Model(config)
        self._forward_iter = 0

        # Bind attention implementation once during initialization.
        action_attn_implementation = config.action_attn_implementation
        if action_attn_implementation == "auto":
            action_attn_implementation = resolve_action_attn_implementation(
                action_attn_implementation,
                self.config,
            )
        self.model.action_expert.set_action_attention_backend(action_attn_implementation)
        self.config.action_attn_implementation = action_attn_implementation
        self.model.config.action_attn_implementation = action_attn_implementation

        self.post_init()

        # Patch decoder layers so gradient checkpointing preserves KV cache.
        patch_decoder_layers(self.model.vlm.model)

        # Merged attention requires matching VLM and AE layer counts.
        vlm_n_layers = len(self.model.vlm.model.language_model.layers)
        ae_n_layers = len(self.model.action_expert.layers)
        assert vlm_n_layers == ae_n_layers, (
            f"VLM has {vlm_n_layers} layers but Action Expert has {ae_n_layers}. "
            f"They must match for merged attention. "
            f"Set action_config['num_hidden_layers'] = {vlm_n_layers}."
        )

        self.enable_gradient_checkpointing(
            vlm_gradient_checkpointing=config.vlm_gradient_checkpointing,
            ae_gradient_checkpointing=config.ae_gradient_checkpointing,
            ae_layers=config.ae_gradient_checkpointing_layers,
        )

        # Optionally freeze VLM token embeddings and the tied lm_head.
        if config.freeze_vlm_embedding:
            self.freeze_vlm_embedding()

    @property
    def lm_head(self):
        return self.model.vlm.lm_head

    @property
    def has_compiled_suffix_layers(self) -> bool:
        return self.model.action_expert.has_compiled_suffix_layers

    def setup_compiled_suffix_layers(
        self,
        *,
        mode: str = "reduce-overhead",
        dynamic: bool = False,
    ) -> None:
        self.model.action_expert.setup_compiled_suffix_layers(
            mode=mode,
            dynamic=dynamic,
        )

    def clear_compiled_suffix_layers(self) -> None:
        self.model.action_expert.clear_compiled_suffix_layers()

    def freeze_vlm_embedding(self):
        """Freeze Gemma VLM token embeddings.

        lm_head is tied to embed_tokens and is frozen with it.
        """
        embed = self.model.vlm.model.language_model.embed_tokens
        for p in embed.parameters():
            p.requires_grad = False
        logger.info("Frozen VLM token embedding (and tied lm_head).")

    def prepare_config_for_save(self):
        return prepare_dm05_config_for_save(
            runtime_model=self.model,
            config=self.config,
        )

    def save_pretrained(self, *args, **kwargs):
        self.prepare_config_for_save()
        return super().save_pretrained(*args, **kwargs)

    def _apply_liger_kernel(self):
        """Apply Liger Kernel optimizations to the VLM and action expert."""
        try:
            from liger_kernel.transformers import (
                apply_liger_kernel_to_gemma3,
                apply_liger_kernel_to_gemma3_text,
            )
        except ImportError:
            logger.warning(
                "liger-kernel is not installed; skipping Liger optimizations. "
                "Install with: pip install liger-kernel"
            )
            return

        # Patch the VLM (Gemma3ForConditionalGeneration).
        # RMSNorm / GeGLU / RoPE / SigLIP LayerNorm
        # LeRobot SFT does not compute AR loss here, so disable fused CE.
        apply_liger_kernel_to_gemma3(
            model=self.model.vlm,
            rope=True,
            rms_norm=True,
            geglu=True,
            layer_norm=True,
            cross_entropy=False,
            fused_linear_cross_entropy=False,
        )

        # Patch the action expert (Gemma3TextModel).
        # RoPE and loss are handled by suffix forward.
        apply_liger_kernel_to_gemma3_text(
            model=self.model.action_expert,
            rope=False,
            rms_norm=True,
            geglu=True,
            cross_entropy=False,
            fused_linear_cross_entropy=False,
        )

        logger.info(
            "Applied Liger Kernel: VLM (RMSNorm+GeGLU+RoPE+LayerNorm+CrossEntropy), AE (RMSNorm+GeGLU)"
        )

    def enable_gradient_checkpointing(
        self,
        *,
        vlm_gradient_checkpointing: bool = True,
        ae_gradient_checkpointing: bool = True,
        ae_layers: int | None = None,
    ):
        """Configure gradient checkpointing for VLM prefix and AE suffix.

        ``ae_layers`` controls AE suffix checkpoint segment size.
        """
        apply_dm05_gradient_checkpointing(
            config=self.config,
            model_config=self.model.config,
            vlm_language_model=self.model.vlm.model.language_model,
            action_expert=self.model.action_expert,
            vlm_enable=vlm_gradient_checkpointing,
            ae_enable=ae_gradient_checkpointing,
            ae_layers=ae_layers,
        )

    # -----------------------------------------------------------------------
    # Training forward.
    # -----------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        # DM05-specific inputs.
        actions: torch.FloatTensor | None = None,
        action_dim_mask: torch.BoolTensor | None = None,
        prefill_actions: torch.FloatTensor | None = None,
        action_prefill_len: torch.LongTensor | None = None,
        **kwargs,
    ) -> Gemma3CausalLMOutputWithPast:
        self._forward_iter += 1
        """Training forward; LeRobot integration only uses flow-matching loss."""
        batch_size = input_ids.shape[0]

        # Step 1: prefix forward fills the KV cache.
        # LeRobot SFT only computes flow-matching action loss here.
        # labels are kept for suffix alignment and masking.
        kv_cache = VLADynamicCache(config=self.model.language_model.config)

        prefix_position_ids = build_prefix_position_ids(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        prefix_model_input_ids = None if inputs_embeds is not None else input_ids

        prefix_outputs = self.model.vlm.model(
            input_ids=prefix_model_input_ids,
            attention_mask=attention_mask,
            position_ids=prefix_position_ids,
            past_key_values=kv_cache,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            token_type_ids=token_type_ids,
            cache_position=cache_position,
            use_cache=True,
        )
        prefix_hidden_states = prefix_outputs.last_hidden_state
        ar_loss = None

        # Step 2: suffix forward for flow matching.
        # The collator provides has_actions and actions together; missing actions
        # are represented by zeros and masked out.
        has_actions = kwargs.get("has_actions")
        if has_actions is None:
            has_actions = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        if actions is None:
            actions = torch.zeros(
                (batch_size, self.config.chunk_size, self.config.action_dim),
                dtype=torch.bfloat16 if self.config.bf16 else torch.float32,
                device=input_ids.device,
            )

        # Build flow-matching inputs.
        noise = torch.randn_like(actions)
        time = (
            torch.distributions.Beta(1.5, 1.0).sample((batch_size,)).to(actions.device, dtype=actions.dtype)
            * 0.999
            + 0.001
        )

        action_prefix_mask = build_action_prefix_mask(
            action_prefill_len,
            horizon=actions.shape[1],
            device=actions.device,
        )
        if action_prefix_mask is not None:
            prefill_actions = prefill_actions.to(
                device=actions.device,
                dtype=actions.dtype,
            )
        if action_prefix_mask is None:
            time_for_suffix = time
            time_expanded = time[:, None, None]
        else:
            time_tokens = time[:, None].expand(batch_size, actions.shape[1])
            time_tokens = torch.where(
                action_prefix_mask,
                torch.zeros((), dtype=time_tokens.dtype, device=time_tokens.device),
                time_tokens,
            )
            time_for_suffix = time_tokens
            time_expanded = time_tokens[..., None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        if action_prefix_mask is not None:
            x_t = torch.where(action_prefix_mask[..., None], prefill_actions, x_t)
        u_t = noise - actions

        # Build suffix embeddings and per-layer time conditioning.
        suffix_embeds = self.model.action_in_proj(x_t)
        adarms_cond = build_action_adarms_cond(
            action_in_proj=self.model.action_in_proj,
            time_mlp_in=self.model.time_mlp_in,
            time_mlp_out=self.model.time_mlp_out,
            time=time_for_suffix,
            dtype=suffix_embeds.dtype,
        )

        # Suffix forward with merged attention.
        prefix_len = prefix_hidden_states.shape[1]
        suffix_len = suffix_embeds.shape[1]

        suffix_attn_mask = make_suffix_attn_mask(
            input_ids=input_ids,
            prefix_len=prefix_len,
            suffix_len=suffix_len,
            batch_size=batch_size,
            device=suffix_embeds.device,
            dtype=suffix_embeds.dtype,
            labels=labels,
            pad_token_id=self.model.vlm.model.language_model.padding_idx,
        )

        suffix_position_ids = build_suffix_position_ids(
            prefix_len,
            suffix_len,
            batch_size,
            suffix_embeds.device,
            input_ids=input_ids,
            labels=labels,
            pad_token_id=self.model.vlm.model.language_model.padding_idx,
        )

        suffix_out = self._suffix_forward(
            suffix_embeds=suffix_embeds,
            attention_mask=suffix_attn_mask,
            position_ids=suffix_position_ids,
            past_key_values=kv_cache,
            adarms_cond=adarms_cond,
        )

        # Compute flow-matching loss.
        v_t = self.model.action_out_proj(suffix_out).to(torch.float32)
        has_actions = has_actions.to(device=v_t.device, dtype=torch.bool)
        u_t = u_t.to(dtype=v_t.dtype)

        elem_mse = torch_nn_functional.mse_loss(v_t, u_t, reduction="none")  # [B, T, D]
        if action_dim_mask is not None:
            # action_dim_mask: [B, D] -> [B, 1, D] for broadcasting with [B, T, D]
            valid_mask = action_dim_mask.to(device=v_t.device, dtype=v_t.dtype).unsqueeze(1)
            valid_mask = valid_mask.expand_as(elem_mse)
        else:
            valid_mask = torch.ones_like(elem_mse, dtype=v_t.dtype)
        if action_prefix_mask is not None:
            suffix_mask = (~action_prefix_mask).to(device=v_t.device, dtype=v_t.dtype)
            valid_mask = valid_mask * suffix_mask[..., None]
        valid_denom = valid_mask.sum(dim=(1, 2)).clamp(min=1)
        per_sample_fm = (elem_mse * valid_mask).sum(dim=(1, 2)) / valid_denom
        fm_loss = (per_sample_fm * has_actions).sum() / torch.clamp(has_actions.sum(), min=1)

        # LeRobot SFT updates parameters with flow-matching loss only.
        loss = torch.zeros((), device=self.device, dtype=torch.float32)
        loss = loss + fm_loss

        return DM05OutputWithPast(
            loss=loss,
            ar_loss=ar_loss,
            fm_loss=fm_loss,
            logits=None,
            past_key_values=None,
        )

    # -----------------------------------------------------------------------
    # Action inference.
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def inference_action(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        states: torch.FloatTensor | None = None,
        image_masks: torch.BoolTensor | None = None,
        diffusion_steps: int = 10,
        past_key_values: DynamicCache | None = None,
        prefill_actions: torch.FloatTensor | None = None,
        action_prefill_len: torch.LongTensor | None = None,
        use_compiled_suffix: bool = False,
        initial_noise: torch.FloatTensor | None = None,
        action_prefix_mask: torch.BoolTensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Infer actions with Euler sampling.

        If past_key_values is provided, reuse the prefix cache.
        """
        if input_ids is not None:
            batch_size = input_ids.shape[0]
            device = input_ids.device
        elif states is not None:
            batch_size = states.shape[0]
            device = states.device
        else:
            raise ValueError("input_ids or states must be provided")

        dtype = torch.bfloat16 if self.model.config.bf16 else torch.float32
        if use_compiled_suffix:
            return self._inference_action_compiled_suffix(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                token_type_ids=token_type_ids,
                diffusion_steps=diffusion_steps,
                past_key_values=past_key_values,
                prefill_actions=prefill_actions,
                initial_noise=initial_noise,
                action_prefix_mask=action_prefix_mask,
                dtype=dtype,
            )

        action_prefix_mask = build_action_prefix_mask(
            action_prefill_len,
            horizon=self.model.config.chunk_size,
            device=device,
        )
        if action_prefix_mask is not None:
            prefill_actions = prefill_actions.to(device=device, dtype=dtype)

        # Step 1: get KV cache.
        if past_key_values is None:
            kv_cache, prefix_len = compute_dm05_prefix_cache(
                vlm_model=self.model.vlm.model,
                language_model_config=self.model.language_model.config,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                pixel_values=pixel_values,
                token_type_ids=token_type_ids,
                cache_cls=DynamicCache,
            )
        else:
            kv_cache = past_key_values
            prefix_len = kv_cache.get_seq_length()

        # Step 2: Euler sampling.
        x_t = torch.randn(
            batch_size,
            self.model.config.chunk_size,
            self.model.config.action_dim,
            device=device,
            dtype=dtype,
        )
        if prefill_actions is not None:
            x_t = torch.where(action_prefix_mask[..., None], prefill_actions, x_t)
        time_val = 1.0
        dt = -1.0 / diffusion_steps

        for _ in range(diffusion_steps):
            time_tensor = torch.full((batch_size,), time_val, device=device, dtype=dtype)
            suffix_embeds, suffix_attn_mask, suffix_position_ids, adarms_cond = build_inference_suffix_inputs(
                action_in_proj=self.model.action_in_proj,
                time_mlp_in=self.model.time_mlp_in,
                time_mlp_out=self.model.time_mlp_out,
                chunk_size=self.model.config.chunk_size,
                x_t=x_t,
                time_tensor=time_tensor,
                action_prefix_mask=action_prefix_mask,
                input_ids=input_ids,
                prefix_len=prefix_len,
                batch_size=batch_size,
                device=device,
                pad_token_id=self.model.vlm.model.language_model.padding_idx,
            )

            suffix_out = self._suffix_forward(
                suffix_embeds=suffix_embeds,
                attention_mask=suffix_attn_mask,
                position_ids=suffix_position_ids,
                past_key_values=kv_cache,
                adarms_cond=adarms_cond,
            )

            v_t = self.model.action_out_proj(suffix_out)
            x_t = x_t + v_t * dt
            if prefill_actions is not None:
                x_t = torch.where(action_prefix_mask[..., None], prefill_actions, x_t)
            time_val += dt
        return x_t

    def _inference_action_compiled_suffix(
        self,
        *,
        input_ids: torch.LongTensor | None,
        attention_mask: torch.Tensor | None,
        pixel_values: torch.Tensor | None,
        token_type_ids: torch.LongTensor | None,
        diffusion_steps: int,
        past_key_values: DynamicCache | None,
        prefill_actions: torch.FloatTensor | None,
        initial_noise: torch.FloatTensor | None,
        action_prefix_mask: torch.BoolTensor | None,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not self.model.action_expert.has_compiled_suffix_layers:
            raise RuntimeError(
                "Compiled suffix layers are not initialized. "
                "Call setup_compiled_suffix_layers() before inference."
            )
        if input_ids is None:
            raise ValueError("compiled suffix inference requires input_ids.")
        if past_key_values is not None:
            raise ValueError("compiled suffix inference does not support external past_key_values.")
        if initial_noise is None or action_prefix_mask is None or prefill_actions is None:
            raise ValueError(
                "compiled suffix inference requires initial_noise, action_prefix_mask, and prefill_actions."
            )

        batch_size = int(input_ids.shape[0])
        device = input_ids.device
        initial_noise = initial_noise.to(device=device, dtype=dtype)
        action_prefix_mask = action_prefix_mask.to(device=device, dtype=torch.bool)
        prefill_actions = prefill_actions.to(device=device, dtype=dtype)

        kv_cache, prefix_len = compute_dm05_prefix_cache(
            vlm_model=self.model.vlm.model,
            language_model_config=self.model.language_model.config,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=None,
            pixel_values=pixel_values,
            token_type_ids=token_type_ids,
            cache_cls=VLADynamicCache,
        )
        prefix_cache_keys, prefix_cache_values = extract_prefix_cache_tensors(
            kv_cache,
            num_layers=len(self.model.action_expert.layers),
        )

        x_t = torch.where(action_prefix_mask[..., None], prefill_actions, initial_noise)
        time_val = 1.0
        dt = -1.0 / diffusion_steps
        padding_idx = int(self.model.vlm.model.language_model.padding_idx)

        for _ in range(diffusion_steps):
            time_tensor = torch.full(
                (batch_size,),
                time_val,
                device=device,
                dtype=dtype,
            )
            suffix_embeds, suffix_attn_mask, suffix_position_ids, adarms_cond = build_inference_suffix_inputs(
                action_in_proj=self.model.action_in_proj,
                time_mlp_in=self.model.time_mlp_in,
                time_mlp_out=self.model.time_mlp_out,
                chunk_size=self.model.config.chunk_size,
                x_t=x_t,
                time_tensor=time_tensor,
                action_prefix_mask=action_prefix_mask,
                input_ids=input_ids,
                prefix_len=prefix_len,
                batch_size=batch_size,
                device=device,
                pad_token_id=padding_idx,
            )
            suffix_out = self.model.action_expert.compiled_suffix_forward(
                suffix_embeds=suffix_embeds,
                attention_mask=suffix_attn_mask,
                position_ids=suffix_position_ids,
                prefix_cache_keys=prefix_cache_keys,
                prefix_cache_values=prefix_cache_values,
                adarms_cond=adarms_cond,
            )
            velocity = self.model.action_out_proj(suffix_out)
            x_t = x_t + velocity * dt
            x_t = torch.where(action_prefix_mask[..., None], prefill_actions, x_t)
            time_val += dt
        return x_t

    # -----------------------------------------------------------------------
    # Helpers.
    # -----------------------------------------------------------------------

    def _suffix_forward(
        self,
        suffix_embeds: torch.Tensor,
        attention_mask: Any,
        position_ids: torch.LongTensor,
        past_key_values: Cache,
        adarms_cond: torch.Tensor,
    ) -> torch.Tensor:
        """Extract cache tensors and run the action expert forward."""
        prefix_cache_keys, prefix_cache_values = extract_prefix_cache_tensors(
            past_key_values,
            num_layers=len(self.model.action_expert.layers),
        )

        return self.model.action_expert(
            suffix_embeds=suffix_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            prefix_cache_keys=prefix_cache_keys,
            prefix_cache_values=prefix_cache_values,
            adarms_cond=adarms_cond,
        )


# Compatibility alias for checkpoints/configs that refer to the historical class name.
DM05Config = DM05CoreModelConfig
