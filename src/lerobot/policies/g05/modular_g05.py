# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea
# Modified for LeRobot in 2026.

"""Reusable neural-network components for the G0.5 policy.

This module contains checkpoint-shaped building blocks. Public policy behavior
lives in ``modeling_g05.py``; keeping the two concerns separate makes the model
entry points readable without hiding the source checkpoint's module structure.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.nn import functional

from lerobot.utils.import_utils import _transformers_available, require_package

from .configuration_g05 import G05Config

if TYPE_CHECKING or _transformers_available:
    from transformers import DynamicCache
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5Attention,
        Qwen3_5GatedDeltaNet,
        Qwen3_5MLP,
        Qwen3_5TextRotaryEmbedding,
        Qwen3_5VisionPatchEmbed,
        Qwen3_5VisionPatchMerger,
    )
else:
    require_package("transformers", extra="g05")


class G05VisionPatchEmbed(Qwen3_5VisionPatchEmbed):
    """Qwen patch projection with the fp32 boundary used by G0.5."""

    def forward(self, hidden_states: Tensor) -> Tensor:
        with torch.autocast(hidden_states.device.type, enabled=False):
            return super().forward(hidden_states.float())


class G05VisionPatchMerger(Qwen3_5VisionPatchMerger):
    """Qwen patch merger with the fp32 boundary used by G0.5."""

    def forward(self, hidden_states: Tensor) -> Tensor:
        with torch.autocast(hidden_states.device.type, enabled=False):
            return super().forward(hidden_states.float())


class G05GatedDeltaNet(Qwen3_5GatedDeltaNet):
    """Qwen linear attention extended for a cached multi-token suffix.

    Transformers handles an uncached prefill and cached one-token decoding, but
    G0.5 also teacher-forces an entire action-token suffix onto an existing VLM
    prefix cache. That path must continue both the causal-convolution and gated
    recurrent states from the prefix. The explicit fp32 decay and gated norm,
    plus chunk size 32, preserve the released checkpoint's numerical behavior.
    """

    def forward(
        self,
        hidden_states: Tensor,
        cache_params: DynamicCache | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        if attention_mask is not None and attention_mask.ndim == 2:
            hidden_states = hidden_states * attention_mask[:, :, None]

        batch_size, sequence_length, _ = hidden_states.shape
        use_cached_state = cache_params is not None and cache_params.has_previous_state(self.layer_idx)
        if use_cached_state:
            conv_state = cache_params.layers[self.layer_idx].conv_states
            recurrent_state = cache_params.layers[self.layer_idx].recurrent_states

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        gate = self.in_proj_z(hidden_states).reshape(
            batch_size, sequence_length, self.num_v_heads, self.head_v_dim
        )
        beta = self.in_proj_b(hidden_states).sigmoid()
        decay = self.in_proj_a(hidden_states)

        if use_cached_state:
            if sequence_length == 1:
                mixed_qkv = self.causal_conv1d_update(
                    mixed_qkv,
                    conv_state,
                    self.conv1d.weight.squeeze(1),
                    self.conv1d.bias,
                    self.activation,
                )
            else:
                # Decode a whole teacher-forced suffix without a Python token
                # loop. Prepending the cached raw projections gives exactly the
                # same depth-wise causal convolution as one full sequence.
                conv_input = torch.cat((conv_state[..., 1:], mixed_qkv), dim=-1)
                mixed_qkv = functional.conv1d(
                    conv_input,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    groups=self.conv1d.groups,
                )
                mixed_qkv = functional.silu(mixed_qkv)
                cache_params.layers[self.layer_idx].conv_states.copy_(
                    conv_input[..., -self.conv_kernel_size :]
                )
        else:
            if cache_params is not None:
                conv_state = functional.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                cache_params.update_conv_state(conv_state, self.layer_idx)
            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=None,
                )
            else:
                mixed_qkv = functional.silu(self.conv1d(mixed_qkv)[:, :, :sequence_length])

        query, key, value = torch.split(
            mixed_qkv.transpose(1, 2),
            (self.key_dim, self.key_dim, self.value_dim),
            dim=-1,
        )
        query = query.reshape(batch_size, sequence_length, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch_size, sequence_length, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch_size, sequence_length, self.num_v_heads, self.head_v_dim)

        # G0.5 computes decay and the gated norm in fp32 even when the outer
        # model runs under bf16 autocast. Both details affect every later layer.
        with torch.autocast(hidden_states.device.type, enabled=False):
            decay = -self.A_log.float().exp() * functional.softplus(decay.float() + self.dt_bias.float())
        head_repeats = self.num_v_heads // self.num_k_heads
        if head_repeats > 1:
            query = query.repeat_interleave(head_repeats, dim=2)
            key = key.repeat_interleave(head_repeats, dim=2)

        if use_cached_state and sequence_length == 1:
            attended, recurrent_state = self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=decay,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            initial_state = recurrent_state.clone() if use_cached_state else None
            attended, recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=decay,
                beta=beta,
                initial_state=initial_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                chunk_size=32,
            )

        if cache_params is not None:
            cache_params.update_recurrent_state(recurrent_state, self.layer_idx)

        attended = attended.reshape(-1, self.head_v_dim)
        gate = gate.reshape(-1, self.head_v_dim)
        with torch.autocast(hidden_states.device.type, enabled=False):
            attended = self.norm(attended.float(), gate.float())
        attended = attended.reshape(batch_size, sequence_length, self.value_dim)
        return self.out_proj(attended)


class SinusoidalTimeEmbedding(nn.Module):
    """Fixed flow-time features used by the action expert."""

    def __init__(self, dim: int, min_period: float = 4e-3, max_period: float = 4.0) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError("time embedding dimension must be even")
        self.dim = dim
        self.min_period = min_period
        self.max_period = max_period
        self.register_buffer("angular_frequency", self._frequencies(), persistent=False)

    def _frequencies(self, device: torch.device | str | None = None) -> Tensor:
        fraction = torch.linspace(0, 1, self.dim // 2, dtype=torch.float64, device=device)
        period = self.min_period * (self.max_period / self.min_period) ** fraction
        return 2 * math.pi / period

    def materialize_buffer(self, device: torch.device | str) -> None:
        """Recreate the non-persistent buffer after meta-device construction."""
        self.angular_frequency = self._frequencies(device)

    def forward(self, time: Tensor) -> Tensor:
        if time.ndim != 1:
            raise ValueError("flow time must have shape [batch]")
        phase = torch.einsum("b,d->bd", time.double(), self.angular_frequency)
        return torch.cat((phase.sin(), phase.cos()), dim=-1).to(time.dtype)


class AdaptiveRMSNorm(nn.Module):
    """AdaLN modulation whose parameter names match the official checkpoint."""

    def __init__(self, dim: int, condition_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.dense = nn.Linear(condition_dim, 3 * dim)

    def forward(self, values: Tensor, condition: Tensor) -> tuple[Tensor, Tensor]:
        # AdaLN is an explicit fp32 island in the released action expert,
        # including the modulation projection weights and bias.
        with torch.autocast(values.device.type, enabled=False):
            values_fp32 = values.float()
            variance = values_fp32.square().mean(dim=-1, keepdim=True)
            normalized = values_fp32 * torch.rsqrt(variance + self.eps)
            modulation = functional.linear(
                condition.float(),
                self.dense.weight.float(),
                self.dense.bias.float() if self.dense.bias is not None else None,
            )
            scale, shift, gate = modulation.unsqueeze(1).chunk(3, dim=-1)
            normalized = normalized * (1 + scale) + shift
        return normalized.to(values.dtype), gate.to(values.dtype)


class ActionExpertLayer(nn.Module):
    """One full-attention action-expert block with time-conditioned AdaLN."""

    def __init__(self, config: Qwen3_5TextConfig, layer_index: int) -> None:
        super().__init__()
        self.self_attn = Qwen3_5Attention(config, layer_index)
        # Action suffix tokens are bidirectional; prefix visibility is carried
        # by the explicit mask rather than decoder causality.
        self.self_attn.is_causal = False
        self.mlp = Qwen3_5MLP(config, intermediate_size=config.intermediate_size)
        self.input_layernorm = AdaptiveRMSNorm(config.hidden_size, config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = AdaptiveRMSNorm(
            config.hidden_size, config.hidden_size, config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: Tensor,
        time_condition: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor | None,
        cache: DynamicCache,
    ) -> Tensor:
        residual = hidden_states
        hidden_states, gate = self.input_layernorm(hidden_states, time_condition)
        hidden_states, _ = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=cache,
        )
        hidden_states = residual + hidden_states * gate

        residual = hidden_states
        hidden_states, gate = self.post_attention_layernorm(hidden_states, time_condition)
        return residual + self.mlp(hidden_states) * gate


class ActionExpert(nn.Module):
    """Flow-matching action decoder conditioned on cached VLM key/value states."""

    def __init__(self, config: G05Config) -> None:
        super().__init__()
        expert_config = Qwen3_5TextConfig(
            vocab_size=1,
            hidden_size=config.expert_hidden_size,
            intermediate_size=config.expert_intermediate_size,
            num_hidden_layers=config.expert_num_layers,
            num_attention_heads=config.expert_num_heads,
            num_key_value_heads=config.expert_num_kv_heads,
            head_dim=config.expert_head_dim,
            max_position_embeddings=262_144,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": config.rope_theta,
                "partial_rotary_factor": 0.25,
                "mrope_section": list(config.mrope_section),
                "mrope_interleaved": True,
            },
            layer_types=["full_attention"] * config.expert_num_layers,
            attention_bias=False,
            rms_norm_eps=1e-6,
        )
        expert_config._attn_implementation = config.attn_implementation
        self.config = expert_config
        self.input_proj = nn.Linear(config.internal_action_dim, config.expert_hidden_size)
        self.output_proj = nn.Linear(config.expert_hidden_size, config.internal_action_dim)
        self.time_embedding = SinusoidalTimeEmbedding(config.expert_hidden_size)
        self.time_mlp_in = nn.Linear(config.expert_hidden_size, config.expert_hidden_size)
        self.time_mlp_out = nn.Linear(config.expert_hidden_size, config.expert_hidden_size)
        self.layers = nn.ModuleList(
            [ActionExpertLayer(expert_config, index) for index in range(config.expert_num_layers)]
        )
        self.norm = AdaptiveRMSNorm(config.expert_hidden_size, config.expert_hidden_size)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(expert_config)

    def encode_time(self, time: Tensor) -> Tensor:
        """Project fixed sinusoidal features into the AdaLN condition space."""
        condition = self.time_embedding(time.float())
        condition = functional.silu(self.time_mlp_in(condition))
        return functional.silu(self.time_mlp_out(condition))

    def copy_prefix_cache(self, prefix_cache: DynamicCache) -> DynamicCache:
        """Create an expert-owned cache while sharing immutable VLM prefix tensors."""
        cache = DynamicCache(config=self.config)
        for index, source_layer in enumerate(prefix_cache.layers):
            if not hasattr(source_layer, "keys") or not source_layer.is_initialized:
                continue
            destination = cache.layers[index]
            destination.keys = source_layer.keys
            destination.values = source_layer.values
            destination.is_initialized = True
        return cache

    def forward(
        self,
        actions: Tensor,
        time: Tensor,
        prefix_cache: DynamicCache,
        prefix_position_ids: Tensor,
        prefix_attention_mask: Tensor,
    ) -> Tensor:
        """Predict flow velocity for all action tokens in parallel."""
        # Action embedding and time conditioning stay fp32 before the bf16
        # transformer projections, matching the source FM training path.
        with torch.autocast(actions.device.type, enabled=False):
            hidden_states = self.input_proj(actions.float())
            time_condition = self.encode_time(time.float())
        batch_size, action_length = actions.shape[:2]
        action_offsets = prefix_position_ids.amax(dim=-1, keepdim=True)
        action_positions = torch.arange(1, action_length + 1, device=actions.device, dtype=torch.long)[
            None, None, :
        ]
        position_embeddings = self.rotary_emb(hidden_states, action_offsets + action_positions)
        cache = self.copy_prefix_cache(prefix_cache)

        # Flash Attention consumes a 2-D valid-token mask, while eager and SDPA
        # consume the equivalent additive mask. Both describe non-causal action
        # suffixes attending to every valid prefix key.
        if self.config._attn_implementation.startswith("flash_attention"):
            action_mask = None
            mask_with_prefix = torch.cat(
                (
                    prefix_attention_mask,
                    torch.ones(
                        batch_size,
                        action_length,
                        dtype=torch.bool,
                        device=actions.device,
                    ),
                ),
                dim=-1,
            )
        else:
            action_mask = hidden_states.new_zeros(batch_size, 1, action_length, action_length)
            prefix_mask = (~prefix_attention_mask).to(hidden_states.dtype)
            prefix_mask = prefix_mask * torch.finfo(hidden_states.dtype).min
            prefix_mask = prefix_mask[:, None, None, :].expand(-1, 1, action_length, -1)
            mask_with_prefix = torch.cat((prefix_mask, action_mask), dim=-1)

        for index, layer in enumerate(self.layers):
            attention_mask = mask_with_prefix if cache.layers[index].is_initialized else action_mask
            hidden_states = layer(
                hidden_states,
                time_condition,
                position_embeddings,
                attention_mask,
                cache,
            )

        hidden_states, _ = self.norm(hidden_states, time_condition)
        with torch.autocast(hidden_states.device.type, enabled=False):
            return self.output_proj(hidden_states.float())
