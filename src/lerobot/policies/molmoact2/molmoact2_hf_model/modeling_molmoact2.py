# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team. All rights reserved.
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


"""Modeling code for MolmoAct2"""

# ruff: noqa: N806

import json
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F  # noqa: N812
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_masks_for_generate
from transformers.modeling_flash_attention_utils import (
    FlashAttentionKwargs,
    _flash_attention_forward,
    flash_attn_supports_top_left_mask,
)
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    ModelOutput,
    TransformersKwargs,
    can_return_tuple,
    logging,
)

from .configuration_molmoact2 import (
    MolmoAct2ActionExpertConfig,
    MolmoAct2AdapterConfig,
    MolmoAct2Config,
    MolmoAct2TextConfig,
    MolmoAct2VitConfig,
)
from .inference import (
    ActionCudaGraphManager,
    DepthDecodeCudaGraphManager,
    _ActionFlowInputs,
    _cache_max_len_int,
    _cache_seq_len_int,
    _iter_cache_key_values,
)

logger = logging.get_logger(__name__)


ACTION_START_TOKEN = "<action_start>"  # nosec B105
ACTION_END_TOKEN = "<action_end>"  # nosec B105
ACTION_OUTPUT_TOKEN = "<action_output>"  # nosec B105
STATE_START_TOKEN = "<state_start>"  # nosec B105
STATE_END_TOKEN = "<state_end>"  # nosec B105
STATE_TOKEN_PREFIX = "<state_"  # nosec B105
DEPTH_START_TOKEN = "<depth_start>"  # nosec B105
DEPTH_END_TOKEN = "<depth_end>"  # nosec B105
DEPTH_OUTPUT_TOKEN = "<depth_output>"  # nosec B105
DEPTH_TOKEN_PREFIX = "<depth_"  # nosec B105
SETUP_START_TOKEN = "<setup_start>"  # nosec B105
SETUP_END_TOKEN = "<setup_end>"  # nosec B105
CONTROL_START_TOKEN = "<control_start>"  # nosec B105
CONTROL_END_TOKEN = "<control_end>"  # nosec B105

_QUESTION_TRAILING_SENTENCE_PUNCTUATION = ".,!?;:,…"
_QUESTION_TRAILING_CLOSERS = "\"'”’)]}"
_QUESTION_SURROUNDING_DELIMITERS = "\"'`“”‘’[](){}"
_QUESTION_PREFIX_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"^(?:task|instruction|language[_ ]instruction|goal)\s*[:\-]\s*",
        r"^(?:the\s+task\s+is\s+to|your\s+task\s+is\s+to)\s+",
    )
)

_DEPTH_REASONING_PATCH_SIZE = 32
_DEPTH_REASONING_THRESHOLD = 0.996


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _round_up_multiple(value: int, multiple_of: int) -> int:
    if multiple_of <= 0:
        return value
    return int(math.ceil(value / multiple_of) * multiple_of)


def _init_linear(linear: nn.Linear, *, zero: bool = False, scale: float = 1.0) -> None:
    if zero:
        nn.init.zeros_(linear.weight)
    else:
        nn.init.xavier_uniform_(linear.weight)
        if scale != 1.0:
            with torch.no_grad():
                linear.weight.mul_(scale)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


@dataclass
class ActionExpertContext:
    kv_contexts: Sequence[tuple[torch.Tensor, torch.Tensor]]
    cross_mask: torch.Tensor | None
    self_mask: torch.Tensor | None
    valid_action: torch.Tensor | None
    rope_cache: tuple[torch.Tensor, torch.Tensor] | None = None


@dataclass
class ActionExpertStepModulation:
    conditioning: torch.Tensor
    block_modulations: Sequence[tuple[torch.Tensor, ...]]
    final_modulation: tuple[torch.Tensor, torch.Tensor]


class ActionExpertRMSNorm(nn.Module):
    def __init__(
        self,
        size: int,
        *,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.size = size
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(size, device=device))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            dtype = x.dtype
            x_float = x.to(torch.float32)
            variance = x_float.pow(2).mean(dim=-1, keepdim=True)
            out = x_float * torch.rsqrt(variance + self.eps)
            out = out.to(dtype)
        if self.weight is not None:
            out = out * self.weight
        return out

    def reset_parameters(self) -> None:
        if self.weight is not None:
            nn.init.ones_(self.weight)


class ActionExpertRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim.")
        self.head_dim = head_dim
        self.base = base

    def build_cache(
        self,
        *,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        half_dim = self.head_dim // 2
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, half_dim, device=device, dtype=torch.float32) / max(half_dim, 1))
        )
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        cos = freqs.cos().to(dtype=dtype).view(1, 1, seq_len, half_dim)
        sin = freqs.sin().to(dtype=dtype).view(1, 1, seq_len, half_dim)
        return cos, sin

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        rope_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if rope_cache is None:
            rope_cache = self.build_cache(seq_len=q.shape[-2], device=q.device, dtype=q.dtype)
        cos, sin = rope_cache
        half_dim = self.head_dim // 2

        def _apply(x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x[..., :half_dim], x[..., half_dim:]
            return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        return _apply(q), _apply(k)


class ActionExpertSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.attn_dropout = attn_dropout
        self.q_norm = ActionExpertRMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        self.k_norm = ActionExpertRMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        self.rope = ActionExpertRotaryEmbedding(self.head_dim) if use_rope else None
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.out_drop = nn.Dropout(proj_dropout)

    def _apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.q_norm is None or self.k_norm is None:
            return q, k
        return self.q_norm(q), self.k_norm(k)

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        dropout_p = self.attn_dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )
        return out.transpose(1, 2).contiguous()

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        rope_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x).view(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].contiguous()
        q, k = self._apply_qk_norm(q, k)
        if self.rope is not None:
            q, k = self.rope(q, k, rope_cache=rope_cache)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        out = self._attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        out = out.reshape(bsz, seq_len, self.hidden_size)
        return self.out_drop(self.out_proj(out))


class ActionExpertCrossAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.attn_dropout = attn_dropout
        self.q_norm = ActionExpertRMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        self.k_norm = ActionExpertRMSNorm(self.head_dim, eps=qk_norm_eps) if qk_norm else None
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.out_drop = nn.Dropout(proj_dropout)

    def _as_heads(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            if x.shape[2] == self.num_heads:
                return x
            if x.shape[1] == self.num_heads:
                return x.transpose(1, 2).contiguous()
            raise ValueError(f"Unexpected cross-attention KV shape {tuple(x.shape)}")
        if x.dim() != 3:
            raise ValueError(f"Expected 3D/4D cross-attention KV, got {tuple(x.shape)}")
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.head_dim)

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dropout_p = self.attn_dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
        )
        return out.transpose(1, 2).contiguous()

    def forward(
        self,
        x: torch.Tensor,
        *,
        kv_k: torch.Tensor,
        kv_v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, tgt_len, _ = x.shape
        q = self.q_proj(x).view(bsz, tgt_len, self.num_heads, self.head_dim)
        k = self._as_heads(kv_k)
        v = self._as_heads(kv_v)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        out = self._attention(q, k, v, attn_mask=attn_mask)
        out = out.reshape(bsz, tgt_len, self.hidden_size)
        return self.out_drop(self.out_proj(out))


class ActionExpertMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        mlp_ratio: float,
        multiple_of: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = _round_up_multiple(int(hidden_size * mlp_ratio), multiple_of)
        self.up_proj = nn.Linear(hidden_size, inner_dim)
        self.gate_proj = nn.Linear(hidden_size, inner_dim)
        self.down_proj = nn.Linear(inner_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return self.dropout(x)


class ActionExpertModulation(nn.Module):
    def __init__(self, hidden_size: int, num_chunks: int) -> None:
        super().__init__()
        self.act = nn.SiLU()
        self.linear = nn.Linear(hidden_size, num_chunks * hidden_size)

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        return self.linear(self.act(conditioning))


class ActionExpertBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        mlp_ratio: float,
        ffn_multiple_of: int,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
        rope: bool = True,
    ) -> None:
        super().__init__()
        self.self_norm = ActionExpertRMSNorm(hidden_size, eps=1e-6)
        self.cross_norm = ActionExpertRMSNorm(hidden_size, eps=1e-6)
        self.ff_norm = ActionExpertRMSNorm(hidden_size, eps=1e-6)
        self.self_attn = ActionExpertSelfAttention(
            hidden_size,
            num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            qk_norm=qk_norm,
            qk_norm_eps=qk_norm_eps,
            use_rope=rope,
        )
        self.cross_attn = ActionExpertCrossAttention(
            hidden_size,
            num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            qk_norm=qk_norm,
            qk_norm_eps=qk_norm_eps,
        )
        self.mlp = ActionExpertMLP(
            hidden_size,
            mlp_ratio=mlp_ratio,
            multiple_of=ffn_multiple_of,
            dropout=dropout,
        )
        self.modulation = ActionExpertModulation(hidden_size, 9)

    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
        *,
        cross_kv: tuple[torch.Tensor, torch.Tensor],
        self_attn_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        modulation: tuple[torch.Tensor, ...] | None = None,
        rope_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if modulation is None:
            modulation = self.modulation(conditioning).chunk(9, dim=1)
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mca,
            scale_mca,
            gate_mca,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = modulation
        x = x + gate_msa.unsqueeze(1) * self.self_attn(
            _modulate(self.self_norm(x), shift_msa, scale_msa),
            attn_mask=self_attn_mask,
            is_causal=is_causal,
            rope_cache=rope_cache,
        )
        x = x + gate_mca.unsqueeze(1) * self.cross_attn(
            _modulate(self.cross_norm(x), shift_mca, scale_mca),
            kv_k=cross_kv[0],
            kv_v=cross_kv[1],
            attn_mask=attn_mask,
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(_modulate(self.ff_norm(x), shift_mlp, scale_mlp))
        return x


class ActionExpertFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, output_dim: int) -> None:
        super().__init__()
        self.norm = ActionExpertRMSNorm(hidden_size, eps=1e-6)
        self.modulation = ActionExpertModulation(hidden_size, 2)
        self.linear = nn.Linear(hidden_size, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor,
        *,
        modulation: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if modulation is None:
            modulation = self.modulation(conditioning).chunk(2, dim=1)
        shift, scale = modulation
        return self.linear(_modulate(self.norm(x), shift, scale))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.dim() > 1:
            timesteps = timesteps.view(timesteps.shape[0], -1)[:, 0]
        half_dim = self.dim // 2
        freq = torch.exp(
            torch.arange(half_dim, device=timesteps.device, dtype=timesteps.dtype)
            * (-math.log(10000.0) / max(half_dim - 1, 1))
        )
        args = timesteps[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ActionExpert(nn.Module):
    """Modern MolmoAct2 action expert embedded in the local LeRobot implementation."""

    def __init__(
        self,
        config: MolmoAct2ActionExpertConfig,
        *,
        llm_dim: int,
        llm_kv_dim: int,
        llm_num_layers: int,
        device=None,
    ):
        super().__init__()
        if config.num_layers != llm_num_layers:
            raise ValueError(
                "MolmoAct2 HF action expert supports only per-layer conditioning with one "
                f"action block per LLM layer (action={config.num_layers}, llm={llm_num_layers})."
            )
        self.config = config
        self.hidden_size = config.hidden_size
        self.llm_dim = llm_dim
        self.llm_kv_dim = llm_kv_dim
        self.action_head_dim = config.hidden_size // config.num_heads

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(config.timestep_embed_dim),
            nn.Linear(config.timestep_embed_dim, config.hidden_size, device=device),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size, device=device),
        )
        self.action_embed = nn.Linear(config.max_action_dim, config.hidden_size, device=device)
        self.context_k_proj = nn.Linear(self.llm_kv_dim, config.hidden_size, bias=False, device=device)
        self.context_v_proj = nn.Linear(self.llm_kv_dim, config.hidden_size, bias=False, device=device)
        self.context_norm = (
            ActionExpertRMSNorm(config.hidden_size, eps=1e-6) if config.context_layer_norm else nn.Identity()
        )
        self._modulation_cache_key: tuple[Any, ...] | None = None
        self._modulation_cache_value: Sequence[ActionExpertStepModulation] | None = None
        self.blocks = nn.ModuleList(
            [
                ActionExpertBlock(
                    config.hidden_size,
                    config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    ffn_multiple_of=config.ffn_multiple_of,
                    attn_dropout=config.attn_dropout,
                    dropout=config.dropout,
                    qk_norm=config.qk_norm,
                    qk_norm_eps=config.qk_norm_eps,
                    rope=config.rope,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_layer = ActionExpertFinalLayer(config.hidden_size, config.max_action_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.time_embed.modules():
            if isinstance(module, nn.Linear):
                _init_linear(module)
        _init_linear(self.action_embed)
        _init_linear(self.context_k_proj)
        _init_linear(self.context_v_proj)
        if isinstance(self.context_norm, ActionExpertRMSNorm):
            self.context_norm.reset_parameters()
        residual_scale = (2 * max(self.config.num_layers, 1)) ** -0.5
        for block in self.blocks:
            _init_linear(block.self_attn.qkv)
            _init_linear(block.self_attn.out_proj, scale=residual_scale)
            _init_linear(block.cross_attn.q_proj)
            _init_linear(block.cross_attn.out_proj, scale=residual_scale)
            _init_linear(block.mlp.up_proj)
            _init_linear(block.mlp.gate_proj)
            _init_linear(block.mlp.down_proj, scale=residual_scale)
            _init_linear(block.modulation.linear, zero=True)
            block.self_norm.reset_parameters()
            block.cross_norm.reset_parameters()
            block.ff_norm.reset_parameters()
            if block.self_attn.q_norm is not None:
                block.self_attn.q_norm.reset_parameters()
            if block.self_attn.k_norm is not None:
                block.self_attn.k_norm.reset_parameters()
            if block.cross_attn.q_norm is not None:
                block.cross_attn.q_norm.reset_parameters()
            if block.cross_attn.k_norm is not None:
                block.cross_attn.k_norm.reset_parameters()
        self.final_layer.norm.reset_parameters()
        _init_linear(self.final_layer.modulation.linear, zero=True)
        _init_linear(self.final_layer.linear, zero=True)

    def _reshape_hidden_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], x.shape[1], self.config.num_heads, self.action_head_dim)

    def _time_conditioning(self, timesteps: torch.Tensor) -> torch.Tensor:
        conditioning = self.time_embed[0](timesteps)
        first_linear = self.time_embed[1]
        if isinstance(first_linear, nn.Linear):
            conditioning = conditioning.to(dtype=first_linear.weight.dtype)
        for module in list(self.time_embed.children())[1:]:
            conditioning = module(conditioning)
        return conditioning

    def _project_kv_tensor(self, x: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        flat = self.context_norm(proj(x))
        return self._reshape_hidden_to_heads(flat)

    def _prepare_kv_context(
        self,
        encoder_kv_states: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ) -> Sequence[tuple[torch.Tensor, torch.Tensor]]:
        if len(encoder_kv_states) != len(self.blocks):
            raise ValueError(
                f"Expected {len(self.blocks)} KV layers for per-layer conditioning, "
                f"got {len(encoder_kv_states)}."
            )
        kv_contexts = []
        for block, (k_in, v_in) in zip(self.blocks, encoder_kv_states, strict=False):
            k_ctx = self._project_kv_tensor(k_in, self.context_k_proj)
            v_ctx = self._project_kv_tensor(v_in, self.context_v_proj)
            k_norm = block.cross_attn.k_norm
            if k_norm is not None:
                k_ctx = k_norm(k_ctx.transpose(1, 2)).transpose(1, 2)
            kv_contexts.append((k_ctx, v_ctx))
        return kv_contexts

    @staticmethod
    def _build_cross_attention_mask(
        encoder_attention_mask: torch.Tensor | None,
        batch_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if encoder_attention_mask is None:
            return None
        mask = encoder_attention_mask[:, None, None, :].to(dtype=dtype)
        return (1.0 - mask) * torch.finfo(dtype).min

    def _build_self_attention_mask(
        self,
        action_attention_mask: torch.Tensor | None,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        mask = None
        if action_attention_mask is not None:
            valid = action_attention_mask.to(device=device, dtype=torch.bool)
            key_mask = (~valid)[:, None, None, :].to(dtype=dtype)
            mask = key_mask * torch.finfo(dtype).min
        if self.config.causal_attn:
            causal = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).triu(diagonal=1)
            causal = causal.unsqueeze(0).unsqueeze(0).to(dtype=dtype) * torch.finfo(dtype).min
            mask = causal if mask is None else mask + causal
        return mask

    def prepare_context(
        self,
        *,
        encoder_kv_states: Sequence[tuple[torch.Tensor, torch.Tensor]],
        encoder_attention_mask: torch.Tensor | None = None,
        action_attention_mask: torch.Tensor | None = None,
        state_embeddings: torch.Tensor | None = None,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> ActionExpertContext:
        if state_embeddings is not None:
            raise ValueError(
                "MolmoAct2 HF action expert supports only discrete state tokens. "
                "Continuous state embeddings are not supported."
            )
        valid_action = None
        if action_attention_mask is not None:
            valid_action = action_attention_mask.to(device=device, dtype=dtype).unsqueeze(-1)
        rope_cache = None
        if len(self.blocks) > 0 and self.blocks[0].self_attn.rope is not None:
            rope_cache = self.blocks[0].self_attn.rope.build_cache(
                seq_len=seq_len,
                device=device,
                dtype=dtype,
            )
        kv_contexts = self._prepare_kv_context(encoder_kv_states)
        cross_mask = self._build_cross_attention_mask(
            encoder_attention_mask,
            batch_size,
            dtype,
        )
        self_mask = self._build_self_attention_mask(action_attention_mask, seq_len, device, dtype)
        return ActionExpertContext(
            kv_contexts=kv_contexts,
            cross_mask=cross_mask,
            self_mask=self_mask,
            valid_action=valid_action,
            rope_cache=rope_cache,
        )

    def prepare_modulation_cache(
        self,
        timesteps: Sequence[torch.Tensor],
    ) -> Sequence[ActionExpertStepModulation]:
        cache = []
        for _idx, step_t in enumerate(timesteps):
            conditioning = self._time_conditioning(step_t)
            block_modulations = []
            for block in self.blocks:
                block_modulations.append(tuple(block.modulation(conditioning).chunk(9, dim=1)))
            final_modulation = tuple(self.final_layer.modulation(conditioning).chunk(2, dim=1))
            cache.append(
                ActionExpertStepModulation(
                    conditioning=conditioning,
                    block_modulations=block_modulations,
                    final_modulation=final_modulation,
                )
            )
        return cache

    def get_or_prepare_modulation_cache(
        self,
        timesteps: Sequence[torch.Tensor],
        *,
        cache_key: tuple[Any, ...] | None = None,
    ) -> Sequence[ActionExpertStepModulation]:
        if self.training or cache_key is None:
            return self.prepare_modulation_cache(timesteps)
        if self._modulation_cache_key == cache_key and self._modulation_cache_value is not None:
            return self._modulation_cache_value
        cached = self.prepare_modulation_cache(timesteps)
        self._modulation_cache_key = cache_key
        self._modulation_cache_value = cached
        return cached

    def forward_with_context(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        context: ActionExpertContext,
        modulation: ActionExpertStepModulation | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = actions.shape
        if seq_len > self.config.max_action_horizon:
            raise ValueError(
                f"Action sequence length {seq_len} exceeds configured max_action_horizon={self.config.max_action_horizon}"
            )
        if modulation is None:
            conditioning = self._time_conditioning(timesteps)
            block_modulations: Sequence[tuple[torch.Tensor, ...] | None] = [None] * len(self.blocks)
            final_modulation = None
        else:
            conditioning = modulation.conditioning
            block_modulations = modulation.block_modulations
            final_modulation = modulation.final_modulation
        x = self.action_embed(actions)
        if context.valid_action is not None:
            x = x * context.valid_action
        for _idx, (block, kv_context, block_modulation) in enumerate(
            zip(self.blocks, context.kv_contexts, block_modulations, strict=False)
        ):
            x = block(
                x,
                conditioning,
                cross_kv=kv_context,
                self_attn_mask=context.self_mask,
                attn_mask=context.cross_mask,
                is_causal=self.config.causal_attn,
                modulation=block_modulation,
                rope_cache=context.rope_cache,
            )
            if context.valid_action is not None:
                x = x * context.valid_action
        out = self.final_layer(x, conditioning, modulation=final_modulation)
        if context.valid_action is not None:
            out = out * context.valid_action
        return out

    def forward(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        encoder_kv_states: Sequence[tuple[torch.Tensor, torch.Tensor]],
        encoder_attention_mask: torch.Tensor | None = None,
        action_attention_mask: torch.Tensor | None = None,
        state_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = actions.shape
        context = self.prepare_context(
            encoder_kv_states=encoder_kv_states,
            encoder_attention_mask=encoder_attention_mask,
            action_attention_mask=action_attention_mask,
            state_embeddings=state_embeddings,
            batch_size=bsz,
            seq_len=seq_len,
            device=actions.device,
            dtype=actions.dtype,
        )
        return self.forward_with_context(actions, timesteps, context=context)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        tensor = value.detach()
        if tensor.dtype in (torch.bfloat16, torch.float16):
            tensor = tensor.float()
        return tensor.cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(value, dtype=np.float32)


def _to_mask(value: Any, fallback_like: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    mask = np.asarray(value, dtype=np.bool_)
    if fallback_like is not None and mask.shape != fallback_like.shape:
        mask = np.broadcast_to(mask, fallback_like.shape)
    return mask


def _feature_dim_from_stats(stats: Mapping[str, Any] | None) -> int | None:
    if not isinstance(stats, Mapping):
        return None
    for key in (
        "mean",
        "std",
        "min",
        "max",
        "q01",
        "q99",
        "q10",
        "q90",
        "mask",
        "names",
    ):
        value = stats.get(key)
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.shape:
            return int(arr.shape[-1])
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return int(len(value))
    return None


class _FeatureNormalizer:
    def __init__(
        self,
        *,
        mode: str,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        min_val: np.ndarray | None = None,
        max_val: np.ndarray | None = None,
        q_low: np.ndarray | None = None,
        q_high: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        zero_mask: np.ndarray | None = None,
    ):
        self.mode = mode
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val
        self.q_low = q_low
        self.q_high = q_high
        self.mask = mask
        self.zero_mask = zero_mask

    @classmethod
    def from_stats(cls, stats: Mapping[str, Any] | None, mode: str) -> Optional["_FeatureNormalizer"]:
        if stats is None:
            return None
        raw_mask = stats.get("mask") if isinstance(stats, Mapping) else None
        if mode == "none":
            fallback = None
            for key in (
                "mean",
                "std",
                "min",
                "max",
                "q01",
                "q99",
                "q10",
                "q90",
                "mask",
            ):
                fallback = _to_array(stats.get(key))
                if fallback is not None:
                    break
            return cls(mode=mode, mask=_to_mask(raw_mask, fallback))
        if mode == "mean_std":
            mean = _to_array(stats.get("mean"))
            std = _to_array(stats.get("std"))
            if mean is None or std is None:
                raise ValueError("norm_mode='mean_std' requires mean and std stats.")
            return cls(mode=mode, mean=mean, std=std, mask=_to_mask(raw_mask, mean))
        if mode == "min_max":
            min_val = _to_array(stats.get("min"))
            max_val = _to_array(stats.get("max"))
            if min_val is None or max_val is None:
                raise ValueError("norm_mode='min_max' requires min and max stats.")
            return cls(
                mode=mode,
                min_val=min_val,
                max_val=max_val,
                mask=_to_mask(raw_mask, min_val),
                zero_mask=(min_val == max_val),
            )
        if mode in {"q01_q99", "q10_q90"}:
            low_key, high_key = ("q01", "q99") if mode == "q01_q99" else ("q10", "q90")
            q_low = _to_array(stats.get(low_key))
            q_high = _to_array(stats.get(high_key))
            if q_low is None or q_high is None:
                raise ValueError(f"norm_mode={mode!r} requires {low_key} and {high_key} stats.")
            min_val = _to_array(stats.get("min"))
            max_val = _to_array(stats.get("max"))
            fallback = min_val if min_val is not None else q_low
            zero_mask = None if min_val is None or max_val is None else (min_val == max_val)
            return cls(
                mode=mode,
                min_val=min_val,
                max_val=max_val,
                q_low=q_low,
                q_high=q_high,
                mask=_to_mask(raw_mask, fallback),
                zero_mask=zero_mask,
            )
        raise ValueError(f"Unsupported robot normalization mode {mode!r}.")

    def normalize(self, x: Any) -> Any:
        arr = _to_array(x)
        if arr is None:
            return None
        eps = 1e-6
        if self.mode == "none":
            normed = arr
        elif self.mode == "mean_std":
            normed = (arr - self.mean) / np.maximum(self.std, eps)
        elif self.mode == "min_max":
            normed = 2.0 * (arr - self.min_val) / np.maximum(self.max_val - self.min_val, eps) - 1.0
        elif self.mode in {"q01_q99", "q10_q90"}:
            normed = 2.0 * (arr - self.q_low) / np.maximum(self.q_high - self.q_low, eps) - 1.0
        else:
            normed = arr
        if self.mode in {"min_max", "q01_q99", "q10_q90"}:
            normed = np.clip(normed, -1.0, 1.0)
        if self.mask is not None:
            normed = np.where(self.mask, normed, arr)
        if self.zero_mask is not None:
            normed = np.where(self.zero_mask, 0.0, normed)
        if torch.is_tensor(x):
            return torch.as_tensor(normed, device=x.device, dtype=x.dtype)
        return normed

    def unnormalize(self, x: Any) -> Any:
        arr = _to_array(x)
        if arr is None:
            return None
        if self.mode in {"min_max", "q01_q99", "q10_q90"}:
            arr = np.clip(arr, -1.0, 1.0)
        if self.mode == "none":
            out = arr
        elif self.mode == "mean_std":
            out = arr * self.std + self.mean
        elif self.mode == "min_max":
            out = (arr + 1.0) * (self.max_val - self.min_val) / 2.0 + self.min_val
        elif self.mode in {"q01_q99", "q10_q90"}:
            out = (arr + 1.0) * (self.q_high - self.q_low) / 2.0 + self.q_low
        else:
            out = arr
        if self.mask is not None:
            out = np.where(self.mask, out, arr)
        if torch.is_tensor(x):
            return torch.as_tensor(out, device=x.device, dtype=x.dtype)
        return out


class _RobotStats:
    def __init__(self, payload: Mapping[str, Any]):
        self.norm_mode = str(payload.get("norm_mode", "min_max"))
        self.metadata_by_tag: dict[str, dict[str, Any]] = {
            str(tag): dict(metadata or {})
            for tag, metadata in dict(payload.get("metadata_by_tag") or {}).items()
        }
        self.action_normalizers = {}
        self.state_normalizers = {}
        for tag, metadata in self.metadata_by_tag.items():
            if metadata.get("action_stats") is not None:
                self.action_normalizers[tag] = _FeatureNormalizer.from_stats(
                    metadata.get("action_stats"),
                    self.norm_mode,
                )
            if metadata.get("state_stats") is not None:
                self.state_normalizers[tag] = _FeatureNormalizer.from_stats(
                    metadata.get("state_stats"),
                    self.norm_mode,
                )

    def validate_tag(self, norm_tag: str | None) -> str:
        tag = str(norm_tag or "").strip()
        if not tag:
            raise ValueError("MolmoAct2 `predict_action` requires `norm_tag`.")
        if tag not in self.metadata_by_tag:
            allowed = ", ".join(sorted(self.metadata_by_tag))
            raise ValueError(f"Unknown MolmoAct2 normalization tag {tag!r}. Allowed tags: {allowed}.")
        return tag

    def get_metadata(self, norm_tag: str | None) -> dict[str, Any]:
        if norm_tag is None:
            return {}
        return dict(self.metadata_by_tag.get(str(norm_tag), {}) or {})

    def normalize_state(self, state: Any, norm_tag: str) -> Any:
        normalizer = self.state_normalizers.get(str(norm_tag))
        return state if normalizer is None else normalizer.normalize(state)

    def unnormalize_action(self, action: Any, norm_tag: str) -> Any:
        normalizer = self.action_normalizers.get(str(norm_tag))
        return action if normalizer is None else normalizer.unnormalize(action)

    def get_action_dim(self, norm_tag: str) -> int | None:
        metadata = self.get_metadata(norm_tag)
        stats = metadata.get("action_stats")
        dim = _feature_dim_from_stats(stats)
        return dim

    def get_state_dim(self, norm_tag: str) -> int | None:
        metadata = self.get_metadata(norm_tag)
        return _feature_dim_from_stats(metadata.get("state_stats"))

    def get_action_horizon(self, norm_tag: str) -> int | None:
        return self._get_positive_int(norm_tag, "action_horizon")

    def get_n_action_steps(self, norm_tag: str) -> int | None:
        return self._get_positive_int(norm_tag, "n_action_steps")

    def _get_positive_int(self, norm_tag: str, key: str) -> int | None:
        value = self.get_metadata(norm_tag).get(key)
        if value is None:
            return None
        value = int(value)
        if value < 1:
            raise ValueError(f"Robot metadata for norm_tag={norm_tag!r} must define {key} >= 1.")
        return value


def _normalize_image_for_cache(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim == 3 and arr.shape[0] in {1, 3, 4} and arr.shape[-1] not in {1, 3, 4}:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.dtype in (np.float32, np.float64):
        if arr.size > 0 and float(arr.max()) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _extract_first_image(images: Any) -> np.ndarray | None:
    if images is None:
        return None
    if isinstance(images, (list, tuple)):
        if not images:
            return None
        return _normalize_image_for_cache(images[0])
    arr = _to_numpy(images)
    if arr.ndim == 4:
        return _normalize_image_for_cache(arr[0])
    return _normalize_image_for_cache(arr)


def _resize_depth_reasoning_image(image: np.ndarray, target_size: int) -> np.ndarray:
    from PIL import Image

    if image.shape[0] == target_size and image.shape[1] == target_size:
        return image
    pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    return np.asarray(pil_image.resize((target_size, target_size), Image.BILINEAR))


def _compute_depth_update_mask(
    current_image: np.ndarray,
    previous_image: np.ndarray,
    *,
    num_depth_codes: int,
) -> np.ndarray:
    grid_side = int(math.isqrt(int(num_depth_codes)))
    if grid_side * grid_side != int(num_depth_codes):
        raise ValueError(
            f"enable_adaptive_depth=True requires a square depth grid, got num_depth_codes={int(num_depth_codes)}."
        )
    target_size = grid_side * _DEPTH_REASONING_PATCH_SIZE
    current_resized = _resize_depth_reasoning_image(current_image, target_size).astype(np.float32)
    previous_resized = _resize_depth_reasoning_image(previous_image, target_size).astype(np.float32)
    current_patches = (
        current_resized.reshape(
            grid_side,
            _DEPTH_REASONING_PATCH_SIZE,
            grid_side,
            _DEPTH_REASONING_PATCH_SIZE,
            3,
        )
        .transpose(0, 2, 1, 3, 4)
        .reshape(grid_side, grid_side, -1)
    )
    previous_patches = (
        previous_resized.reshape(
            grid_side,
            _DEPTH_REASONING_PATCH_SIZE,
            grid_side,
            _DEPTH_REASONING_PATCH_SIZE,
            3,
        )
        .transpose(0, 2, 1, 3, 4)
        .reshape(grid_side, grid_side, -1)
    )
    dot = np.sum(current_patches * previous_patches, axis=-1)
    norm_current = np.linalg.norm(current_patches, axis=-1)
    norm_previous = np.linalg.norm(previous_patches, axis=-1)
    denom = norm_current * norm_previous
    similarity = np.where(denom < 1e-8, 1.0, dot / (denom + 1e-12))
    return np.asarray(similarity < _DEPTH_REASONING_THRESHOLD, dtype=np.bool_).reshape(-1)


def _build_depth_update_spans(
    update_mask: Sequence[bool],
) -> list[tuple[int, int, bool]]:
    flat_mask = np.asarray(update_mask, dtype=np.bool_).reshape(-1)
    if flat_mask.size == 0:
        return []
    spans: list[tuple[int, int, bool]] = []
    start = 0
    current_value = bool(flat_mask[0])
    for idx in range(1, int(flat_mask.shape[0])):
        next_value = bool(flat_mask[idx])
        if next_value == current_value:
            continue
        spans.append((start, idx, current_value))
        start = idx
        current_value = next_value
    spans.append((start, int(flat_mask.shape[0]), current_value))
    return spans


def _wrap_setup_text(setup_type: str, add_setup_tokens: bool = False) -> str:
    setup_type = str(setup_type or "")
    if setup_type.startswith(SETUP_START_TOKEN) and setup_type.endswith(SETUP_END_TOKEN):
        return setup_type
    if not setup_type or not add_setup_tokens:
        return setup_type
    return f"{SETUP_START_TOKEN}{setup_type}{SETUP_END_TOKEN}"


def _wrap_control_text(control_mode: str, add_control_tokens: bool = False) -> str:
    control_mode = str(control_mode or "")
    if control_mode.startswith(CONTROL_START_TOKEN) and control_mode.endswith(CONTROL_END_TOKEN):
        return control_mode
    if not control_mode or not add_control_tokens:
        return control_mode
    return f"{CONTROL_START_TOKEN}{control_mode}{CONTROL_END_TOKEN}"


def _discretize_normalized_state(state: np.ndarray, num_state_tokens: int) -> np.ndarray:
    arr = np.asarray(state, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    arr = np.clip(arr, -1.0, 1.0)
    scaled = (arr + 1.0) / 2.0 * float(num_state_tokens - 1)
    return np.clip(np.rint(scaled).astype(np.int64), 0, int(num_state_tokens) - 1)


def _build_discrete_state_string(state: np.ndarray | None, num_state_tokens: int) -> str:
    if state is None:
        return ""
    token_ids = _discretize_normalized_state(state, num_state_tokens).reshape(-1)
    return f"{STATE_START_TOKEN}{''.join(f'{STATE_TOKEN_PREFIX}{int(token_id)}>' for token_id in token_ids)}{STATE_END_TOKEN}"


def _normalize_question_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return ""
    previous = None
    while normalized and normalized != previous:
        previous = normalized
        normalized = normalized.strip().strip(_QUESTION_SURROUNDING_DELIMITERS).strip()
        for pattern in _QUESTION_PREFIX_PATTERNS:
            normalized = pattern.sub("", normalized, count=1).strip()
        normalized = normalized.rstrip(_QUESTION_TRAILING_SENTENCE_PUNCTUATION).rstrip()
        normalized = normalized.rstrip(_QUESTION_TRAILING_CLOSERS).rstrip()
        normalized = normalized.rstrip(_QUESTION_TRAILING_SENTENCE_PUNCTUATION).rstrip()
    sentence_chunks = [chunk.strip() for chunk in re.split(r"[.!?]+", normalized) if chunk.strip()]
    if len(sentence_chunks) > 1:
        normalized = "; ".join(sentence_chunks)
    normalized = normalized.lower()
    return normalized


def _build_robot_text(
    *,
    task: str,
    style: str,
    discrete_state_string: str,
    setup_type: str,
    control_mode: str,
    add_setup_tokens: bool,
    add_control_tokens: bool,
    num_images: int,
) -> str:
    setup_text = _wrap_setup_text(setup_type, add_setup_tokens=add_setup_tokens)
    control_text = _wrap_control_text(control_mode, add_control_tokens=add_control_tokens)
    state_clause = (
        f" The current state of the robot is {discrete_state_string}." if discrete_state_string else ""
    )
    if style == "robot_depth_action":
        prompt = (
            f"The task is to {task}. The setup is {setup_text}.{state_clause} "
            f"The expected control mode is {control_text}. Given these, first predict the depth map of the main image "
            "and then predict the action the robot should take to complete the task?"
        )
        trigger = f"{DEPTH_OUTPUT_TOKEN}{ACTION_OUTPUT_TOKEN}"
    else:
        prompt = (
            f"The task is to {task}. The setup is {setup_text}.{state_clause} "
            f"The expected control mode is {control_text}. Given these, what action should the robot take to complete the task?"
        )
        trigger = ACTION_OUTPUT_TOKEN
    if num_images <= 0:
        image_prefix = ""
    elif num_images == 1:
        image_prefix = "<|image|>"
    else:
        image_prefix = "".join(f"Image {idx + 1}<|image|>" for idx in range(num_images))
    return f"{image_prefix}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{trigger}"


def _flatten_generated_token_ids(token_ids: torch.Tensor) -> list[int]:
    if token_ids.ndim == 3:
        return [int(x) for x in token_ids[0, 0].detach().cpu().tolist()]
    if token_ids.ndim == 2:
        return [int(x) for x in token_ids[0].detach().cpu().tolist()]
    if token_ids.ndim == 1:
        return [int(x) for x in token_ids.detach().cpu().tolist()]
    raise ValueError(f"Unexpected generated token tensor shape {tuple(token_ids.shape)}")


def _extract_discrete_token_bins(
    generated_ids: list[int],
    start_token_id: int,
    end_token_id: int,
    token_id_to_bin: dict[int, int],
) -> list[int]:
    start_idx = None
    end_idx = None
    for idx, token_id in enumerate(generated_ids):
        if token_id == start_token_id:
            start_idx = idx
            break
    if start_idx is not None:
        for idx in range(start_idx + 1, len(generated_ids)):
            if generated_ids[idx] == end_token_id:
                end_idx = idx
                break
    span_start = 0 if start_idx is None else start_idx + 1
    span_end = len(generated_ids) if end_idx is None else end_idx
    return [
        int(token_id_to_bin[token_id])
        for token_id in generated_ids[span_start:span_end]
        if token_id in token_id_to_bin
    ]


@dataclass
class MolmoAct2ActionOutput(ModelOutput):
    actions: torch.FloatTensor | None = None
    generated_token_ids: torch.LongTensor | None = None
    depth_bins: torch.LongTensor | None = None
    depth_cache: dict[str, Any] | None = None


@dataclass
class _DepthPrefix:
    token_ids: torch.Tensor
    depth_bins: torch.Tensor
    full_input_ids: torch.Tensor
    attention_mask: torch.Tensor | None
    encoder_kv_states: Sequence[tuple[torch.Tensor, torch.Tensor]]
    next_output: Any
    past_key_values: Cache | None


@dataclass
class MolmoAct2CausalLMOutputWithPast(ModelOutput):
    """
    Base class for MolmoAct2 causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        image_hidden_states (`torch.FloatTensor`, *optional*):
            A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    image_hidden_states: torch.FloatTensor | None = None


@dataclass
class MolmoAct2ModelOutputWithPast(BaseModelOutputWithPast):
    """
    Base class for MolmoAct2 outputs, with hidden states and attentions.

    Args:
        image_hidden_states (`torch.FloatTensor`, *optional*):
            A `torch.FloatTensor` of size `(batch_num_patches, hidden_size)`.
            image_hidden_states of the model produced by the vision backbone
    """

    last_hidden_state: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    image_hidden_states: torch.FloatTensor | None = None


class ViTMLP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        hidden_act: str,
        device: str | torch.device = None,
    ):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=True, device=device)
        self.act = ACT2FN[hidden_act]
        self.w2 = nn.Linear(hidden_dim, dim, bias=True, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act(self.w1(x)))


class ViTMultiHeadDotProductAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        use_bias: bool = True,
        input_dim: int | None = None,
        float32_attention: bool = True,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        device: str | torch.device = None,
        attn_implementation: str = "eager",
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attn_implementation = attn_implementation
        self.is_causal = False

        input_dim = input_dim or hidden_size

        self.wq = nn.Linear(
            input_dim,
            self.num_heads * self.head_dim,
            bias=use_bias,
            device=device,
        )
        self.wk = nn.Linear(
            input_dim,
            self.num_key_value_heads * self.head_dim,
            bias=use_bias,
            device=device,
        )
        self.wv = nn.Linear(
            input_dim,
            self.num_key_value_heads * self.head_dim,
            bias=use_bias,
            device=device,
        )
        self.wo = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
        )
        self.float32_attention = float32_attention
        self.attention_dropout = attention_dropout
        self.residual_dropout = nn.Dropout(residual_dropout)
        self.sdpa_backend_list = [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.CUDNN_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]

    def _split_heads(self, hidden_states, num_heads) -> torch.Tensor:
        return hidden_states.reshape(hidden_states.shape[:2] + (num_heads, self.head_dim))

    def _merge_heads(self, hidden_states) -> torch.Tensor:
        return hidden_states.reshape(hidden_states.shape[:2] + (self.hidden_size,))

    def forward(
        self,
        inputs_q: torch.Tensor,
        inputs_kv: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_kv is not None:
            inputs_k = inputs_kv
            inputs_v = inputs_kv
        else:
            inputs_k = inputs_q
            inputs_v = inputs_q

        xq, xk, xv = self.wq(inputs_q), self.wk(inputs_k), self.wv(inputs_v)

        xq = self._split_heads(xq, self.num_heads)
        xk = self._split_heads(xk, self.num_key_value_heads)
        xv = self._split_heads(xv, self.num_key_value_heads)

        if self.num_heads != self.num_key_value_heads:
            xk = xk.repeat_interleave(self.num_key_value_groups, dim=2, output_size=self.num_heads)
            xv = xv.repeat_interleave(self.num_key_value_groups, dim=2, output_size=self.num_heads)

        og_dtype = xq.dtype

        if self.float32_attention:
            xq = xq.to(torch.float)
            xk = xk.to(torch.float)

        dropout_p = 0.0 if not self.training else self.attention_dropout

        if self.attn_implementation == "eager":
            attn_weights = torch.einsum("...qhd,...khd->...hqk", xq / math.sqrt(xq.size(-1)), xk)
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(xq.dtype)
            attn_weights = F.dropout(attn_weights, p=dropout_p, training=self.training)
            attn_output = torch.einsum("...hqk,...khd->...qhd", attn_weights.to(xv.dtype), xv)

        elif self.attn_implementation == "sdpa":
            if self.float32_attention:
                xv = xv.to(torch.float32)

            query = xq.transpose(1, 2).contiguous()
            key = xk.transpose(1, 2).contiguous()
            value = xv.transpose(1, 2).contiguous()
            if inputs_kv is not None:
                with sdpa_kernel(self.sdpa_backend_list):
                    attn_output = F.scaled_dot_product_attention(
                        query,
                        key,
                        value,
                        attn_mask=attn_mask,
                        is_causal=False,
                        dropout_p=dropout_p,
                    ).transpose(1, 2)
            else:
                attn_output = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    is_causal=False,
                    dropout_p=dropout_p,
                ).transpose(1, 2)

        elif self.attn_implementation == "flash_attention_2":
            if xq.dtype == torch.float32:
                if torch.is_autocast_enabled():
                    target_dtype = torch.get_autocast_gpu_dtype()
                else:
                    target_dtype = self.wq.weight.dtype
            attn_output = _flash_attention_forward(
                xq,
                xk,
                xv,
                attention_mask=attn_mask,
                query_length=inputs_q.shape[1],
                is_causal=False,
                dropout=dropout_p,
                softmax_scale=xq.shape[-1] ** -0.5,
                use_top_left_mask=flash_attn_supports_top_left_mask(),
                target_dtype=target_dtype,
                implementation=self.attn_implementation,
            )
        else:
            raise ValueError(f"Attention implementation {self.attn_implementation} not supported")

        attn_output = attn_output.to(og_dtype)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.wo(attn_output)
        attn_output = self.residual_dropout(attn_output)

        return attn_output


class MolmoAct2VisionBlock(nn.Module):
    def __init__(self, config: MolmoAct2VitConfig, device: str | torch.device = None):
        super().__init__()
        self.attention = ViTMultiHeadDotProductAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            float32_attention=config.float32_attention,
            attention_dropout=config.attention_dropout,
            residual_dropout=config.residual_dropout,
            device=device,
            attn_implementation=config._attn_implementation,
        )
        self.feed_forward = ViTMLP(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            device=device,
        )
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, device=device)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class MolmoAct2VisionBlockCollection(nn.Module):
    def __init__(self, config: MolmoAct2VitConfig, device: str | torch.device = None):
        super().__init__()
        self.config = config
        self.resblocks = nn.ModuleList(
            [MolmoAct2VisionBlock(config, device) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        hidden_states = []
        for r in self.resblocks:
            x = r(x)
            hidden_states.append(x)
        return hidden_states


class MolmoAct2VisionTransformer(nn.Module):
    def __init__(self, config: MolmoAct2VitConfig, device: str | torch.device = None):
        super().__init__()
        self.config = config

        # positional embeddings
        self.scale = config.hidden_size**-0.5
        self.num_prefix_tokens: int = 0  # no class embeddings
        self.positional_embedding = nn.Parameter(
            torch.zeros(config.image_num_pos, config.hidden_size, device=device),
        )

        image_patch_size = config.image_patch_size
        self.patch_embedding = nn.Linear(
            image_patch_size * image_patch_size * 3,
            config.hidden_size,
            bias=True,
            device=device,
        )

        self.transformer = MolmoAct2VisionBlockCollection(config, device)

    def add_pos_emb(self, x: torch.Tensor, patch_num: int) -> torch.Tensor:
        pos_emb = self.positional_embedding

        pos_emb = pos_emb.reshape(
            (
                int(math.sqrt(pos_emb.shape[0])),
                int(math.sqrt(pos_emb.shape[0])),
                pos_emb.shape[1],
            )
        )

        (patch_num_0, patch_num_1) = patch_num

        if pos_emb.shape[0] != patch_num_0 or pos_emb.shape[1] != patch_num_1:
            # Derived from https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
            # antialias: default True in jax.image.resize
            pos_emb = pos_emb.unsqueeze(0).permute(0, 3, 1, 2)
            pos_emb = F.interpolate(
                pos_emb,
                size=(patch_num_0, patch_num_1),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            pos_emb = pos_emb.permute(0, 2, 3, 1).squeeze(0)

        pos_emb = pos_emb.reshape(-1, pos_emb.shape[-1])
        x = x + pos_emb[None, :, :].to(x.dtype)
        return x

    def forward(self, x: torch.Tensor, patch_num: int = None) -> list[torch.Tensor]:
        """
        : param x: (batch_size, num_patch, n_pixels)
        """
        if patch_num is None:
            patch_num = self.config.image_num_patch

        B, N, D = x.shape

        x = self.patch_embedding(x)

        # class embeddings and positional embeddings
        x = self.add_pos_emb(x, patch_num)

        hidden_states = self.transformer(x)
        return hidden_states


class ImageProjectorMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        hidden_act: str,
        device: str | torch.device = None,
    ):
        super().__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim, bias=False, device=device)
        self.w2 = nn.Linear(hidden_dim, output_dim, bias=False, device=device)
        self.w3 = nn.Linear(input_dim, hidden_dim, bias=False, device=device)
        self.act = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act(self.w1(x)) * self.w3(x))


class MolmoAct2VisionBackbone(nn.Module):
    def __init__(self, vit_config: MolmoAct2VitConfig, adapter_config: MolmoAct2AdapterConfig):
        super().__init__()
        self.vit_config = vit_config
        self.adapter_config = adapter_config

        self.vit_layers = []
        for layer in adapter_config.vit_layers:
            if layer >= 0:
                self.vit_layers.append(layer)
            else:
                self.vit_layers.append(layer + vit_config.num_hidden_layers)

        last_layer_needed = max(self.vit_layers) + 1
        if last_layer_needed < vit_config.num_hidden_layers:
            new_vit_config = deepcopy(vit_config)
            new_vit_config.num_hidden_layers = last_layer_needed
            self.image_vit = MolmoAct2VisionTransformer(new_vit_config)
        else:
            self.image_vit = MolmoAct2VisionTransformer(vit_config)

        self.num_prefix_tokens: int = self.image_vit.num_prefix_tokens

        pool_dim = vit_config.hidden_size * len(adapter_config.vit_layers)
        self.image_pooling_2d = ViTMultiHeadDotProductAttention(
            hidden_size=adapter_config.hidden_size,
            num_heads=adapter_config.num_attention_heads,
            num_key_value_heads=adapter_config.num_key_value_heads,
            head_dim=adapter_config.head_dim,
            input_dim=pool_dim,
            float32_attention=adapter_config.float32_attention,
            attention_dropout=adapter_config.attention_dropout,
            residual_dropout=adapter_config.residual_dropout,
            attn_implementation=adapter_config._attn_implementation,
        )
        self.image_projector = ImageProjectorMLP(
            adapter_config.hidden_size,
            adapter_config.intermediate_size,
            adapter_config.text_hidden_size,
            adapter_config.hidden_act,
        )
        self.image_feature_dropout = nn.Dropout(adapter_config.image_feature_dropout)
        self.gradient_checkpointing = False

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        : param images: (batch_size, num_crops, num_patch, n_pixels)
        """
        batch_size, num_crops, num_patches, patch_dim = images.shape
        images = images.view(batch_size * num_crops, num_patches, patch_dim)

        x = self.image_vit.patch_embedding(images)
        x = self.image_vit.add_pos_emb(x, self.image_vit.config.image_num_patch)

        needed_layers = {int(layer) for layer in self.vit_layers}
        selected_features: dict[int, torch.Tensor] = {}
        use_checkpoint = bool(self.gradient_checkpointing and self.training and torch.is_grad_enabled())
        for layer_idx, block in enumerate(self.image_vit.transformer.resblocks):
            if use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
            if layer_idx in needed_layers:
                selected_features[layer_idx] = x

        missing = needed_layers - set(selected_features)
        if missing:
            raise RuntimeError(
                f"MolmoAct2 vision backbone did not produce requested layers: {sorted(missing)}."
            )

        image_features = torch.cat([selected_features[int(layer)] for layer in self.vit_layers], dim=-1)

        if self.num_prefix_tokens > 0:
            image_features = image_features[:, 1:]
        image_features = image_features.view(batch_size, num_crops, num_patches, -1)
        return image_features

    @property
    def dtype(self) -> torch.dtype:
        return self.image_vit.patch_embedding.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.image_vit.patch_embedding.weight.device

    def forward(
        self,
        images: torch.Tensor,
        pooled_patches_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # image_features: (batch_size, num_crops(=num_image), num_patch, nximage_emb_dim)
        batch_size, num_image = images.shape[:2]
        images = images.to(device=self.device)
        if images.dtype == torch.uint8:
            images = images.to(dtype=torch.float32) / 255.0
            images = images * 2.0 - 1.0
        elif torch.is_floating_point(images):
            # Native MolmoAct2 eval keeps resized SigLIP pixels as uint8 and normalizes
            # on device. Canonicalize HF processor floats to that exact grid.
            images = torch.round(((images.to(dtype=torch.float32) + 1.0) * 0.5) * 255.0)
            images = torch.clamp(images, 0.0, 255.0) / 255.0
            images = images * 2.0 - 1.0
        images = images.to(dtype=self.dtype)
        image_features = self.encode_image(images)

        image_features = self.image_feature_dropout(image_features)
        dim = image_features.shape[-1]
        valid = pooled_patches_idx >= 0
        valid_token = torch.any(valid, -1)

        # Use `pooled_patches_idx` to arange the features for image pooling
        batch_idx = torch.arange(
            pooled_patches_idx.shape[0],
            dtype=torch.long,
            device=pooled_patches_idx.device,
        )
        batch_idx = torch.tile(
            batch_idx.view(batch_size, 1, 1),
            [1, pooled_patches_idx.shape[1], pooled_patches_idx.shape[2]],
        )

        # Now [batch, num_high_res_features, pool_dim, dim]
        to_pool = image_features.reshape(batch_size, -1, dim)[batch_idx, torch.clip(pooled_patches_idx, 0)]
        to_pool = to_pool * valid.to(self.dtype)[:, :, :, None]
        to_pool = to_pool.reshape([-1, pooled_patches_idx.shape[-1], dim])
        if self.adapter_config.pooling_attention_mask:
            attn_mask = valid.reshape([-1, 1, 1, valid.shape[-1]])
            denom = valid.view(-1, to_pool.shape[-2]).float().sum(-1)
            denom = torch.where(denom == 0, 1, denom)
            query = to_pool.sum(-2, keepdim=True) / denom[:, None, None].to(to_pool.dtype)
        else:
            attn_mask = None
            query = to_pool.mean(-2, keepdim=True)
        pooled_features = self.image_pooling_2d(query, to_pool, attn_mask=attn_mask)
        pooled_features = pooled_features.reshape([batch_size, -1, pooled_features.shape[-1]])

        # MLP layer to map the feature.
        pooled_features = self.image_projector(pooled_features)
        return pooled_features.view(-1, pooled_features.shape[-1])[valid_token.flatten()]


# Copied from transformers.models.llama.modeling_llama.rotate_half


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MolmoAct2RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(
        self,
        config: MolmoAct2TextConfig,
        device: str | torch.device = None,
        rope_type: str | None = None,
    ):
        super().__init__()
        if rope_type is not None:
            self.rope_type = rope_type
        elif hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            # BC: "rope_type" was originally "type"
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        if self.rope_type == "default":
            self.rope_init_fn = self._default_rope_init
        else:
            self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=True)
        self.original_inv_freq = self.inv_freq
        self.register_buffer("_pos_sin_cache", torch.empty(0), persistent=False)
        self.register_buffer("_pos_cos_cache", torch.empty(0), persistent=False)

    @staticmethod
    def _default_rope_init(
        config: MolmoAct2TextConfig, device: str | torch.device = None, **_
    ) -> tuple[torch.Tensor, float]:
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32, device=device) / config.head_dim)
        )
        return inv_freq, 1.0

    def _target_cache_seq_len(self, x: torch.Tensor, position_ids: torch.Tensor | None) -> int:
        if self.config.max_position_embeddings:
            return int(self.config.max_position_embeddings)
        if position_ids is not None:
            return int(position_ids.max().item()) + 1
        return int(x.shape[-2])

    def _rope_cache_ready(self, device: torch.device, seq_len: int) -> bool:
        return (
            self._pos_sin_cache.numel() > 0
            and self._pos_sin_cache.device == device
            and self._pos_cos_cache.device == device
            and self._pos_sin_cache.shape[-2] >= seq_len
            and self._pos_cos_cache.shape[-2] >= seq_len
        )

    def _refresh_inv_freq_if_needed(self, device: torch.device) -> None:
        device = torch.device(device)
        expected = int(self.config.head_dim) // 2
        needs_refresh = (
            self.inv_freq is None
            or self._pos_sin_cache.numel() == 0
            or self.inv_freq.device.type == "meta"
            or self.inv_freq.device != device
            or self.inv_freq.numel() != expected
        )
        if not needs_refresh:
            inv_freq_cpu = self.inv_freq.detach()
            needs_refresh = (
                not bool(torch.isfinite(inv_freq_cpu).all().item())
                or bool((inv_freq_cpu <= 0).any().item())
                or not bool(torch.isclose(inv_freq_cpu[0].cpu(), torch.tensor(1.0)).item())
            )
        if needs_refresh:
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
            self.register_buffer("inv_freq", inv_freq, persistent=True)
            self.original_inv_freq = self.inv_freq
            self._pos_sin_cache = torch.empty(0, device=device)
            self._pos_cos_cache = torch.empty(0, device=device)

    def _build_rope_cache(self, device: torch.device, seq_len: int) -> None:
        device_type = device.type if device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            seq = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = torch.einsum("i,j->ij", seq, self.inv_freq.to(device=device, dtype=torch.float))
            emb = torch.cat((freqs, freqs), dim=-1)
            self._pos_sin_cache = emb.sin()[None, None, :, :] * self.attention_scaling
            self._pos_cos_cache = emb.cos()[None, None, :, :] * self.attention_scaling

    @torch.no_grad()
    def prepare_rope_cache(
        self,
        *,
        device: str | torch.device,
        max_seq_len: int | None = None,
    ) -> None:
        if self.rope_type != "default":
            return
        device = torch.device(device)
        seq_len = int(max_seq_len or self.config.max_position_embeddings or 0)
        if seq_len <= 0:
            raise ValueError("RoPE cache preparation requires a positive max sequence length.")
        if self._rope_cache_ready(device, seq_len):
            return
        self._refresh_inv_freq_if_needed(device)
        self._build_rope_cache(device, seq_len)

    def _select_rope_cache(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos_sin = self._pos_sin_cache[:, :, :seq_len, :]
        pos_cos = self._pos_cos_cache[:, :, :seq_len, :]
        if position_ids is None:
            sin = pos_sin[0, 0, : x.shape[-2], :]
            cos = pos_cos[0, 0, : x.shape[-2], :]
        else:
            sin = pos_sin[0, 0][position_ids].view(position_ids.shape + (pos_sin.shape[-1],))
            cos = pos_cos[0, 0][position_ids].view(position_ids.shape + (pos_cos.shape[-1],))
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = self._target_cache_seq_len(x, position_ids)
        if not self._rope_cache_ready(x.device, seq_len):
            self._refresh_inv_freq_if_needed(x.device)
            self._build_rope_cache(x.device, seq_len)
        return self._select_rope_cache(x, position_ids, seq_len)


class MolmoAct2RMSNorm(nn.Module):
    def __init__(
        self,
        size: int,
        eps: float = 1e-6,
        device: str | torch.device = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size, device=device))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            x = x.to(og_dtype)

        return self.weight * x

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class MolmoAct2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: MolmoAct2TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.is_causal = True

        self.fused_dims = (
            config.num_attention_heads * config.head_dim,
            config.head_dim * config.num_key_value_heads,
            config.head_dim * config.num_key_value_heads,
        )
        self.att_proj = nn.Linear(
            config.hidden_size,
            sum(self.fused_dims),
            bias=config.qkv_bias,
        )

        # Layer norms.
        self.k_norm: MolmoAct2RMSNorm | None = None
        self.q_norm: MolmoAct2RMSNorm | None = None
        self.qk_norm_type: str | None = None
        if config.use_qk_norm:
            k_norm_size = (
                config.head_dim
                if config.qk_norm_type == "qwen3"
                else config.num_key_value_heads * config.head_dim
            )
            self.k_norm = MolmoAct2RMSNorm(k_norm_size, eps=config.layer_norm_eps)
            q_norm_size = (
                config.head_dim
                if config.qk_norm_type == "qwen3"
                else config.num_attention_heads * config.head_dim
            )
            self.q_norm = MolmoAct2RMSNorm(q_norm_size, eps=config.layer_norm_eps)
            self.qk_norm_type = config.qk_norm_type

        self.attention_dropout = config.attention_dropout

        self.attn_out = nn.Linear(
            config.head_dim * config.num_attention_heads,
            config.hidden_size,
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        collect_layer_kv_states = bool(kwargs.pop("collect_layer_kv_states", False))
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        qkv = self.att_proj(hidden_states)
        query_states, key_states, value_states = qkv.split(self.fused_dims, dim=-1)
        value_states = value_states.view(hidden_shape)

        # Optionally apply layer norm to keys and queries.
        if self.q_norm is not None and self.k_norm is not None and self.qk_norm_type != "qwen3":
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.view(hidden_shape)
        key_states = key_states.view(hidden_shape)
        if self.q_norm is not None and self.k_norm is not None and self.qk_norm_type == "qwen3":
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        collected_key_states = key_states
        collected_value_states = value_states

        dropout_p = 0.0 if not self.training else self.attention_dropout
        if self.config._attn_implementation == "sdpa" and (
            attention_mask is None or torch.is_tensor(attention_mask)
        ):
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=dropout_p,
                is_causal=attention_mask is None,
            )
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_weights = None
        else:
            attention_interface: Callable = eager_attention_forward
            if self.config._attn_implementation != "eager":
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=dropout_p,
                scaling=self.scaling,
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.attn_out(attn_output)
        if collect_layer_kv_states:
            return attn_output, attn_weights, collected_key_states, collected_value_states
        return attn_output, attn_weights


class LanguageModelMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        intermediate_size: int,
        hidden_act: str,
        device: str | torch.device = None,
    ):
        super().__init__()
        self.ff_proj = nn.Linear(input_dim, intermediate_size * 2, bias=False, device=device)
        self.ff_out = nn.Linear(intermediate_size, input_dim, bias=False, device=device)
        self.act = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ff_proj(x)
        x, gate = x.chunk(2, dim=-1)
        x = self.act(gate) * x
        x = self.ff_out(x)
        return x


class MolmoAct2DecoderLayer(GradientCheckpointingLayer):
    def __init__(
        self,
        config: MolmoAct2TextConfig,
        layer_idx: int | None = None,
        device: str | torch.device = None,
    ):
        super().__init__()
        self.config = config

        self.self_attn = MolmoAct2Attention(config, layer_idx)
        self.attn_norm = MolmoAct2RMSNorm(config.hidden_size, eps=config.layer_norm_eps, device=device)
        self.dropout = nn.Dropout(config.residual_dropout)
        self.mlp = LanguageModelMLP(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            device=device,
        )
        self.ff_norm = MolmoAct2RMSNorm(config.hidden_size, eps=config.layer_norm_eps, device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        collect_layer_kv_states = bool(kwargs.pop("collect_layer_kv_states", False))

        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)

        # Self Attention
        attention_outputs = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            collect_layer_kv_states=collect_layer_kv_states,
            **kwargs,
        )
        hidden_states = attention_outputs[0]
        self_attn_weights = attention_outputs[1]

        hidden_states = residual + self.dropout(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.ff_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + self.dropout(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)
        if collect_layer_kv_states:
            outputs += (attention_outputs[2], attention_outputs[3])

        return outputs


class MolmoAct2PostNormDecoderLayer(MolmoAct2DecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        collect_layer_kv_states = bool(kwargs.pop("collect_layer_kv_states", False))

        residual = hidden_states

        # Self Attention
        attention_outputs = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            collect_layer_kv_states=collect_layer_kv_states,
            **kwargs,
        )
        hidden_states = attention_outputs[0]
        self_attn_weights = attention_outputs[1]
        hidden_states = self.attn_norm(hidden_states)

        hidden_states = residual + self.dropout(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.ff_norm(hidden_states)

        hidden_states = residual + self.dropout(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)
        if collect_layer_kv_states:
            outputs += (attention_outputs[2], attention_outputs[3])

        return outputs


class MolmoAct2Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        num_new_embeddings: int,
        features: int,
        device: str | torch.device = None,
    ):
        super().__init__()
        self.embedding = nn.Parameter(
            torch.zeros(num_embeddings, features, device=device),
        )
        self.new_embedding = nn.Parameter(
            torch.zeros(num_new_embeddings, features, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, torch.cat([self.embedding, self.new_embedding], dim=0))


class MolmoAct2PreTrainedModel(PreTrainedModel):
    config: MolmoAct2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "MolmoAct2DecoderLayer",
        "MolmoAct2PostNormDecoderLayer",
        "MolmoAct2VisionBlock",
        "ViTMultiHeadDotProductAttention",
    ]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": MolmoAct2DecoderLayer,
        "attentions": MolmoAct2Attention,
    }

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear,)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, MolmoAct2Embedding):
            module.embedding.data.normal_(mean=0.0, std=std)
            module.new_embedding.data.normal_(mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, MolmoAct2RMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()


class MolmoAct2TextModel(MolmoAct2PreTrainedModel):
    config: MolmoAct2TextConfig
    _no_split_modules = ["MolmoAct2DecoderLayer", "MolmoAct2PostNormDecoderLayer"]

    def __init__(self, config: MolmoAct2TextConfig):
        super().__init__(config)
        if config.additional_vocab_size is not None:
            self.wte = MolmoAct2Embedding(
                config.vocab_size,
                config.additional_vocab_size,
                config.hidden_size,
            )
        else:
            self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.emb_drop = nn.Dropout(config.embedding_dropout)
        decoder_layer = MolmoAct2PostNormDecoderLayer if config.norm_after else MolmoAct2DecoderLayer
        self.blocks = nn.ModuleList(
            [decoder_layer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.ln_f = MolmoAct2RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        if config.rope_scaling_layers is not None:
            self.rotary_embs = nn.ModuleDict(
                {
                    "default": MolmoAct2RotaryEmbedding(config, rope_type="default"),
                    "scaling": MolmoAct2RotaryEmbedding(config),
                }
            )
        else:
            self.rotary_emb = MolmoAct2RotaryEmbedding(config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @torch.no_grad()
    def prepare_rope_cache(
        self,
        *,
        device: str | torch.device,
        max_seq_len: int | None = None,
    ) -> None:
        if self.config.rope_scaling_layers is not None:
            for rotary_emb in self.rotary_embs.values():
                rotary_emb.prepare_rope_cache(device=device, max_seq_len=max_seq_len)
            return
        self.rotary_emb.prepare_rope_cache(device=device, max_seq_len=max_seq_len)

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.wte

    def set_input_embeddings(self, value: torch.nn.Module) -> None:
        self.wte = value

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        collect_layer_kv_states = bool(kwargs.pop("collect_layer_kv_states", False))
        if collect_layer_kv_states and past_key_values is not None:
            raise ValueError("collect_layer_kv_states cannot be used with past_key_values.")
        if collect_layer_kv_states:
            use_cache = False

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            input_ids = input_ids * (input_ids != -1).to(input_ids.dtype)
            inputs_embeds = self.wte(input_ids)

        # torch.jit.trace() doesn't support cache objects in the output
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if torch.is_tensor(attention_mask) and attention_mask.ndim == 4:
            causal_mask_mapping = attention_mask
        elif not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }

            # Create the mask
            causal_mask_mapping = create_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        if self.config.rope_scaling_layers is not None:
            position_embeddings_mapping = {
                "default": self.rotary_embs["default"](hidden_states, position_ids),
                "scaling": self.rotary_embs["scaling"](hidden_states, position_ids),
            }
        else:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        collected_kv_states = [] if collect_layer_kv_states else None

        for layer_idx, decoder_block in enumerate(self.blocks[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.config.rope_scaling_layers is not None:
                position_embeddings_i = (
                    position_embeddings_mapping["scaling"]
                    if layer_idx in self.config.rope_scaling_layers
                    else position_embeddings_mapping["default"]
                )
            else:
                position_embeddings_i = position_embeddings

            layer_outputs = decoder_block(
                hidden_states,
                attention_mask=causal_mask_mapping,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings_i,
                collect_layer_kv_states=collect_layer_kv_states,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            output_idx = 1
            if output_attentions:
                all_self_attns += (layer_outputs[output_idx],)
                output_idx += 1
            if collect_layer_kv_states:
                collected_kv_states.append((layer_outputs[output_idx], layer_outputs[output_idx + 1]))

        hidden_states = self.ln_f(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=tuple(collected_kv_states) if collect_layer_kv_states else past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# Adapted from transformers.models.gemma3.modeling_gemma3
def token_type_ids_mask_function(
    token_type_ids: torch.Tensor | None = None,
) -> Callable | None:
    """
    This function adds the correct offsets to the `q_idx` and `kv_idx` as the torch API can only accept lengths,
    not start and end indices.
    """
    # Do not return an additional mask in this case
    if token_type_ids is None:
        return None

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        # If it's 1 for both query and key/value, we are in an image block
        # NOTE: static cache shape goes beyond input seq length, while token_type_ids.shape[1] == input seq length
        # Since vmap doesn't support `if statement` we workaround it with `torch.where`
        safe_idx = torch.where(kv_idx < token_type_ids.shape[1], kv_idx, 0)
        token_type_ids_at_kv_idx = token_type_ids[batch_idx, safe_idx]
        token_type_ids_at_kv_idx = torch.where(kv_idx < token_type_ids.shape[1], token_type_ids_at_kv_idx, 0)

        is_image_block = (token_type_ids[batch_idx, q_idx] == 1) & (token_type_ids_at_kv_idx == 1)

        # This is bidirectional attention whenever we are dealing with image tokens
        return is_image_block & is_image_block

    return inner_mask


class MolmoAct2Model(MolmoAct2PreTrainedModel):
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {}
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: MolmoAct2Config

    def __init__(self, config: MolmoAct2Config):
        super().__init__(config)
        self.transformer: MolmoAct2TextModel = MolmoAct2TextModel(config.text_config)
        self.vision_backbone: MolmoAct2VisionBackbone | None = None
        if config.vit_config is not None and config.adapter_config is not None:
            self.vision_backbone = MolmoAct2VisionBackbone(config.vit_config, config.adapter_config)
        llm_kv_dim = config.text_config.num_key_value_heads * config.text_config.head_dim
        if config.add_action_expert:
            self.action_expert = ActionExpert(
                config.action_expert_config,
                llm_dim=config.hidden_size,
                llm_kv_dim=llm_kv_dim,
                llm_num_layers=config.num_hidden_layers,
            )
        else:
            self.action_expert = None
        if config.add_action_expert and config.action_expert_depth_gate:
            if config.action_expert_depth_gate_per_layer:
                self.action_expert_depth_gate = nn.ModuleList(
                    nn.Linear(llm_kv_dim, 1) for _ in range(config.action_expert_config.num_layers)
                )
            else:
                self.action_expert_depth_gate = nn.Linear(llm_kv_dim, 1)
            self.reset_action_expert_depth_gate_parameters()
        else:
            self.action_expert_depth_gate = None
        self._depth_gate_token_ids = self._resolve_depth_gate_token_ids()
        self.action_cuda_graph_manager: ActionCudaGraphManager | None = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.transformer.wte

    def set_input_embeddings(self, value: torch.nn.Module) -> None:
        self.transformer.wte = value

    def set_decoder(self, decoder):
        self.transformer = decoder

    def get_decoder(self):
        return self.transformer

    @property
    def device(self) -> torch.device:
        return self.transformer.ln_f.weight.device

    def reset_action_expert_depth_gate_parameters(self) -> None:
        if self.action_expert_depth_gate is None:
            return
        gates = (
            self.action_expert_depth_gate
            if isinstance(self.action_expert_depth_gate, nn.ModuleList)
            else [self.action_expert_depth_gate]
        )
        for gate in gates:
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, float(self.config.action_expert_depth_gate_init_bias))

    def _resolve_depth_gate_token_ids(self) -> tuple[int, ...]:
        if not self.config.action_expert_depth_gate:
            return ()
        token_ids = []
        for token_id in (
            self.config.depth_output_token_id,
            self.config.depth_start_token_id,
            self.config.depth_end_token_id,
        ):
            if token_id is not None:
                token_ids.append(int(token_id))
        if self.config.depth_token_start_id is not None and int(self.config.num_depth_tokens or 0) > 0:
            start = int(self.config.depth_token_start_id)
            token_ids.extend(range(start, start + int(self.config.num_depth_tokens)))
        return tuple(dict.fromkeys(token_ids))

    def _require_action_expert(self) -> ActionExpert:
        if self.action_expert is None:
            raise RuntimeError("This MolmoAct2 checkpoint does not include an action expert.")
        return self.action_expert

    def _cache_to_sequence(self, cache: torch.Tensor) -> torch.Tensor:
        if cache.dim() != 4:
            raise ValueError(f"Expected KV cache tensor with 4 dims, got shape {tuple(cache.shape)}")
        head_candidates = {
            self.config.text_config.num_key_value_heads,
            self.config.text_config.num_attention_heads,
        }
        if cache.shape[1] in head_candidates:
            bsz, n_heads, seq_len, head_dim = cache.shape
            return cache.permute(0, 2, 1, 3).reshape(bsz, seq_len, n_heads * head_dim)
        if cache.shape[2] in head_candidates:
            bsz, seq_len, n_heads, head_dim = cache.shape
            return cache.reshape(bsz, seq_len, n_heads * head_dim)
        if cache.shape[1] <= cache.shape[2]:
            bsz, n_heads, seq_len, head_dim = cache.shape
            return cache.permute(0, 2, 1, 3).reshape(bsz, seq_len, n_heads * head_dim)
        bsz, seq_len, n_heads, head_dim = cache.shape
        return cache.reshape(bsz, seq_len, n_heads * head_dim)

    def _extract_kv_states(self, past_key_values: Cache) -> Sequence[tuple[torch.Tensor, torch.Tensor]]:
        if past_key_values is None:
            raise RuntimeError("Action generation requires past_key_values from the VLM forward pass.")
        seq_len = _cache_seq_len_int(past_key_values)
        kv_states = []
        for key, value in _iter_cache_key_values(past_key_values):
            if key is None or value is None:
                continue
            if key.shape[-2] > seq_len:
                key = key[..., :seq_len, :]
                value = value[..., :seq_len, :]
            kv_states.append((self._cache_to_sequence(key), self._cache_to_sequence(value)))
        if len(kv_states) != self.config.action_expert_config.num_layers:
            raise RuntimeError(
                f"Expected {self.config.action_expert_config.num_layers} KV layers, got {len(kv_states)}."
            )
        return kv_states

    @staticmethod
    def _mask_discrete_output_span(
        row_ids: torch.Tensor,
        row_mask: torch.Tensor,
        start_id: int | None,
        end_id: int | None,
    ) -> None:
        if start_id is None or end_id is None:
            return
        start_positions = (row_ids == start_id).nonzero(as_tuple=False).flatten().tolist()
        if not start_positions:
            return
        end_positions = (row_ids == end_id).nonzero(as_tuple=False).flatten().tolist()
        end_ptr = 0
        for start_pos in start_positions:
            while end_ptr < len(end_positions) and end_positions[end_ptr] < start_pos:
                end_ptr += 1
            if end_ptr >= len(end_positions):
                row_mask[start_pos:] = False
                break
            end_pos = end_positions[end_ptr]
            row_mask[start_pos : end_pos + 1] = False
            end_ptr += 1

    def _get_encoder_attention_mask(
        self,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if attention_mask is not None:
            mask = attention_mask.to(dtype=torch.bool).clone()
        elif input_ids is not None:
            mask = input_ids != -1
        else:
            return None
        if self.config.action_mode != "both" or input_ids is None:
            return mask
        eos_id = getattr(self.config, "eos_token_id", None)
        if eos_id is not None:
            mask &= input_ids != int(eos_id)
        for batch_idx in range(input_ids.shape[0]):
            self._mask_discrete_output_span(
                input_ids[batch_idx],
                mask[batch_idx],
                self.config.action_start_token_id,
                self.config.action_end_token_id,
            )
        return mask

    def _get_depth_token_mask(
        self,
        input_ids: torch.Tensor | None,
        encoder_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if not self.config.action_expert_depth_gate or input_ids is None or not self._depth_gate_token_ids:
            return None
        depth_token_ids = torch.as_tensor(
            self._depth_gate_token_ids,
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        depth_mask = (input_ids.unsqueeze(-1) == depth_token_ids).any(dim=-1)
        if encoder_attention_mask is not None:
            depth_mask = depth_mask & encoder_attention_mask.to(device=input_ids.device, dtype=torch.bool)
        return depth_mask

    @staticmethod
    def _depth_gate_from_source(
        gate_head: nn.Linear,
        *,
        source: torch.Tensor,
        depth_mask: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if source.ndim == 4:
            source = source.reshape(source.shape[0], source.shape[1], -1)
        if source.ndim != 3:
            raise ValueError(f"Depth gate expected a 3D sequence tensor, got {tuple(source.shape)}.")
        if encoder_attention_mask is not None:
            valid_mask = encoder_attention_mask.to(device=source.device, dtype=torch.bool)
        else:
            valid_mask = torch.ones(depth_mask.shape, device=source.device, dtype=torch.bool)
        depth_mask = depth_mask.to(device=source.device, dtype=torch.bool)
        pool_mask = valid_mask & ~depth_mask
        has_pool = pool_mask.any(dim=-1, keepdim=True)
        pool_mask = torch.where(has_pool, pool_mask, valid_mask)
        weights = pool_mask.to(dtype=source.dtype).unsqueeze(-1)
        pooled = (source * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        gate_logits = gate_head(pooled.to(dtype=gate_head.weight.dtype))
        return torch.sigmoid(gate_logits).to(dtype=source.dtype)

    def _depth_gate_from_condition(
        self,
        *,
        input_ids: torch.Tensor | None,
        encoder_attention_mask: torch.Tensor | None,
        layer_kv_states: Sequence[tuple[torch.Tensor, torch.Tensor]] | None,
    ) -> tuple[torch.Tensor | Sequence[torch.Tensor] | None, torch.Tensor | None]:
        gate_head = self.action_expert_depth_gate
        if gate_head is None:
            return None, None
        depth_mask = self._get_depth_token_mask(input_ids, encoder_attention_mask)
        if depth_mask is None or layer_kv_states is None:
            return None, depth_mask
        sources = [value for _, value in layer_kv_states]
        if isinstance(gate_head, nn.ModuleList):
            if len(gate_head) != len(sources):
                raise ValueError(
                    f"Depth gate layer count mismatch: gates={len(gate_head)}, sources={len(sources)}."
                )
            gates = [
                self._depth_gate_from_source(
                    gate,
                    source=source,
                    depth_mask=depth_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
                for gate, source in zip(gate_head, sources, strict=False)
            ]
            return gates, depth_mask
        gate = self._depth_gate_from_source(
            gate_head,
            source=sources[-1],
            depth_mask=depth_mask,
            encoder_attention_mask=encoder_attention_mask,
        )
        return gate, depth_mask

    @staticmethod
    def _depth_gate_for_layer(
        gate: torch.Tensor | Sequence[torch.Tensor],
        layer_idx: int,
        *,
        num_layers: int,
    ) -> torch.Tensor:
        if isinstance(gate, torch.Tensor):
            return gate
        if len(gate) != num_layers:
            raise ValueError(f"Depth gate layer count mismatch: gates={len(gate)}, layers={num_layers}.")
        return gate[layer_idx]

    def _apply_depth_gate_to_layer_kv_states(
        self,
        layer_kv_states: Sequence[tuple[torch.Tensor, torch.Tensor]] | None,
        depth_mask: torch.Tensor | None,
        gate: torch.Tensor | Sequence[torch.Tensor] | None,
    ) -> Sequence[tuple[torch.Tensor, torch.Tensor]] | None:
        if layer_kv_states is None or depth_mask is None or gate is None:
            return layer_kv_states
        gated_kv = []
        for layer_idx, (key, value) in enumerate(layer_kv_states):
            layer_gate = self._depth_gate_for_layer(gate, layer_idx, num_layers=len(layer_kv_states))
            mask = depth_mask.to(device=key.device, dtype=torch.bool)
            view_shape = [mask.shape[0], mask.shape[1]] + [1] * (key.ndim - 2)
            scale = torch.ones(view_shape, device=key.device, dtype=key.dtype)
            gate_view = layer_gate.to(device=key.device, dtype=key.dtype).view(
                layer_gate.shape[0],
                *([1] * (key.ndim - 1)),
            )
            scale = torch.where(mask.view(view_shape), gate_view, scale)
            gated_kv.append((key * scale, value * scale))
        return gated_kv

    @staticmethod
    def _action_dim_valid_mask(
        target: torch.Tensor,
        action_dim_is_pad: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if action_dim_is_pad is None:
            return None
        mask = ~action_dim_is_pad.to(device=target.device, dtype=torch.bool)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.shape[-1] != target.shape[-1]:
            raise ValueError(
                f"action_dim_is_pad width {mask.shape[-1]} does not match target width {target.shape[-1]}."
            )
        if mask.shape[0] == 1 and target.shape[0] != 1:
            mask = mask.expand(target.shape[0], -1)
        if mask.shape[0] != target.shape[0]:
            raise ValueError(
                f"action_dim_is_pad batch {mask.shape[0]} does not match target batch {target.shape[0]}."
            )
        while mask.ndim < target.ndim:
            mask = mask.unsqueeze(1)
        return mask

    @classmethod
    def _mask_action_dim_tensor(
        cls,
        tensor: torch.Tensor,
        *,
        action_dim_is_pad: torch.Tensor | None,
        enabled: bool,
    ) -> torch.Tensor:
        if not enabled:
            return tensor
        valid_mask = cls._action_dim_valid_mask(tensor, action_dim_is_pad)
        if valid_mask is None:
            return tensor
        return tensor.masked_fill(~valid_mask, 0)

    def _run_action_flow_loop(self, inputs: _ActionFlowInputs, steps: int) -> torch.Tensor:
        action_expert = self._require_action_expert()
        dt = 1.0 / steps
        trajectory = inputs.trajectory
        action_dim_is_pad = inputs.action_dim_is_pad
        mask_enabled = self.config.mask_action_dim_padding
        for idx in range(steps):
            velocity = action_expert.forward_with_context(
                trajectory,
                inputs.modulations[idx].conditioning,
                context=inputs.context,
                modulation=inputs.modulations[idx],
            )
            velocity = self._mask_action_dim_tensor(
                velocity,
                action_dim_is_pad=action_dim_is_pad,
                enabled=mask_enabled,
            )
            trajectory = trajectory + dt * velocity
            trajectory = self._mask_action_dim_tensor(
                trajectory,
                action_dim_is_pad=action_dim_is_pad,
                enabled=mask_enabled,
            )
        return trajectory

    def _resolve_action_horizon(self, action_horizon: int | None = None) -> int:
        max_action_horizon = int(self.config.max_action_horizon or 1)
        resolved = max_action_horizon if action_horizon is None else int(action_horizon)
        if resolved < 1:
            raise ValueError(f"action_horizon must be >= 1, got {resolved}.")
        if resolved > max_action_horizon:
            raise ValueError(
                f"Requested action_horizon={resolved} exceeds checkpoint max_action_horizon={max_action_horizon}."
            )
        return resolved

    @torch.no_grad()
    def generate_actions_from_inputs(
        self,
        *,
        input_ids: torch.LongTensor,
        pixel_values: torch.Tensor | None = None,
        image_token_pooling: torch.Tensor | None = None,
        image_grids: torch.Tensor | None = None,
        image_num_crops: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_token_pooling: torch.Tensor | None = None,
        video_grids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        states: torch.Tensor | None = None,
        action_dim_is_pad: torch.Tensor | None = None,
        action_horizon: int | None = None,
        num_steps: int | None = None,
        generator: torch.Generator | None = None,
        encoder_kv_states: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        action_expert = self._require_action_expert()
        if encoder_kv_states is None:
            outputs = self(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_token_pooling=image_token_pooling,
                image_grids=image_grids,
                image_num_crops=image_num_crops,
                pixel_values_videos=pixel_values_videos,
                video_token_pooling=video_token_pooling,
                video_grids=video_grids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                use_cache=True,
            )
            encoder_kv_states = self._extract_kv_states(outputs.past_key_values)
            encoder_attention_mask = self._get_encoder_attention_mask(input_ids, attention_mask)
        elif encoder_attention_mask is None:
            encoder_attention_mask = self._get_encoder_attention_mask(input_ids, attention_mask)

        depth_gate, depth_mask = self._depth_gate_from_condition(
            input_ids=input_ids,
            encoder_attention_mask=encoder_attention_mask,
            layer_kv_states=encoder_kv_states,
        )
        encoder_kv_states = self._apply_depth_gate_to_layer_kv_states(
            encoder_kv_states,
            depth_mask,
            depth_gate,
        )
        steps = int(num_steps or self.config.flow_matching_num_steps)
        if steps <= 0:
            raise ValueError(f"num_steps must be >= 1, got {steps}.")
        source_tensor = encoder_kv_states[0][0]
        batch_size = source_tensor.shape[0]
        device = source_tensor.device
        action_horizon = self._resolve_action_horizon(action_horizon)
        trajectory_dtype = action_expert.action_embed.weight.dtype
        trajectory = torch.randn(
            (batch_size, action_horizon, self.config.max_action_dim),
            device=device,
            dtype=trajectory_dtype,
            generator=generator,
        )
        trajectory = self._mask_action_dim_tensor(
            trajectory,
            action_dim_is_pad=action_dim_is_pad,
            enabled=self.config.mask_action_dim_padding,
        )
        action_context = action_expert.prepare_context(
            encoder_kv_states=encoder_kv_states,
            encoder_attention_mask=encoder_attention_mask,
            state_embeddings=states,
            batch_size=batch_size,
            seq_len=trajectory.shape[1],
            device=device,
            dtype=trajectory.dtype,
        )
        flow_timesteps = [
            torch.full((batch_size,), idx / steps, device=device, dtype=torch.float32) for idx in range(steps)
        ]
        modulation_cache = action_expert.get_or_prepare_modulation_cache(
            flow_timesteps,
            cache_key=(steps, batch_size, device, trajectory.dtype),
        )
        flow_inputs = _ActionFlowInputs(
            trajectory=trajectory,
            context=action_context,
            modulations=modulation_cache,
            action_dim_is_pad=action_dim_is_pad,
        )
        action_cuda_graph_manager = self.action_cuda_graph_manager
        if action_cuda_graph_manager is not None and action_cuda_graph_manager.can_use_action_flow(
            flow_inputs
        ):
            trajectory = action_cuda_graph_manager.run_action_flow(
                flow_inputs, steps, self._run_action_flow_loop
            )
        else:
            trajectory = self._run_action_flow_loop(flow_inputs, steps)
        return trajectory

    def build_batched_images(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.Tensor,
        image_token_pooling: torch.Tensor,
        image_grids: torch.Tensor,
        image_num_crops: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1) Count the number of images in each example
        raw_counts = (input_ids == self.config.image_end_token_id).sum(1)  # [N]
        total_images = int(image_grids.size(0))
        total_end_tokens = int(raw_counts.sum().item())
        if total_images <= 0:
            counts = raw_counts.new_zeros(raw_counts.shape)
        elif total_end_tokens == total_images:
            counts = raw_counts
        elif total_end_tokens == 2 * total_images:
            counts = raw_counts // 2
        else:
            raise ValueError(
                "Could not infer image counts from image end tokens: "
                f"end_tokens={total_end_tokens}, image_grids={total_images}."
            )
        N = counts.size(0)
        device = input_ids.device

        # Total number of images in the batch
        num_images = total_images

        # Sanity check
        assert image_grids.size(0) == num_images, (
            f"Expected {num_images} image grids, but got {image_grids.size(0)}"
        )
        assert image_num_crops.size(0) == num_images, (
            f"Expected {num_images} image num crops, but got {image_num_crops.size(0)}"
        )

        # 1-1) Compute per-image pooled patch count from image grids
        with torch.no_grad():
            first_prod = image_grids[:, :2].prod(dim=1)  # [num_images]
            second_prod = image_grids[:, 2:].prod(dim=1)  # [num_images]
            num_pooled_patches_per_image = (first_prod + second_prod).to(
                image_num_crops.dtype
            )  # [num_images]

        # pixel_values: [n_crops, n_patches, pixels_per_patch]
        n_crops, n_patches, pixels_per_patch = pixel_values.shape

        # 2) Map each image index → example index
        # Example: if counts = [2, 1, 3], then this becomes [0,0,1,2,2,2]
        example_ids_for_image = torch.arange(N, device=device).repeat_interleave(counts)  # [num_images]
        assert example_ids_for_image.numel() == num_images

        # 2-1) Compute crops_per_example by summing per-image crop counts
        crops_per_example = torch.zeros(N, dtype=image_num_crops.dtype, device=image_num_crops.device)
        crops_per_example.index_add_(0, example_ids_for_image, image_num_crops)  # [N]

        # 2-2) Per-image number of patches = (crops per image) * n_patches
        patches_per_image = image_num_crops * n_patches  # [num_images]

        # 2-3) Compute per-example per-image patch offsets
        counts_list = counts.tolist()
        index_offset_per_example_list = []
        offset_img = 0
        for c in counts_list:
            per_img_patches = patches_per_image[offset_img : offset_img + c]  # [c]
            # Offsets: [0, img0_total_patches, img0+img1_total_patches, ...]
            index_offset = [0] + per_img_patches.cumsum(0).tolist()[:-1]
            index_offset_per_example_list.append(index_offset)
            offset_img += c

        # 2-4) Compute num_pooled_patches_per_example
        num_pooled_patches_per_example = torch.zeros(
            N,
            dtype=num_pooled_patches_per_image.dtype,
            device=num_pooled_patches_per_image.device,
        )
        num_pooled_patches_per_example.index_add_(0, example_ids_for_image, num_pooled_patches_per_image)

        # Sanity checks
        total_crops = int(crops_per_example.sum().item())
        assert total_crops == n_crops, f"Expected {total_crops} crops, but got {n_crops}"

        total_num_pooled_patches = int(num_pooled_patches_per_example.sum().item())
        assert total_num_pooled_patches == image_token_pooling.size(0), (
            f"Expected {total_num_pooled_patches} pooled patches, but got {image_token_pooling.size(0)}"
        )

        # 3) Build images tensor filled with -1
        M = int(crops_per_example.max().item())
        images = torch.full(
            (N, M, n_patches, pixels_per_patch),
            fill_value=-1,
            dtype=pixel_values.dtype,
            device=pixel_values.device,
        )

        # 4) Fill images with per-example slices from pixel_values
        offset_crop = 0
        for i in range(N):
            num = int(crops_per_example[i].item())
            cur = pixel_values[offset_crop : offset_crop + num]  # [num, n_patches, pixels_per_patch]
            images[i, :num] = cur
            offset_crop += num

        # Sanity check
        assert offset_crop == n_crops

        # 5) Build new_token_pooling tensor filled with -1
        P = int(num_pooled_patches_per_example.max().item())
        _, dim = image_token_pooling.shape
        new_token_pooling = torch.full(
            (N, P, dim),
            fill_value=-1,
            dtype=image_token_pooling.dtype,
            device=image_token_pooling.device,
        )

        # 6) Fill token_pooling with per-example slices, adding per-image patch offsets
        patch_offset = 0
        img_offset = 0

        for i, c in enumerate(counts_list):
            num_patches = int(num_pooled_patches_per_example[i].item())

            # Subsequence of pooled tokens belonging to this example
            cur = image_token_pooling[patch_offset : patch_offset + num_patches].clone()  # [num_patches, dim]

            index_offset_per_example = index_offset_per_example_list[i]  # length = c
            per_img_pooled = num_pooled_patches_per_image[img_offset : img_offset + c]  # [c]

            assert len(index_offset_per_example) == per_img_pooled.numel()

            # Apply per-image offsets to the (ragged) subsequence
            offset = 0
            for j in range(c):
                index_offset = int(index_offset_per_example[j])
                n = int(per_img_pooled[j].item())
                cur_slice = cur[offset : offset + n]

                # Apply offset across all columns
                cur[offset : offset + n] = torch.where(
                    cur_slice >= 0,
                    cur_slice + index_offset,
                    cur_slice,
                )
                offset += n

            new_token_pooling[i, :num_patches] = cur

            patch_offset += num_patches
            img_offset += c

        # Final sanity checks
        assert patch_offset == total_num_pooled_patches
        assert img_offset == num_images

        return images, new_token_pooling

    def build_batched_videos(
        self,
        input_ids: torch.LongTensor,
        pixel_values_videos: torch.Tensor,
        video_token_pooling: torch.Tensor,
        video_grids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1) Count the number of videos in each example
        if self.config.use_frame_special_tokens:
            end_token_id = self.config.frame_end_token_id
        else:
            end_token_id = self.config.image_end_token_id
        counts = (input_ids == end_token_id).any(dim=1).long()  # [N]
        N = counts.size(0)
        device = input_ids.device

        # Total number of videos in the batch
        num_videos = int(counts.sum().item())

        # Sanity check
        assert video_grids.size(0) == num_videos, (
            f"Expected {num_videos} videos, but got {video_grids.size(0)}"
        )

        video_num_frames = video_grids[:, 0]  # [num_videos]
        num_pooled_patches_per_video = video_grids.prod(dim=1)  # [num_videos]

        # pixel_values_videos: [n_frames, n_patches, pixels_per_patch]
        n_frames, n_patches, pixels_per_patch = pixel_values_videos.shape

        # 2) Map each video index -> example index
        # Example: if counts = [2, 1, 3], then this becomes [0,0,1,2,2,2]
        example_ids_for_video = torch.arange(N, device=device).repeat_interleave(counts)  # [num_videos]
        assert example_ids_for_video.numel() == num_videos

        # 2-1) Compute frames_per_example by summing per-video frame counts
        frames_per_example = torch.zeros(
            N,
            dtype=video_num_frames.dtype,
            device=device,
        )
        frames_per_example.index_add_(0, example_ids_for_video, video_num_frames)  # [N]

        # 2-2) Compute num_pooled_patches_per_example
        num_pooled_patches_per_example = torch.zeros(
            N,
            dtype=num_pooled_patches_per_video.dtype,
            device=num_pooled_patches_per_video.device,
        )
        num_pooled_patches_per_example.index_add_(
            0,
            example_ids_for_video,
            num_pooled_patches_per_video,
        )

        # Sanity checks
        total_frames = int(frames_per_example.sum().item())
        assert total_frames == n_frames, f"Expected {total_frames} frames, but got {n_frames}"

        total_num_pooled_patches = int(num_pooled_patches_per_example.sum().item())
        assert total_num_pooled_patches == video_token_pooling.size(0), (
            f"Expected {total_num_pooled_patches} pooled patches, but got {video_token_pooling.size(0)}"
        )

        # 3) Build videos tensor filled with -1
        M = int(frames_per_example.max().item())
        videos = torch.full(
            (N, M, n_patches, pixels_per_patch),
            fill_value=-1,
            dtype=pixel_values_videos.dtype,
            device=device,
        )

        # 4) Fill videos with per-examples slices from pixel_values_videos
        offset_frame = 0
        for i in range(N):
            num = int(frames_per_example[i].item())
            cur = pixel_values_videos[offset_frame : offset_frame + num]  # [num, n_patches, pixels_per_patch]
            videos[i, :num] = cur
            offset_frame += num

        # Sanity check
        assert offset_frame == n_frames

        # 5) Build new token_pooling tensor filled with -1
        P = int(num_pooled_patches_per_example.max().item())
        _, dim = video_token_pooling.shape
        new_token_pooling = torch.full(
            (N, P, dim),
            fill_value=-1,
            dtype=video_token_pooling.dtype,
            device=video_token_pooling.device,
        )

        # 6) Fill new token_pooling with per-examples slices from video_token_pooling
        patch_offset = 0
        for i in range(N):
            num_patches = int(num_pooled_patches_per_example[i].item())
            cur = video_token_pooling[patch_offset : patch_offset + num_patches]  # [num_patches, dim]
            new_token_pooling[i, :num_patches] = cur
            patch_offset += num_patches

        # Final sanity checks
        assert patch_offset == total_num_pooled_patches

        return videos, new_token_pooling

    def merge_visual_inputs(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_token_pooling: torch.Tensor | None = None,
        image_grids: torch.Tensor | None = None,
        image_num_crops: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_token_pooling: torch.Tensor | None = None,
        video_grids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if pixel_values is not None and pixel_values_videos is not None:
            raise ValueError("pixel_values and pixel_values_videos are provided at the same time")
        elif pixel_values is not None:
            assert input_ids is not None
            images, token_pooling = self.build_batched_images(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_token_pooling=image_token_pooling,
                image_grids=image_grids,
                image_num_crops=image_num_crops,
            )
        elif pixel_values_videos is not None:
            assert input_ids is not None
            images, token_pooling = self.build_batched_videos(
                input_ids=input_ids,
                pixel_values_videos=pixel_values_videos,
                video_token_pooling=video_token_pooling,
                video_grids=video_grids,
            )
        else:
            images, token_pooling = None, None
        return images, token_pooling

    def build_input_embeddings(
        self,
        input_ids: torch.LongTensor,
        images: torch.FloatTensor | None = None,  # image inputs
        token_pooling: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Get embeddings of input.
        # shape: (batch_size, seq_len, d_model)
        input_ids = input_ids * (input_ids != -1).to(input_ids.dtype)
        x = self.transformer.wte(input_ids)

        image_features: torch.FloatTensor | None = None
        if images is not None:
            image_features = self.vision_backbone(images, token_pooling).to(x.device)
            is_image_patch = input_ids.reshape(-1) == self.config.image_patch_id
            if is_image_patch.sum() != len(image_features):
                raise RuntimeError(
                    f"Expected {int(is_image_patch.sum())} image patch embeddings, got {len(image_features)}."
                )
            flat_x = x.reshape(-1, x.shape[-1]).clone()
            flat_x[is_image_patch] = flat_x[is_image_patch] + image_features
            x = flat_x.reshape_as(x)

        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.emb_drop(x)  # type: ignore

        return x, image_features

    def _build_native_attention_bias(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        token_type_ids: torch.Tensor | None,
        past_key_values: Cache | None,
    ) -> torch.Tensor:
        if attention_mask is not None and attention_mask.ndim == 4:
            return attention_mask.to(device=inputs_embeds.device)
        batch_size, seq_len = inputs_embeds.shape[:2]
        past_length = _cache_seq_len_int(past_key_values)
        current_length = past_length + int(seq_len)
        max_cache_len = _cache_max_len_int(past_key_values)
        attention_mask_len = max_cache_len if max_cache_len > 0 else current_length
        device = inputs_embeds.device

        if attention_mask is None:
            positions = torch.arange(attention_mask_len, device=device)
            valid_mask = positions.unsqueeze(0) < current_length
            valid_mask = valid_mask.expand(batch_size, -1)
        elif attention_mask.ndim == 2:
            valid_mask = torch.zeros((batch_size, attention_mask_len), device=device, dtype=torch.bool)
            source_mask = attention_mask.to(device=device, dtype=torch.bool)
            copy_len = min(int(source_mask.shape[-1]), attention_mask_len)
            if copy_len > 0:
                valid_mask[:, :copy_len] = source_mask[:, :copy_len]
            if attention_mask_len > current_length:
                valid_mask[:, current_length:] = False
        else:
            raise ValueError(f"Unsupported attention_mask shape for MolmoAct2: {tuple(attention_mask.shape)}")

        valid_mask = valid_mask[:, None, None, :]
        causal_mask = torch.tril(
            torch.ones(attention_mask_len, attention_mask_len, device=device, dtype=torch.bool)
        )[None, None, past_length:current_length, :attention_mask_len]

        if token_type_ids is not None and past_length == 0:
            causal_mask = causal_mask.expand(batch_size, -1, -1, -1).clone()
            image_mask = token_type_ids.to(device=device, dtype=torch.bool)
            can_attend_back = image_mask[:, :, None] & image_mask[:, None, :]
            image_len = min(int(token_type_ids.shape[1]), attention_mask_len)
            causal_mask[:, :, :, :image_len] = (
                causal_mask[:, :, :, :image_len] | can_attend_back[:, None, :, :image_len]
            )

        allowed = valid_mask & causal_mask
        return torch.where(
            allowed,
            torch.zeros((), device=device, dtype=inputs_embeds.dtype),
            torch.full(
                (),
                torch.finfo(inputs_embeds.dtype).min,
                device=device,
                dtype=inputs_embeds.dtype,
            ),
        )

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_token_pooling: torch.Tensor | None = None,
        image_grids: torch.Tensor | None = None,
        image_num_crops: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_token_pooling: torch.Tensor | None = None,
        video_grids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        token_type_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MolmoAct2ModelOutputWithPast:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        images, token_pooling = self.merge_visual_inputs(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_token_pooling=image_token_pooling,
            image_grids=image_grids,
            image_num_crops=image_num_crops,
            pixel_values_videos=pixel_values_videos,
            video_token_pooling=video_token_pooling,
            video_grids=video_grids,
        )

        if images is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both images and inputs_embeds at the same time.")

        if inputs_embeds is None:
            inputs_embeds, image_features = self.build_input_embeddings(
                input_ids,
                images,
                token_pooling,
            )

        if cache_position is None:
            past_seen_tokens = _cache_seq_len_int(past_key_values)
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if isinstance(attention_mask, dict):
            causal_mask_mapping = attention_mask
        else:
            causal_mask_mapping = self._build_native_attention_bias(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                past_key_values=past_key_values,
            )

        outputs = self.transformer(
            attention_mask=causal_mask_mapping,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        return MolmoAct2ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if images is not None else None,
        )


class MolmoAct2ForConditionalGeneration(MolmoAct2PreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = []  # Weights are not tied
    # Reference: fix gemma3 grad acc #37208
    accepts_loss_kwargs = False
    config: MolmoAct2Config

    def __init__(self, config: MolmoAct2Config):
        super().__init__(config)

        self.model = MolmoAct2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vocab_size = config.vocab_size
        self.model.action_cuda_graph_manager = ActionCudaGraphManager(self.model)
        self.depth_decode_cuda_graph_manager = DepthDecodeCudaGraphManager(self)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.transformer.wte

    def set_input_embeddings(self, value: torch.nn.Module) -> None:
        self.model.transformer.wte = value

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    # Make modules available through conditional class for BC
    @property
    def language_model(self) -> torch.nn.Module:
        return self.model.transformer

    @property
    def vision_backbone(self) -> torch.nn.Module:
        return self.model.vision_backbone

    def _get_robot_stats(self) -> _RobotStats:
        stats = getattr(self, "_molmoact2_robot_stats", None)
        if stats is not None:
            return stats
        filename = getattr(self.config, "norm_stats_filename", "norm_stats.json")
        base_dir = getattr(self.config, "_name_or_path", None) or getattr(self, "name_or_path", None)
        if not base_dir:
            raise ValueError(
                "MolmoAct2 normalization stats are not loaded and config._name_or_path is empty; "
                "load the model from a converted HF directory containing norm_stats.json."
            )
        stats_path = os.path.join(str(base_dir), filename)
        if not os.path.isfile(stats_path):
            try:
                from huggingface_hub import hf_hub_download

                stats_path = hf_hub_download(str(base_dir), filename, repo_type="model")
            except Exception as exc:
                raise FileNotFoundError(
                    f"MolmoAct2 normalization stats file is missing: {stats_path}. "
                    "Converted checkpoints must include norm_stats.json."
                ) from exc
        with open(stats_path, encoding="utf-8") as f:
            payload = json.load(f)
        stats = _RobotStats(payload)
        self._molmoact2_robot_stats = stats
        return stats

    @staticmethod
    def _move_inputs_to_device(inputs: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
        out = {}
        for key, value in inputs.items():
            out[key] = value.to(device) if torch.is_tensor(value) else value
        return out

    @staticmethod
    def _drop_trivial_attention_mask(inputs: Mapping[str, Any]) -> dict[str, Any]:
        out = dict(inputs)
        attention_mask = out.get("attention_mask")
        if torch.is_tensor(attention_mask) and bool(attention_mask.to(dtype=torch.bool).all().item()):
            out.pop("attention_mask", None)
        return out

    @staticmethod
    def _count_images(images: Any) -> int:
        if images is None:
            return 0
        if isinstance(images, (list, tuple)):
            return len(images)
        arr = np.asarray(images) if not torch.is_tensor(images) else images
        if getattr(arr, "ndim", 0) == 4:
            return int(arr.shape[0])
        return 1

    @staticmethod
    def _build_action_dim_is_pad(
        *,
        action_dim: int,
        max_action_dim: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if int(action_dim) > int(max_action_dim):
            raise ValueError(
                f"Requested action_dim {int(action_dim)} exceeds checkpoint max_action_dim {int(max_action_dim)}."
            )
        if int(action_dim) == int(max_action_dim):
            return None
        mask = torch.ones((int(batch_size), int(max_action_dim)), device=device, dtype=torch.bool)
        mask[:, : int(action_dim)] = False
        return mask

    @staticmethod
    def _slice_action_dim(actions: torch.Tensor, action_dim: int) -> torch.Tensor:
        if actions.shape[-1] < int(action_dim):
            raise ValueError(
                f"Requested action_dim {int(action_dim)} but chunk only has width {actions.shape[-1]}."
            )
        return actions[..., : int(action_dim)]

    @staticmethod
    def _slice_action_chunk(
        actions: torch.Tensor, n_obs_steps: int, n_action_steps: int | None
    ) -> torch.Tensor:
        if n_action_steps is None:
            return actions
        start = int(n_obs_steps) - 1
        end = start + int(n_action_steps)
        if end > actions.shape[1]:
            raise ValueError(f"Requested actions up to {end} but model produced horizon {actions.shape[1]}.")
        return actions[:, start:end]

    def _depth_token_id_to_bin(self) -> dict[int, int]:
        if self.config.depth_token_start_id is None or int(self.config.num_depth_tokens or 0) <= 0:
            return {}
        start = int(self.config.depth_token_start_id)
        return {start + idx: idx for idx in range(int(self.config.num_depth_tokens))}

    def _action_token_id_to_bin(self) -> dict[int, int]:
        if self.config.action_token_start_id is None or int(self.config.num_action_tokens or 0) <= 0:
            return {}
        start = int(self.config.action_token_start_id)
        return {start + idx: idx for idx in range(int(self.config.num_action_tokens))}

    def _require_eos_token_id(self) -> int:
        eos_token_id = getattr(self.config, "eos_token_id", None)
        if eos_token_id is None and getattr(self, "generation_config", None) is not None:
            eos_token_id = getattr(self.generation_config, "eos_token_id", None)
        if isinstance(eos_token_id, (list, tuple)):
            eos_token_id = eos_token_id[0] if eos_token_id else None
        if eos_token_id is None:
            raise RuntimeError(
                "Discrete action generation requires `eos_token_id` in the converted HF config."
            )
        return int(eos_token_id)

    def _decode_depth_bins_from_token_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.config.depth_start_token_id is None or self.config.depth_end_token_id is None:
            raise RuntimeError("Depth generation requires <depth_start>/<depth_end> token IDs.")
        token_id_to_bin = self._depth_token_id_to_bin()
        if not token_id_to_bin:
            raise RuntimeError("Depth generation requires indexed depth tokens in the converted config.")
        depth_token_bins = _extract_discrete_token_bins(
            _flatten_generated_token_ids(token_ids),
            int(self.config.depth_start_token_id),
            int(self.config.depth_end_token_id),
            token_id_to_bin,
        )
        if not depth_token_bins:
            raise RuntimeError("Model generated no decodable depth tokens between <depth_start>/<depth_end>.")
        return torch.as_tensor([depth_token_bins], device=self.device, dtype=torch.long)

    def _consume_generation_tokens(
        self,
        token_ids: torch.Tensor,
        *,
        past_key_values: Cache | None,
        attention_mask: torch.Tensor | None,
    ) -> tuple[MolmoAct2CausalLMOutputWithPast, torch.Tensor | None]:
        if token_ids.ndim == 1:
            next_input_ids = token_ids.unsqueeze(1)
        elif token_ids.ndim == 2:
            next_input_ids = token_ids
        else:
            raise ValueError(f"Expected token_ids to have rank 1 or 2, got {tuple(token_ids.shape)}.")
        next_attention_mask = attention_mask
        if next_attention_mask is not None:
            past_length = _cache_seq_len_int(past_key_values)
            required_len = int(past_length) + int(next_input_ids.shape[1])
            if int(next_attention_mask.shape[-1]) < required_len:
                pad_len = required_len - int(next_attention_mask.shape[-1])
                next_attention_mask = torch.cat(
                    (
                        next_attention_mask,
                        next_attention_mask.new_ones((next_input_ids.shape[0], pad_len)),
                    ),
                    dim=-1,
                )
        past_length = _cache_seq_len_int(past_key_values)
        output = self(
            input_ids=next_input_ids,
            attention_mask=next_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=(
                torch.arange(
                    past_length,
                    past_length + int(next_input_ids.shape[1]),
                    device=next_input_ids.device,
                )
                if past_key_values is not None
                else None
            ),
        )
        return output, next_attention_mask

    def _make_depth_decode_attention_bias(
        self, inputs: Mapping[str, Any], past_key_values: Cache
    ) -> torch.Tensor:
        layers = getattr(past_key_values, "layers", None)
        max_cache_len = int(getattr(layers[0], "max_cache_len", 0)) if layers else 0
        if max_cache_len <= 0:
            raise RuntimeError("Depth decode fast path requires a cache with a fixed maximum length.")
        input_ids = inputs["input_ids"]
        batch_size = int(input_ids.shape[0])
        device = input_ids.device
        dtype = self.lm_head.weight.dtype

        positions = torch.arange(max_cache_len, device=device, dtype=torch.long)
        valid_mask = torch.ones((batch_size, max_cache_len), device=device, dtype=torch.bool)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            source_mask = attention_mask.to(device=device, dtype=torch.bool)
            copy_len = min(int(source_mask.shape[-1]), max_cache_len)
            if copy_len > 0:
                valid_mask[:, :copy_len] = source_mask[:, :copy_len]
        causal_mask = positions[None, :] <= positions[:, None]
        allowed = causal_mask.unsqueeze(0) & valid_mask[:, None, :]
        attention_bias = torch.where(
            allowed[:, None, :, :],
            torch.zeros((), device=device, dtype=dtype),
            torch.full((), torch.finfo(dtype).min, device=device, dtype=dtype),
        )
        return attention_bias

    def _embed_base_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Skips MolmoAct2Embedding's per-call cat([base, new]); safe only for IDs
        # below text_config.vocab_size. This includes released depth/action tokens.
        wte = self.model.transformer.wte
        base_embedding = getattr(wte, "embedding", None)
        if base_embedding is None:
            return wte(input_ids)
        return F.embedding(input_ids, base_embedding)

    def _run_ar_decode_step(
        self,
        token_ids: torch.Tensor,
        *,
        past_key_values: Cache,
        attention_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, Cache]:
        if token_ids.ndim == 1:
            next_input_ids = token_ids.unsqueeze(1)
        elif token_ids.ndim == 2:
            next_input_ids = token_ids
        else:
            raise ValueError(f"Expected token_ids to have rank 1 or 2, got {tuple(token_ids.shape)}.")
        past_length = _cache_seq_len_int(past_key_values)
        end = past_length + int(next_input_ids.shape[1])
        if self.depth_decode_cuda_graph_manager.can_use(
            next_input_ids,
            past_key_values=past_key_values,
            attention_bias=attention_bias,
        ):
            return self.depth_decode_cuda_graph_manager.run(
                next_input_ids,
                past_key_values=past_key_values,
                attention_bias=attention_bias,
                past_length=past_length,
            )
        cache_position = torch.arange(past_length, end, device=next_input_ids.device, dtype=torch.long)
        attention_bias = attention_bias[:, :, past_length:end, :end]
        inputs_embeds = self._embed_base_tokens(next_input_ids)
        outputs = self.model.transformer(
            attention_mask=attention_bias,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            cache_position=cache_position,
        )
        return outputs.last_hidden_state[:, -1:, :], outputs.past_key_values

    def _run_depth_decode_step(
        self,
        token_ids: torch.Tensor,
        *,
        past_key_values: Cache,
        attention_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, Cache]:
        return self._run_ar_decode_step(
            token_ids,
            past_key_values=past_key_values,
            attention_bias=attention_bias,
        )

    def _project_depth_logits(self, last_hidden: torch.Tensor) -> torch.Tensor:
        start = int(self.config.depth_token_start_id)
        end_id = start + int(self.config.num_depth_tokens)
        return F.linear(last_hidden, self.lm_head.weight[start:end_id])

    def _max_depth_decode_steps(self) -> int:
        return max(
            int(self.config.num_depth_codes or 0) + 8,
            self.model._resolve_action_horizon() * 16,
            1,
        )

    def _make_ar_decode_static_cache(self, inputs: Mapping[str, Any], max_steps: int) -> Cache:
        prompt_len = inputs["input_ids"].shape[1]
        return self.depth_decode_cuda_graph_manager.make_static_cache(
            max_cache_len=prompt_len + max(1, int(max_steps)),
        )

    def _make_depth_static_cache(self, inputs: Mapping[str, Any]) -> Cache:
        prompt_len = inputs["input_ids"].shape[1]
        action_horizon = self.model._resolve_action_horizon()
        max_end_steps = max(8, action_horizon)
        action_token_budget = max(1, action_horizon * 16)
        return self.depth_decode_cuda_graph_manager.make_static_cache(
            max_cache_len=prompt_len + self._max_depth_decode_steps() + max_end_steps + action_token_budget,
        )

    def _continue_discrete_generation_from_output(
        self,
        initial_output: MolmoAct2CausalLMOutputWithPast,
        *,
        past_key_values: Cache | None,
        attention_mask: torch.Tensor | None,
        end_token_id: int,
        max_steps: int,
        attention_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        generated_tokens: list[torch.Tensor] = []
        current_output = initial_output
        current_past_key_values = past_key_values
        current_attention_mask = attention_mask
        hit_end = False
        for _ in range(int(max_steps)):
            next_token = torch.argmax(current_output.logits[:, -1, :], dim=-1)
            generated_tokens.append(next_token)
            if bool((next_token == int(end_token_id)).all()):
                hit_end = True
                break
            if attention_bias is None:
                current_output, current_attention_mask = self._consume_generation_tokens(
                    next_token,
                    past_key_values=current_past_key_values,
                    attention_mask=current_attention_mask,
                )
                current_past_key_values = current_output.past_key_values
            else:
                last_hidden, current_past_key_values = self._run_ar_decode_step(
                    next_token,
                    past_key_values=current_past_key_values,
                    attention_bias=attention_bias,
                )
                current_output = MolmoAct2CausalLMOutputWithPast(
                    logits=self.lm_head(last_hidden),
                    past_key_values=current_past_key_values,
                )
        if not generated_tokens:
            raise RuntimeError("Discrete continuation generated no tokens.")
        if not hit_end:
            raise RuntimeError(
                f"Discrete continuation did not emit end token {int(end_token_id)} within {int(max_steps)} steps."
            )
        return torch.stack(generated_tokens, dim=1)

    def _generate_depth_prefix(
        self,
        inputs: Mapping[str, Any],
        *,
        latest_first_image: np.ndarray | None,
        depth_cache: Mapping[str, Any] | None,
        enable_adaptive_depth: bool,
    ) -> _DepthPrefix:
        if self.config.depth_start_token_id is None or self.config.depth_end_token_id is None:
            raise RuntimeError("Depth reasoning requires single-token <depth_start>/<depth_end>.")
        if self.config.depth_token_start_id is None or int(self.config.num_depth_tokens or 0) <= 0:
            raise RuntimeError("Depth reasoning requires indexed depth tokens.")
        batch_size = int(inputs["input_ids"].shape[0])
        if batch_size != 1 and enable_adaptive_depth:
            raise ValueError("enable_adaptive_depth=True currently supports batch size 1.")
        static_cache = self._make_depth_static_cache(inputs)
        output = self(**inputs, use_cache=True, past_key_values=static_cache)
        current_output = output
        current_past_key_values = output.past_key_values
        current_attention_mask = inputs.get("attention_mask")
        generated_tokens: list[torch.Tensor] = []

        if not enable_adaptive_depth:
            hit_depth_end = False
            max_steps = self._max_depth_decode_steps()
            for _ in range(max_steps):
                next_token = torch.argmax(current_output.logits[:, -1, :], dim=-1)
                generated_tokens.append(next_token)
                current_output, current_attention_mask = self._consume_generation_tokens(
                    next_token,
                    past_key_values=current_past_key_values,
                    attention_mask=current_attention_mask,
                )
                current_past_key_values = current_output.past_key_values
                if bool((next_token == int(self.config.depth_end_token_id)).all()):
                    hit_depth_end = True
                    break
            if not generated_tokens:
                raise RuntimeError("Depth generation produced no tokens.")
            if not hit_depth_end:
                raise RuntimeError(f"Depth generation did not emit <depth_end> within {max_steps} steps.")
            depth_token_ids = torch.stack(generated_tokens, dim=1)
            full_input_ids = torch.cat([inputs["input_ids"], depth_token_ids], dim=1)
            full_attention_mask = None
            if current_attention_mask is not None:
                full_attention_mask = current_attention_mask[:, : full_input_ids.shape[1]]
            encoder_kv_states = self.model._extract_kv_states(current_past_key_values)
            return _DepthPrefix(
                token_ids=depth_token_ids,
                depth_bins=self._decode_depth_bins_from_token_ids(depth_token_ids),
                full_input_ids=full_input_ids,
                attention_mask=full_attention_mask,
                encoder_kv_states=encoder_kv_states,
                next_output=current_output,
                past_key_values=current_past_key_values,
            )

        depth_start = torch.full(
            (batch_size,),
            int(self.config.depth_start_token_id),
            device=self.device,
            dtype=torch.long,
        )
        code_token_ids = torch.arange(
            int(self.config.depth_token_start_id),
            int(self.config.depth_token_start_id) + int(self.config.num_depth_tokens),
            device=self.device,
            dtype=torch.long,
        )
        depth_attention_bias = self._make_depth_decode_attention_bias(inputs, current_past_key_values)
        generated_tokens.append(depth_start)
        last_hidden, current_past_key_values = self._run_depth_decode_step(
            depth_start,
            past_key_values=current_past_key_values,
            attention_bias=depth_attention_bias,
        )
        previous_image = None
        previous_bins = None
        if depth_cache is not None:
            previous_image = depth_cache.get("image")
            previous_bins = depth_cache.get("depth_bins")
        selective = (
            bool(enable_adaptive_depth)
            and latest_first_image is not None
            and previous_image is not None
            and previous_bins is not None
        )
        update_mask = None
        previous_buffer_t = None
        if selective:
            previous_buffer = np.asarray(previous_bins, dtype=np.int64).reshape(-1)
            if previous_buffer.shape[0] == int(self.config.num_depth_codes):
                update_mask = _compute_depth_update_mask(
                    latest_first_image,
                    _normalize_image_for_cache(previous_image),
                    num_depth_codes=int(self.config.num_depth_codes),
                )
                previous_buffer_t = (
                    torch.from_numpy(previous_buffer)
                    .to(
                        device=self.device,
                        dtype=torch.long,
                    )
                    .unsqueeze(0)
                )
            else:
                selective = False

        depth_bins = torch.zeros(
            (batch_size, int(self.config.num_depth_codes)),
            device=self.device,
            dtype=torch.long,
        )
        num_depth_codes = int(self.config.num_depth_codes)
        if not selective or update_mask is None or previous_buffer_t is None:
            for depth_idx in range(num_depth_codes):
                depth_logits = self._project_depth_logits(last_hidden)
                predicted_bins = depth_logits.squeeze(1).argmax(dim=-1)
                depth_bins[:, depth_idx] = predicted_bins
                chosen_token_ids = code_token_ids[predicted_bins]
                generated_tokens.append(chosen_token_ids)
                last_hidden, current_past_key_values = self._run_depth_decode_step(
                    chosen_token_ids,
                    past_key_values=current_past_key_values,
                    attention_bias=depth_attention_bias,
                )
        else:
            for start_idx, end_idx, should_generate in _build_depth_update_spans(update_mask):
                if should_generate:
                    for depth_idx in range(start_idx, end_idx):
                        depth_logits = self._project_depth_logits(last_hidden)
                        predicted_bins = depth_logits.squeeze(1).argmax(dim=-1)
                        depth_bins[:, depth_idx] = predicted_bins
                        chosen_token_ids = code_token_ids[predicted_bins]
                        generated_tokens.append(chosen_token_ids)
                        last_hidden, current_past_key_values = self._run_depth_decode_step(
                            chosen_token_ids,
                            past_key_values=current_past_key_values,
                            attention_bias=depth_attention_bias,
                        )
                    continue
                replay_bins = previous_buffer_t[:, start_idx:end_idx].expand(batch_size, -1)
                depth_bins[:, start_idx:end_idx] = replay_bins
                replay_token_ids = code_token_ids[replay_bins]
                generated_tokens.extend(replay_token_ids.unbind(dim=1))
                last_hidden, current_past_key_values = self._run_depth_decode_step(
                    replay_token_ids,
                    past_key_values=current_past_key_values,
                    attention_bias=depth_attention_bias,
                )
        hit_depth_end = False
        max_depth_end_steps = max(8, self.model._resolve_action_horizon())
        full_logits = self.lm_head(last_hidden)
        for _ in range(max_depth_end_steps):
            next_token = full_logits.squeeze(1).argmax(dim=-1)
            generated_tokens.append(next_token)
            last_hidden, current_past_key_values = self._run_depth_decode_step(
                next_token,
                past_key_values=current_past_key_values,
                attention_bias=depth_attention_bias,
            )
            full_logits = self.lm_head(last_hidden)
            if bool((next_token == int(self.config.depth_end_token_id)).all()):
                hit_depth_end = True
                break
        if not hit_depth_end:
            raise RuntimeError(
                f"Depth generation did not emit <depth_end> within {max_depth_end_steps} steps "
                "after adaptive depth tokens."
            )

        depth_token_ids = torch.stack(generated_tokens, dim=1)
        full_input_ids = torch.cat([inputs["input_ids"], depth_token_ids], dim=1)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            full_attention_mask = torch.cat(
                (attention_mask, attention_mask.new_ones(depth_token_ids.shape)),
                dim=-1,
            )[:, : full_input_ids.shape[1]]
        else:
            full_attention_mask = None
        current_output = MolmoAct2CausalLMOutputWithPast(
            logits=full_logits,
            past_key_values=current_past_key_values,
        )
        encoder_kv_states = self.model._extract_kv_states(current_past_key_values)
        return _DepthPrefix(
            token_ids=depth_token_ids,
            depth_bins=depth_bins,
            full_input_ids=full_input_ids,
            attention_mask=full_attention_mask,
            encoder_kv_states=encoder_kv_states,
            next_output=current_output,
            past_key_values=current_past_key_values,
        )

    def _decode_discrete_action_chunk(
        self,
        generated_token_ids: torch.Tensor,
        *,
        action_tokenizer: Any,
        action_dim: int,
        action_horizon: int,
    ) -> torch.Tensor:
        if action_tokenizer is None:
            raise ValueError("inference_action_mode='discrete' requires an `action_tokenizer` input.")
        if self.config.action_start_token_id is None or self.config.action_end_token_id is None:
            raise RuntimeError("Discrete action generation requires <action_start>/<action_end> token IDs.")
        token_id_to_bin = self._action_token_id_to_bin()
        if not token_id_to_bin:
            raise RuntimeError(
                "Discrete action generation requires indexed action tokens in the converted config."
            )
        discrete_token_ids = _extract_discrete_token_bins(
            _flatten_generated_token_ids(generated_token_ids),
            int(self.config.action_start_token_id),
            int(self.config.action_end_token_id),
            token_id_to_bin,
        )
        if not discrete_token_ids:
            raise RuntimeError(
                "Model generated no decodable action tokens between <action_start>/<action_end>."
            )
        try:
            decoded = action_tokenizer.decode(
                [discrete_token_ids],
                time_horizon=int(action_horizon),
                action_dim=int(action_dim),
            )
        except TypeError:
            decoded = action_tokenizer.decode([discrete_token_ids])
        action_chunk = np.asarray(decoded, dtype=np.float32)
        if action_chunk.ndim == 1:
            action_chunk = action_chunk[None, None, :]
        elif action_chunk.ndim == 2:
            action_chunk = action_chunk[None, :, :]
        elif action_chunk.ndim > 3:
            action_chunk = action_chunk.reshape(1, action_chunk.shape[-2], action_chunk.shape[-1])
        if action_chunk.ndim != 3:
            raise RuntimeError(f"Decoded action chunk has unexpected shape {action_chunk.shape}.")
        return torch.as_tensor(action_chunk, device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def predict_action(
        self,
        *,
        processor: Any,
        images: Any,
        task: str,
        state: Any,
        norm_tag: str,
        inference_action_mode: str | None = None,
        enable_depth_reasoning: bool = False,
        enable_adaptive_depth: bool = True,
        depth_cache: Mapping[str, Any] | None = None,
        action_tokenizer: Any = None,
        num_steps: int | None = None,
        n_action_steps: int | None = None,
        generator: torch.Generator | None = None,
        normalize_language: bool = True,
        enable_cuda_graph: bool = True,
        return_dict: bool = True,
    ) -> MolmoAct2ActionOutput | torch.Tensor:
        if state is None:
            raise ValueError("MolmoAct2 `predict_action` requires `state` for discrete state prompting.")
        if inference_action_mode is None:
            raise ValueError(
                "`inference_action_mode` must be provided explicitly as either 'continuous' or 'discrete'."
            )
        inference_action_mode = str(inference_action_mode)
        if inference_action_mode not in {"continuous", "discrete"}:
            raise ValueError("inference_action_mode must be either 'continuous' or 'discrete'.")
        if inference_action_mode == "continuous" and not bool(self.config.add_action_expert):
            raise RuntimeError(
                "inference_action_mode='continuous' requires an action expert, but this checkpoint "
                "was converted with add_action_expert=False."
            )
        if inference_action_mode == "continuous" and self.config.action_mode not in {
            "continuous",
            "both",
        }:
            raise ValueError(
                "inference_action_mode='continuous' requires checkpoint action_mode in "
                f"{{'continuous', 'both'}}, got {self.config.action_mode!r}."
            )
        if inference_action_mode == "discrete":
            if action_tokenizer is None:
                raise ValueError("inference_action_mode='discrete' requires an `action_tokenizer` input.")
            if self.config.action_mode not in {"discrete", "both"}:
                raise ValueError(
                    "inference_action_mode='discrete' requires checkpoint action_mode in "
                    f"{{'discrete', 'both'}}, got {self.config.action_mode!r}."
                )
        if enable_depth_reasoning and not bool(self.config.enable_depth_reasoning):
            raise ValueError("this model was not trained with `--enable_depth_reasoning`.")

        stats = self._get_robot_stats()
        norm_tag = stats.validate_tag(norm_tag)
        metadata = stats.get_metadata(norm_tag)
        normalized_state = np.asarray(stats.normalize_state(state, norm_tag), dtype=np.float32)
        num_state_tokens = int(self.config.num_state_tokens or 0)
        if num_state_tokens <= 0:
            raise RuntimeError(
                "Discrete state prompting requires indexed state tokens in the converted config."
            )
        discrete_state_string = _build_discrete_state_string(normalized_state, num_state_tokens)
        style = "robot_depth_action" if enable_depth_reasoning else "robot_action"
        task_text = str(task or "")
        if normalize_language:
            task_text = _normalize_question_text(task_text)
        text = _build_robot_text(
            task=task_text,
            style=style,
            discrete_state_string=discrete_state_string,
            setup_type=str(metadata.get("setup_type", "") or ""),
            control_mode=str(metadata.get("control_mode", "") or ""),
            add_setup_tokens=bool(self.config.add_setup_tokens),
            add_control_tokens=bool(self.config.add_control_tokens),
            num_images=self._count_images(images),
        )
        inputs = processor(text=text, images=images, return_tensors="pt")
        inputs = self._move_inputs_to_device(inputs, self.device)
        inputs = self._drop_trivial_attention_mask(inputs)

        action_dim = stats.get_action_dim(norm_tag)
        if action_dim is None:
            action_dim = int(self.config.max_action_dim)
        action_dim = int(action_dim)
        max_action_horizon = self.model._resolve_action_horizon()
        action_horizon = stats.get_action_horizon(norm_tag) or max_action_horizon
        if int(action_horizon) > max_action_horizon:
            raise ValueError(
                f"Tag action_horizon={int(action_horizon)} exceeds checkpoint max_action_horizon={max_action_horizon}."
            )
        generation_horizon = int(action_horizon)
        resolved_n_action_steps = n_action_steps
        if resolved_n_action_steps is None:
            resolved_n_action_steps = stats.get_n_action_steps(norm_tag)
        if resolved_n_action_steps is None:
            resolved_n_action_steps = int(action_horizon)
        resolved_n_action_steps = int(resolved_n_action_steps)
        if resolved_n_action_steps < 1:
            raise ValueError(f"n_action_steps must be >= 1, got {resolved_n_action_steps}.")
        if resolved_n_action_steps > int(action_horizon):
            raise ValueError(
                f"Requested n_action_steps={resolved_n_action_steps} exceeds tag action_horizon={int(action_horizon)}."
            )
        batch_size = int(inputs["input_ids"].shape[0])
        action_dim_is_pad = self._build_action_dim_is_pad(
            action_dim=action_dim,
            max_action_dim=int(self.config.max_action_dim),
            batch_size=batch_size,
            device=self.device,
        )
        self.model.action_cuda_graph_manager.set_enabled(enable_cuda_graph)
        self.depth_decode_cuda_graph_manager.set_enabled(enable_cuda_graph)

        generated_token_ids = None
        depth_bins = None
        updated_depth_cache = depth_cache
        if inference_action_mode == "continuous":
            if enable_depth_reasoning:
                latest_first_image = _extract_first_image(images)
                depth_prefix = self._generate_depth_prefix(
                    inputs,
                    latest_first_image=latest_first_image,
                    depth_cache=depth_cache,
                    enable_adaptive_depth=bool(enable_adaptive_depth),
                )
                generated_token_ids = depth_prefix.token_ids
                depth_bins = depth_prefix.depth_bins
                actions = self.model.generate_actions_from_inputs(
                    input_ids=depth_prefix.full_input_ids,
                    attention_mask=depth_prefix.attention_mask,
                    action_dim_is_pad=action_dim_is_pad,
                    action_horizon=generation_horizon,
                    num_steps=num_steps,
                    generator=generator,
                    encoder_kv_states=depth_prefix.encoder_kv_states,
                    encoder_attention_mask=self.model._get_encoder_attention_mask(
                        depth_prefix.full_input_ids,
                        depth_prefix.attention_mask,
                    ),
                )
                if latest_first_image is not None:
                    updated_depth_cache = {
                        "image": latest_first_image,
                        "depth_bins": depth_bins.detach().cpu().reshape(-1).numpy().astype(np.int64),
                    }
            else:
                actions = self.model.generate_actions_from_inputs(
                    **inputs,
                    action_dim_is_pad=action_dim_is_pad,
                    action_horizon=generation_horizon,
                    num_steps=num_steps,
                    generator=generator,
                )
        else:
            if enable_depth_reasoning:
                latest_first_image = _extract_first_image(images)
                depth_prefix = self._generate_depth_prefix(
                    inputs,
                    latest_first_image=latest_first_image,
                    depth_cache=depth_cache,
                    enable_adaptive_depth=bool(enable_adaptive_depth),
                )
                action_token_ids = self._continue_discrete_generation_from_output(
                    depth_prefix.next_output,
                    past_key_values=depth_prefix.past_key_values,
                    attention_mask=depth_prefix.attention_mask,
                    end_token_id=self._require_eos_token_id(),
                    max_steps=max(1, int(generation_horizon * 16)),
                )
                generated_token_ids = torch.cat([depth_prefix.token_ids, action_token_ids], dim=1)
                depth_bins = depth_prefix.depth_bins
                if latest_first_image is not None:
                    updated_depth_cache = {
                        "image": latest_first_image,
                        "depth_bins": depth_bins.detach().cpu().reshape(-1).numpy().astype(np.int64),
                    }
            else:
                max_action_decode_steps = max(1, int(generation_horizon * 16))
                action_attention_bias = None
                if enable_cuda_graph:
                    action_static_cache = self._make_ar_decode_static_cache(
                        inputs,
                        max_steps=max_action_decode_steps,
                    )
                    action_attention_bias = self._make_depth_decode_attention_bias(
                        inputs,
                        action_static_cache,
                    )
                    prefill_output = self(
                        **inputs,
                        use_cache=True,
                        past_key_values=action_static_cache,
                    )
                else:
                    prefill_output = self(**inputs, use_cache=True)
                action_token_ids = self._continue_discrete_generation_from_output(
                    prefill_output,
                    past_key_values=prefill_output.past_key_values,
                    attention_mask=inputs.get("attention_mask"),
                    end_token_id=self._require_eos_token_id(),
                    max_steps=max_action_decode_steps,
                    attention_bias=action_attention_bias,
                )
                generated_token_ids = action_token_ids
            actions = self._decode_discrete_action_chunk(
                generated_token_ids,
                action_tokenizer=action_tokenizer,
                action_dim=action_dim,
                action_horizon=generation_horizon,
            )

        actions = self._slice_action_dim(actions, action_dim)
        actions = self._slice_action_chunk(actions, int(self.config.n_obs_steps), resolved_n_action_steps)
        actions = stats.unnormalize_action(actions, norm_tag)
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        else:
            actions = actions.to(device=self.device, dtype=torch.float32)
        output = MolmoAct2ActionOutput(
            actions=actions,
            generated_token_ids=generated_token_ids,
            depth_bins=depth_bins,
            depth_cache=updated_depth_cache,
        )
        if return_dict:
            return output
        return actions

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.Tensor | None = None,
        image_token_pooling: torch.Tensor | None = None,
        image_grids: torch.Tensor | None = None,
        image_num_crops: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_token_pooling: torch.Tensor | None = None,
        video_grids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        token_type_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | MolmoAct2CausalLMOutputWithPast:
        r"""
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from lerobot.policies.molmoact2.molmoact2_hf_model.modeling_molmoact2 import MolmoAct2ForConditionalGeneration
        >>> from lerobot.policies.molmoact2.processor_molmoact2 import _load_local_molmoact2_processor

        >>> model = MolmoAct2ForConditionalGeneration.from_pretrained("...")
        >>> processor = _load_local_molmoact2_processor("...")

        >>> prompt = "What's the content of the image?"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "image": image}]}]

        >>> inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)

        >>> # Generate
        >>> generated_ids = model.generate(**inputs, max_new_tokens=15)
        >>> generated_tokens = generated_ids[:, inputs['input_ids'].size(1):]
        >>> processor.post_process_image_text_to_text(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a bustling street scene in what appears to be a Chinatown area. There's ..."
        ```"""
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_token_pooling=image_token_pooling,
            image_grids=image_grids,
            image_num_crops=image_num_crops,
            pixel_values_videos=pixel_values_videos,
            video_token_pooling=video_token_pooling,
            video_grids=video_grids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.vocab_size)

        return MolmoAct2CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        image_token_pooling: torch.Tensor | None = None,
        image_grids: torch.Tensor | None = None,
        image_num_crops: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_token_pooling: torch.Tensor | None = None,
        video_grids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor | None = None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            token_type_ids=token_type_ids,
            **kwargs,
        )

        include_visual_inputs = past_key_values is None
        if past_key_values is not None and hasattr(past_key_values, "get_seq_length"):
            include_visual_inputs = int(past_key_values.get_seq_length()) == 0
        if include_visual_inputs:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_token_pooling"] = image_token_pooling
            model_inputs["image_grids"] = image_grids
            model_inputs["image_num_crops"] = image_num_crops
            model_inputs["pixel_values_videos"] = pixel_values_videos
            model_inputs["video_token_pooling"] = video_token_pooling
            model_inputs["video_grids"] = video_grids

        return model_inputs

    # Adapted from transformers.models.gemma3.modeling_gemma3
    @staticmethod
    def create_masks_for_generate(
        config: PretrainedConfig,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cache_position: torch.Tensor,
        past_key_values: Cache | None,
        position_ids: torch.Tensor | None,
        token_type_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        # Prepare mask arguments
        mask_kwargs = {
            "config": config.get_text_config(),
            "input_embeds": input_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        # Add the token type ids mask for generate as well
        if token_type_ids is not None and input_embeds.shape[1] != 1:
            # We need to pass an additional mask function to account for token type ids, and it needs to be an `or`
            mask_kwargs["or_mask_function"] = token_type_ids_mask_function(
                token_type_ids.to(cache_position.device)
            )

        return create_masks_for_generate(**mask_kwargs)
