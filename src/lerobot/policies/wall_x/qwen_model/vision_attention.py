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

"""Wall-X vision attention backends.

Qwen2.5-VL's native non-Flash vision path splits a packed image sequence into
Python-level chunks before calling attention. Wall-X batches many camera frames,
so that path launches thousands of tiny attention operations per training step.
This module keeps the native SDPA path as a portable fallback and adds a packed
``torch.nn.attention.varlen`` path that consumes Qwen's existing ``cu_seqlens``
metadata directly.
"""

from __future__ import annotations

import inspect
import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn

from lerobot.utils.import_utils import _transformers_available

if TYPE_CHECKING or _transformers_available:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLVisionAttention,
        apply_rotary_pos_emb_vision,
    )
else:
    Qwen2_5_VLVisionAttention = nn.Module
    apply_rotary_pos_emb_vision = None

try:
    from torch.nn.attention.varlen import varlen_attn as _varlen_attn
except ImportError:  # torch<2.10
    _varlen_attn = None

_VARLEN_USES_WINDOW_SIZE = (
    _varlen_attn is not None and "window_size" in inspect.signature(_varlen_attn).parameters
)


VisionAttentionBackend = Literal["auto", "sdpa", "varlen"]

logger = logging.getLogger(__name__)


@lru_cache
def _log_resolved_backend(requested: str, resolved: str) -> None:
    logger.info("Wall-X vision attention backend: %s (requested: %s)", resolved, requested)


def _varlen_unavailable_reason(
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
) -> str | None:
    if _varlen_attn is None:
        return "torch.nn.attention.varlen is unavailable (PyTorch 2.10 or newer is required)"
    if position_embeddings is None:
        return "precomputed vision position embeddings were not provided"
    if hidden_states.device.type != "cuda" or torch.version.cuda is None:
        return "packed varlen attention requires an NVIDIA CUDA device"
    if hidden_states.dtype not in {torch.float16, torch.bfloat16}:
        return f"packed varlen attention requires float16 or bfloat16 inputs, got {hidden_states.dtype}"
    major, _minor = torch.cuda.get_device_capability(hidden_states.device)
    if major < 8:
        return "packed varlen attention requires an NVIDIA Ampere GPU or newer"
    return None


def _supports_varlen_attention(
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
) -> bool:
    return _varlen_unavailable_reason(hidden_states, position_embeddings) is None


class WallXVisionAttention(Qwen2_5_VLVisionAttention):
    """Qwen2.5-VL vision attention with packed varlen and native SDPA fallback."""

    def __init__(self, config, backend: VisionAttentionBackend):
        super().__init__(config)
        self.wallx_backend = backend
        self._resolved_backend_key = None
        self._resolved_backend = None

    def _resolve_backend(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> str:
        key = (
            hidden_states.device.type,
            hidden_states.device.index,
            hidden_states.dtype,
            position_embeddings is not None,
        )
        if self._resolved_backend_key == key:
            return self._resolved_backend

        use_varlen = self.wallx_backend != "sdpa" and _supports_varlen_attention(
            hidden_states, position_embeddings
        )
        if self.wallx_backend == "varlen" and not use_varlen:
            reason = _varlen_unavailable_reason(hidden_states, position_embeddings)
            raise RuntimeError(f"Wall-X vision_attn_implementation='varlen' cannot be used: {reason}")

        resolved_backend = "varlen" if use_varlen else "sdpa"
        self._resolved_backend_key = key
        self._resolved_backend = resolved_backend
        _log_resolved_backend(self.wallx_backend, resolved_backend)
        return resolved_backend

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del rotary_pos_emb

        if self._resolve_backend(hidden_states, position_embeddings) == "sdpa":
            return super().forward(
                hidden_states=hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(
            query_states,
            key_states,
            cos,
            sin,
        )

        if cu_seqlens.dtype != torch.int32:
            cu_seqlens = cu_seqlens.to(dtype=torch.int32)
        max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
        varlen_kwargs = {"scale": self.scaling}
        if _VARLEN_USES_WINDOW_SIZE:
            varlen_kwargs["window_size"] = (-1, -1)
        else:  # Stable PyTorch 2.10 API; pre-release variants used window_size.
            varlen_kwargs["is_causal"] = False
        attn_output = _varlen_attn(
            query_states,
            key_states,
            value_states,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            **varlen_kwargs,
        )
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)


def configure_wall_x_vision_attention(
    vision_model: nn.Module,
    backend: VisionAttentionBackend,
) -> None:
    """Install Wall-X's scoped packed attention without changing checkpoint keys."""
    if backend == "sdpa":
        _log_resolved_backend(backend, "sdpa")
        return
    if backend == "varlen" and _varlen_attn is None:
        raise RuntimeError(
            "Wall-X vision_attn_implementation='varlen' requires torch.nn.attention.varlen "
            "from PyTorch 2.10 or newer"
        )
    if backend == "auto" and _varlen_attn is None:
        _log_resolved_backend(backend, "sdpa")
        return

    for block in vision_model.blocks:
        previous_attention = block.attn
        replacement = WallXVisionAttention(previous_attention.config, backend=backend)
        replacement.to(
            device=previous_attention.qkv.weight.device,
            dtype=previous_attention.qkv.weight.dtype,
        )
        replacement.load_state_dict(previous_attention.state_dict(), strict=True)
        replacement.train(previous_attention.training)
        block.attn = replacement
