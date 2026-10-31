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

"""IMPRINT memory math, in PyTorch.

The equations follow the Apache-2.0 reference at
https://github.com/amazon-far/imprint (arXiv:2610.14406). This file is a
bounded reimplementation, not a JAX port and not a checkpoint loader.

Source map:

- ``CosmoBottleneckMemory``: :class:`FullAttentionMemory`
- language FiLM ``q * (1 + gamma) + beta``, skipped when language is missing:
  :func:`language_condition_queries`
- ``GatedDeltaBottleneckMemory`` event order
  ``action_proprio_then_finalized_visual_v1``: :func:`chunk_type_ids`
- ``_delta_recurrent_step``: :func:`gated_delta_step`
- ``update_incremental`` carried state equals one full event scan:
  :meth:`RecurrentMemory.forward_sequence` and :meth:`RecurrentMemory.update_incremental`
- stage-1 heads and weights 1, 0.5, 0.5, 0.5: :class:`ImprintDecoder`,
  :func:`memory_pretraining_loss`
- action-expert cross-attention plus FiLM, identity when disabled:
  :class:`MemoryModulation`
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.configs.types import PolicyFeature

from .configuration_imprint import ImprintConfig

# Stage-1 loss weights from the paper training recipe.
VISUAL_LOSS_WEIGHT = 1.0
ACTION_LOSS_WEIGHT = 0.5
PROPRIO_LOSS_WEIGHT = 0.5
PROGRESS_LOSS_WEIGHT = 0.5

ARCHITECTURE_NAME = "gated_delta_hybrid_v1"
EVENT_ORDER = "action_proprio_then_finalized_visual_v1"


def _vector_width(feature: PolicyFeature | None, name: str) -> int:
    if feature is None:
        raise ValueError(f"IMPRINT memory requires {name}.")
    return int(feature.shape[0])


def chunk_type_ids(chunk_frames: int, tokens_per_chunk: int) -> Tensor:
    """Event types inside one finalized chunk: action, proprio, then visual tokens."""
    action_proprio = torch.tensor([0, 1], dtype=torch.long).repeat(chunk_frames)
    visual = torch.full((tokens_per_chunk,), 2, dtype=torch.long)
    return torch.cat([action_proprio, visual], dim=0)


def language_condition_queries(queries: Tensor, lang: Tensor | None, film: nn.Linear) -> Tensor:
    """Apply language FiLM. Missing language leaves the queries unchanged."""
    if lang is None:
        return queries
    gamma, beta = film(lang).chunk(2, dim=-1)
    return queries * (1.0 + gamma[:, None, :]) + beta[:, None, :]


def l2_normalize(value: Tensor, eps: float = 1e-6) -> Tensor:
    square_sum = value.float().square().sum(dim=-1, keepdim=True)
    return value.float() * torch.rsqrt(square_sum + eps)


def gated_delta_step(
    state: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    beta: Tensor,
    decay: Tensor,
    valid: Tensor,
) -> tuple[Tensor, Tensor]:
    """One event of the gated delta rule.

    ``state`` is ``(B, heads, head_dim, head_dim)`` stored as ``(v, k)``.
    ``valid`` is ``(B,)`` for this event, not one bit per head. Invalid rows
    keep the previous state and write a zero output.
    """
    retained = state * decay[:, :, None, None]
    prediction = torch.einsum("bhvk,bhk->bhv", retained, key)
    error = value - prediction
    candidate = retained + beta[:, :, None, None] * torch.einsum("bhv,bhk->bhvk", error, key)
    next_state = torch.where(valid[:, None, None, None], candidate, state)
    output = torch.einsum("bhvk,bhk->bhv", next_state, query)
    output = torch.where(valid[:, None, None], output, torch.zeros_like(output))
    return next_state, output


def causal_action_stream(executed: Tensor) -> Tensor:
    """Shift actions so index ``t`` stores the action executed at ``t - 1``.

    The first step is zero. ``executed[:, t]`` is the action commanded from frame ``t``.
    """
    shifted = torch.zeros_like(executed)
    if executed.shape[1] > 1:
        shifted[:, 1:] = executed[:, :-1]
    return shifted


class _MlpTokenizer(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden)
        self.l2 = nn.Linear(hidden, out_dim)
        nn.init.normal_(self.l1.weight, std=0.02)
        nn.init.normal_(self.l2.weight, std=0.02)
        nn.init.zeros_(self.l1.bias)
        nn.init.zeros_(self.l2.bias)

    def forward(self, features: Tensor) -> Tensor:
        return self.l2(F.gelu(self.l1(features)))


class _CrossBlock(nn.Module):
    """Perceiver block: cross-attend, self-attend, then a tanh residual."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_s = nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn = nn.Linear(dim, dim)

    def forward(self, queries: Tensor, key_values: Tensor, valid: Tensor) -> Tensor:
        attended, _ = self.cross(
            self.norm_q(queries),
            self.norm_kv(key_values),
            self.norm_kv(key_values),
            key_padding_mask=~valid,
            need_weights=False,
        )
        queries = queries + attended
        normalized = self.norm_s(queries)
        attended, _ = self.self_attn(normalized, normalized, normalized, need_weights=False)
        return queries + attended + torch.tanh(self.ffn(queries + attended))


class FullAttentionMemory(nn.Module):
    """Learned queries attend to chunk, recent, action, and proprio streams."""

    def __init__(self, config: ImprintConfig) -> None:
        super().__init__()
        dim = config.memory_dim
        self.num_latents = config.num_latents
        self.proj_chunk = nn.Linear(config.chunk_feature_dim, dim)
        self.proj_recent = nn.Linear(config.recent_feature_dim, dim)
        self.tok_action = _MlpTokenizer(
            _vector_width(config.action_feature, "action"), config.tokenizer_hidden, dim
        )
        self.tok_proprio = _MlpTokenizer(
            _vector_width(config.robot_state_feature, "observation.state"), config.tokenizer_hidden, dim
        )
        self.lang_film = nn.Linear(config.lang_dim, 2 * dim)
        nn.init.normal_(self.lang_film.weight, std=0.02)
        nn.init.zeros_(self.lang_film.bias)
        self.latents = nn.Parameter(torch.randn(config.num_latents, dim) * 0.02)
        self.blocks = nn.ModuleList(
            [_CrossBlock(dim, config.num_heads) for _ in range(config.num_cross_blocks)]
        )

    def forward(
        self,
        chunk: Tensor,
        recent: Tensor,
        action: Tensor,
        proprio: Tensor,
        lang: Tensor | None,
        chunk_valid: Tensor | None = None,
        recent_valid: Tensor | None = None,
        action_valid: Tensor | None = None,
        proprio_valid: Tensor | None = None,
    ) -> Tensor:
        parts: list[Tensor] = []
        valids: list[Tensor] = []
        self._append_stream(parts, valids, self.proj_chunk(chunk), chunk_valid)
        self._append_stream(parts, valids, self.proj_recent(recent), recent_valid)
        self._append_stream(parts, valids, self.tok_action(action), action_valid)
        self._append_stream(parts, valids, self.tok_proprio(proprio), proprio_valid)
        if not parts:
            raise ValueError("Full-attention IMPRINT memory received no history tokens.")
        key_values = torch.cat(parts, dim=1)
        valid = torch.cat(valids, dim=1)
        if not bool(valid.any()):
            raise ValueError("Full-attention IMPRINT memory has no valid history tokens.")
        queries = self.latents.unsqueeze(0).expand(key_values.shape[0], -1, -1)
        queries = language_condition_queries(queries, lang, self.lang_film)
        for block in self.blocks:
            queries = block(queries, key_values, valid)
        return queries

    @staticmethod
    def _append_stream(
        parts: list[Tensor],
        valids: list[Tensor],
        tokens: Tensor,
        valid: Tensor | None,
    ) -> None:
        if tokens.shape[1] == 0:
            return
        if valid is None:
            valid = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        parts.append(tokens)
        valids.append(valid)


class _CausalDepthwiseConv(nn.Module):
    def __init__(self, dim: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        kernel = torch.zeros(kernel_size, dim)
        kernel[-1] = 1.0
        self.kernel = nn.Parameter(kernel)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, features: Tensor, cache: Tensor | None) -> tuple[Tensor, Tensor]:
        batch, length, dim = features.shape
        cache_length = self.kernel_size - 1
        if cache is None:
            cache = features.new_zeros(batch, cache_length, dim)
        if cache_length == 0:
            mixed = features * self.kernel[0] + self.bias
            return mixed, cache
        joined = torch.cat([cache.to(dtype=features.dtype), features], dim=1)
        mixed = features.new_zeros(batch, length, dim)
        for offset in range(self.kernel_size):
            mixed = mixed + joined[:, offset : offset + length] * self.kernel[offset]
        return mixed + self.bias, joined[:, -cache_length:]


class _GatedDeltaBlock(nn.Module):
    def __init__(self, config: ImprintConfig) -> None:
        super().__init__()
        dim = config.memory_dim
        self.dim = dim
        self.num_heads = config.delta_num_heads
        self.head_dim = config.delta_head_dim
        self.decay_bias = config.delta_decay_bias
        self.beta_bias = config.delta_beta_bias
        self.norm_in = nn.LayerNorm(dim)
        self.conv = _CausalDepthwiseConv(dim, config.delta_conv_kernel)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.beta_proj = nn.Linear(dim, self.num_heads)
        self.decay_proj = nn.Linear(dim, self.num_heads)
        self.out_gate = nn.Linear(dim, dim)
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn_in = nn.Linear(dim, config.delta_ffn_inner)
        self.ffn_out = nn.Linear(config.delta_ffn_inner, dim)
        for linear in (
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.beta_proj,
            self.decay_proj,
            self.out_gate,
            self.out_proj,
            self.ffn_in,
            self.ffn_out,
        ):
            nn.init.normal_(linear.weight, std=0.02)
            nn.init.zeros_(linear.bias)

    def forward(
        self,
        features: Tensor,
        valid: Tensor,
        matrix: Tensor,
        conv_cache: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, length, _ = features.shape
        mixed, next_cache = self.conv(self.norm_in(features), conv_cache)
        mixed = F.silu(mixed)

        def split_heads(value: Tensor) -> Tensor:
            return value.reshape(batch, length, self.num_heads, self.head_dim)

        query = l2_normalize(split_heads(self.q_proj(mixed)))
        key = l2_normalize(split_heads(self.k_proj(mixed)))
        value = split_heads(self.v_proj(mixed)).float()
        beta = torch.sigmoid(self.beta_proj(mixed).float() + self.beta_bias)
        decay = torch.exp(-F.softplus(self.decay_proj(mixed).float() + self.decay_bias))

        outputs: list[Tensor] = []
        for step in range(length):
            matrix, output = gated_delta_step(
                matrix,
                query[:, step],
                key[:, step],
                value[:, step],
                beta[:, step],
                decay[:, step],
                valid[:, step],
            )
            outputs.append(output)
        stacked = torch.stack(outputs, dim=1).reshape(batch, length, self.dim).to(dtype=features.dtype)
        gated = self.out_norm(stacked) * torch.sigmoid(self.out_gate(mixed))
        hidden = features + self.out_proj(gated)
        hidden = hidden + self.ffn_out(F.gelu(self.ffn_in(self.ffn_norm(hidden))))
        hidden = torch.where(valid.unsqueeze(-1), hidden, torch.zeros_like(hidden))
        return hidden, matrix, next_cache


class _RecentFusionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.cross_q_norm = nn.LayerNorm(dim)
        self.cross_kv_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn_in = nn.Linear(dim, 2 * dim)
        self.ffn_out = nn.Linear(2 * dim, dim)

    def forward(self, queries: Tensor, key_values: Tensor, valid: Tensor) -> Tensor:
        attended, _ = self.cross_attn(
            self.cross_q_norm(queries),
            self.cross_kv_norm(key_values),
            self.cross_kv_norm(key_values),
            key_padding_mask=~valid,
            need_weights=False,
        )
        queries = queries + attended
        normalized = self.self_norm(queries)
        attended, _ = self.self_attn(normalized, normalized, normalized, need_weights=False)
        queries = queries + attended
        return queries + self.ffn_out(F.gelu(self.ffn_in(self.ffn_norm(queries))))


class RecurrentState:
    """Runtime gated-delta state. It is not part of the checkpoint."""

    def __init__(self, matrices: Tensor, conv_caches: Tensor, chunk_count: Tensor) -> None:
        self.matrices = matrices
        self.conv_caches = conv_caches
        self.chunk_count = chunk_count


class RecurrentMemory(nn.Module):
    """Gated-delta long-term memory plus an exact recent-window read."""

    architecture_name = ARCHITECTURE_NAME

    def __init__(self, config: ImprintConfig, tokens_per_chunk: int) -> None:
        super().__init__()
        if tokens_per_chunk < 1:
            raise ValueError(f"tokens_per_chunk must be positive, got {tokens_per_chunk}.")
        self.config = config
        self.chunk_frames = config.chunk_frames
        self.tokens_per_chunk = tokens_per_chunk
        self.events_per_chunk = 2 * config.chunk_frames + tokens_per_chunk
        self.max_chunks = config.max_chunks
        dim = config.memory_dim
        action_dim = _vector_width(config.action_feature, "action")
        proprio_dim = _vector_width(config.robot_state_feature, "observation.state")
        self.proj_chunk = nn.Linear(config.chunk_feature_dim, dim)
        self.proj_recent = nn.Linear(config.recent_feature_dim, dim)
        self.tok_action = _MlpTokenizer(action_dim, config.tokenizer_hidden, dim)
        self.tok_proprio = _MlpTokenizer(proprio_dim, config.tokenizer_hidden, dim)
        self.lang_film = nn.Linear(config.lang_dim, 2 * dim)
        nn.init.normal_(self.lang_film.weight, std=0.02)
        nn.init.zeros_(self.lang_film.bias)
        self.latents = nn.Parameter(torch.randn(config.num_latents, dim) * 0.02)
        self.event_type = nn.Parameter(torch.randn(3, dim) * 0.02)
        self.event_slot = nn.Parameter(torch.randn(self.events_per_chunk, dim) * 0.02)
        self.time_proj = nn.Linear(1, dim)
        self.blocks = nn.ModuleList([_GatedDeltaBlock(config) for _ in range(config.delta_num_layers)])
        self.read_norm = nn.LayerNorm(dim)
        self.read_q = nn.Linear(dim, dim)
        self.read_out = nn.Linear(dim, dim)
        self.recent_null = nn.Parameter(torch.zeros(1, dim))
        self.recent_blocks = nn.ModuleList(
            [_RecentFusionBlock(dim, config.num_heads) for _ in range(config.recent_fusion_blocks)]
        )
        type_ids = chunk_type_ids(config.chunk_frames, tokens_per_chunk)
        self.register_buffer("type_ids", type_ids, persistent=False)

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> RecurrentState:
        head = self.config.delta_head_dim
        heads = self.config.delta_num_heads
        matrices = torch.zeros(
            batch_size,
            self.config.delta_num_layers,
            heads,
            head,
            head,
            dtype=torch.float32,
            device=device,
        )
        cache_length = self.config.delta_conv_kernel - 1
        caches = torch.zeros(
            batch_size,
            self.config.delta_num_layers,
            cache_length,
            self.config.memory_dim,
            dtype=dtype,
            device=device,
        )
        count = torch.zeros(batch_size, dtype=torch.long, device=device)
        return RecurrentState(matrices, caches, count)

    def forward_sequence(
        self,
        chunk_visual: Tensor,
        chunk_action: Tensor,
        chunk_proprio: Tensor,
        recent_visual: Tensor,
        recent_action: Tensor,
        recent_proprio: Tensor,
        lang: Tensor | None,
    ) -> tuple[Tensor, RecurrentState]:
        """Scan every finalized chunk, then read with the recent tail.

        ``chunk_visual`` is ``(B, chunks, tokens, dim)``. Action and proprio chunks
        are ``(B, chunks, frames, dim)``. Recent tensors may have length 0.
        """
        batch = chunk_visual.shape[0]
        state = self.init_state(batch, chunk_visual.device, chunk_visual.dtype)
        if chunk_visual.shape[1] > 0:
            events, valid = self._decorate_all(chunk_visual, chunk_action, chunk_proprio)
            _, matrices, caches = self._scan(events, valid, state)
            count = torch.full(
                (batch,),
                chunk_visual.shape[1],
                dtype=torch.long,
                device=chunk_visual.device,
            )
            state = RecurrentState(matrices, caches, count)
        tokens = self.read(state, recent_visual, recent_action, recent_proprio, lang)
        return tokens, state

    def update_incremental(
        self,
        state: RecurrentState,
        chunk_visual: Tensor,
        chunk_action: Tensor,
        chunk_proprio: Tensor,
        chunk_valid: Tensor | None = None,
    ) -> RecurrentState:
        """Fold one finalized chunk into ``state``.

        ``chunk_visual`` is ``(B, tokens, dim)``. Actions and proprio are
        ``(B, frames, dim)``.
        """
        self._check_incremental_shapes(chunk_visual, chunk_action, chunk_proprio)
        events, valid = self._events_from_chunks(
            chunk_visual.unsqueeze(1),
            chunk_action.unsqueeze(1),
            chunk_proprio.unsqueeze(1),
            state.chunk_count,
        )
        if chunk_valid is not None:
            token_valid = chunk_valid
        else:
            token_valid = torch.ones(chunk_visual.shape[:2], dtype=torch.bool, device=chunk_visual.device)
        _, matrices, caches = self._scan(events, valid, state)
        accepted = token_valid.any(dim=1)
        count = state.chunk_count + accepted.to(dtype=torch.long)
        if int(count.max()) > self.max_chunks:
            raise ValueError(
                f"IMPRINT recurrent memory exceeded max_chunks={self.max_chunks}. "
                "Raise max_chunks or reset the episode."
            )
        return RecurrentState(matrices, caches, count)

    def read(
        self,
        state: RecurrentState,
        recent_visual: Tensor,
        recent_action: Tensor,
        recent_proprio: Tensor,
        lang: Tensor | None,
        recent_visual_valid: Tensor | None = None,
        recent_action_valid: Tensor | None = None,
        recent_proprio_valid: Tensor | None = None,
    ) -> Tensor:
        queries = self._language_queries(state.matrices.shape[0], state.matrices.device, lang)
        queries = self._read_matrix(queries, state.matrices[:, -1])
        return self._fuse_recent(
            queries,
            recent_visual,
            recent_action,
            recent_proprio,
            recent_visual_valid,
            recent_action_valid,
            recent_proprio_valid,
        )

    def _decorate_all(
        self, chunk_visual: Tensor, chunk_action: Tensor, chunk_proprio: Tensor
    ) -> tuple[Tensor, Tensor]:
        batch, chunks, _, _ = chunk_visual.shape
        positions = torch.arange(chunks, device=chunk_visual.device).unsqueeze(0).expand(batch, -1)
        return self._events_from_chunks(chunk_visual, chunk_action, chunk_proprio, positions)

    def _events_from_chunks(
        self,
        chunk_visual: Tensor,
        chunk_action: Tensor,
        chunk_proprio: Tensor,
        positions: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if positions.ndim == 1:
            positions = positions.unsqueeze(1)
        if chunk_visual.shape[2] != self.tokens_per_chunk:
            raise ValueError(
                f"Expected {self.tokens_per_chunk} tokens per chunk, got {chunk_visual.shape[2]}."
            )
        if chunk_action.shape[2] != self.chunk_frames or chunk_proprio.shape[2] != self.chunk_frames:
            raise ValueError(
                f"Expected {self.chunk_frames} action and proprio frames per chunk, "
                f"got {chunk_action.shape[2]} and {chunk_proprio.shape[2]}."
            )
        if int(positions.max()) >= self.max_chunks:
            raise ValueError(f"Chunk index {int(positions.max())} is outside max_chunks={self.max_chunks}.")
        action = self.tok_action(chunk_action)
        proprio = self.tok_proprio(chunk_proprio)
        visual = self.proj_chunk(chunk_visual)
        action_proprio = torch.stack([action, proprio], dim=3).flatten(2, 3)
        events = torch.cat([action_proprio, visual], dim=2)
        type_ids = self.type_ids.to(device=events.device)
        type_emb = self.event_type[type_ids]
        denom = max(self.max_chunks - 1, 1)
        time = self.time_proj((positions.float() / denom).unsqueeze(-1))
        events = events + type_emb.view(1, 1, self.events_per_chunk, -1)
        events = events + self.event_slot.view(1, 1, self.events_per_chunk, -1)
        events = events + time.unsqueeze(2)
        valid = torch.ones(events.shape[:3], dtype=torch.bool, device=events.device)
        flat_events = events.flatten(1, 2)
        flat_valid = valid.flatten(1, 2)
        return flat_events, flat_valid

    def _scan(self, events: Tensor, valid: Tensor, state: RecurrentState) -> tuple[Tensor, Tensor, Tensor]:
        hidden = events
        matrices = []
        caches = []
        for layer, block in enumerate(self.blocks):
            hidden, matrix, cache = block(
                hidden,
                valid,
                state.matrices[:, layer],
                state.conv_caches[:, layer],
            )
            matrices.append(matrix)
            caches.append(cache)
        return hidden, torch.stack(matrices, dim=1), torch.stack(caches, dim=1)

    def _language_queries(self, batch: int, device: torch.device, lang: Tensor | None) -> Tensor:
        queries = self.latents.to(device=device).unsqueeze(0).expand(batch, -1, -1)
        return language_condition_queries(queries, lang, self.lang_film)

    def _read_matrix(self, queries: Tensor, matrix: Tensor) -> Tensor:
        batch, latents, dim = queries.shape
        heads = self.config.delta_num_heads
        head_dim = self.config.delta_head_dim
        projected = self.read_q(self.read_norm(queries)).reshape(batch, latents, heads, head_dim)
        projected = l2_normalize(projected)
        read = torch.einsum("bhvk,bmhk->bmhv", matrix.float(), projected)
        read = read.reshape(batch, latents, dim).to(dtype=queries.dtype)
        return queries + self.read_out(read)

    def _fuse_recent(
        self,
        queries: Tensor,
        recent_visual: Tensor,
        recent_action: Tensor,
        recent_proprio: Tensor,
        recent_visual_valid: Tensor | None,
        recent_action_valid: Tensor | None,
        recent_proprio_valid: Tensor | None,
    ) -> Tensor:
        batch = queries.shape[0]
        null = self.recent_null.to(dtype=queries.dtype).view(1, 1, -1).expand(batch, 1, -1)
        parts = [
            null,
            self.proj_recent(recent_visual),
            self.tok_action(recent_action),
            self.tok_proprio(recent_proprio),
        ]
        # Drop empty optional streams but keep the null token.
        kept = [parts[0]]
        valids = [torch.ones(batch, 1, dtype=torch.bool, device=queries.device)]
        streams = (
            (parts[1], recent_visual_valid),
            (parts[2], recent_action_valid),
            (parts[3], recent_proprio_valid),
        )
        for tokens, mask in streams:
            if tokens.shape[1] == 0:
                continue
            if mask is None:
                mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
            kept.append(tokens)
            valids.append(mask)
        key_values = torch.cat(kept, dim=1)
        valid = torch.cat(valids, dim=1)
        for block in self.recent_blocks:
            queries = block(queries, key_values, valid)
        return queries

    def _check_incremental_shapes(self, visual: Tensor, action: Tensor, proprio: Tensor) -> None:
        if visual.shape[1] != self.tokens_per_chunk:
            raise ValueError(
                f"Expected {self.tokens_per_chunk} visual tokens in a chunk, got {visual.shape[1]}."
            )
        if action.shape[1] != self.chunk_frames or proprio.shape[1] != self.chunk_frames:
            raise ValueError(
                f"Expected {self.chunk_frames} action and proprio steps per chunk, "
                f"got {action.shape[1]} and {proprio.shape[1]}."
            )


class _DecoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_inner: int) -> None:
        super().__init__()
        self.n_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.n_cross_q = nn.LayerNorm(dim)
        self.n_cross_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.n_ffn = nn.LayerNorm(dim)
        self.ffn_in = nn.Linear(dim, ffn_inner)
        self.ffn_out = nn.Linear(ffn_inner, dim)

    def forward(self, queries: Tensor, memory: Tensor) -> Tensor:
        batch, steps, tokens, dim = queries.shape
        local = self.n_self(queries).reshape(batch * steps, tokens, dim)
        attended, _ = self.self_attn(local, local, local, need_weights=False)
        queries = queries + attended.reshape(batch, steps, tokens, dim)
        flat = self.n_cross_q(queries).reshape(batch, steps * tokens, dim)
        keys = self.n_cross_kv(memory)
        attended, _ = self.cross_attn(flat, keys, keys, need_weights=False)
        queries = queries + attended.reshape(batch, steps, tokens, dim)
        return queries + self.ffn_out(F.gelu(self.ffn_in(self.n_ffn(queries))))


class ImprintDecoder(nn.Module):
    """Stage-1 decoder. The paper discards these heads before joint finetuning."""

    def __init__(self, config: ImprintConfig, num_patches: int) -> None:
        super().__init__()
        dim = config.memory_dim
        self.num_patches = num_patches
        self.max_query_steps = config.max_query_steps
        self.step_emb = nn.Embedding(config.max_query_steps, dim)
        self.patch_emb = nn.Embedding(num_patches, dim)
        self.summary_emb = nn.Parameter(torch.randn(dim) * 0.02)
        self.blocks = nn.ModuleList(
            [
                _DecoderBlock(dim, config.num_heads, config.decoder_ffn_inner)
                for _ in range(config.num_decoder_blocks)
            ]
        )
        self.out_norm = nn.LayerNorm(dim)
        self.head_align = nn.Linear(dim, config.align_dim)
        self.head_action = nn.Linear(dim, _vector_width(config.action_feature, "action"))
        self.head_proprio = nn.Linear(dim, _vector_width(config.robot_state_feature, "observation.state"))
        self.reward_query = nn.Parameter(torch.randn(dim) * 0.02)
        self.reward_norm_q = nn.LayerNorm(dim)
        self.reward_norm_kv = nn.LayerNorm(dim)
        self.reward_cross = nn.MultiheadAttention(dim, config.num_heads, batch_first=True)
        self.head_reward = nn.Linear(dim, 1)

    def forward(self, memory: Tensor, query_steps: Tensor) -> dict[str, Tensor]:
        if int(query_steps.max()) >= self.max_query_steps or int(query_steps.min()) < 0:
            raise ValueError(
                f"query_steps must lie in 0..{self.max_query_steps - 1}, "
                f"got min {int(query_steps.min())} and max {int(query_steps.max())}."
            )
        batch, steps = query_steps.shape
        patches = self.num_patches
        step = self.step_emb(query_steps)
        token_pos = torch.cat([self.patch_emb.weight, self.summary_emb.unsqueeze(0)], dim=0)
        queries = step[:, :, None, :] + token_pos.view(1, 1, patches + 1, -1)
        for block in self.blocks:
            queries = block(queries, memory)
        queries = self.out_norm(queries)
        patch_tokens = queries[:, :, :patches, :]
        summary = queries[:, :, patches, :]
        reward_query = self.reward_query.view(1, 1, -1).expand(batch, 1, -1)
        attended, _ = self.reward_cross(
            self.reward_norm_q(reward_query),
            self.reward_norm_kv(memory),
            self.reward_norm_kv(memory),
            need_weights=False,
        )
        progress = torch.sigmoid(self.head_reward(reward_query + attended)[:, 0, 0])
        return {
            "align": self.head_align(patch_tokens),
            "action": self.head_action(summary),
            "proprio": self.head_proprio(summary),
            "progress": progress,
        }


def memory_pretraining_loss(
    prediction: dict[str, Tensor],
    targets: dict[str, Tensor],
    reduction: str = "mean",
) -> tuple[Tensor, dict[str, float]]:
    """Stage-1 objective. Weights are visual 1, action 0.5, proprio 0.5, progress 0.5."""
    visual = (prediction["align"] - targets["align"]).square().flatten(1).mean(dim=-1)
    action = (prediction["action"] - targets["action"]).square().flatten(1).mean(dim=-1)
    proprio = (prediction["proprio"] - targets["proprio"]).square().flatten(1).mean(dim=-1)
    progress = (prediction["progress"] - targets["progress"]).square()
    per_sample = (
        VISUAL_LOSS_WEIGHT * visual
        + ACTION_LOSS_WEIGHT * action
        + PROPRIO_LOSS_WEIGHT * proprio
        + PROGRESS_LOSS_WEIGHT * progress
    )
    logs = {
        "loss_visual": float(visual.mean().detach()),
        "loss_action": float(action.mean().detach()),
        "loss_proprio": float(proprio.mean().detach()),
        "loss_progress": float(progress.mean().detach()),
    }
    if reduction == "none":
        return per_sample, logs
    if reduction == "mean":
        return per_sample.mean(), logs
    raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}.")


class MemoryModulation(nn.Module):
    """Cross-attend action tokens to memory, then FiLM the features.

    Disabled modulation returns the input unchanged. That is the no-memory path.
    """

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cond = nn.Linear(dim, 2 * dim)

    def forward(
        self, features: Tensor, memory: Tensor | None, valid: Tensor | None, *, enabled: bool
    ) -> Tensor:
        if not enabled:
            return features
        if memory is None or valid is None:
            raise ValueError("Memory modulation requires memory tokens and a validity mask.")
        attended, _ = self.cross(
            self.q_norm(features),
            self.kv_norm(memory),
            self.kv_norm(memory),
            key_padding_mask=~valid,
            need_weights=False,
        )
        variance = features.float().square().mean(dim=-1, keepdim=True)
        normalized = features * torch.rsqrt(variance + 1e-6)
        scale, shift = self.cond(attended).chunk(2, dim=-1)
        return (normalized * (1.0 + scale) + shift).to(dtype=features.dtype)
