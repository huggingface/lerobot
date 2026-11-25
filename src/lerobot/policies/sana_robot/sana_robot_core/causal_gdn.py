# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://github.com/NVlabs/Sana/tree/robot) at commit 1e7d6cc51,
# original path sana_robot/policy/attention/causal_gdn.py. Import paths are rewritten to the LeRobot package;
# the recurrence in PyTorch; the fused kernels and the packed row of several windows are not vendored.
# Keep this file in sync with the upstream file.

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import torch
import torch.nn as nn

from lerobot.policies.sana_robot.sana_robot_core.gdn import GatedDeltaNet
from lerobot.policies.sana_robot.sana_robot_core.teacher_forcing import (
    teacher_forcing_observation_index,
    teacher_forcing_prediction_index,
)


def _compute_dtype(reference: torch.Tensor) -> torch.dtype:
    return torch.float64 if reference.dtype == torch.float64 else torch.float32


def _rotate(features: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary phases to [B, heads, N, head_dim] features in float32 complex arithmetic."""
    pairs = torch.view_as_complex(features.to(torch.float32).contiguous().unflatten(-1, (-1, 2)))
    freqs = freqs.to(torch.complex64) if freqs.is_complex() else freqs.float()
    return torch.view_as_real(pairs * freqs).flatten(-2).type_as(features)


def chunk_read(q, k, v, beta, decay, state):
    """Return the output of one chunk [B, heads, N, head_dim] that reads ``state`` scaled by ``decay`` [B, heads] together with its own tokens. Keys are scaled by 1 / sqrt(head_dim * tokens)."""
    dtype = _compute_dtype(q)
    (q, k, v, beta, decay) = (tensor.to(dtype) for tensor in (q, k, v, beta, decay))
    key_scale = torch.full((), q.shape[-1] * q.shape[2], dtype=dtype, device=q.device).rsqrt()
    own = torch.einsum("bhld,bhl,bhlv->bhdv", k * key_scale, beta, v)
    return torch.einsum("bhld,bhdv->bhlv", q, state.to(dtype) * decay[..., None, None] + own).to(dtype)


def chunk_write(k, v, beta, decay, state):
    """Return the state after one chunk writes its tokens by the delta rule: ``S' = g S + K^T diag(beta) (V - K g S)`` with ``g`` the chunk's ``decay`` [B, heads]."""
    dtype = _compute_dtype(k)
    (k, v, beta, decay) = (tensor.to(dtype) for tensor in (k, v, beta, decay))
    key_scale = torch.full((), k.shape[-1] * k.shape[2], dtype=dtype, device=k.device).rsqrt()
    k = k * key_scale
    decayed = state.to(dtype) * decay[..., None, None]
    correction = (v - torch.einsum("bhld,bhdv->bhlv", k, decayed)) * beta[..., None]
    return decayed + torch.einsum("bhld,bhlv->bhdv", k, correction)


def observation_write(k, v, beta, decay, state):
    """Return the state a chunk's observation tokens leave for its denoising steps: ``chunk_write`` of the observation into the committed ``state``, which stays untouched."""
    dtype = _compute_dtype(k)
    key_scale = float(k.shape[-1] * k.shape[2]) ** (-0.5)
    k = k.to(dtype) * key_scale
    decayed = state.to(dtype) * decay.to(dtype)[..., None, None]
    correction = (v.to(dtype) - torch.einsum("bhld,bhdv->bhlv", k, decayed)) * beta.to(dtype)[..., None]
    return decayed + torch.einsum("bhld,bhlv->bhdv", k, correction)


class SegmentLayout(NamedTuple):
    """Token positions of every segment padded to the longest one: ``index`` [S, T] (N, one past the last token, marks a padding slot), ``slot`` [S, T] true on real tokens, and ``order`` [N], the flat [S * T] position of every token."""

    index: torch.Tensor
    slot: torch.Tensor
    order: torch.Tensor


@lru_cache(maxsize=512)
def segment_layout(bounds: tuple, device: torch.device) -> SegmentLayout:
    """Return the ``SegmentLayout`` of the token bounds [0, t_1, ..., N], cached per (bounds, device)."""
    starts = torch.tensor(bounds[:-1], dtype=torch.long)
    sizes = torch.tensor(bounds[1:], dtype=torch.long) - starts
    width = int(sizes.max()) if sizes.numel() else 0
    offsets = torch.arange(width, dtype=torch.long)
    slot = offsets[None] < sizes[:, None]
    index = torch.where(
        slot, starts[:, None] + offsets[None], torch.full_like(slot, bounds[-1], dtype=torch.long)
    )
    order = torch.arange(index.numel(), dtype=torch.long).reshape(index.shape)[slot]
    return SegmentLayout(index.to(device), slot.to(device), order.to(device))


def gather_segments(tensor: torch.Tensor, index: torch.Tensor, dim: int) -> torch.Tensor:
    """Return ``tensor`` with its token dimension ``dim`` [N] replaced by padded segments [S, T]; padding slots are zero."""
    padding_shape = list(tensor.shape)
    padding_shape[dim] = 1
    padded = torch.cat([tensor, tensor.new_zeros(padding_shape)], dim=dim)
    return padded.index_select(dim, index.reshape(-1)).unflatten(dim, tuple(index.shape))


def scatter_segments(segmented: torch.Tensor, layout: SegmentLayout) -> torch.Tensor:
    """Return padded segments [B, heads, S, T, D] back in token order [B, heads, N, D]."""
    return segmented.flatten(2, 3).index_select(2, layout.order)


def _segment_features(q, k, v, beta, layout: SegmentLayout):
    """Return (q, scaled k, v, beta) of every segment at once, each padded to [.., S, T, ..]; keys are scaled by 1 / sqrt(head_dim * segment tokens)."""
    sizes = layout.slot.to(q.dtype).sum(dim=-1).clamp_min(1.0)
    key_scale = (q.shape[-1] * sizes).rsqrt()[None, None, :, None, None]
    return (
        gather_segments(q, layout.index, 2),
        gather_segments(k, layout.index, 2) * key_scale,
        gather_segments(v, layout.index, 2),
        gather_segments(beta, layout.index, 2),
    )


def _segment_read(q, k, v, beta, decayed_state):
    """Return the output of every segment [B, heads, S, T, D] against its own decayed state [B, heads, S, D, D]."""
    own = torch.einsum("bhstd,bhst,bhstv->bhsdv", k, beta, v)
    return torch.einsum("bhstd,bhsdv->bhstv", q, decayed_state + own).to(decayed_state.dtype)


def _segment_means(x, bounds: tuple, groups: tuple):
    """Return the token mean of ``x`` [B, N, C] over every segment of ``bounds`` and then over every group of each (token index, group bounds) pair in ``groups``, [B, S + G, C] in float32 or wider, as one matmul."""
    sum_dtype = torch.promote_types(x.dtype, torch.float32)
    (indicator, counts) = _segment_indicator(bounds, x.device)
    for index, group_bounds in groups:
        (rows, sizes) = _group_rows(tuple(group_bounds), x.device)
        group = torch.zeros(sizes.numel(), x.shape[1], dtype=torch.float32, device=x.device)
        group[rows, index] = 1.0
        (indicator, counts) = (torch.cat([indicator, group]), torch.cat([counts, sizes]))
    with torch.autocast(x.device.type, enabled=False):
        sums = torch.matmul(indicator.to(sum_dtype), x.to(sum_dtype))
    return sums / counts.to(sum_dtype).clamp_min(1.0)[:, None]


@lru_cache(maxsize=512)
def _segment_indicator(bounds: tuple, device: torch.device) -> tuple:
    """Return the indicator [S, N] of every segment's tokens and the segment sizes [S], both float32, cached per (bounds, device)."""
    sizes = torch.tensor(bounds[1:], dtype=torch.long) - torch.tensor(bounds[:-1], dtype=torch.long)
    indicator = torch.zeros(sizes.numel(), bounds[-1], dtype=torch.float32)
    indicator[torch.repeat_interleave(torch.arange(sizes.numel()), sizes), torch.arange(bounds[-1])] = 1.0
    return (indicator.to(device), sizes.to(torch.float32).to(device))


@lru_cache(maxsize=512)
def _group_rows(group_bounds: tuple, device: torch.device) -> tuple:
    """Return the group of every position and the group sizes (float32) of the group bounds (0, n_0, ..., n), cached per (bounds, device)."""
    sizes = torch.tensor(group_bounds[1:], dtype=torch.long) - torch.tensor(
        group_bounds[:-1], dtype=torch.long
    )
    return (
        torch.repeat_interleave(torch.arange(sizes.numel()), sizes).to(device),
        sizes.to(torch.float32).to(device),
    )


class TeacherForcingObservation(NamedTuple):
    """The split of every noisy chunk of the teacher-forcing sequence into its observation tokens and its prediction tokens (target latents and actions): the token index and per-chunk bounds of both, and ``decay`` [B, heads, L], the decay every chunk's observation applies to the state it reads."""

    index: torch.Tensor
    bounds: tuple
    decay: torch.Tensor
    prediction_index: torch.Tensor
    prediction_bounds: tuple


@lru_cache(maxsize=256)
def _noisy_chunk_split(bounds: tuple, clean_segments: int, observation_spans: tuple, device: torch.device):
    """Return (observation index, its per-chunk bounds, prediction index, its per-chunk bounds) of the noisy chunks, cached per layout."""
    (observation, observation_bounds) = teacher_forcing_observation_index(
        bounds, clean_segments, observation_spans
    )
    (prediction, prediction_bounds) = teacher_forcing_prediction_index(
        bounds, clean_segments, observation_spans
    )
    return (observation.to(device), observation_bounds, prediction.to(device), prediction_bounds)


def gdn_teacher_forcing(
    q, k, v, beta, segment_decay, bounds, clean_segments: int, observation: TeacherForcingObservation
):
    """Run the recurrence on the teacher-forcing sequence [sink | clean chunks | noisy chunks] and return the output [B, heads, N, head_dim]. The clean segments read the state before them and write it in turn; the observation tokens of noisy chunk c read the state after the clean segments before it and write a copy, which the chunk's prediction tokens read."""
    dtype = _compute_dtype(q)
    output_dtype = v.dtype
    (q, k, v, beta) = (tensor.to(dtype) for tensor in (q, k, v, beta))
    (batch, heads, _, dim) = q.shape
    clean_end = bounds[clean_segments]

    clean_layout = segment_layout(tuple(bounds[: clean_segments + 1]), q.device)
    clean = _segment_features(
        *(tensor.narrow(2, 0, clean_end) for tensor in (q, k, v)), beta[:, :, :clean_end], clean_layout
    )
    clean_decay = segment_decay[:, :, :clean_segments].to(dtype)
    key_outer = torch.einsum("bhstd,bhst,bhste->bhsde", clean[1], clean[3], clean[1])
    key_value = torch.einsum("bhstd,bhst,bhstv->bhsdv", clean[1], clean[3], clean[2])
    state = torch.zeros(batch, heads, dim, v.shape[-1], dtype=dtype, device=q.device)
    (states_before, states_after) = ([], [])
    for decay, outer, written in zip(
        clean_decay.unbind(2), key_outer.unbind(2), key_value.unbind(2), strict=False
    ):
        states_before.append(state)
        decayed = state * decay[..., None, None]
        state = decayed - outer @ decayed + written
        states_after.append(state)
    clean_out = _segment_read(*clean, torch.stack(states_before, dim=2) * clean_decay[..., None, None])

    noisy_chunks = len(bounds) - 1 - clean_segments
    memory = torch.stack(states_after[:noisy_chunks], dim=2)

    def gathered(index, group_bounds):
        layout = segment_layout(tuple(int(bound) for bound in group_bounds), index.device)
        features = _segment_features(
            *(tensor.index_select(2, index) for tensor in (q, k, v)), beta.index_select(2, index), layout
        )
        return (layout, features)

    (observation_layout, observed) = gathered(observation.index, observation.bounds)
    decayed = memory * observation.decay.to(dtype)[..., None, None]
    observation_out = _segment_read(*observed, decayed)
    key_outer = torch.einsum("bhstd,bhst,bhste->bhsde", observed[1], observed[3], observed[1])
    key_value = torch.einsum("bhstd,bhst,bhstv->bhsdv", observed[1], observed[3], observed[2])
    observed_state = decayed - key_outer @ decayed + key_value
    (prediction_layout, predicted) = gathered(observation.prediction_index, observation.prediction_bounds)
    prediction_decay = segment_decay[:, :, clean_segments:].to(dtype)
    prediction_out = _segment_read(*predicted, observed_state * prediction_decay[..., None, None])

    noisy_out = memory.new_zeros(batch, heads, bounds[-1] - clean_end, memory.shape[-1])
    noisy_out = noisy_out.index_copy(
        2, observation.index - clean_end, scatter_segments(observation_out, observation_layout)
    )
    noisy_out = noisy_out.index_copy(
        2, observation.prediction_index - clean_end, scatter_segments(prediction_out, prediction_layout)
    )
    return torch.cat((scatter_segments(clean_out, clean_layout), noisy_out), dim=2).to(output_dtype)


class CausalGatedDeltaNet(GatedDeltaNet):
    """Gated DeltaNet of SANA-Robot-AR: a recurrent state that every clean chunk decays and then writes by the delta rule, read by the chunks after it. A streaming window keeps the committed state in the layer's cache under ``"S"`` and the state after the current chunk's observation under ``"observation"``."""

    def reset_recurrence_parameters(self) -> None:
        """Set the decay, the write strength and the output gate to their starting values."""
        super().reset_recurrence_parameters()
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.constant_(self.beta_proj.bias, 5.0)
        nn.init.zeros_(self.output_gate.weight)
        nn.init.constant_(self.output_gate.bias, 5.0)

    def forward(
        self,
        x,
        rotary_emb,
        chunk_bounds=None,
        teacher_forcing_clean_segments=None,
        observation_spans=None,
        kv_cache=None,
        save_kv_cache: bool = False,
        observation_prefill: bool = False,
    ):
        if kv_cache is None:
            return self._forward_teacher_forcing(
                x, rotary_emb, chunk_bounds, teacher_forcing_clean_segments, observation_spans
            )
        (q, k, v, beta) = self._features(x, rotary_emb)
        decay = self.decay_from_gate(self.gate_proj(x.mean(dim=1)).float())
        state = kv_cache.get("S")
        if state is None:
            state = torch.zeros(
                q.shape[0], self.heads, self.dim, self.dim, dtype=_compute_dtype(q), device=q.device
            )
        if observation_prefill:
            out = chunk_read(q, k, v, beta, decay, state)
            kv_cache["observation"] = observation_write(k, v, beta, decay, state)
        elif save_kv_cache:
            out = chunk_read(q, k, v, beta, decay, state)
            kv_cache["S"] = chunk_write(k, v, beta, decay, state).detach()
            kv_cache.pop("observation", None)
        else:
            out = chunk_read(q, k, v, beta, decay, kv_cache["observation"])
        return self._normalize_gate_project(out.to(v.dtype), x)

    def _forward_teacher_forcing(self, x, rotary_emb, chunk_bounds, clean_segments, observation_spans):
        """Run every chunk of the teacher-forcing sequence."""
        bounds = tuple(int(bound) for bound in chunk_bounds)
        clean_segments = int(clean_segments)
        (q, k, v, beta) = self._features(x, rotary_emb)
        (segment_decay, observation) = self._teacher_forcing_decay(
            x, bounds, clean_segments, observation_spans
        )
        out = gdn_teacher_forcing(q, k, v, beta, segment_decay, bounds, clean_segments, observation)
        return self._normalize_gate_project(out, x)

    def _teacher_forcing_decay(self, x, bounds: tuple, clean_segments: int, observation_spans):
        """Return the decay [B, heads, S] of every segment, a noisy chunk's from its prediction tokens, and the ``TeacherForcingObservation`` carrying the decay of every noisy chunk's observation tokens; each from the mean of its tokens."""
        (observation_index, observation_bounds, prediction_index, prediction_bounds) = _noisy_chunk_split(
            bounds, clean_segments, tuple(observation_spans), x.device
        )
        segments = len(bounds) - 1
        noisy_chunks = segments - clean_segments
        pooled = _segment_means(
            x, bounds, ((observation_index, observation_bounds), (prediction_index, prediction_bounds))
        )
        decay = self.decay_from_gate(self.gate_proj(pooled.to(x.dtype)).float()).permute(0, 2, 1)
        segment_decay = torch.cat(
            [decay[..., :clean_segments], decay[..., segments + noisy_chunks :]], dim=-1
        )
        observation = TeacherForcingObservation(
            observation_index,
            observation_bounds,
            decay[..., segments : segments + noisy_chunks],
            prediction_index,
            prediction_bounds,
        )
        return (segment_decay, observation)

    def _features(self, x: torch.Tensor, rotary_emb):
        """Return the rotated queries and keys and the values [B, heads, N, head_dim] and the write strength beta [B, heads, N] of ``x``."""
        (batch, tokens, channels) = x.shape
        (q, k, v) = self.qkv(x).reshape(batch, tokens, 3, channels).unbind(2)

        def heads_first(tensor):
            return tensor.reshape(batch, tokens, self.heads, self.dim).permute(0, 2, 1, 3)

        q = _rotate(heads_first(self.q_norm(q)), rotary_emb)
        k = _rotate(heads_first(self.k_norm(k)), rotary_emb)
        beta = self.beta_proj(x).sigmoid().permute(0, 2, 1).contiguous()
        return (q, k, heads_first(v), beta)

    def _normalize_gate_project(self, out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Normalize the recurrence output [B, heads, N, head_dim] per head, gate it by the layer input ``x`` and project it."""
        (batch, tokens, channels) = x.shape
        out = self.o_norm(out.permute(0, 1, 3, 2))
        out = out.permute(0, 3, 1, 2).reshape(batch, tokens, channels)
        out = out * torch.sigmoid(self.output_gate(x))
        return self.proj(out.to(self.proj.weight.dtype))
