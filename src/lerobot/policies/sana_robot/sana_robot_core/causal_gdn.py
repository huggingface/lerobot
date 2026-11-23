# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://github.com/NVlabs/Sana/tree/robot) at commit 1e7d6cc51,
# original path sana_robot/policy/attention/causal_gdn.py. Import paths are rewritten to the LeRobot package;
# the streaming window; the recurrence over the teacher-forcing sequence is not vendored.
# Keep this file in sync with the upstream file.

from __future__ import annotations

import torch
import torch.nn as nn

from lerobot.policies.sana_robot.sana_robot_core.gdn import GatedDeltaNet


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
        kv_cache,
        save_kv_cache: bool = False,
        observation_prefill: bool = False,
    ):
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
