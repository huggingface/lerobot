# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://github.com/NVlabs/Sana/tree/robot) at commit 1e7d6cc51,
# original path sana_robot/policy/attention/softmax.py. Import paths are rewritten to the LeRobot package.
# Keep this file in sync with the upstream file.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from lerobot.policies.sana_robot.sana_robot_core.norms import RMSNorm


def rotate(features: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary phases to [B, N, heads, head_dim] features in float64 complex arithmetic."""
    pairs = torch.view_as_complex(features.transpose(1, 2).to(torch.float64).unflatten(3, (-1, 2)))
    rotated = torch.view_as_real(pairs * freqs).flatten(3, 4).transpose(1, 2)
    return rotated.type_as(features)


class GatedSoftmaxAttention(nn.Module):
    """Parameters and attention kernel shared by the softmax layers of both policies: RMS-normalized queries and keys, rotary positions and a sigmoid output gate. The policies' subclasses decide which keys every query reads."""

    def __init__(self, hidden_size: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.dim = hidden_size // heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.q_norm = RMSNorm(hidden_size, scale_factor=1.0, eps=1e-05)
        self.k_norm = RMSNorm(hidden_size, scale_factor=1.0, eps=1e-05)
        self.output_gate = nn.Linear(hidden_size, hidden_size, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def project(self, x: torch.Tensor):
        """Return the normalized queries and keys and the values of ``x``, each [B, N, heads, head_dim]."""
        (batch, tokens, channels) = x.shape
        (q, k, v) = self.qkv(x).reshape(batch, tokens, 3, channels).unbind(2)
        q = self.q_norm(q).reshape(batch, tokens, self.heads, self.dim)
        k = self.k_norm(k).reshape(batch, tokens, self.heads, self.dim)
        return (q, k, v.reshape(batch, tokens, self.heads, self.dim))

    def attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Return softmax attention of the queries over every given key, all [B, N, heads, head_dim]."""
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return out.transpose(1, 2)

    def gate_and_project(self, out: torch.Tensor, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Gate the attention output [B, N, heads, head_dim], cast to ``dtype``, by the layer input ``x`` and project it."""
        out = out.reshape(x.shape).to(dtype)
        return self.proj(out * torch.sigmoid(self.output_gate(x)))
