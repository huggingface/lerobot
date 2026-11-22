# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://nvlabs.github.io/Sana/Robot/) at commit bac6da0e4,
# original path sana_robot/policy/attention/causal_cross_attention.py. Import paths are rewritten to the LeRobot package;
# the variable-length attention call of flash-attn is one scaled_dot_product_attention call per text span, and the shared text rows of training are not vendored.
# Keep this file in sync with the upstream file.

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F  # noqa: N812

from lerobot.policies.sana_robot.sana_robot_core.cross_attention import MultiHeadCrossAttention


class ChunkTextSpans(NamedTuple):
    """Descriptor that routes each chunk's queries to its own caption span in cross-attention. ``q_lens``/``kv_lens`` pair query segments with caption spans one to one; cu_seqlens are int32 prefix sums."""

    q_lens: tuple
    kv_lens: tuple
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int


def chunk_text_spans(q_lens, kv_lens, device) -> ChunkTextSpans:
    """Return the ChunkTextSpans descriptor for host-int segment lengths."""
    q_lens = tuple(int(v) for v in q_lens)
    kv_lens = tuple(int(v) for v in kv_lens)
    offsets_q = [0]
    for length in q_lens:
        offsets_q.append(offsets_q[-1] + length)
    offsets_k = [0]
    for length in kv_lens:
        offsets_k.append(offsets_k[-1] + length)
    return ChunkTextSpans(
        q_lens=q_lens,
        kv_lens=kv_lens,
        cu_seqlens_q=torch.tensor(offsets_q, dtype=torch.int32, device=device),
        cu_seqlens_k=torch.tensor(offsets_k, dtype=torch.int32, device=device),
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(kv_lens),
    )


class ChunkTextCrossAttention(MultiHeadCrossAttention):
    """Text cross-attention of SANA-Robot-AR: every (chunk, text group) token span reads its own text span."""

    def forward(self, x, text, text_spans: ChunkTextSpans):
        (batch, tokens, channels) = x.shape
        (k, v) = self.kv_linear(text).view(1, -1, 2, channels).unbind(2)
        q = self.q_norm(self.q_linear(x)).view(batch, tokens, self.num_heads, self.head_dim)
        k = self.k_norm(k).view(1, -1, self.num_heads, self.head_dim)
        v = v.view(1, -1, self.num_heads, self.head_dim)
        q = q.reshape(-1, self.num_heads, self.head_dim)
        k = k.reshape(-1, self.num_heads, self.head_dim).to(q.dtype)
        v = v.reshape(-1, self.num_heads, self.head_dim).to(q.dtype)
        (query_offset, text_offset, outputs) = (0, 0, [])
        for query_length, text_length in zip(text_spans.q_lens, text_spans.kv_lens, strict=False):
            span = F.scaled_dot_product_attention(
                q[query_offset : query_offset + query_length].transpose(0, 1).unsqueeze(0),
                k[text_offset : text_offset + text_length].transpose(0, 1).unsqueeze(0),
                v[text_offset : text_offset + text_length].transpose(0, 1).unsqueeze(0),
            )
            outputs.append(span[0].transpose(0, 1))
            (query_offset, text_offset) = (query_offset + query_length, text_offset + text_length)
        return self.proj(torch.cat(outputs, dim=0).reshape(batch, tokens, channels))
