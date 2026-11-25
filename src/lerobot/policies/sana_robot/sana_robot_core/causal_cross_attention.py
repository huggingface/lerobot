# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://github.com/NVlabs/Sana/tree/robot) at commit 1e7d6cc51,
# original path sana_robot/policy/attention/causal_cross_attention.py. Import paths are rewritten to the LeRobot package;
# the variable-length attention call of flash-attn is one scaled_dot_product_attention call per text span, or per text when the spans share the rows of their text (kv_starts).
# Keep this file in sync with the upstream file.

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import torch
import torch.nn.functional as F  # noqa: N812

from lerobot.policies.sana_robot.sana_robot_core.cross_attention import MultiHeadCrossAttention


class ChunkTextSpans(NamedTuple):
    """Descriptor that routes each chunk's queries to its own caption span in cross-attention. ``q_lens``/``kv_lens`` pair query segments with caption spans one to one; cu_seqlens are int32 prefix sums. ``kv_starts`` holds the first text row of every span when spans share the rows of one text; without it the spans lie end to end."""

    q_lens: tuple
    kv_lens: tuple
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    kv_starts: tuple | None = None


def chunk_text_spans(q_lens, kv_lens, device, kv_starts=None) -> ChunkTextSpans:
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
        kv_starts=None if kv_starts is None else tuple(int(v) for v in kv_starts),
    )


def merge_text_spans(window_texts) -> tuple:
    """Return the packed text and the ``ChunkTextSpans`` of several windows laid end to end, from every window's own (packed text, spans with ``kv_starts``)."""
    (q_lens, kv_lens, kv_starts, text_rows) = ([], [], [], 0)
    for text, spans in window_texts:
        q_lens.extend(spans.q_lens)
        kv_lens.extend(spans.kv_lens)
        kv_starts.extend(start + text_rows for start in spans.kv_starts)
        text_rows += int(text.shape[1])
    packed_text = torch.cat([text for (text, _spans) in window_texts], dim=1)
    return (packed_text, chunk_text_spans(q_lens, kv_lens, packed_text.device, kv_starts))


@lru_cache(maxsize=256)
def shared_text_groups(q_lens: tuple, kv_lens: tuple, kv_starts: tuple, device: torch.device) -> tuple:
    """Return (groups, restore) of spans that share text rows, cached per layout and device: ``groups`` lists (query rows, first text row, text rows) per distinct text, the query rows being those of every span that reads it; ``restore`` puts the concatenated outputs of the groups back in token order."""
    (queries, offset) = ({}, 0)
    for query_length, text_length, start in zip(q_lens, kv_lens, kv_starts, strict=False):
        queries.setdefault((start, text_length), []).append(torch.arange(offset, offset + query_length))
        offset += query_length
    groups = [(torch.cat(rows), start, length) for ((start, length), rows) in queries.items()]
    restore = torch.argsort(torch.cat([rows for (rows, _start, _length) in groups]))
    return (tuple((rows.to(device), start, length) for (rows, start, length) in groups), restore.to(device))


class ChunkTextCrossAttention(MultiHeadCrossAttention):
    """Text cross-attention of SANA-Robot-AR: every (chunk, text group) token span reads its own text span, or the spans that share a text read it in one call."""

    def forward(self, x, text, text_spans: ChunkTextSpans):
        (batch, tokens, channels) = x.shape
        (k, v) = self.kv_linear(text).view(1, -1, 2, channels).unbind(2)
        q = self.q_norm(self.q_linear(x)).view(batch, tokens, self.num_heads, self.head_dim)
        k = self.k_norm(k).view(1, -1, self.num_heads, self.head_dim)
        v = v.view(1, -1, self.num_heads, self.head_dim)
        q = q.reshape(-1, self.num_heads, self.head_dim)
        k = k.reshape(-1, self.num_heads, self.head_dim).to(q.dtype)
        v = v.reshape(-1, self.num_heads, self.head_dim).to(q.dtype)
        if text_spans.kv_starts is not None:
            (groups, restore) = shared_text_groups(
                text_spans.q_lens, text_spans.kv_lens, text_spans.kv_starts, x.device
            )
            outputs = [
                F.scaled_dot_product_attention(
                    q.index_select(0, rows).transpose(0, 1).unsqueeze(0),
                    k[start : start + length].transpose(0, 1).unsqueeze(0),
                    v[start : start + length].transpose(0, 1).unsqueeze(0),
                )[0].transpose(0, 1)
                for (rows, start, length) in groups
            ]
            out = torch.cat(outputs, dim=0).index_select(0, restore)
            return self.proj(out.reshape(batch, tokens, channels))
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
