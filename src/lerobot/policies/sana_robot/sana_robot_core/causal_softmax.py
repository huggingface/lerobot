# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://github.com/NVlabs/Sana/tree/robot) at commit 1e7d6cc51,
# original path sana_robot/policy/attention/causal_softmax.py. Import paths are rewritten to the LeRobot package;
# the teacher-forcing sequence is attended with one scaled_dot_product_attention call per group instead of FlashAttention over variable-length groups, a packed row window by window.
# Keep this file in sync with the upstream file.

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F  # noqa: N812

from lerobot.policies.sana_robot.sana_robot_core.softmax import GatedSoftmaxAttention, rotate
from lerobot.policies.sana_robot.sana_robot_core.teacher_forcing import teacher_forcing_observation_index


def teacher_forcing_reads(bounds: tuple, clean_segments: int, window: int, sink: int) -> list:
    """Return, per segment of the teacher-forcing sequence, the segments it reads and the window start of the rotated sink copy it reads (None when it reads the sink in place). A segment reads the sink, the last ``window`` clean chunks and itself; once its window has moved past the first chunk it reads the sink through a copy rotated to that window."""
    sink_end = bounds[sink]
    reads = []
    for segment in range(len(bounds) - 1):
        if segment < clean_segments:
            first = sink if segment < sink else max(sink, segment - window)
            visible = list(range(0 if first == sink else first, segment + 1))
        else:
            chunk = segment - clean_segments
            first = sink + max(0, chunk - window)
            visible = list(range(0 if first == sink else first, sink + chunk)) + [segment]
        evicted = bool(sink) and first != sink and (bounds[first] > sink_end)
        reads.append((visible, first - sink if evicted else None))
    return reads


def rotated_sink_keys(sink_keys, sink_phases, starts):
    """Return the sink keys [B, S, heads, head_dim] rotated to every window start in ``starts``, [B, len(starts) * S, heads, head_dim] in start order."""
    keys = torch.view_as_complex(sink_keys.to(torch.float64).unflatten(-1, (-1, 2)))
    phases = sink_phases[:, 0].index_select(1, starts)
    rotated = keys[:, None] * phases[:, :, None, None, :]
    return torch.view_as_real(rotated).flatten(-2).flatten(1, 2).type_as(sink_keys)


@lru_cache(maxsize=256)
def teacher_forcing_groups(
    bounds: tuple, clean_segments: int, window: int, sink: int, observation_spans: tuple, device: torch.device
) -> tuple:
    """Return (groups, sink starts, restore) of one teacher-forcing sequence, cached per layout and device. ``groups`` lists the (query rows, key rows, value rows) of every segment, a noisy chunk as two groups: its prediction tokens, then its observation tokens, which read their own segment's observation tokens only. Key rows address [sequence | one rotated sink copy per window start], value rows the sequence; ``restore`` puts the concatenated outputs of the groups back in token order."""
    reads = teacher_forcing_reads(bounds, clean_segments, window, sink)
    starts = sorted({start for (_visible, start) in reads if start is not None})
    (total_tokens, sink_end) = (bounds[-1], bounds[sink])
    observation = torch.zeros(total_tokens, dtype=torch.bool)
    if observation_spans:
        observation[teacher_forcing_observation_index(bounds, clean_segments, observation_spans)[0]] = True
    groups = []
    for segment, (visible, start) in enumerate(reads):
        tokens = torch.arange(bounds[segment], bounds[segment + 1])
        others = [torch.arange(bounds[other], bounds[other + 1]) for other in visible if other != segment]
        copy = (
            [total_tokens + starts.index(start) * sink_end + torch.arange(sink_end)]
            if start is not None
            else []
        )
        own = observation[tokens]
        if bool((~own).any()):
            groups.append((tokens[~own], torch.cat(others + [tokens] + copy)))
        if bool(own.any()):
            groups.append((tokens[own], torch.cat(others + [tokens[own]] + copy)))
    restore = torch.argsort(torch.cat([queries for (queries, _keys) in groups]))
    rows = tuple(
        (
            queries.to(device),
            keys.to(device),
            torch.where(keys < total_tokens, keys, (keys - total_tokens) % max(sink_end, 1)).to(device),
        )
        for (queries, keys) in groups
    )
    return (rows, torch.tensor(starts, dtype=torch.long, device=device), restore.to(device))


class CausalSoftmaxAttention(GatedSoftmaxAttention):
    """Softmax attention of SANA-Robot-AR. On the teacher-forcing sequence every segment reads the sink, the clean chunks inside its sliding window and itself. A streaming window reads the keys and values that the layer's cache manager (the layer cache's ``"kv"`` entry) selects, rotated at the positions the manager assigns."""

    def forward(
        self,
        x,
        rotary_emb,
        chunk_bounds=None,
        teacher_forcing_clean_segments=None,
        observation_spans=(),
        kv_cache=None,
        save_kv_cache: bool = False,
        observation_prefill: bool = False,
        window_chunks: int = 1,
        sink_phases=None,
        windows=None,
    ):
        (q, k, v) = self.project(x)
        dtype = v.dtype
        if windows is not None:
            (q, k) = (rotate(q, rotary_emb), rotate(k, rotary_emb))
            out = self._packed_row_attention(q, k, v, windows, window_chunks, tuple(observation_spans))
            return self.gate_and_project(out, x, dtype)
        if kv_cache is None:
            (q, k) = (rotate(q, rotary_emb), rotate(k, rotary_emb))
            bounds = [int(bound) for bound in chunk_bounds]
            out = self._teacher_forcing_attention(
                q,
                k,
                v,
                bounds,
                int(teacher_forcing_clean_segments),
                window_chunks,
                sink_phases,
                tuple(observation_spans),
            )
            return self.gate_and_project(out, x, dtype)
        manager = kv_cache["kv"]
        if observation_prefill:
            (keys, values, valid, phases) = manager.observe(k.detach(), v.detach(), q)
        elif save_kv_cache:
            (keys, values, valid, phases) = manager.commit(k.detach(), v.detach(), q)
        else:
            (keys, values, valid, phases) = manager.read(k, v, q)
        if phases is not None:
            query_phases = phases.narrow(2, keys.shape[1] - q.shape[1], q.shape[1])
            (q, keys) = (rotate(q, query_phases), rotate(keys, phases))
        if valid is None:
            out = self.attend(q, keys, values)
        else:
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2), attn_mask=valid
            ).transpose(1, 2)
        return self.gate_and_project(out, x, dtype)

    def _packed_row_attention(self, q, k, v, windows, window, observation_spans):
        """Attend a packed row of teacher-forcing windows, every window on its own tokens."""
        outputs = [
            self._teacher_forcing_attention(
                *(tensor[:, row.start : row.end] for tensor in (q, k, v)),
                list(row.bounds),
                row.num_chunks,
                window,
                row.sink_phases,
                observation_spans,
            )
            for row in windows
        ]
        return torch.cat(outputs, dim=1)

    def _teacher_forcing_attention(
        self, q, k, v, bounds, clean_segments, window, sink_phases, observation_spans
    ):
        """Attend every segment of the teacher-forcing sequence to the segments it reads, one attention call per group of ``teacher_forcing_groups``."""
        (groups, starts, restore) = teacher_forcing_groups(
            tuple(bounds), int(clean_segments), int(window), 1, tuple(observation_spans), q.device
        )
        if starts.numel():
            k = torch.cat([k, rotated_sink_keys(k[:, : bounds[1]], sink_phases, starts)], dim=1)
        outputs = [
            F.scaled_dot_product_attention(
                q.index_select(1, queries).transpose(1, 2),
                k.index_select(1, keys).transpose(1, 2),
                v.index_select(1, values).transpose(1, 2),
            ).transpose(1, 2)
            for (queries, keys, values) in groups
        ]
        return torch.cat(outputs, dim=1).index_select(1, restore)
