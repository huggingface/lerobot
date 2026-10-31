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
"""NATTEN's neighborhood attention in plain PyTorch, for the Video VAE without NATTEN and on any device."""

import itertools
import math
from collections.abc import Sequence

import torch
from torch import Tensor

# Target size of the keys one chunk of queries gathers, in elements.
_NEIGHBORHOOD_CHUNK_ELEMENTS = 2**28


def _window(size: int, kernel: int, causal: bool, device: torch.device) -> tuple[Tensor, Tensor]:
    """The ``(size, kernel)`` key positions each query of one axis attends to, and which of them it sees.

    These are NATTEN's rules with dilation 1: the window is centered on the query (an even window reaches
    one key further back) and shifted inward at the borders. A causal window ends at the query, so the first
    queries see fewer keys.
    """
    if size < kernel:
        raise ValueError(f"A window of {kernel} keys needs at least {kernel} positions per axis, got {size}")
    query = torch.arange(size, device=device)
    start = (query - (kernel - 1 if causal else kernel // 2)).clamp(0, size - kernel)
    keys = start[:, None] + torch.arange(kernel, device=device)
    return keys, keys <= query[:, None] if causal else torch.ones_like(keys, dtype=torch.bool)


def _neighbors(x: Tensor, windows: list[Tensor]) -> Tensor:
    """Gather ``(batch, *spatial, heads, dim)`` into ``(batch, *queries, heads, window, dim)``."""
    n = len(windows)
    batch, *spatial, heads, dim = x.shape
    # Gather whole rows by a flat (position, head) index, ordered so the window axes merge without a copy.
    row = torch.arange(heads, device=x.device).view([1] * n + [-1] + [1] * n)
    stride = heads
    for axis in reversed(range(n)):
        shape = [1] * (2 * n + 1)
        shape[axis], shape[n + 1 + axis] = windows[axis].shape
        row = row + windows[axis].view(shape) * stride
        stride *= spatial[axis]
    return x.reshape(batch, -1, dim)[:, row].flatten(n + 2, 2 * n + 1)


def neighborhood_attention(
    q: Tensor, k: Tensor, v: Tensor, kernel_size: Sequence[int], is_causal: Sequence[bool]
) -> Tensor:
    """NATTEN's neighborhood attention in plain PyTorch, so the VAE runs without NATTEN.

    ``q``, ``k`` and ``v`` are ``(batch, *spatial, heads, head_dim)``. Along every spatial axis each query
    attends to the keys of its window (see ``_window``). Computed in float32, returned in the dtype of ``v``.
    Autocast is off here, or the bfloat16 autocast the policy encodes videos under would multiply in bfloat16.
    Chunks are planned in Python from the spatial sizes, so compiling or exporting fixes those sizes.
    """
    n = q.dim() - 3
    sizes = q.shape[1:-2]
    windows = [_window(*axis, q.device) for axis in zip(sizes, kernel_size, is_causal, strict=True)]
    visible = None
    if any(is_causal):
        # (*spatial, heads, *window) like the logits, of size 1 along the heads and the non-causal axes
        visible = torch.ones((), dtype=torch.bool, device=q.device)
        for axis, (_, seen) in enumerate(windows):
            if is_causal[axis]:
                shape = [1] * (2 * n + 1)
                shape[axis], shape[n + 1 + axis] = seen.shape
                visible = visible & seen.view(shape)
    # Queries per chunk along each axis. The last axes stay whole while they fit, so a frame too large for one
    # chunk is split by rows rather than gathered at once.
    fit = max(1, _NEIGHBORHOOD_CHUNK_ELEMENTS // math.prod([q.shape[0], *q.shape[-2:], *kernel_size]))
    chunks: list[int] = []
    for size in reversed(sizes):
        chunks.insert(0, min(size, fit))
        fit = max(1, fit // size)
    out = v.new_empty(v.shape)
    ranges = [range(0, size, chunk) for size, chunk in zip(sizes, chunks, strict=True)]
    with torch.autocast(q.device.type, enabled=False):
        for starts in itertools.product(*ranges):
            queries, slab, keys = [slice(None)], [slice(None)], []
            for start, chunk, (window, _), kernel in zip(starts, chunks, windows, kernel_size, strict=True):
                # The keys of these queries lie within kernel - 1 positions of them, so only those are upcast.
                first = max(0, start - kernel + 1)
                queries.append(slice(start, start + chunk))
                slab.append(slice(first, start + chunk + kernel - 1))
                keys.append(window[start : start + chunk] - first)
            query = (q[tuple(queries)].float() * q.shape[-1] ** -0.5).unsqueeze(-2)
            logits = (query @ _neighbors(k[tuple(slab)].float(), keys).transpose(-1, -2)).squeeze(-2)
            if visible is not None:
                # The mask has size 1 along the non-causal axes, so only the causal axes are sliced.
                causal_slices = [s if c else slice(None) for s, c in zip(queries[1:], is_causal, strict=True)]
                mask = visible[tuple(causal_slices)]
                logits = logits.unflatten(-1, kernel_size).masked_fill(~mask, float("-inf")).flatten(-n)
            weights = logits.softmax(-1).unsqueeze(-2)
            out[tuple(queries)] = (weights @ _neighbors(v[tuple(slab)].float(), keys)).squeeze(-2)
    return out
