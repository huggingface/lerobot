#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Test that the cached SmolVLA RoPE matches the closed form it replaces."""

import pytest
import torch

from lerobot.policies.smolvla.smolvlm_with_expert import RotaryPositionalEmbeddings
from lerobot.utils.random_utils import set_seed


def apply_rope_reference(x, positions, max_wavelength=10_000):
    """The per-call implementation `RotaryPositionalEmbeddings` replaces, kept verbatim."""
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)

    radians = radians[..., None, :]

    sin = torch.sin(radians)
    cos = torch.cos(radians)

    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin

    return res.to(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_rope_cache_matches_closed_form(dtype, head_dim):
    """Cached lookups are bitwise-identical to recomputing the tables per call."""
    set_seed(1000)
    batch, seq_len, num_heads = 2, 48, 4
    x = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype)
    positions = torch.arange(seq_len).unsqueeze(0).expand(batch, seq_len)

    rope = RotaryPositionalEmbeddings(head_dim)

    assert torch.equal(rope(x, positions), apply_rope_reference(x, positions))


def test_rope_cache_matches_closed_form_with_offset_positions():
    """Position ids of a suffix (offset, non-contiguous across the batch) also match."""
    set_seed(1000)
    head_dim = 64
    x = torch.randn(2, 8, 4, head_dim)
    positions = torch.tensor([[600, 601, 602, 603, 604, 605, 606, 607], [12, 13, 14, 15, 16, 17, 18, 19]])

    rope = RotaryPositionalEmbeddings(head_dim)

    assert torch.equal(rope(x, positions), apply_rope_reference(x, positions))


def test_rope_rebase_matches_explicit_shift():
    """`rebase=True` is the caller-side `positions - positions.min()` it replaces."""
    set_seed(1000)
    head_dim = 64
    x = torch.randn(2, 8, 4, head_dim)
    positions = torch.tensor([[600, 601, 602, 603, 604, 605, 606, 607], [12, 13, 14, 15, 16, 17, 18, 19]])
    shifted = positions - torch.min(positions, dim=1, keepdim=True).values

    rope = RotaryPositionalEmbeddings(head_dim)

    assert torch.equal(rope(x, positions, rebase=True), apply_rope_reference(x, shifted))


def test_rope_cache_is_not_in_state_dict():
    """The cache is derived from position ids only, so checkpoints stay unchanged."""
    rope = RotaryPositionalEmbeddings(64)

    assert rope.state_dict() == {}


def test_rope_cache_survives_dtype_cast():
    """A cast rebuilds the cache in float32, keeping the rotation exact in low precision."""
    set_seed(1000)
    head_dim = 64
    x = torch.randn(2, 16, 4, head_dim, dtype=torch.bfloat16)
    positions = torch.arange(16).unsqueeze(0).expand(2, 16)

    rope = RotaryPositionalEmbeddings(head_dim).to(torch.bfloat16)

    assert rope.sin.dtype == torch.float32
    assert torch.equal(rope(x, positions), apply_rope_reference(x, positions))
