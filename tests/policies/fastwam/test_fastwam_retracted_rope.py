#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""Unit tests for retracted 1D-RoPE (Faster-WAM, arXiv 2608.02365, Eq. 3 / Appx. B.4).

The de-rotation must undo the exact rotary transform `rope_apply` applies in the video
backbone (this codebase slices one shared frequency grid into t/y/x pair blocks, rather
than the paper's per-axis grids). The roundtrip test on nonzero t/y/x coordinates settles
that mapping: apply `rope_apply`, then `retract_video_rope`, and the original keys must
come back. A wrong pair split, axis assignment or coordinate order breaks it.
"""

import pytest
import torch

pytest.importorskip("transformers", reason="fastwam requires the `fastwam` extra (transformers)")
pytest.importorskip("diffusers", reason="fastwam requires the `fastwam` extra (diffusers)")

from lerobot.policies.fastwam.wan.model import rope_apply
from lerobot.policies.fastwam.wan.video_dit import (
    _rope_pair_axis_counts,
    apply_dense_rope,
    precompute_freqs_cis,
    retract_video_rope,
)


def test_axis_split_matches_rope_apply():
    # head_dim=128 -> 64 pairs -> (22, 21, 21), matching the paper's (44, 42, 42) dims.
    assert _rope_pair_axis_counts(64) == (22, 21, 21)
    # Small head: 6 pairs -> (2, 2, 2); each axis keeps at least one pair here.
    assert _rope_pair_axis_counts(6) == (2, 2, 2)


def test_roundtrip_cancellation_on_nonzero_temporal_spatial_coordinates():
    torch.manual_seed(0)
    heads, head_dim = 2, 12
    freqs = precompute_freqs_cis(head_dim, end=64)
    grid = torch.tensor([[2, 2, 3]])  # tokens at t in {0,1}, y in {0,1}, x in {0,1,2}
    seq = 12
    keys = torch.randn(1, seq, heads * head_dim)

    rotated = rope_apply(keys.view(1, seq, heads, head_dim), grid, freqs).flatten(2)
    retracted = retract_video_rope(keys=rotated, video_freqs=freqs, grid_sizes=grid)
    # float32 keys rotated in float64: cancellation must be exact to float32 precision.
    torch.testing.assert_close(retracted, keys, rtol=1e-5, atol=1e-5)


def test_roundtrip_covers_each_axis_separately():
    """A token moved ONLY along one axis must still roundtrip; this fails if the
    de-rotation reads the wrong coordinate per pair block (e.g. swaps y and x)."""
    torch.manual_seed(1)
    heads, head_dim = 2, 12
    freqs = precompute_freqs_cis(head_dim, end=64)
    grid = torch.tensor([[1, 3, 2]])  # single frame, y in {0,1,2}, x in {0,1}
    seq = 6
    keys = torch.randn(1, seq, heads * head_dim)
    rotated = rope_apply(keys.view(1, seq, heads, head_dim), grid, freqs).flatten(2)
    retracted = retract_video_rope(keys=rotated, video_freqs=freqs, grid_sizes=grid)
    torch.testing.assert_close(retracted, keys, rtol=1e-5, atol=1e-5)

    # Multi-frame grids put temporal rotation on the first pair block: verify it too.
    grid = torch.tensor([[3, 1, 2]])
    seq = 6
    keys = torch.randn(1, seq, heads * head_dim)
    rotated = rope_apply(keys.view(1, seq, heads, head_dim), grid, freqs).flatten(2)
    retracted = retract_video_rope(keys=rotated, video_freqs=freqs, grid_sizes=grid)
    torch.testing.assert_close(retracted, keys, rtol=1e-5, atol=1e-5)


def test_wrong_axis_assignment_does_not_cancel():
    """Negative control: retracting with y/x grid dims swapped must NOT cancel the
    rotation, proving the test can detect a wrong positional axis assignment."""
    torch.manual_seed(2)
    heads, head_dim = 2, 12
    freqs = precompute_freqs_cis(head_dim, end=64)
    grid = torch.tensor([[1, 3, 2]])  # h != w so a swap is observable
    seq = 6
    keys = torch.randn(1, seq, heads * head_dim)
    rotated = rope_apply(keys.view(1, seq, heads, head_dim), grid, freqs).flatten(2)
    swapped = retract_video_rope(keys=rotated, video_freqs=freqs, grid_sizes=torch.tensor([[1, 2, 3]]))
    assert (swapped - keys).abs().max() > 1e-3


def test_reapplication_uses_the_action_1d_grid():
    """k_v->a = R_a(t_p) * R_v(p)^T * k. For t_p=2 every token's pairs must be rotated by
    row 2 of the action grid; for t_p=0 reapplication is the identity (the Faster-WAM
    observed-frame runtime case), so the result equals pure de-rotation."""
    torch.manual_seed(3)
    heads, head_dim = 2, 12
    freqs = precompute_freqs_cis(head_dim, end=64)
    grid = torch.tensor([[2, 2, 3]])
    seq = 12
    keys = torch.randn(1, seq, heads * head_dim)
    rotated = rope_apply(keys.view(1, seq, heads, head_dim), grid, freqs).flatten(2)

    derotated = retract_video_rope(keys=rotated, video_freqs=freqs, grid_sizes=grid)

    tau = 2
    reapplied = retract_video_rope(
        keys=rotated,
        video_freqs=freqs,
        grid_sizes=grid,
        action_freqs=freqs,
        temporal_positions=torch.full((1, seq), tau),
    )
    expected = apply_dense_rope(derotated, freqs[tau].view(1, 1, -1).expand(seq, 1, -1).contiguous(), heads)
    torch.testing.assert_close(reapplied, expected, rtol=1e-5, atol=1e-5)

    identity = retract_video_rope(keys=rotated, video_freqs=freqs, grid_sizes=grid, action_freqs=freqs)
    torch.testing.assert_close(identity, derotated, rtol=1e-6, atol=1e-6)


def test_unrotated_padding_tokens_pass_through():
    """Tokens beyond a sample's f*h*w grid are unrotated in rope_apply; retraction must
    leave them untouched (per-sample handling), not apply any rotation to them."""
    torch.manual_seed(4)
    heads, head_dim = 2, 12
    freqs = precompute_freqs_cis(head_dim, end=64)
    grid = torch.tensor([[1, 2, 3]])  # 6 rotated tokens
    seq = 8  # 2 padding tokens
    keys = torch.randn(1, seq, heads * head_dim)
    rotated = rope_apply(keys.view(1, seq, heads, head_dim), grid, freqs).flatten(2)
    retracted = retract_video_rope(keys=rotated, video_freqs=freqs, grid_sizes=grid)
    torch.testing.assert_close(retracted[:, :6], keys[:, :6], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(retracted[:, 6:], rotated[:, 6:])  # untouched
