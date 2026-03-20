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
"""Unit tests for UniformActionTokenizer (AWM policy)."""

import pytest
import torch

from lerobot.policies.awm.tokenizer_awm import UniformActionTokenizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tok_2d_v4():
    """2-D tokenizer, V=4, ranges [[-1,1],[-1,1]]."""
    return UniformActionTokenizer([[-1.0, 1.0], [-1.0, 1.0]], vocab_size=4)


@pytest.fixture
def tok_3d_v128():
    """3-D tokenizer, V=128, ranges [[-1,1],[-1,1],[-1,1]]."""
    return UniformActionTokenizer([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]], vocab_size=128)


# ---------------------------------------------------------------------------
# Test 1 — Encode range
# ---------------------------------------------------------------------------


def test_encode_range(tok_3d_v128):
    """Encoded tokens are always in [0, total_vocab_size - 1]."""
    actions = torch.rand(50, 3) * 2 - 1  # uniform in [-1, 1]
    token_ids = tok_3d_v128.encode(actions)
    assert token_ids.min() >= 0
    assert token_ids.max() < tok_3d_v128.total_vocab_size


# ---------------------------------------------------------------------------
# Test 2 — Boundary: lower end maps to bin 0
# ---------------------------------------------------------------------------


def test_boundary_lower(tok_2d_v4):
    """Action at lo for each dimension → bin 0 for that dimension (joint token 0)."""
    actions = torch.tensor([[-1.0, -1.0]])  # (1, 2) — both dims at lo
    token_ids = tok_2d_v4.encode(actions)
    # Both dims are in bin 0 → joint token = 0*1 + 0*4 = 0
    assert token_ids.item() == 0


# ---------------------------------------------------------------------------
# Test 3 — Boundary: upper end maps to bin vocab_size - 1
# ---------------------------------------------------------------------------


def test_boundary_upper(tok_2d_v4):
    """Action at hi for each dimension → bin vocab_size-1 for that dimension."""
    actions = torch.tensor([[1.0, 1.0]])  # (1, 2) — both dims at hi
    token_ids = tok_2d_v4.encode(actions)
    V = tok_2d_v4.vocab_size
    # Both dims are in bin V-1 → joint token = (V-1)*1 + (V-1)*V = (V-1)*(1+V)
    expected = (V - 1) + (V - 1) * V
    assert token_ids.item() == expected


# ---------------------------------------------------------------------------
# Test 4 — Out-of-range clamping (no error)
# ---------------------------------------------------------------------------


def test_out_of_range_clamping(tok_2d_v4):
    """Actions outside [lo, hi] are clamped, no exception is raised."""
    actions = torch.tensor([[-5.0, 10.0]])  # well outside range
    token_ids = tok_2d_v4.encode(actions)  # should not raise
    assert token_ids.min() >= 0
    assert token_ids.max() < tok_2d_v4.total_vocab_size


# ---------------------------------------------------------------------------
# Test 5 — Decode gives bin centres
# ---------------------------------------------------------------------------


def test_decode_bin_centers(tok_2d_v4):
    """For a known token index, decoded value equals lo + (bin + 0.5) * bin_width."""
    V = tok_2d_v4.vocab_size
    lo, hi = -1.0, 1.0
    bin_width = (hi - lo) / V

    # Manually pick bin 1 for dim-0 and bin 2 for dim-1.
    # joint token = 1*1 + 2*V = 1 + 8 = 9
    token_id = torch.tensor([1 + 2 * V])
    decoded = tok_2d_v4.decode(token_id)  # (1, 2)

    expected_d0 = lo + (1 + 0.5) * bin_width
    expected_d1 = lo + (2 + 0.5) * bin_width
    assert decoded.shape == (1, 2)
    assert torch.isclose(decoded[0, 0], torch.tensor(expected_d0), atol=1e-6)
    assert torch.isclose(decoded[0, 1], torch.tensor(expected_d1), atol=1e-6)


# ---------------------------------------------------------------------------
# Test 6 — Round-trip accuracy within half a bin width
# ---------------------------------------------------------------------------


def test_round_trip_accuracy(tok_3d_v128):
    """decode(encode(x)) is within 0.5 * bin_width of x for all dims."""
    V = tok_3d_v128.vocab_size
    lo, hi = -1.0, 1.0
    bin_width = (hi - lo) / V

    actions = torch.rand(100, 3) * 2 - 1  # (100, 3) uniform in [-1, 1]
    token_ids = tok_3d_v128.encode(actions)
    recovered = tok_3d_v128.decode(token_ids)
    assert (recovered - actions).abs().max() <= 0.5 * bin_width + 1e-6


# ---------------------------------------------------------------------------
# Test 7 — Known values: hand-verify a specific 2-D multi-radix joint token
# ---------------------------------------------------------------------------


def test_known_values_2d():
    """Hand-verify encode and decode for V=4, 2-D, specific bin indices."""
    tok = UniformActionTokenizer([[-1.0, 1.0], [0.0, 2.0]], vocab_size=4)
    V = 4
    # Choose bins: dim-0 → bin 2, dim-1 → bin 3.
    # joint = 2 + 3*4 = 14
    # dim-0 centre: -1 + (2 + 0.5) * 0.5 = -1 + 1.25 = 0.25
    # dim-1 centre:  0 + (3 + 0.5) * 0.5 =  0 + 1.75 = 1.75
    token_id = torch.tensor([2 + 3 * V])
    decoded = tok.decode(token_id)

    assert torch.isclose(decoded[0, 0], torch.tensor(0.25), atol=1e-6)
    assert torch.isclose(decoded[0, 1], torch.tensor(1.75), atol=1e-6)

    # Verify that an action at those centres encodes back to the same token.
    action = torch.tensor([[0.25, 1.75]])
    assert tok.encode(action).item() == token_id.item()


# ---------------------------------------------------------------------------
# Test 8 — Batch shapes: 1D, 2D, 3D
# ---------------------------------------------------------------------------


def test_batch_shapes(tok_2d_v4):
    """encode/decode work correctly for (D,), (B, D), and (B, T, D) inputs."""
    # 1-D: (D,)
    a1 = torch.tensor([0.0, 0.5])
    ids1 = tok_2d_v4.encode(a1)
    assert ids1.shape == ()
    dec1 = tok_2d_v4.decode(ids1)
    assert dec1.shape == (2,)

    # 2-D: (B, D)
    a2 = torch.zeros(5, 2)
    ids2 = tok_2d_v4.encode(a2)
    assert ids2.shape == (5,)
    dec2 = tok_2d_v4.decode(ids2)
    assert dec2.shape == (5, 2)

    # 3-D: (B, T, D)
    a3 = torch.zeros(3, 7, 2)
    ids3 = tok_2d_v4.encode(a3)
    assert ids3.shape == (3, 7)
    dec3 = tok_2d_v4.decode(ids3)
    assert dec3.shape == (3, 7, 2)


# ---------------------------------------------------------------------------
# Test 9 — Monotonicity: larger action value → larger bin index per dimension
# ---------------------------------------------------------------------------


def test_monotonicity():
    """Increasing action value along a single dimension increases (or maintains) its bin index."""
    tok = UniformActionTokenizer([[-1.0, 1.0]], vocab_size=16)
    values = torch.linspace(-1.0, 1.0, 100).unsqueeze(-1)  # (100, 1)
    token_ids = tok.encode(values)  # (100,) — single-dim, so joint == bin index
    # Sequence should be non-decreasing.
    diffs = token_ids[1:] - token_ids[:-1]
    assert (diffs >= 0).all(), "Token indices are not monotonically non-decreasing"


# ---------------------------------------------------------------------------
# Test 10 — Different vocab sizes (V=4 and V=128)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("vocab_size", [4, 128])
def test_different_vocab_sizes(vocab_size):
    """Tokenizer works correctly for multiple different vocab sizes."""
    tok = UniformActionTokenizer([[-1.0, 1.0], [-1.0, 1.0]], vocab_size=vocab_size)
    assert tok.total_vocab_size == vocab_size**2

    # Encode a batch and check range.
    actions = torch.rand(20, 2) * 2 - 1
    token_ids = tok.encode(actions)
    assert token_ids.min() >= 0
    assert token_ids.max() < tok.total_vocab_size

    # Round-trip within half a bin width.
    bin_width = 2.0 / vocab_size
    recovered = tok.decode(token_ids)
    assert (recovered - actions).abs().max() <= 0.5 * bin_width + 1e-6
