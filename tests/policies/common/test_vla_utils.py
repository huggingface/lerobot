#!/usr/bin/env python

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

"""Behavior-pinning tests for the shared VLA helpers.

These helpers are the canonical versions of functions that used to be copy-pasted across
the openpi-derived policies (pi0, pi05, pi0_fast, smolvla, eo1, xvla). The expected
values below encode the historical per-policy behavior exactly; a failure here means a
behavior change that would silently affect released checkpoints.
"""

import math

import pytest
import torch

from lerobot.policies.common.vla_utils import (
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    pad_vector,
    prepare_attention_masks_4d,
    resize_with_pad,
    resize_with_pad_torch,
)
from lerobot.utils.constants import OPENPI_ATTENTION_MASK_VALUE


def test_create_sinusoidal_pos_embedding_matches_openpi_formula():
    time = torch.tensor([0.0, 0.25, 1.0])
    dim, min_period, max_period = 8, 4e-3, 4.0
    emb = create_sinusoidal_pos_embedding(time, dim, min_period, max_period, device=torch.device("cpu"))

    assert emb.shape == (3, dim)
    # Independent recomputation of the openpi formula in float64.
    fraction = torch.linspace(0.0, 1.0, dim // 2, dtype=torch.float64)
    period = min_period * (max_period / min_period) ** fraction
    scaling = 1.0 / period * 2 * math.pi
    sin_input = scaling[None, :] * time.to(torch.float64)[:, None]
    expected = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    torch.testing.assert_close(emb, expected, rtol=1e-9, atol=1e-9)


def test_create_sinusoidal_pos_embedding_validation():
    with pytest.raises(ValueError, match="divisible by 2"):
        create_sinusoidal_pos_embedding(torch.zeros(2), 7, 4e-3, 4.0, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="batch_size"):
        create_sinusoidal_pos_embedding(torch.zeros(2, 2), 8, 4e-3, 4.0, device=torch.device("cpu"))


def test_make_att_2d_masks_docstring_cases():
    # Pure causal attention: [[1 1 1]]
    pad = torch.ones(1, 3, dtype=torch.bool)
    att = torch.tensor([[1, 1, 1]], dtype=torch.int32)
    expected = torch.tensor([[[1, 0, 0], [1, 1, 0], [1, 1, 1]]], dtype=torch.bool)
    assert torch.equal(make_att_2d_masks(pad, att), expected)

    # Prefix-LM: [[0 0 1 1]] -> first two tokens attend bidirectionally, rest causal.
    att = torch.tensor([[0, 0, 1, 1]], dtype=torch.int32)
    pad = torch.ones(1, 4, dtype=torch.bool)
    expected = torch.tensor([[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]], dtype=torch.bool)
    assert torch.equal(make_att_2d_masks(pad, att), expected)

    # Padding removes rows and columns.
    pad = torch.tensor([[True, True, False]])
    att = torch.tensor([[0, 1, 1]], dtype=torch.int32)
    out = make_att_2d_masks(pad, att)
    assert not out[0, :, 2].any() and not out[0, 2, :].any()


def test_make_att_2d_masks_validation():
    with pytest.raises(ValueError):
        make_att_2d_masks(torch.ones(3, dtype=torch.bool), torch.ones(1, 3, dtype=torch.int32))
    with pytest.raises(ValueError):
        make_att_2d_masks(torch.ones(1, 3, dtype=torch.bool), torch.ones(3, dtype=torch.int32))


def test_prepare_attention_masks_4d():
    masks = torch.tensor([[[True, False], [False, True]]])
    out = prepare_attention_masks_4d(masks)
    assert out.shape == (1, 1, 2, 2)
    expected = torch.tensor([[[[0.0, OPENPI_ATTENTION_MASK_VALUE], [OPENPI_ATTENTION_MASK_VALUE, 0.0]]]])
    assert torch.equal(out, expected)

    out_bf16 = prepare_attention_masks_4d(masks, dtype=torch.bfloat16)
    assert out_bf16.dtype == torch.bfloat16
    assert torch.equal(out_bf16, expected.to(torch.bfloat16))


def test_pad_vector_openpi_semantics():
    v = torch.arange(6.0).reshape(2, 3)
    padded = pad_vector(v, 5)
    assert padded.shape == (2, 5)
    assert torch.equal(padded[:, :3], v) and not padded[:, 3:].any()
    # Already large enough (>=): returned unchanged, same object.
    assert pad_vector(v, 3) is v
    assert pad_vector(v, 2) is v
    # 3D input.
    v3 = torch.ones(2, 4, 3)
    assert pad_vector(v3, 7).shape == (2, 4, 7)


def test_pad_vector_truncate_semantics():
    v = torch.arange(6.0).reshape(2, 3)
    out = pad_vector(v, 2, truncate=True)
    assert out.shape == (2, 2) and torch.equal(out, v[:, :2])
    out = pad_vector(v, 5, truncate=True)
    assert out.shape == (2, 5) and torch.equal(out[:, :3], v) and not out[:, 3:].any()
    assert pad_vector(v, 0, truncate=True).shape == (2, 0)
    assert pad_vector(v, 3, truncate=True) is v


@pytest.mark.parametrize("channels_last", [True, False])
def test_resize_with_pad_torch_centered(channels_last):
    img = torch.rand(2, 3, 30, 60) if not channels_last else torch.rand(2, 30, 60, 3)
    out = resize_with_pad_torch(img, 64, 64)
    if channels_last:
        assert out.shape == (2, 64, 64, 3)
        # Aspect ratio preserved: 30x60 -> 32x64, padded 16 top and 16 bottom (centered).
        assert not out[:, :16].any() and not out[:, -16:].any()
        assert out[:, 16:48].abs().sum() > 0
    else:
        assert out.shape == (2, 3, 64, 64)
        assert not out[:, :, :16].any() and not out[:, :, -16:].any()


def test_resize_with_pad_torch_uint8_roundtrip():
    img = (torch.rand(1, 3, 20, 20) * 255).to(torch.uint8)
    out = resize_with_pad_torch(img, 40, 40)
    assert out.dtype == torch.uint8 and out.shape == (1, 3, 40, 40)
    with pytest.raises(ValueError, match="Unsupported image dtype"):
        resize_with_pad_torch(torch.rand(1, 3, 8, 8, dtype=torch.float64), 16, 16)


def test_resize_with_pad_top_left():
    img = torch.rand(2, 3, 30, 60)
    out = resize_with_pad(img, 64, 64, pad_value=-1.0)
    assert out.shape == (2, 3, 64, 64)
    # 30x60 -> 32x64; this variant pads on the TOP only (32 rows of pad_value).
    assert torch.equal(out[:, :, :32], torch.full((2, 3, 32, 64), -1.0))
    assert out[:, :, 32:].min() >= 0
    # No-op fast path returns the same object.
    assert resize_with_pad(img, 30, 60, pad_value=0.0) is img
    with pytest.raises(ValueError, match="expected"):
        resize_with_pad(torch.rand(3, 8, 8), 16, 16, pad_value=0.0)


def test_clone_past_key_values():
    pytest.importorskip("transformers")
    from transformers import DynamicCache

    from lerobot.policies.common.vla_utils import clone_past_key_values

    cache = DynamicCache()
    keys, values = torch.rand(1, 2, 4, 8), torch.rand(1, 2, 4, 8)
    cache.update(keys, values, 0)
    cloned = clone_past_key_values(cache)
    (ck, cv, _), (ok, ov, _) = next(iter(cloned)), next(iter(cache))
    assert torch.equal(ck, ok) and torch.equal(cv, ov)
    # Deep copy: mutating the clone must not touch the original.
    ck.zero_()
    assert not torch.equal(ck, ok)


def test_clone_past_key_values_is_fullgraph_compilable():
    pytest.importorskip("transformers")
    from transformers import DynamicCache

    from lerobot.policies.common.vla_utils import clone_past_key_values

    cache = DynamicCache()
    keys, values = torch.rand(1, 2, 4, 8), torch.rand(1, 2, 4, 8)
    cache.update(keys, values, 0)

    compiled_clone = torch.compile(clone_past_key_values, backend="eager", fullgraph=True)
    cloned = compiled_clone(cache)

    (cloned_keys, cloned_values, _), (original_keys, original_values, _) = (
        next(iter(cloned)),
        next(iter(cache)),
    )
    assert torch.equal(cloned_keys, original_keys)
    assert torch.equal(cloned_values, original_values)
