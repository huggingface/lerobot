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

from __future__ import annotations

import pytest
import torch

from lerobot.policies.vla_jepa.world_model import (
    ACRoPEAttention,
    ActionConditionedVideoPredictor,
    rotate_queries_or_keys,
)

_ACTION_EMBED_DIM = 8


def _make_predictor(
    embed_dim: int = 8,
    action_embed_dim: int = _ACTION_EMBED_DIM,
    predictor_embed_dim: int = 24,
    num_action_tokens: int = 2,
    tokens_per_frame: int = 1,
) -> ActionConditionedVideoPredictor:
    return ActionConditionedVideoPredictor(
        num_frames=1,
        img_size=(1, tokens_per_frame),
        patch_size=1,
        tubelet_size=1,
        embed_dim=embed_dim,
        action_embed_dim=action_embed_dim,
        predictor_embed_dim=predictor_embed_dim,
        depth=1,
        num_heads=2,
        mlp_ratio=2.0,
        num_action_tokens_per_step=num_action_tokens,
    )


@pytest.mark.parametrize(
    "batch,num_steps,tokens_per_frame,embed_dim",
    [
        (1, 2, 1, 8),
        (2, 3, 4, 8),
        (4, 5, 2, 16),
    ],
)
def test_predictor_output_shape(batch: int, num_steps: int, tokens_per_frame: int, embed_dim: int) -> None:
    predictor = _make_predictor(
        embed_dim=embed_dim, action_embed_dim=_ACTION_EMBED_DIM, tokens_per_frame=tokens_per_frame
    )
    frame_tokens = torch.randn(batch, num_steps * tokens_per_frame, embed_dim)
    action_tokens = torch.randn(batch, num_steps * 2, _ACTION_EMBED_DIM)
    out = predictor(frame_tokens, action_tokens)
    assert tuple(out.shape) == (batch, num_steps * tokens_per_frame, embed_dim)
    assert torch.isfinite(out).all()


def test_predictor_step_mismatch_raises() -> None:
    predictor = _make_predictor(tokens_per_frame=4)
    frame_tokens = torch.randn(2, 3 * 4, 8)  # 3 steps, 4 tokens each
    with pytest.raises(RuntimeError):
        predictor(frame_tokens, torch.randn(2, 2 * 2, 8))  # 2 steps → mismatch


def _reference_pair_rotation(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Multiply each adjacent coordinate pair by exp(i * position * frequency)."""
    pairs = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2).contiguous())
    frequencies = 10000 ** (
        -torch.arange(x.shape[-1] // 2, device=x.device, dtype=x.dtype) / (x.shape[-1] / 2)
    )
    angles = pos[..., None] * frequencies
    phases = torch.polar(torch.ones_like(angles), angles)
    return torch.view_as_real(pairs * phases).flatten(-2)


@pytest.mark.parametrize("dim", [2, 4, 20])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batched_positions", [False, True])
def test_rotary_pair_reference(dim: int, dtype: torch.dtype, batched_positions: bool) -> None:
    generator = torch.Generator().manual_seed(12)
    values = torch.randn(2, 3, 5, dim, generator=generator, dtype=dtype)
    positions = torch.tensor([0.0, 1.0, -2.0, 3.5, 8.0], dtype=dtype)
    if batched_positions:
        positions = positions.expand(2, 3, 5).clone()
        positions[1] += 2
    actual = rotate_queries_or_keys(values, positions)
    torch.testing.assert_close(actual, _reference_pair_rotation(values, positions))
    torch.testing.assert_close(actual.square().sum(-1), values.square().sum(-1))


@pytest.mark.parametrize("dim", [4, 20])
def test_rotary_same_position_preserves_inner_product(dim: int) -> None:
    generator = torch.Generator().manual_seed(21)
    queries = torch.randn(2, 3, 4, dim, generator=generator, dtype=torch.float64)
    keys = torch.randn(2, 3, 4, dim, generator=generator, dtype=torch.float64)
    positions = torch.tensor([0.0, 1.0, 3.0, -4.0], dtype=queries.dtype)
    rotated_queries = rotate_queries_or_keys(queries, positions)
    rotated_keys = rotate_queries_or_keys(keys, positions)
    torch.testing.assert_close((rotated_queries * rotated_keys).sum(-1), (queries * keys).sum(-1))


def test_rotary_inner_product_depends_only_on_relative_position() -> None:
    generator = torch.Generator().manual_seed(31)
    queries = torch.randn(2, 3, 4, 20, generator=generator, dtype=torch.float64)
    keys = torch.randn(2, 3, 4, 20, generator=generator, dtype=torch.float64)
    query_positions = torch.tensor([0.0, 1.0, 4.0, 8.0], dtype=queries.dtype)
    key_positions = torch.tensor([2.0, 3.0, 5.0, 9.0], dtype=queries.dtype)
    dot_products = []
    for shift in (0.0, 2.0):
        q = rotate_queries_or_keys(queries, query_positions + shift)
        k = rotate_queries_or_keys(keys, key_positions + shift)
        dot_products.append((q * k).sum(-1))
    torch.testing.assert_close(dot_products[0], dot_products[1])


def test_rotary_gradients_match_pair_reference() -> None:
    generator = torch.Generator().manual_seed(41)
    values = torch.randn(2, 3, 4, 20, generator=generator, dtype=torch.float64, requires_grad=True)
    positions = torch.tensor([0.0, 1.0, 2.5, -3.0], dtype=values.dtype, requires_grad=True)
    weights = torch.randn(values.shape, generator=generator, dtype=values.dtype)
    actual = (rotate_queries_or_keys(values, positions) * weights).sum()
    expected = (_reference_pair_rotation(values, positions) * weights).sum()
    actual_gradients = torch.autograd.grad(actual, (values, positions))
    expected_gradients = torch.autograd.grad(expected, (values, positions))
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient)
    assert torch.autograd.gradcheck(rotate_queries_or_keys, (values, positions))


@pytest.mark.parametrize("use_sdpa", [False, True])
@pytest.mark.parametrize("action_tokens", [0, 2])
def test_attention_matches_pair_rotary_reference(monkeypatch, use_sdpa: bool, action_tokens: int) -> None:
    from lerobot.policies.vla_jepa import world_model

    torch.manual_seed(51)
    attention = ACRoPEAttention(dim=120, num_heads=2, grid_size=2, use_sdpa=use_sdpa).double().eval()
    tokens = 3 * (action_tokens + 2 * 2)
    values = torch.randn(2, tokens, 120, dtype=torch.float64, requires_grad=True)
    kwargs = {"num_frames": 3, "grid_height": 2, "grid_width": 2, "action_tokens": action_tokens}
    actual = attention(values, **kwargs)
    weights = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad((actual * weights).sum(), (values, attention.qkv.weight))
    monkeypatch.setattr(world_model, "rotate_queries_or_keys", _reference_pair_rotation)
    expected = attention(values, **kwargs)
    expected_gradients = torch.autograd.grad((expected * weights).sum(), (values, attention.qkv.weight))
    torch.testing.assert_close(actual, expected)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_predictor_backward_matches_pair_rotary_reference(monkeypatch) -> None:
    from lerobot.policies.vla_jepa import world_model

    torch.manual_seed(61)
    predictor = _make_predictor(tokens_per_frame=4).double()
    frames = torch.randn(2, 3 * 4, 8, dtype=torch.float64, requires_grad=True)
    actions = torch.randn(2, 3 * 2, _ACTION_EMBED_DIM, dtype=torch.float64, requires_grad=True)
    actual = predictor(frames, actions)
    weights = torch.randn_like(actual)
    inputs = (frames, actions, *predictor.parameters())
    actual_gradients = torch.autograd.grad((actual * weights).sum(), inputs)
    monkeypatch.setattr(world_model, "rotate_queries_or_keys", _reference_pair_rotation)
    expected = predictor(frames, actions)
    expected_gradients = torch.autograd.grad((expected * weights).sum(), inputs)
    torch.testing.assert_close(actual, expected)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient)
