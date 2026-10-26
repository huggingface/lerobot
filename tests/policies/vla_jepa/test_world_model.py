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
    ActionConditionedVideoPredictor,
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


def _rope_2e88044(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """`rotate_queries_or_keys` exactly as it was at 2e88044, before #4835."""
    _, _, _, dim = x.size()
    omega = torch.arange(dim // 2, dtype=x.dtype, device=x.device)
    omega /= dim / 2.0
    omega = 1.0 / 10000**omega
    freqs = torch.einsum("..., f -> ... f", pos, omega)
    emb_sin = freqs.sin().squeeze(-1).repeat(1, 1, 1, 2)
    emb_cos = freqs.cos().squeeze(-1).repeat(1, 1, 1, 2)
    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1).flatten(-2)
    return x * emb_cos + y * emb_sin


def _rope_pairwise(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """`rotate_queries_or_keys` as merged in #4835 (each coordinate pair rotated by one frequency)."""
    _, _, _, dim = x.size()
    omega = torch.arange(dim // 2, dtype=x.dtype, device=x.device)
    omega /= dim / 2.0
    omega = 1.0 / 10000**omega
    freqs = torch.einsum("..., f -> ... f", pos, omega)
    emb_sin = freqs.sin().repeat_interleave(2, dim=-1)
    emb_cos = freqs.cos().repeat_interleave(2, dim=-1)
    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1).flatten(-2)
    return x * emb_cos + y * emb_sin


@pytest.mark.parametrize("pos_shape", ["frames", "grid"])
def test_rope_legacy_flag_matches_both_references(pos_shape: str) -> None:
    from lerobot.policies.vla_jepa.world_model import rotate_queries_or_keys

    torch.manual_seed(0)
    x = torch.randn(2, 4, 6, 16, dtype=torch.float64)
    pos = (
        torch.arange(6, dtype=torch.float64)
        if pos_shape == "frames"
        else torch.rand(6, dtype=torch.float64) * 8
    )

    assert torch.equal(rotate_queries_or_keys(x, pos, legacy=True), _rope_2e88044(x, pos))
    assert torch.equal(rotate_queries_or_keys(x, pos), _rope_pairwise(x, pos))
    assert torch.equal(rotate_queries_or_keys(x, pos, legacy=False), _rope_pairwise(x, pos))
    assert not torch.allclose(rotate_queries_or_keys(x, pos, legacy=True), rotate_queries_or_keys(x, pos))


def test_rope_legacy_flag_reaches_predictor(monkeypatch: pytest.MonkeyPatch) -> None:
    import lerobot.policies.vla_jepa.world_model as world_model

    torch.manual_seed(0)
    fixed = _make_predictor(tokens_per_frame=4)
    legacy = _make_predictor(tokens_per_frame=4)
    legacy.load_state_dict(fixed.state_dict())
    for block in legacy.predictor_blocks:
        block.attn.rope_legacy = True
    frame_tokens = torch.randn(2, 3 * 4, 8)
    action_tokens = torch.randn(2, 3 * 2, _ACTION_EMBED_DIM)
    with torch.no_grad():
        out_fixed = fixed(frame_tokens, action_tokens)
        out_legacy = legacy(frame_tokens, action_tokens)
        # The pre-#4835 predictor: the module-level function swapped for the 2e88044 one.
        monkeypatch.setattr(
            world_model, "rotate_queries_or_keys", lambda x, pos, legacy=False: _rope_2e88044(x, pos)
        )
        out_reference = fixed(frame_tokens, action_tokens)

    assert torch.equal(out_legacy, out_reference)
    assert not torch.allclose(out_legacy, out_fixed)


def test_rope_legacy_constructor_argument() -> None:
    predictor = ActionConditionedVideoPredictor(
        num_frames=1,
        img_size=(1, 4),
        patch_size=1,
        tubelet_size=1,
        embed_dim=8,
        action_embed_dim=_ACTION_EMBED_DIM,
        predictor_embed_dim=24,
        depth=2,
        num_heads=2,
        mlp_ratio=2.0,
        num_action_tokens_per_step=2,
        rope_legacy=True,
    )
    assert all(block.attn.rope_legacy for block in predictor.predictor_blocks)
    assert not any(block.attn.rope_legacy for block in _make_predictor().predictor_blocks)
