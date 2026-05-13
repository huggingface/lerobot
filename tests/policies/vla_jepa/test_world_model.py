#!/usr/bin/env python

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
    predictor_embed_dim: int = 16,
    num_action_tokens: int = 2,
) -> ActionConditionedVideoPredictor:
    return ActionConditionedVideoPredictor(
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
    predictor = _make_predictor(embed_dim=embed_dim, action_embed_dim=_ACTION_EMBED_DIM)
    frame_tokens = torch.randn(batch, num_steps, tokens_per_frame, embed_dim)
    action_tokens = torch.randn(batch, num_steps, 2, _ACTION_EMBED_DIM)
    out = predictor(frame_tokens, action_tokens)
    assert tuple(out.shape) == (batch, num_steps, tokens_per_frame, embed_dim)
    assert torch.isfinite(out).all()


def test_predictor_step_mismatch_raises() -> None:
    predictor = _make_predictor()
    frame_tokens = torch.randn(2, 3, 4, 8)
    with pytest.raises(ValueError, match="Expected 3 action steps"):
        predictor(frame_tokens, torch.randn(2, 2, 2, 8))
