#!/usr/bin/env python

from __future__ import annotations

import pytest
import torch

pytest.importorskip("diffusers")

from conftest import (
    ACTION_DIM,
    ACTION_HORIZON,
    BATCH_SIZE,
    QWEN_HIDDEN_SIZE,
    STATE_DIM,
    make_config,
    set_seed_all,
)  # noqa: E402

from lerobot.policies.vla_jepa.action_head import (  # noqa: E402
    VLAJEPAActionHead,
)

# ---------------------------------------------------------------------------
# VLAJEPAActionHead
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "action_dim,state_dim,action_horizon",
    [
        (3, 4, 4),  # default test dims
        (7, 0, 16),  # no proprioceptive state, production-like action space
        (6, 8, 8),  # medium dims
    ],
)
def test_action_head_sample_time_range(action_dim: int, state_dim: int, action_horizon: int) -> None:
    config = make_config(action_dim=action_dim, state_dim=state_dim, action_horizon=action_horizon)
    head = VLAJEPAActionHead(config, cross_attention_dim=QWEN_HIDDEN_SIZE)
    t = head.sample_time(batch_size=200, device=torch.device("cpu"), dtype=torch.float32)
    assert t.shape == (200,)
    assert torch.isfinite(t).all()


@pytest.mark.parametrize(
    "action_dim,state_dim,action_horizon",
    [
        (3, 4, 4),
        (7, 0, 16),
        (6, 8, 8),
    ],
)
def test_action_head_build_inputs_shape(action_dim: int, state_dim: int, action_horizon: int) -> None:
    config = make_config(action_dim=action_dim, state_dim=state_dim, action_horizon=action_horizon)
    head = VLAJEPAActionHead(config, cross_attention_dim=QWEN_HIDDEN_SIZE)
    conditioning = torch.randn(2, 4, QWEN_HIDDEN_SIZE)
    actions = torch.randn(2, action_horizon, action_dim)
    timesteps = torch.randint(0, 100, (2,))

    state = torch.randn(2, state_dim) if state_dim > 0 else None
    out_with = head._build_inputs(conditioning, actions, state, timesteps)
    out_none = head._build_inputs(conditioning, actions, None, timesteps)

    assert out_with.ndim == 3 and out_none.ndim == 3
    if state_dim > 0:
        assert out_with.shape[1] > out_none.shape[1]
    assert torch.isfinite(out_with).all() and torch.isfinite(out_none).all()


@pytest.mark.parametrize(
    "action_dim,state_dim,action_horizon",
    [
        (3, 4, 4),
        (7, 0, 16),
        (6, 8, 8),
    ],
)
def test_action_head_forward_loss_valid(action_dim: int, state_dim: int, action_horizon: int) -> None:
    set_seed_all(42)
    config = make_config(action_dim=action_dim, state_dim=state_dim, action_horizon=action_horizon)
    head = VLAJEPAActionHead(config, cross_attention_dim=QWEN_HIDDEN_SIZE)
    conditioning = torch.randn(2, 4, QWEN_HIDDEN_SIZE)
    actions = torch.randn(2, action_horizon, action_dim)
    state = torch.randn(2, state_dim) if state_dim > 0 else None
    loss = head.forward(conditioning, actions, state)
    assert loss.shape == ()
    assert torch.isfinite(loss) and loss > 0


def test_action_head_forward_gradient_flows() -> None:
    set_seed_all(42)
    config = make_config()
    head = VLAJEPAActionHead(config, cross_attention_dim=QWEN_HIDDEN_SIZE)
    conditioning = torch.randn(BATCH_SIZE, 4, QWEN_HIDDEN_SIZE)
    actions = torch.randn(BATCH_SIZE, ACTION_HORIZON, ACTION_DIM)
    state = torch.randn(BATCH_SIZE, STATE_DIM)
    loss = head.forward(conditioning, actions, state)
    loss.backward()
    assert any(p.grad is not None for p in head.parameters() if p.requires_grad)


@torch.no_grad()
@pytest.mark.parametrize(
    "action_dim,state_dim,action_horizon",
    [
        (3, 4, 4),
        (7, 0, 16),
        (6, 8, 8),
    ],
)
def test_action_head_predict_action_shape(action_dim: int, state_dim: int, action_horizon: int) -> None:
    set_seed_all(42)
    config = make_config(action_dim=action_dim, state_dim=state_dim, action_horizon=action_horizon)
    head = VLAJEPAActionHead(config, cross_attention_dim=QWEN_HIDDEN_SIZE)
    conditioning = torch.randn(2, 4, QWEN_HIDDEN_SIZE)
    state = torch.randn(2, state_dim) if state_dim > 0 else None
    pred = head.predict_action(conditioning, state)
    assert tuple(pred.shape) == (2, action_horizon, action_dim)
    assert torch.isfinite(pred).all()


# ---------------------------------------------------------------------------
# action_is_pad masking
# ---------------------------------------------------------------------------


def test_action_head_loss_fully_padded_is_zero() -> None:
    """Loss is 0 when every timestep is padded (exercises the clamp_min guard)."""
    set_seed_all(42)
    config = make_config()
    head = VLAJEPAActionHead(config, cross_attention_dim=QWEN_HIDDEN_SIZE)
    conditioning = torch.randn(BATCH_SIZE, 4, QWEN_HIDDEN_SIZE)
    actions = torch.randn(BATCH_SIZE, ACTION_HORIZON, ACTION_DIM)
    state = torch.randn(BATCH_SIZE, STATE_DIM)

    action_is_pad = torch.ones(BATCH_SIZE, ACTION_HORIZON, dtype=torch.bool)
    loss = head.forward(conditioning, actions, state, action_is_pad)
    assert loss.item() == 0.0


def test_action_head_loss_none_matches_no_padding() -> None:
    """action_is_pad=None is equivalent to an all-False (no padding) mask."""
    set_seed_all(42)
    config = make_config()
    head = VLAJEPAActionHead(config, cross_attention_dim=QWEN_HIDDEN_SIZE)
    conditioning = torch.randn(BATCH_SIZE, 4, QWEN_HIDDEN_SIZE)
    actions = torch.randn(BATCH_SIZE, ACTION_HORIZON, ACTION_DIM)
    state = torch.randn(BATCH_SIZE, STATE_DIM)

    set_seed_all(0)
    loss_none = head.forward(conditioning, actions, state, action_is_pad=None)

    set_seed_all(0)
    no_pad = torch.zeros(BATCH_SIZE, ACTION_HORIZON, dtype=torch.bool)
    loss_zeros = head.forward(conditioning, actions, state, action_is_pad=no_pad)

    assert torch.isclose(loss_none, loss_zeros)
