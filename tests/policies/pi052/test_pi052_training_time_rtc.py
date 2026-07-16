#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Unit coverage for Pi052 training-time RTC conditioning."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from lerobot.policies.pi05.modeling_pi05 import (  # noqa: E402
    _prepare_trained_rtc_prefix,
    create_sinusoidal_pos_embedding,
)
from lerobot.policies.pi052.configuration_pi052 import PI052Config  # noqa: E402
from lerobot.policies.pi052.modeling_pi052 import (  # noqa: E402
    PI05Pytorch as PI052Pytorch,
    _build_flow_matching_inputs,
    _flow_loss_per_sample,
    _reduce_flow_loss,
)


def test_training_rtc_uses_clean_prefix_and_per_token_time():
    actions = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    noise = torch.tensor([[[10.0], [20.0], [30.0], [40.0]]])
    time = torch.tensor([0.25])
    prefix_mask = torch.tensor([[True, True, False, False]])

    x_t, model_time = _build_flow_matching_inputs(actions, noise, time, prefix_mask)

    assert model_time.tolist() == [[0.0, 0.0, 0.25, 0.25]]
    assert torch.equal(x_t[:, :2], actions[:, :2])
    assert torch.equal(x_t[:, 2:], 0.25 * noise[:, 2:] + 0.75 * actions[:, 2:])


def test_training_rtc_loss_averages_over_postfix_only():
    flow_loss = torch.tensor([[[100.0], [100.0], [2.0], [4.0]]])
    prefix_mask = torch.tensor([[True, True, False, False]])

    per_sample = _flow_loss_per_sample(flow_loss, prefix_mask)

    assert per_sample.tolist() == [3.0]


def test_training_rtc_mean_loss_uses_global_postfix_normalization():
    flow_loss = torch.tensor(
        [
            [[1.0], [1.0], [1.0], [1.0]],
            [[9.0], [9.0], [9.0], [9.0]],
        ]
    )
    prefix_mask = torch.tensor(
        [
            [False, False, False, False],
            [True, True, True, False],
        ]
    )

    loss = _reduce_flow_loss(flow_loss, prefix_mask, predict_actions_t=None, reduction="mean")

    assert loss.item() == pytest.approx(13 / 5)


def test_per_token_time_embedding_preserves_action_axis():
    time = torch.tensor([[0.0, 0.5, 1.0]])

    embedding = create_sinusoidal_pos_embedding(time, 8, 4e-3, 4.0, time.device)

    assert embedding.shape == (1, 3, 8)
    assert not torch.equal(embedding[:, 0], embedding[:, 1])


def test_action_expert_embeds_per_token_flow_times():
    model = PI052Pytorch.__new__(PI052Pytorch)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(chunk_size=3, min_period=4e-3, max_period=4.0)
    model.gradient_checkpointing_enabled = False
    model.action_in_proj = nn.Linear(2, 8)
    model.time_mlp_in = nn.Linear(8, 8)
    model.time_mlp_out = nn.Linear(8, 8)

    suffix, _, _, adarms_cond = model.embed_suffix(
        torch.randn(2, 3, 2),
        torch.tensor([[0.0, 0.5, 0.5], [0.0, 0.0, 0.5]]),
    )

    assert suffix.shape == (2, 3, 8)
    assert adarms_cond.shape == (2, 3, 8)


def test_trained_rtc_prefix_is_padded_and_masked():
    x_t = torch.randn(1, 5, 4)
    previous = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    prefix, mask = _prepare_trained_rtc_prefix(x_t, previous, inference_delay=2, training_max_delay=3)

    assert prefix.shape == x_t.shape
    assert mask.shape == x_t.shape
    assert torch.equal(prefix[0, :2, :2], previous[:2])
    assert torch.count_nonzero(prefix[0, :2, 2:]) == 0
    assert mask[0, :2].all()
    assert not mask[0, 2:].any()


def test_trained_rtc_rejects_delay_outside_training_distribution():
    with pytest.raises(ValueError, match="exceeds the checkpoint"):
        _prepare_trained_rtc_prefix(
            torch.randn(1, 5, 4),
            torch.randn(4, 2),
            inference_delay=4,
            training_max_delay=3,
        )


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf")])
def test_trained_rtc_rejects_non_finite_prefix(non_finite):
    previous = torch.randn(4, 2)
    previous[0, 0] = non_finite

    with pytest.raises(ValueError, match="NaN or Inf"):
        _prepare_trained_rtc_prefix(
            torch.randn(1, 5, 4),
            previous,
            inference_delay=2,
            training_max_delay=3,
        )


@pytest.mark.parametrize("max_delay", [-1, 5])
def test_pi052_config_rejects_invalid_training_rtc_delay(max_delay):
    with pytest.raises(ValueError, match="rtc_training_max_delay"):
        PI052Config(chunk_size=5, n_action_steps=5, rtc_training_max_delay=max_delay)
