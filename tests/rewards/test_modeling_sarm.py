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

"""Tests for the SARM reward model training loss."""

import torch

from lerobot.rewards.sarm.configuration_sarm import SARMConfig
from lerobot.rewards.sarm.modeling_sarm import SARMRewardModel


def _make_model(stage_loss_weight: float) -> SARMRewardModel:
    config = SARMConfig(
        annotation_mode="dual",
        num_sparse_stages=3,
        num_dense_stages=3,
        hidden_dim=32,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
        image_dim=16,
        text_dim=16,
        max_state_dim=8,
        stage_loss_weight=stage_loss_weight,
        device="cpu",
    )
    return SARMRewardModel(config)


def _train_step(model: SARMRewardModel) -> dict[str, torch.Tensor]:
    batch_size, seq_len = 2, 5
    config = model.config
    torch.manual_seed(0)
    img_emb = torch.randn(batch_size, 1, seq_len, config.image_dim)
    lang_emb = torch.randn(batch_size, config.text_dim)
    state = torch.randn(batch_size, seq_len, config.max_state_dim)
    lengths = torch.full((batch_size,), seq_len, dtype=torch.int32)
    targets = torch.rand(batch_size, seq_len) * config.num_sparse_stages
    return model._train_step(img_emb, lang_emb, state, lengths, targets, scheme="sparse")


def test_stage_loss_weight_scales_stage_loss():
    """`stage_loss_weight` should scale the stage classification loss (issue: dead config field)."""
    model = _make_model(stage_loss_weight=2.5)
    result = _train_step(model)

    assert result["stage_loss"] > 0, "test is vacuous if the stage loss is zero"
    expected = 2.5 * result["stage_loss"] + result["subtask_loss"]
    assert torch.isclose(result["total_loss"], expected)


def test_default_stage_loss_weight_preserves_behavior():
    """The default weight of 1.0 must leave the total loss unchanged."""
    model = _make_model(stage_loss_weight=1.0)
    result = _train_step(model)

    assert torch.isclose(result["total_loss"], result["stage_loss"] + result["subtask_loss"])
