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
"""Tests for the encoder-freezing contract of PreTrainedPolicy.

Freezing means two things that must travel together: no gradients, and no training mode.
The second one is the easy one to lose, because `nn.Module.train()` recurses over every
submodule and the training loop calls `policy.train()` again after each evaluation.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from tests.fixtures.dummy_checkpoint_policy import DummyCheckpointConfig, make_dummy_policy


class _PolicyWithEncoder(PreTrainedPolicy):
    """A tiny policy whose encoder carries the two layers that care about training mode."""

    config_class = DummyCheckpointConfig
    name = "dummy_frozen_encoder"

    def __init__(self, config: DummyCheckpointConfig, freeze_encoder: bool = True):
        super().__init__(config)
        self.freeze_encoder = freeze_encoder
        self.encoder = nn.Sequential(
            nn.Linear(config.hidden, config.hidden),
            nn.BatchNorm1d(config.hidden),
            nn.Dropout(p=1.0),
        )
        self.head = nn.Linear(config.hidden, config.hidden)
        self.apply_freezing()

    def get_frozen_modules(self) -> dict[str, nn.Module]:
        return {"encoder": self.encoder} if self.freeze_encoder else {}

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self) -> None:
        pass

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        return self.head(self.encoder(batch["observation.state"])).mean(), None

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        return self.head(self.encoder(batch["observation.state"]))

    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        return self.predict_action_chunk(batch)


def _make_policy(freeze_encoder: bool = True) -> _PolicyWithEncoder:
    return _PolicyWithEncoder(DummyCheckpointConfig(device="cpu"), freeze_encoder=freeze_encoder)


def test_apply_freezing_removes_gradients_and_training_mode():
    policy = _make_policy()

    assert all(not p.requires_grad for p in policy.encoder.parameters())
    assert not policy.encoder.training
    assert all(p.requires_grad for p in policy.head.parameters())


def test_frozen_encoder_survives_the_train_eval_train_cycle():
    """The training loop calls policy.train() again after every evaluation."""
    policy = _make_policy()

    policy.train()
    policy.eval()
    policy.train()

    assert not policy.encoder.training, "policy.train() thawed the frozen encoder"
    assert policy.head.training, "the trainable part must be in training mode"
    assert all(not p.requires_grad for p in policy.encoder.parameters())


def test_frozen_batchnorm_statistics_do_not_move_during_training():
    """The consequence of losing eval mode: running statistics drift silently."""
    policy = _make_policy()
    policy.train()
    batchnorm = policy.encoder[1]
    before = batchnorm.running_mean.clone()

    loss, _ = policy.forward({"observation.state": torch.randn(8, policy.config.hidden)})
    loss.backward()

    torch.testing.assert_close(batchnorm.running_mean, before)
    assert all(p.grad is None for p in policy.encoder.parameters())


def test_set_module_trainable_round_trip():
    policy = _make_policy()
    policy.train()

    policy.set_module_trainable("encoder", True)
    assert all(p.requires_grad for p in policy.encoder.parameters())
    assert policy.encoder.training

    policy.set_module_trainable("encoder", False)
    assert all(not p.requires_grad for p in policy.encoder.parameters())
    assert not policy.encoder.training


def test_policy_without_declaration_is_unaffected():
    """A policy that declares nothing keeps the stock nn.Module.train() behaviour."""
    policy = make_dummy_policy()

    assert policy.get_frozen_modules() == {}
    policy.train()
    assert policy.training
    assert all(module.training for module in policy.modules())
    assert all(p.requires_grad for p in policy.parameters())


def test_unfrozen_encoder_trains_normally():
    """The same policy without the declaration: nothing is held back."""
    policy = _make_policy(freeze_encoder=False)
    policy.train()

    assert policy.encoder.training
    assert all(p.requires_grad for p in policy.encoder.parameters())
