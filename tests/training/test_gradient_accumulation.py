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

import pytest
import torch
from accelerate import Accelerator
from torch import nn

from lerobot.scripts.lerobot_train import update_policy
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 1, bias=False)

    def forward(self, batch):
        loss = self.projection(batch["x"]).square().mean()
        return loss, {}


def test_gradient_accumulation_steps_optimizer_and_scheduler_once():
    accelerator = Accelerator(
        cpu=True,
        gradient_accumulation_steps=2,
        step_scheduler_with_optimizer=False,
    )
    policy = TinyPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    policy, optimizer, scheduler = accelerator.prepare(policy, optimizer, scheduler)
    metrics = {
        "loss": AverageMeter("loss"),
        "grad_norm": AverageMeter("grad_norm"),
        "lr": AverageMeter("lr"),
        "update_s": AverageMeter("update_s"),
    }
    tracker = MetricsTracker(1, 2, 1, metrics, accelerator=accelerator)
    batch = {"x": torch.ones(1, 2)}
    before = policy.projection.weight.detach().clone()

    with accelerator.accumulate(policy):
        update_policy(
            tracker,
            policy,
            batch,
            optimizer,
            grad_clip_norm=0,
            accelerator=accelerator,
            lr_scheduler=scheduler,
            log_metrics=False,
        )
    after_first_microbatch = policy.projection.weight.detach().clone()

    with accelerator.accumulate(policy):
        update_policy(
            tracker,
            policy,
            batch,
            optimizer,
            grad_clip_norm=0,
            accelerator=accelerator,
            lr_scheduler=scheduler,
            log_metrics=False,
        )
    after_optimizer_step = policy.projection.weight.detach().clone()

    torch.testing.assert_close(after_first_microbatch, before)
    assert not torch.equal(after_optimizer_step, after_first_microbatch)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)
