#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import torch
from accelerate import Accelerator

from lerobot.scripts.lerobot_train import update_policy
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker


class DummyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, batch):
        loss = self.linear(batch["observation"]).sum()
        output_dict = {"prediction": loss}
        return loss, output_dict


def create_policy_and_optimizer():
    torch.manual_seed(42)
    policy = DummyPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    return policy, optimizer, lr_scheduler


def create_metrics_tracker(batch_size: int):
    return MetricsTracker(
        batch_size=batch_size,
        num_frames=100,
        num_episodes=10,
        metrics={
            "loss": AverageMeter("loss", ":.3f"),
            "grad_norm": AverageMeter("grad_norm", ":.3f"),
            "lr": AverageMeter("lr", ":.2e"),
            "update_s": AverageMeter("update_s", ":.3f"),
        },
    )


def test_update_policy_sync_gradients():
    gradient_accumulation_steps = 4
    batch_size = 4

    policy, optimizer, lr_scheduler = create_policy_and_optimizer()
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        step_scheduler_with_optimizer=False,
    )
    policy, optimizer, lr_scheduler = accelerator.prepare(policy, optimizer, lr_scheduler)

    train_metrics = create_metrics_tracker(batch_size)
    for i in range(gradient_accumulation_steps):
        batch = {"observation": torch.randn(batch_size, 10).to(accelerator.device)}
        train_metrics, _ = update_policy(
            train_metrics=train_metrics,
            policy=policy,
            batch=batch,
            optimizer=optimizer,
            grad_clip_norm=1.0,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
        )

        if i < gradient_accumulation_steps - 1:
            assert not accelerator.sync_gradients, f"Step {i}: Gradients should not be synced yet"
        else:
            assert accelerator.sync_gradients, "Final step: Gradients should be synced"


def test_update_policy_gradient_accumulation_equivalence():
    gradient_accumulation_steps = 4
    batch_size = 4

    policy_no_accum, optimizer_no_accum, lr_scheduler_no_accum = create_policy_and_optimizer()
    policy_with_accum, optimizer_with_accum, lr_scheduler_with_accum = create_policy_and_optimizer()

    batches_small = [{"observation": torch.randn(batch_size, 10)} for _ in range(gradient_accumulation_steps)]
    combined_batch = {"observation": torch.cat([b["observation"] for b in batches_small], dim=0)}

    accelerator_no_accum = Accelerator(
        gradient_accumulation_steps=1,
        step_scheduler_with_optimizer=False,
    )
    policy_no_accum, optimizer_no_accum, lr_scheduler_no_accum = accelerator_no_accum.prepare(
        policy_no_accum, optimizer_no_accum, lr_scheduler_no_accum
    )

    train_metrics_no_accum = create_metrics_tracker(batch_size * gradient_accumulation_steps)
    batch_prepared = {k: v.to(accelerator_no_accum.device) for k, v in combined_batch.items()}
    update_policy(
        train_metrics=train_metrics_no_accum,
        policy=policy_no_accum,
        batch=batch_prepared,
        optimizer=optimizer_no_accum,
        grad_clip_norm=1.0,
        accelerator=accelerator_no_accum,
        lr_scheduler=lr_scheduler_no_accum,
    )

    accelerator_with_accum = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        step_scheduler_with_optimizer=False,
    )
    policy_with_accum, optimizer_with_accum, lr_scheduler_with_accum = accelerator_with_accum.prepare(
        policy_with_accum, optimizer_with_accum, lr_scheduler_with_accum
    )

    train_metrics_with_accum = create_metrics_tracker(batch_size)
    for batch in batches_small:
        batch_prepared = {k: v.to(accelerator_with_accum.device) for k, v in batch.items()}
        train_metrics_with_accum, _ = update_policy(
            train_metrics=train_metrics_with_accum,
            policy=policy_with_accum,
            batch=batch_prepared,
            optimizer=optimizer_with_accum,
            grad_clip_norm=1.0,
            accelerator=accelerator_with_accum,
            lr_scheduler=lr_scheduler_with_accum,
        )

    params_no_accum = list(accelerator_no_accum.unwrap_model(policy_no_accum).parameters())
    params_with_accum = list(accelerator_with_accum.unwrap_model(policy_with_accum).parameters())

    for p1, p2 in zip(params_no_accum, params_with_accum, strict=True):
        assert torch.allclose(p1, p2), "Parameters should be equal with gradient accumulation"
