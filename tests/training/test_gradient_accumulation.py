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

import importlib.util
import sys
import types

import pytest
import torch

from lerobot.utils.logging_utils import AverageMeter, MetricsTracker


def import_update_policy():
    # The trainer module imports the dataset package for the full CLI path, but
    # this unit test only exercises update_policy. Keep the test runnable with
    # the base test extra, where Hugging Face datasets is not always installed.
    if importlib.util.find_spec("datasets") is None:
        fake_datasets_module = types.ModuleType("lerobot.datasets")
        fake_datasets_module.__path__ = []
        fake_datasets_module.EpisodeAwareSampler = object
        fake_datasets_module.make_dataset = None
        fake_language_module = types.ModuleType("lerobot.datasets.language")
        fake_language_module.LANGUAGE_COLUMNS = set()
        sys.modules.setdefault("lerobot.datasets", fake_datasets_module)
        sys.modules.setdefault("lerobot.datasets.language", fake_language_module)

    from lerobot.scripts.lerobot_train import update_policy

    return update_policy


class StepCounter:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(self, batch):
        loss = ((self.weight * batch["x"] - batch["y"]) ** 2).mean()
        return loss, {"policy_loss": loss.detach()}


def make_tracker():
    return MetricsTracker(
        batch_size=2,
        num_frames=100,
        num_episodes=10,
        metrics={
            "loss": AverageMeter("loss", ":.3f"),
            "grad_norm": AverageMeter("grdn", ":.3f"),
            "lr": AverageMeter("lr", ":0.1e"),
            "update_s": AverageMeter("updt_s", ":.3f"),
        },
    )


def test_update_policy_accumulates_before_optimizer_step():
    accelerate = pytest.importorskip("accelerate")
    update_policy = import_update_policy()
    accelerator = accelerate.Accelerator(cpu=True, gradient_accumulation_steps=2)
    policy = TinyPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    scheduler = StepCounter()
    policy, optimizer = accelerator.prepare(policy, optimizer)
    tracker = make_tracker()

    batch = {"x": torch.ones(2), "y": torch.zeros(2)}
    initial_weight = accelerator.unwrap_model(policy).weight.detach().clone()

    with accelerator.accumulate(policy):
        update_policy(
            tracker,
            policy,
            batch,
            optimizer,
            grad_clip_norm=0.0,
            accelerator=accelerator,
            lr_scheduler=scheduler,
        )

    assert scheduler.steps == 0
    assert torch.equal(accelerator.unwrap_model(policy).weight.detach(), initial_weight)

    with accelerator.accumulate(policy):
        update_policy(
            tracker,
            policy,
            batch,
            optimizer,
            grad_clip_norm=0.0,
            accelerator=accelerator,
            lr_scheduler=scheduler,
        )

    assert scheduler.steps == 1
    assert accelerator.unwrap_model(policy).weight.item() < initial_weight.item()
