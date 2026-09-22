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
"""What `update_policy` does when the fp16 scaler skips an optimizer update.

The numerical work is accelerate's; what LeRobot owns is the bookkeeping around a skipped step,
which is what this pins. A stub accelerator (rather than a real one) keeps it in the CPU lane:
accelerate builds no GradScaler on CPU, so the skipped-step branch is otherwise unreachable
without a GPU. The real end-to-end behaviour is covered by the multigpu suite.
"""

from contextlib import nullcontext

import pytest
import torch

# Importing the training script pulls in the dataset stack; the fast-test tiers that do not
# install `lerobot[dataset]` skip this module rather than fail collection.
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.scripts.lerobot_train import update_policy  # noqa: E402
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker  # noqa: E402
from tests.fixtures.dummy_checkpoint_policy import make_dummy_policy  # noqa: E402


class StubAccelerator:
    """The `update_policy` surface, with the scaler outcome under the test's control."""

    def __init__(self, *, step_was_skipped: bool, scale: float | None = 256.0):
        self.sync_gradients = True
        self.optimizer_step_was_skipped = step_was_skipped
        self.scaler = None if scale is None else _StubScaler(scale)

    def accumulate(self, _model):
        return nullcontext()

    def autocast(self):
        return nullcontext()

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, parameters, max_norm):
        # Mirrors the real return on an overflowed step: the norm of inf gradients.
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def unwrap_model(self, model, keep_fp32_wrapper: bool = True):
        return model


class _StubScaler:
    def __init__(self, scale: float):
        self._scale = scale

    def get_scale(self) -> float:
        return self._scale


class UpdatableDummyPolicy(torch.nn.Module):
    """Wraps the dummy policy with the `update()` hook real policies use for EMA/targets."""

    def __init__(self):
        super().__init__()
        self.inner = make_dummy_policy()
        self.updates = 0

    def forward(self, batch):
        return self.inner.forward(batch)

    def update(self):
        self.updates += 1


def _tracker(*, with_scale: bool) -> MetricsTracker:
    metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "gpu_mem_gb": AverageMeter("mem_gb", ":.2f", reduction="max"),
    }
    if with_scale:
        metrics["grad_scale"] = AverageMeter("scale", ":.0f")
    return MetricsTracker(1, 1, 1, metrics, initial_step=0)


def _run(*, step_was_skipped: bool, make_grads_nonfinite: bool = False):
    policy = UpdatableDummyPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    accelerator = StubAccelerator(step_was_skipped=step_was_skipped)
    batch = {"observation.state": torch.randn(2, 4)}
    if make_grads_nonfinite:
        batch["observation.state"][0, 0] = float("inf")

    tracker, _ = update_policy(
        _tracker(with_scale=True),
        policy,
        batch,
        optimizer,
        grad_clip_norm=1.0,
        accelerator=accelerator,
        lr_scheduler=scheduler,
    )
    return policy, scheduler, tracker


def test_skipped_update_does_not_advance_policy_state():
    policy, _, _ = _run(step_was_skipped=True, make_grads_nonfinite=True)
    # EMA shadows and target networks must track applied updates, not attempted ones.
    assert policy.updates == 0


def test_applied_update_advances_policy_state():
    policy, _, _ = _run(step_was_skipped=False)
    assert policy.updates == 1


@pytest.mark.parametrize("step_was_skipped", [True, False])
def test_scheduler_keeps_its_micro_batch_cadence(step_was_skipped):
    """LeRobot's schedule is a function of micro-batches consumed, so it is deliberately NOT
    gated on the scaler — under gradient accumulation it already advances on micro-batches that
    apply no update at all."""
    _, scheduler, _ = _run(step_was_skipped=step_was_skipped, make_grads_nonfinite=step_was_skipped)
    assert scheduler.last_epoch == 1


def test_skipped_update_keeps_the_inf_norm_out_of_the_metrics():
    """A skipped step's norm is inf by construction; averaging it in would destroy the window."""
    _, _, skipped = _run(step_was_skipped=True, make_grads_nonfinite=True)
    _, _, applied = _run(step_was_skipped=False)

    assert skipped.grad_norm.count == 0
    assert applied.grad_norm.count == 1
    # The loss scale is reported whenever a scaler exists, skipped step or not — a falling
    # scale is how a user sees that updates are being discarded.
    assert skipped.grad_scale.count == 1
    assert applied.grad_scale.val == 256.0
