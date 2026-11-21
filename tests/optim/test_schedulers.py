# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from pathlib import Path

import pytest
import torch
from packaging.version import Version
from torch.optim.lr_scheduler import LambdaLR

from lerobot.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
    DiffuserSchedulerConfig,
    VQBeTSchedulerConfig,
    WarmupStableDecaySchedulerConfig,
    load_scheduler_state,
    save_scheduler_state,
)
from lerobot.utils.constants import SCHEDULER_STATE
from lerobot.utils.import_utils import is_package_available


@pytest.mark.skipif(not is_package_available("diffusers"), reason="diffusers not installed")
def test_diffuser_scheduler(optimizer):
    config = DiffuserSchedulerConfig(name="cosine", num_warmup_steps=5)
    scheduler = config.build(optimizer, num_training_steps=100)
    assert isinstance(scheduler, LambdaLR)

    optimizer.step()  # so that we don't get torch warning
    scheduler.step()
    expected_state_dict = {
        "_get_lr_called_within_step": False,
        "_last_lr": [0.0002],
        "_step_count": 2,
        "base_lrs": [0.001],
        "last_epoch": 1,
        "lr_lambdas": [None],
    }

    if Version(torch.__version__) >= Version("2.8"):
        expected_state_dict["_is_initial"] = False

    assert scheduler.state_dict() == expected_state_dict


def test_vqbet_scheduler(optimizer):
    config = VQBeTSchedulerConfig(num_warmup_steps=10, num_vqvae_training_steps=20, num_cycles=0.5)
    scheduler = config.build(optimizer, num_training_steps=100)
    assert isinstance(scheduler, LambdaLR)

    optimizer.step()
    scheduler.step()
    expected_state_dict = {
        "_get_lr_called_within_step": False,
        "_last_lr": [0.001],
        "_step_count": 2,
        "base_lrs": [0.001],
        "last_epoch": 1,
        "lr_lambdas": [None],
    }

    if Version(torch.__version__) >= Version("2.8"):
        expected_state_dict["_is_initial"] = False

    assert scheduler.state_dict() == expected_state_dict


def test_cosine_decay_with_warmup_scheduler(optimizer):
    config = CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=10, num_decay_steps=90, peak_lr=0.01, decay_lr=0.001
    )
    scheduler = config.build(optimizer, num_training_steps=100)
    assert isinstance(scheduler, LambdaLR)

    optimizer.step()
    scheduler.step()
    expected_state_dict = {
        "_get_lr_called_within_step": False,
        "_last_lr": [0.0001818181818181819],
        "_step_count": 2,
        "base_lrs": [0.001],
        "last_epoch": 1,
        "lr_lambdas": [None],
    }

    if Version(torch.__version__) >= Version("2.8"):
        expected_state_dict["_is_initial"] = False

    assert scheduler.state_dict() == expected_state_dict


@pytest.mark.parametrize(
    "warmup_steps,decay_steps,training_steps,effective_warmup,effective_end",
    [
        (50, 100, 100, 50, 100),
        (100, 200, 100, 50, 100),  # Scale both phases for a shorter run.
        (0, 100, 100, 0, 100),
        (1, 1000, 10, 0, 10),  # Scaling rounds warmup down to zero.
        (1, 2, 2, 1, 2),  # A single decay step.
        (50, 100, 200, 50, 100),  # Hold the floor after the configured end.
    ],
)
def test_cosine_decay_warmup_boundaries(
    warmup_steps: int, decay_steps: int, training_steps: int, effective_warmup: int, effective_end: int
) -> None:
    peak_lr, decay_lr = 1e-4, 1e-5
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=peak_lr)
    config = CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=warmup_steps,
        num_decay_steps=decay_steps,
        peak_lr=peak_lr,
        decay_lr=decay_lr,
    )
    scheduler = config.build(optimizer, num_training_steps=training_steps)
    rates = [scheduler.get_last_lr()[0]]
    for _ in range(training_steps + 2):
        optimizer.step()
        scheduler.step()
        rates.append(scheduler.get_last_lr()[0])

    assert rates[0] == pytest.approx(peak_lr / (effective_warmup + 1))
    assert rates[effective_warmup] == pytest.approx(peak_lr)
    assert rates[effective_end:] == pytest.approx([decay_lr] * len(rates[effective_end:]))
    assert all(a <= b for a, b in zip(rates[:effective_warmup], rates[1 : effective_warmup + 1], strict=True))
    assert all(a >= b for a, b in zip(rates[effective_warmup:-1], rates[effective_warmup + 1 :], strict=True))
    duration = effective_end - effective_warmup
    if duration % 2 == 0:
        assert rates[effective_warmup + duration // 2] == pytest.approx((peak_lr + decay_lr) / 2)


@pytest.mark.parametrize("resume_step", [25, 50, 75, 100])
def test_cosine_decay_resume_preserves_group_rates(resume_step: int, tmp_path: Path) -> None:
    config = CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=50, num_decay_steps=100, peak_lr=1e-4, decay_lr=1e-5
    )

    def make_optimizer() -> torch.optim.SGD:
        return torch.optim.SGD(
            [
                {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 1e-4},
                {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": 1e-5},
            ]
        )

    optimizer = make_optimizer()
    scheduler = config.build(optimizer, num_training_steps=100)
    for _ in range(resume_step):
        optimizer.step()
        scheduler.step()
    save_scheduler_state(scheduler, tmp_path)

    restored_optimizer = make_optimizer()
    restored_scheduler = config.build(restored_optimizer, num_training_steps=100)
    restored_optimizer.load_state_dict(optimizer.state_dict())
    load_scheduler_state(restored_scheduler, tmp_path)
    for _ in range(105 - resume_step):
        optimizer.step()
        scheduler.step()
        restored_optimizer.step()
        restored_scheduler.step()
        assert restored_scheduler.get_last_lr() == pytest.approx(scheduler.get_last_lr())
        assert restored_scheduler.get_last_lr()[1] == pytest.approx(scheduler.get_last_lr()[0] * 0.1)


@pytest.mark.parametrize("warmup_steps,decay_steps,training_steps", [(10, 10, 10), (11, 10, 10), (1, 10, 0)])
def test_cosine_decay_requires_decay_interval(
    warmup_steps: int, decay_steps: int, training_steps: int
) -> None:
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-4)
    config = CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=warmup_steps, num_decay_steps=decay_steps, peak_lr=1e-4, decay_lr=1e-5
    )
    with pytest.raises(ValueError, match="at least one step after warmup"):
        config.build(optimizer, num_training_steps=training_steps)


def test_warmup_stable_decay_scheduler(optimizer):
    config = WarmupStableDecaySchedulerConfig(
        num_warmup_steps=10, num_decay_steps=80, num_stable_steps=10, decay_type="linear"
    )
    scheduler = config.build(optimizer, num_training_steps=100)
    assert isinstance(scheduler, LambdaLR)

    optimizer.step()
    scheduler.step()
    expected_state_dict = {
        "_get_lr_called_within_step": False,
        "_last_lr": [0.0001],
        "_step_count": 2,
        "base_lrs": [0.001],
        "last_epoch": 1,
        "lr_lambdas": [None],
    }

    if Version(torch.__version__) >= Version("2.8"):
        expected_state_dict["_is_initial"] = False

    assert scheduler.state_dict() == expected_state_dict


def test_save_scheduler_state(scheduler, tmp_path):
    save_scheduler_state(scheduler, tmp_path)
    assert (tmp_path / SCHEDULER_STATE).is_file()


def test_save_load_scheduler_state(scheduler, tmp_path):
    save_scheduler_state(scheduler, tmp_path)
    loaded_scheduler = load_scheduler_state(scheduler, tmp_path)

    assert scheduler.state_dict() == loaded_scheduler.state_dict()
