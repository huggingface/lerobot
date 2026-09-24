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
from copy import deepcopy

import pytest
import torch
from packaging.version import Version
from torch.optim.lr_scheduler import LambdaLR

from lerobot.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
    DiffuserSchedulerConfig,
    VQBeTSchedulerConfig,
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


@pytest.mark.parametrize("vqvae_steps, warmup_steps", [(20, 10), (0, 10), (20, 0), (0, 0)])
def test_vqbet_scheduler_decay_finishes_with_training(optimizer, vqvae_steps, warmup_steps):
    config = VQBeTSchedulerConfig(num_warmup_steps=warmup_steps, num_vqvae_training_steps=vqvae_steps)
    total_steps = 100
    scheduler = config.build(optimizer, num_training_steps=total_steps)
    peak_lr = scheduler.base_lrs[0]
    rates = [scheduler.get_last_lr()[0]]
    for _ in range(total_steps):
        optimizer.step()
        scheduler.step()
        rates.append(scheduler.get_last_lr()[0])

    assert rates[:vqvae_steps] == pytest.approx([peak_lr] * vqvae_steps)
    if warmup_steps:
        assert rates[vqvae_steps] == pytest.approx(0.0)
        assert rates[vqvae_steps + warmup_steps // 2] == pytest.approx(peak_lr / 2)
    decay_start = vqvae_steps + warmup_steps
    assert rates[decay_start] == pytest.approx(peak_lr)
    assert rates[(decay_start + total_steps) // 2] == pytest.approx(peak_lr / 2)
    assert rates[total_steps] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "vqvae_steps, warmup_steps, total_steps, expected",
    [(0, 0, 1, [1, 0]), (2, 1, 4, [1, 1, 0, 1, 0]), (2, 1, 2, [1, 1, 0]), (2, 1, 3, [1, 1, 0, 1])],
)
def test_vqbet_scheduler_short_training(optimizer, vqvae_steps, warmup_steps, total_steps, expected):
    config = VQBeTSchedulerConfig(num_warmup_steps=warmup_steps, num_vqvae_training_steps=vqvae_steps)
    scheduler = config.build(optimizer, num_training_steps=total_steps)
    peak_lr = scheduler.base_lrs[0]
    rates = [scheduler.get_last_lr()[0] / peak_lr]
    for _ in range(total_steps):
        optimizer.step()
        scheduler.step()
        rates.append(scheduler.get_last_lr()[0] / peak_lr)
    assert rates == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("num_cycles, expected", [(0, [1, 1, 1]), (0.5, [1, 0.5, 0]), (1, [1, 0, 1])])
def test_vqbet_scheduler_cycle_count(optimizer, num_cycles, expected):
    config = VQBeTSchedulerConfig(num_warmup_steps=10, num_vqvae_training_steps=20, num_cycles=num_cycles)
    scheduler = config.build(optimizer, num_training_steps=100)
    peak_lr = scheduler.base_lrs[0]
    rates = []
    for step in range(1, 101):
        optimizer.step()
        scheduler.step()
        if step in (30, 65, 100):
            rates.append(scheduler.get_last_lr()[0] / peak_lr)
    assert rates == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("resume_step", [10, 25, 60])
def test_vqbet_scheduler_checkpoint_continuation(optimizer, tmp_path, resume_step):
    config = VQBeTSchedulerConfig(num_warmup_steps=10, num_vqvae_training_steps=20)
    scheduler = config.build(optimizer, num_training_steps=100)
    for _ in range(resume_step):
        optimizer.step()
        scheduler.step()
    optimizer_state = deepcopy(optimizer.state_dict())
    save_scheduler_state(scheduler, tmp_path)

    expected = []
    for _ in range(resume_step, 100):
        optimizer.step()
        scheduler.step()
        expected.append(scheduler.get_last_lr()[0])

    restored = config.build(optimizer, num_training_steps=100)
    optimizer.load_state_dict(optimizer_state)
    load_scheduler_state(restored, tmp_path)
    assert restored.get_last_lr()[0] == optimizer.param_groups[0]["lr"]
    actual = []
    for _ in range(resume_step, 100):
        optimizer.step()
        restored.step()
        actual.append(restored.get_last_lr()[0])
    assert actual == expected
    assert actual[-1] == pytest.approx(0.0, abs=1e-12)


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


def test_save_scheduler_state(scheduler, tmp_path):
    save_scheduler_state(scheduler, tmp_path)
    assert (tmp_path / SCHEDULER_STATE).is_file()


def test_save_load_scheduler_state(scheduler, tmp_path):
    save_scheduler_state(scheduler, tmp_path)
    loaded_scheduler = load_scheduler_state(scheduler, tmp_path)

    assert scheduler.state_dict() == loaded_scheduler.state_dict()
