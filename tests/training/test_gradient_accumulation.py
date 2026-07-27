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

from types import SimpleNamespace

import pytest

from lerobot.configs import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.lerobot_train import _compute_samples_per_s, _validate_accelerator_configuration


class MockAccelerator:
    def __init__(self, gradient_accumulation_steps: int, *, sync_with_dataloader: bool):
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_state = SimpleNamespace(sync_with_dataloader=sync_with_dataloader)


@pytest.mark.parametrize("gradient_accumulation_steps", [1, 4])
def test_accelerator_configuration_accepts_matching_steps(gradient_accumulation_steps):
    accelerator = MockAccelerator(
        gradient_accumulation_steps,
        sync_with_dataloader=False,
    )

    assert (
        _validate_accelerator_configuration(accelerator, gradient_accumulation_steps)
        == gradient_accumulation_steps
    )


def test_accelerator_configuration_rejects_mismatched_steps():
    accelerator = MockAccelerator(4, sync_with_dataloader=False)

    with pytest.raises(ValueError, match="These values must match"):
        _validate_accelerator_configuration(accelerator, configured_steps=2)


def test_accelerator_configuration_rejects_dataloader_sync():
    accelerator = MockAccelerator(4, sync_with_dataloader=True)

    with pytest.raises(ValueError, match="sync_with_dataloader=False"):
        _validate_accelerator_configuration(accelerator, configured_steps=4)


def test_samples_per_s_is_unaffected_by_accumulation():
    # Throughput is a property of the hardware, not of how many micro-batches are folded into one
    # optimizer step: accumulating 4 x 32 samples at the same per-micro-batch time processes exactly
    # as many samples per second as a single 32-sample step does.
    baseline = _compute_samples_per_s(32, micro_batch_time_s=0.5, gradient_accumulation_steps=1)

    assert baseline == pytest.approx(64.0)
    assert _compute_samples_per_s(
        32 * 4, micro_batch_time_s=0.5, gradient_accumulation_steps=4
    ) == pytest.approx(baseline)


@pytest.mark.parametrize("gradient_accumulation_steps", [0, -1])
def test_train_config_rejects_invalid_gradient_accumulation_steps(gradient_accumulation_steps):
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="user/repo"),
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    with pytest.raises(ValueError, match="greater than or equal to 1"):
        cfg.validate()
