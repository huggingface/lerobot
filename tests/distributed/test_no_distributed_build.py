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
"""A torch built without distributed support (`USE_DISTRIBUTED=0`, e.g. the NVIDIA JetPack
wheels) ships a `torch.distributed` stub: `is_available()` is False, `is_initialized` and
`get_rank` do not exist, and `torch.distributed.fsdp` does not import. The single-process
code paths must keep working there."""

import builtins
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.common import train_utils
from lerobot.configs.parallelism import ParallelismConfig
from lerobot.distributed import (
    checkpoint as dist_checkpoint,
    parallel_dims as parallel_dims_module,
    utils as dist_utils,
)
from lerobot.distributed.parallel_dims import ParallelDims
from lerobot.optim.ema import EMA, ConstantEMAConfig
from lerobot.utils import logging_utils
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker

STUB = SimpleNamespace(is_available=lambda: False)  # no is_initialized, no get_rank


@pytest.fixture
def no_distributed(monkeypatch):
    """Mimic a torch without distributed support everywhere lerobot looks."""
    monkeypatch.setattr(torch.distributed, "is_available", lambda: False)
    for module in (dist_utils, parallel_dims_module, train_utils, logging_utils):
        monkeypatch.setattr(module, "dist", STUB)

    real_import = builtins.__import__

    def refuse_fsdp(name, *args, **kwargs):
        if name.startswith("torch.distributed.fsdp") or name.startswith("torch.distributed.checkpoint"):
            raise ModuleNotFoundError("No module named 'torch._C._distributed_c10d'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse_fsdp)


def test_is_main_process(no_distributed):
    assert dist_utils.is_main_process() is True
    assert dist_utils.is_distributed_initialized() is False


def test_dp_rank(no_distributed):
    cfg = ParallelismConfig()
    cfg.resolve(1)
    assert ParallelDims.from_config(cfg, world_size=1, device_type="cpu").dp_rank == 0


def test_full_state_dict_of_a_plain_module(no_distributed):
    model = nn.Linear(2, 2)
    assert dist_checkpoint.is_sharded_module(model) is False
    assert dist_checkpoint.full_model_state_dict(model).keys() == model.state_dict().keys()


@pytest.mark.parametrize("exceptional_exit", [False, True])
def test_ema_average_parameters_swaps_and_restores(no_distributed, exceptional_exit):
    model = nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.fill_(2)
        model.bias.fill_(3)
    ema = EMA(model, ConstantEMAConfig())
    with torch.no_grad():
        model.weight.fill_(7)
        model.bias.fill_(11)

    def evaluate():
        with ema.average_parameters(model):
            torch.testing.assert_close(model.weight, torch.full_like(model.weight, 2))
            torch.testing.assert_close(model.bias, torch.full_like(model.bias, 3))
            if exceptional_exit:
                raise RuntimeError("evaluation failed")

    if exceptional_exit:
        with pytest.raises(RuntimeError, match="evaluation failed"):
            evaluate()
    else:
        evaluate()
    torch.testing.assert_close(model.weight, torch.full_like(model.weight, 7))
    torch.testing.assert_close(model.bias, torch.full_like(model.bias, 11))


def test_metrics_reduce_is_a_no_op(no_distributed):
    tracker = MetricsTracker(
        batch_size=1, num_frames=1, num_episodes=1, metrics={"loss": AverageMeter("loss", ":.3f")}
    )
    tracker.loss = 1.0
    tracker.reduce_across_ranks()
    assert tracker.loss.avg == 1.0
