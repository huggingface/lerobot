#!/usr/bin/env python

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

from safetensors.torch import load_file
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lerobot.common.logger import TRAINING_STATE
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.utils.random_utils import deserialize_rng_state
from lerobot.configs.train import TrainPipelineConfig


def make_optimizer_and_scheduler(
    cfg: TrainPipelineConfig, policy: PreTrainedPolicy
) -> tuple[Optimizer, LRScheduler | None]:
    """Generates the optimizer and scheduler based on configs.

    Args:
        cfg (TrainPipelineConfig): The training config that contains optimizer and scheduler configs
        policy (PreTrainedPolicy): The policy config from which parameters and presets must be taken from.

    Returns:
        tuple[Optimizer, LRScheduler | None]: The couple (Optimizer, Scheduler). Scheduler can be `None`.
    """
    params = policy.get_optim_params() if cfg.use_policy_training_preset else policy.parameters()
    optimizer = cfg.optimizer.build(params)
    lr_scheduler = cfg.scheduler.build(optimizer, cfg.offline.steps) if cfg.scheduler is not None else None
    return optimizer, lr_scheduler


def load_training_state(checkpoint_dir: Path, optimizer: Optimizer, scheduler: LRScheduler | None) -> int:
    """
    Given the checkpoint directory, load the optimizer state, scheduler state, and random state, and
    return the global training step.
    """
    if not (checkpoint_dir / TRAINING_STATE).is_file():
        raise FileNotFoundError(checkpoint_dir / TRAINING_STATE)

    training_state_dict = load_file(checkpoint_dir / TRAINING_STATE)

    deserialize_rng_state(training_state_dict)
    step = training_state_dict["step"].item()

    optimizer.load_state_dict(training_state_dict["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(training_state_dict["scheduler"])
    elif "scheduler" in training_state_dict:
        raise ValueError("The checkpoint contains a scheduler state_dict, but no LRScheduler was provided.")

    return step, optimizer, scheduler
