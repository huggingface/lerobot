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


from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies import PreTrainedPolicy


def make_optimizer_and_scheduler(
    cfg: TrainPipelineConfig, policy: PreTrainedPolicy
) -> tuple[Optimizer, LRScheduler | None]:
    """Build the optimizer and, if configured, the learning rate scheduler for training a policy.

    Args:
        cfg (`TrainPipelineConfig`):
            The training config, whose `optimizer` and `scheduler` fields are built.
        policy (`PreTrainedPolicy`):
            The policy being trained; its parameters (or optimizer-preset groups, if
            `cfg.use_policy_training_preset` is `True`) are passed to the optimizer.

    Returns:
        `tuple[Optimizer, LRScheduler | None]`: The built optimizer, and scheduler if one was configured.

    Raises:
        ValueError: If `cfg.optimizer` is `None`.
    """
    params = policy.get_optim_params() if cfg.use_policy_training_preset else policy.parameters()
    if cfg.optimizer is None:
        raise ValueError("Optimizer config is required but not provided in TrainPipelineConfig")
    optimizer = cfg.optimizer.build(params)
    lr_scheduler = cfg.scheduler.build(optimizer, cfg.steps) if cfg.scheduler is not None else None
    return optimizer, lr_scheduler
