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


from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies import PreTrainedPolicy


def make_loraplus_param_groups(
    model: torch.nn.Module, lr: float, loraplus_lr_ratio: float
) -> list[dict[str, Any]]:
    """Split a PEFT model's trainable parameters into LoRA+ learning-rate groups.

    LoRA+ (https://arxiv.org/abs/2402.12354) trains the LoRA ``B`` matrices at
    ``lr * loraplus_lr_ratio`` while the ``A`` matrices and any other trainable parameters keep the
    base ``lr``, which can speed up convergence at no extra compute cost.

    Args:
        model: The (PEFT-wrapped) policy whose ``named_parameters`` are grouped.
        lr: Base learning rate used for the A matrices and any other trainable parameters.
        loraplus_lr_ratio: Multiplier applied to ``lr`` for the LoRA B matrices. Must be > 0.

    Returns:
        Parameter groups (with per-group ``lr``) suitable for an optimizer's ``build`` method.
    """
    if loraplus_lr_ratio <= 0:
        raise ValueError(f"`loraplus_lr_ratio` must be > 0, got {loraplus_lr_ratio}.")

    lora_b_params: list[torch.nn.Parameter] = []
    other_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_B" in name or "lora_embedding_B" in name:
            lora_b_params.append(param)
        else:
            other_params.append(param)

    param_groups: list[dict[str, Any]] = []
    if other_params:
        param_groups.append({"params": other_params, "lr": lr})
    if lora_b_params:
        param_groups.append({"params": lora_b_params, "lr": lr * loraplus_lr_ratio})
    return param_groups


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
    if cfg.optimizer is None:
        raise ValueError("Optimizer config is required but not provided in TrainPipelineConfig")

    if cfg.peft is not None and cfg.peft.loraplus_lr_ratio is not None:
        params = make_loraplus_param_groups(policy, cfg.optimizer.lr, cfg.peft.loraplus_lr_ratio)
    else:
        params = policy.get_optim_params() if cfg.use_policy_training_preset else policy.parameters()

    optimizer = cfg.optimizer.build(params)
    lr_scheduler = cfg.scheduler.build(optimizer, cfg.steps) if cfg.scheduler is not None else None
    return optimizer, lr_scheduler
