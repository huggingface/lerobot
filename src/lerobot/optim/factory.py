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


def apply_loraplus_lr_ratio(optimizer: Optimizer, model: torch.nn.Module, loraplus_lr_ratio: float) -> None:
    """Give the LoRA B matrices a higher learning rate (LoRA+) on an already-built optimizer.

    LoRA+ (https://arxiv.org/abs/2402.12354) trains the LoRA ``B`` matrices at
    ``lr * loraplus_lr_ratio`` while the ``A`` matrices keep the base ``lr``, which can speed up
    convergence at no extra compute cost. Rather than replacing the optimizer's inputs, this splits
    the ``B`` parameters out of each existing parameter group into a sibling group with the scaled
    learning rate. It therefore composes with any optimizer, including presets that build their own
    (named-parameter) groups such as ``XVLAAdamWConfig``.

    Args:
        optimizer: An optimizer already built from the model's parameters.
        model: The (PEFT-wrapped) policy, used to identify the LoRA B parameters by name.
        loraplus_lr_ratio: Multiplier applied to each group's ``lr`` for its LoRA B matrices.
            Must be > 0.
    """
    if loraplus_lr_ratio <= 0:
        raise ValueError(f"`loraplus_lr_ratio` must be > 0, got {loraplus_lr_ratio}.")

    lora_b_param_ids = {
        id(param)
        for name, param in model.named_parameters()
        if param.requires_grad and ("lora_B" in name or "lora_embedding_B" in name)
    }
    if not lora_b_param_ids:
        return

    scaled_b_groups: list[dict[str, Any]] = []
    for group in optimizer.param_groups:
        b_params = [p for p in group["params"] if id(p) in lora_b_param_ids]
        if not b_params:
            continue
        other_params = [p for p in group["params"] if id(p) not in lora_b_param_ids]
        if not other_params:
            # The whole group is LoRA B: scale its learning rate in place.
            group["lr"] = group["lr"] * loraplus_lr_ratio
            continue
        # Keep the non-B params in this group and move the B params to a scaled sibling group.
        group["params"] = other_params
        b_group = {key: value for key, value in group.items() if key != "params"}
        b_group["params"] = b_params
        b_group["lr"] = group["lr"] * loraplus_lr_ratio
        scaled_b_groups.append(b_group)

    for b_group in scaled_b_groups:
        optimizer.add_param_group(b_group)


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

    params = policy.get_optim_params() if cfg.use_policy_training_preset else policy.parameters()
    optimizer = cfg.optimizer.build(params)

    if cfg.peft is not None and cfg.peft.loraplus_lr_ratio is not None:
        if not isinstance(optimizer, Optimizer):
            raise ValueError(
                "LoRA+ (`peft.loraplus_lr_ratio`) is not supported with multi-optimizer configs."
            )
        apply_loraplus_lr_ratio(optimizer, policy, cfg.peft.loraplus_lr_ratio)

    lr_scheduler = cfg.scheduler.build(optimizer, cfg.steps) if cfg.scheduler is not None else None
    return optimizer, lr_scheduler
