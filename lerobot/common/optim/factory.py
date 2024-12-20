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

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lerobot.common.policies import Policy
from lerobot.configs.default import MainConfig


def make_optimizer_and_scheduler(cfg: MainConfig, policy: Policy) -> tuple[Optimizer, LRScheduler | None]:
    params = policy.get_optim_params() if cfg.use_policy_optimizer_preset else policy.parameters()
    optimizer = cfg.optimizer.build(params)

    if not cfg.use_policy_optimizer_preset:
        lr_scheduler = None
        return optimizer, lr_scheduler

    if cfg.policy.type == "act":
        lr_scheduler = None
    elif cfg.policy.type == "diffusion":
        from diffusers.optimization import get_scheduler

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=cfg.training.offline_steps,
        )
    elif cfg.policy.type == "tdmpc":
        optimizer = torch.optim.Adam(params, cfg.training.lr)
        lr_scheduler = None
    elif cfg.policy.type == "vqbet":
        # from lerobot.common.policies.vqbet.modeling_vqbet import VQBeTOptimizer, VQBeTScheduler
        # optimizer = VQBeTOptimizer(policy, cfg)
        optimizer = torch.optim.Adam(
            params,
            cfg.optimizer.lr,
            cfg.optimizer.adam_betas,
            cfg.optimizer.adam_eps,
        )
        from lerobot.common.policies.vqbet.modeling_vqbet import VQBeTScheduler

        lr_scheduler = VQBeTScheduler(optimizer, cfg)
    else:
        raise NotImplementedError()

    return optimizer, lr_scheduler
