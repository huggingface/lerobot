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
"""RLT algorithm configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from lerobot.rl.algorithms.base import RLAlgorithmConfig

if TYPE_CHECKING:
    from lerobot.rl.algorithms.rlt.rlt_algorithm import RLTAlgorithm


@RLAlgorithmConfig.register_subclass("rlt")
@dataclass
class RLTAlgorithmConfig(RLAlgorithmConfig):
    """RLT-specific hyper-parameters that control the update loop."""

    # ── Action chunks ──
    chunk_size: int = 10
    chunk_stride: int = 2

    # ── Update cadence ──
    utd_ratio: int = 5
    policy_update_freq: int = 2
    clip_grad_norm: float = 10.0

    # ── Learning rates ──
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    rl_token_lr: float = 1e-4

    # ── TD learning ──
    discount: float = 0.99
    tau: float = 0.005
    num_critics: int = 2

    # ── Policy constraint (paper Eq. 5) ──
    bc_reg_coeff: float = 0.1
    ref_dropout: float = 0.5

    # ── Offline RL-token training ──
    vla_finetune_weight: float = 0.0

    @classmethod
    def from_policy_config(cls, policy_cfg) -> RLTAlgorithmConfig:
        """Build from an existing ``RLTConfig`` (cfg.policy)."""
        return cls(
            chunk_size=policy_cfg.chunk_size,
            chunk_stride=policy_cfg.chunk_stride,
            utd_ratio=policy_cfg.utd_ratio,
            policy_update_freq=policy_cfg.policy_update_freq,
            clip_grad_norm=policy_cfg.clip_grad_norm,
            actor_lr=policy_cfg.actor_lr,
            critic_lr=policy_cfg.critic_lr,
            rl_token_lr=policy_cfg.rl_token_lr,
            discount=policy_cfg.discount,
            tau=policy_cfg.tau,
            num_critics=policy_cfg.num_critics,
            bc_reg_coeff=policy_cfg.bc_reg_coeff,
            ref_dropout=policy_cfg.ref_dropout,
            vla_finetune_weight=policy_cfg.vla_finetune_weight,
        )

    def build_algorithm(self, policy: torch.nn.Module) -> RLTAlgorithm:
        from lerobot.rl.algorithms.rlt.rlt_algorithm import RLTAlgorithm

        return RLTAlgorithm(policy=policy, config=self)
