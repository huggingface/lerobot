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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.flow_actor.configuration_flow_actor import FlowActorConfig

from ..configs import RLAlgorithmConfig


@RLAlgorithmConfig.register_subclass("qf3")
@dataclass
class QF3AlgorithmConfig(RLAlgorithmConfig):
    """QF3 hyperparameters for a state-only flow actor.

    ``alpha`` and ``lambda_cfm`` default to the shared MuJoCo setting named in the
    analysis section (1 and 0.3). Table ``tab:mujoco-hyperparameters`` selects
    ``(1, 0.2)`` for Hopper. The actor learning rate 3e-4 is the default named
    in that section. ``polyak_tau``, ``policy_noise``, ``noise_clip``, and
    ``policy_frequency`` are CleanRL TD3 defaults. The paper says the MuJoCo critic
    matches that baseline and does not restate those four numbers. ``gamma`` is not
    restated for MuJoCo either. The Robomimic table uses 0.99 on Square.

    This config does not add ``lambda_base``. That anchor is only in the fine-tuning
    experiments. The critic is a scalar twin Q, not the locomotion distributional critic
    and not the Robomimic ensemble of 10.
    """

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha: float = 1.0
    lambda_cfm: float = 0.3
    gamma: float = 0.99
    polyak_tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_frequency: int = 2
    # True is the velocity clip in eq:actor-loss. False is the direct-Q ablation:
    # the same objective with that clip removed. The MuJoCo figure does not plot
    # this ablation. The Humanoid ablation does.
    clip_velocity: bool = True
    euler_steps: int = 5
    # Critic width. The paper's MuJoCo critic is 256. The flow actor width lives on
    # FlowActorConfig.
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    policy_config: PreTrainedConfig | None = None

    @classmethod
    def from_policy_config(cls, policy_cfg: Any) -> QF3AlgorithmConfig:
        if not isinstance(policy_cfg, FlowActorConfig):
            raise TypeError(f"QF3 requires a flow_actor policy config, got {type(policy_cfg).__name__}.")
        return cls(policy_config=policy_cfg, euler_steps=policy_cfg.euler_steps)
