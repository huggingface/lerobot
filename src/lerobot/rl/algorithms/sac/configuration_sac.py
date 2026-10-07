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

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.gaussian_actor.configuration_gaussian_actor import (
    CriticNetworkConfig,
    GaussianActorConfig,
)

from ..configs import RLAlgorithmConfig


@RLAlgorithmConfig.register_subclass("sac")
@dataclass
class SACAlgorithmConfig(RLAlgorithmConfig):
    """Soft Actor-Critic (SAC) algorithm configuration.

    SAC is an off-policy actor-critic deep RL algorithm based on the maximum
    entropy reinforcement learning framework. It learns a policy and a Q-function
    simultaneously using experience collected from the environment.

    This configuration class contains the algorithm-side hyperparameters: critic
    ensemble, target networks, temperature / entropy tuning, and the Bellman
    update loop. The policy-side (actor + observation encoder) lives in
    :class:`~lerobot.policies.gaussian_actor.GaussianActorConfig` and is
    referenced via :attr:`policy_config`.
    """

    # Optimizer learning rates
    # Learning rate for the actor network
    actor_lr: float = 3e-4
    # Learning rate for the critic network
    critic_lr: float = 3e-4
    # Learning rate for the temperature parameter
    temperature_lr: float = 3e-4

    # Bellman update
    # Discount factor for the SAC algorithm
    discount: float = 0.99
    # Whether to use backup entropy for the SAC algorithm
    use_backup_entropy: bool = True
    # Weight for the critic target update
    critic_target_update_weight: float = 0.005

    # Critic ensemble
    # Number of critics in the ensemble
    num_critics: int = 2
    # Number of subsampled critics for training
    num_subsample_critics: int | None = None
    # Configuration for the critic network architecture
    critic_network_kwargs: CriticNetworkConfig = field(default_factory=CriticNetworkConfig)
    # Configuration for the discrete critic network
    discrete_critic_network_kwargs: CriticNetworkConfig = field(default_factory=CriticNetworkConfig)

    # Temperature / entropy
    # Initial temperature value
    temperature_init: float = 1.0
    # Target entropy for automatic temperature tuning. If ``None``, defaults to
    # ``-|A|/2`` where ``|A|`` is the total action dimension (continuous + 1 if
    # there is a discrete action head).
    target_entropy: float | None = None

    # Update loop
    # Update-to-data ratio. Set to >1 to enable extra critic updates per env step.
    utd_ratio: int = 1
    # Frequency of policy updates
    policy_update_freq: int = 1
    # Gradient clipping norm for the SAC algorithm
    grad_clip_norm: float = 40.0

    # Optimizations
    # torch.compile is currently disabled by default
    use_torch_compile: bool = False

    # Policy config
    policy_config: PreTrainedConfig | None = None

    @classmethod
    def from_policy_config(cls, policy_cfg: GaussianActorConfig) -> SACAlgorithmConfig:
        """Build an algorithm config with default hyperparameters for a given policy."""
        return cls(
            policy_config=policy_cfg,
            discrete_critic_network_kwargs=policy_cfg.discrete_critic_network_kwargs,
        )
