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

    Args:
        actor_lr (`float`, *optional*, defaults to 0.0003):
            Learning rate for the actor network.
        critic_lr (`float`, *optional*, defaults to 0.0003):
            Learning rate for the critic network.
        temperature_lr (`float`, *optional*, defaults to 0.0003):
            Learning rate for the temperature parameter.
        discount (`float`, *optional*, defaults to 0.99):
            Discount factor for the Bellman update.
        use_backup_entropy (`bool`, *optional*, defaults to `True`):
            Whether to use backup entropy in the Bellman target.
        critic_target_update_weight (`float`, *optional*, defaults to 0.005):
            Polyak-averaging weight for the critic target update.
        num_critics (`int`, *optional*, defaults to 2):
            Number of critics in the ensemble.
        num_subsample_critics (`int | None`, *optional*):
            Number of critics to subsample from the ensemble for each Bellman target computation.
            `None` uses the full ensemble.
        critic_network_kwargs (`CriticNetworkConfig`, *optional*):
            Configuration for the (continuous-action) critic network architecture.
        discrete_critic_network_kwargs (`CriticNetworkConfig`, *optional*):
            Configuration for the discrete-action critic network architecture.
        temperature_init (`float`, *optional*, defaults to 1.0):
            Initial value of the entropy temperature.
        target_entropy (`float | None`, *optional*):
            Target entropy for automatic temperature tuning. If `None`, defaults to `-|A|/2` where
            `|A|` is the total action dimension (continuous + 1 if there is a discrete action head).
        utd_ratio (`int`, *optional*, defaults to 1):
            Update-to-data ratio. Set to `>1` to enable extra critic updates per env step.
        policy_update_freq (`int`, *optional*, defaults to 1):
            Frequency of policy updates, in units of critic updates.
        grad_clip_norm (`float`, *optional*, defaults to 40.0):
            Gradient-clipping norm applied during optimization.
        use_torch_compile (`bool`, *optional*, defaults to `False`):
            Whether to `torch.compile` the algorithm's forward passes. Currently disabled by default.
        policy_config (`PreTrainedConfig | None`, *optional*):
            The policy (actor) config this algorithm trains. Populated via `from_policy_config` or by
            `TrainRLServerPipelineConfig.validate` before the algorithm is constructed.
    """

    # Optimizer learning rates
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temperature_lr: float = 3e-4

    # Bellman update
    discount: float = 0.99
    use_backup_entropy: bool = True
    critic_target_update_weight: float = 0.005

    # Critic ensemble
    num_critics: int = 2
    num_subsample_critics: int | None = None
    critic_network_kwargs: CriticNetworkConfig = field(default_factory=CriticNetworkConfig)
    discrete_critic_network_kwargs: CriticNetworkConfig = field(default_factory=CriticNetworkConfig)

    # Temperature / entropy
    temperature_init: float = 1.0
    target_entropy: float | None = None

    # Update loop
    utd_ratio: int = 1
    policy_update_freq: int = 1
    grad_clip_norm: float = 40.0

    # Optimizations
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
