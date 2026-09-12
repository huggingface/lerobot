#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RLTActorCriticConfig:
    rl_token_dim: int
    proprio_dim: int
    action_dim: int
    chunk_length: int = 10
    hidden_dim: int = 256
    hidden_layers: int = 2
    fixed_std: float = 0.05
    reference_dropout: float = 0.5
    num_critics: int = 2

    def __post_init__(self) -> None:
        for name in ("rl_token_dim", "proprio_dim", "action_dim", "chunk_length"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.hidden_dim <= 0 or self.hidden_layers <= 0:
            raise ValueError("hidden_dim and hidden_layers must be positive")
        if not 0.0 <= self.reference_dropout <= 1.0:
            raise ValueError("reference_dropout must be in [0, 1]")
        if self.fixed_std < 0.0:
            raise ValueError("fixed_std must be non-negative")
        if self.num_critics < 2:
            raise ValueError("RLT requires at least two critics")

    @property
    def state_dim(self) -> int:
        return self.rl_token_dim + self.proprio_dim


@dataclass(frozen=True)
class RLTOnlineConfig:
    actor_critic: RLTActorCriticConfig
    discount: float = 0.99
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    tau: float = 0.005
    reference_regularization: float = 1.0
    batch_size: int = 256
    utd_ratio: int = 5
    critic_updates_per_actor: int = 2
    replay_capacity: int = 200_000
    expert_replay_capacity: int = 100_000
    online_sample_ratio: float = 0.5
    stride: int = 2
    warmup_env_steps: int = 2_000
    total_env_steps: int = 100_000
    max_episode_steps: int = 400
    seed: int = 0
    device: str = "cuda"

    def __post_init__(self) -> None:
        if not 0.0 < self.discount <= 1.0:
            raise ValueError("discount must be in (0, 1]")
        if self.stride <= 0 or self.actor_critic.chunk_length % self.stride != 0:
            raise ValueError("stride must be a positive divisor of chunk_length")
        if self.batch_size <= 0 or self.utd_ratio < 0:
            raise ValueError("batch_size must be positive and utd_ratio non-negative")
        if self.critic_updates_per_actor <= 0:
            raise ValueError("critic_updates_per_actor must be positive")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be at least batch_size")
        if self.expert_replay_capacity <= 0:
            raise ValueError("expert_replay_capacity must be positive")
        if not 0.0 <= self.online_sample_ratio <= 1.0:
            raise ValueError("online_sample_ratio must be in [0, 1]")
        if self.warmup_env_steps < 0 or self.total_env_steps <= 0:
            raise ValueError("invalid environment step budget")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if self.actor_lr <= 0.0 or self.critic_lr <= 0.0:
            raise ValueError("actor_lr and critic_lr must be positive")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")
        if self.reference_regularization < 0.0:
            raise ValueError("reference_regularization must be non-negative")
