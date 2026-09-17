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

import copy
from contextlib import contextmanager

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from .configuration_rlt import RLTActorCriticConfig, RLTOnlineConfig


def _mlp(input_dim: int, output_dim: int, hidden_dim: int, layers: int) -> nn.Sequential:
    modules: list[nn.Module] = [nn.LayerNorm(input_dim)]
    current_dim = input_dim
    for _ in range(layers):
        modules.extend([nn.Linear(current_dim, hidden_dim), nn.ReLU()])
        current_dim = hidden_dim
    modules.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*modules)


def _horizon_mask(valid_horizon: Tensor, chunk_length: int, dtype: torch.dtype) -> Tensor:
    steps = torch.arange(chunk_length, device=valid_horizon.device)
    return (steps.unsqueeze(0) < valid_horizon.unsqueeze(1)).to(dtype).unsqueeze(-1)


class GaussianChunkActor(nn.Module):
    """Fixed-variance Gaussian actor conditioned on x=(z_rl, proprio) and the PI0 action."""

    def __init__(self, config: RLTActorCriticConfig) -> None:
        super().__init__()
        self.config = config
        chunk_dim = config.chunk_length * config.action_dim
        self.network = _mlp(
            config.state_dim + chunk_dim,
            chunk_dim,
            config.hidden_dim,
            config.hidden_layers,
        )

    def mean(self, state: Tensor, reference: Tensor) -> Tensor:
        self._validate(state, reference)
        inputs = torch.cat([state, reference.flatten(start_dim=1)], dim=-1)
        return self.network(inputs).reshape(-1, self.config.chunk_length, self.config.action_dim)

    def sample(self, state: Tensor, reference: Tensor, deterministic: bool = False) -> Tensor:
        action = self.mean(state, reference)
        if not deterministic and self.config.fixed_std > 0.0:
            action = action + self.config.fixed_std * torch.randn_like(action)
        return action

    def drop_reference(self, reference: Tensor) -> Tensor:
        if self.config.reference_dropout == 0.0:
            return reference
        keep = torch.rand(reference.shape[0], 1, 1, device=reference.device)
        keep = keep >= self.config.reference_dropout
        return reference * keep.to(reference.dtype)

    def _validate(self, state: Tensor, reference: Tensor) -> None:
        if state.ndim != 2 or state.shape[1] != self.config.state_dim:
            raise ValueError("state has the wrong shape")
        expected = (state.shape[0], self.config.chunk_length, self.config.action_dim)
        if reference.shape != expected:
            raise ValueError("reference has the wrong shape")


class TwinChunkCritic(nn.Module):
    def __init__(self, config: RLTActorCriticConfig) -> None:
        super().__init__()
        chunk_dim = config.chunk_length * config.action_dim
        self.critics = nn.ModuleList(
            [
                _mlp(config.state_dim + chunk_dim, 1, config.hidden_dim, config.hidden_layers)
                for _ in range(config.num_critics)
            ]
        )

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        inputs = torch.cat([state, action.flatten(start_dim=1)], dim=-1)
        return torch.stack([critic(inputs).squeeze(-1) for critic in self.critics], dim=0)

    def minimum(self, state: Tensor, action: Tensor) -> Tensor:
        return self(state, action).min(dim=0).values

    def first(self, state: Tensor, action: Tensor) -> Tensor:
        inputs = torch.cat([state, action.flatten(start_dim=1)], dim=-1)
        return self.critics[0](inputs).squeeze(-1)


@contextmanager
def _frozen(module: nn.Module):
    requires_grad = [parameter.requires_grad for parameter in module.parameters()]
    try:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, original in zip(module.parameters(), requires_grad, strict=True):
            parameter.requires_grad_(original)


class RLTAgent:
    """RLT actor, twin critic, target critic, and the paper-specific update schedule."""

    def __init__(self, config: RLTOnlineConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.actor = GaussianChunkActor(config.actor_critic).to(self.device)
        self.critic = TwinChunkCritic(config.actor_critic).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.critic_updates = 0
        self.actor_updates = 0

    @torch.no_grad()
    def act(self, state: Tensor, reference: Tensor, deterministic: bool = False) -> Tensor:
        return self.actor.sample(
            state.to(self.device), reference.to(self.device), deterministic=deterministic
        )

    def actor_state_dict(self) -> dict[str, Tensor]:
        """Return an immutable CPU snapshot suitable for actor/learner IPC."""
        return {name: tensor.detach().cpu().clone() for name, tensor in self.actor.state_dict().items()}

    def load_actor_state_dict(self, state: dict[str, Tensor]) -> None:
        self.actor.load_state_dict(state)
        self.actor.eval()

    def update_critic(self, batch: dict[str, Tensor]) -> dict[str, float]:
        with torch.no_grad():
            next_action = self.actor.sample(batch["next_state"], batch["next_reference"])
            next_q = self.target_critic.minimum(batch["next_state"], next_action)
            target = batch["discounted_return"] + batch["bootstrap_discount"] * next_q

        values = self.critic(batch["state"], batch["action"])
        loss = F.mse_loss(values, target.unsqueeze(0).expand_as(values))
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.critic_optimizer.step()
        self._update_targets()
        return {
            "critic_loss": loss.item(),
            "q_mean": values.mean().item(),
            "target_mean": target.mean().item(),
        }

    def update_actor(self, batch: dict[str, Tensor]) -> dict[str, float]:
        reference_input = self.actor.drop_reference(batch["reference"])
        action = self.actor.sample(batch["state"], reference_input)
        valid_mask = _horizon_mask(
            batch["valid_horizon"], self.config.actor_critic.chunk_length, action.dtype
        )
        masked_action = action * valid_mask

        with _frozen(self.critic):
            q_value = self.critic.first(batch["state"], masked_action)
            reference_distance = ((action - batch["reference"]).square() * valid_mask).sum(dim=(1, 2))
            loss = (-q_value + self.config.reference_regularization * reference_distance).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.actor_optimizer.step()
        return {
            "actor_loss": loss.item(),
            "actor_q": q_value.mean().item(),
            "reference_distance": reference_distance.mean().item(),
        }

    def update(self, sample_batch) -> dict[str, float]:
        metrics = self.update_critic(sample_batch())
        self.critic_updates += 1
        if self.critic_updates % self.config.critic_updates_per_actor == 0:
            metrics.update(self.update_actor(sample_batch()))
            self.actor_updates += 1
        return metrics

    @torch.no_grad()
    def _update_targets(self) -> None:
        for online, target in zip(self.critic.parameters(), self.target_critic.parameters(), strict=True):
            target.lerp_(online, self.config.tau)

    def training_state_dict(self) -> dict[str, object]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "critic_updates": self.critic_updates,
            "actor_updates": self.actor_updates,
        }

    def load_training_state_dict(self, state: dict[str, object]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.target_critic.load_state_dict(state["target_critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self._optimizer_to_device(self.actor_optimizer)
        self._optimizer_to_device(self.critic_optimizer)
        self.critic_updates = int(state["critic_updates"])
        self.actor_updates = int(state["actor_updates"])

    def _optimizer_to_device(self, optimizer: torch.optim.Optimizer) -> None:
        for optimizer_state in optimizer.state.values():
            for key, value in optimizer_state.items():
                if isinstance(value, Tensor):
                    optimizer_state[key] = value.to(self.device)
