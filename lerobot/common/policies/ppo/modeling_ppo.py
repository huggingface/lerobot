#!/usr/bin/env python

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
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812

from lerobot.common.policies.pretrained import PreTrainedPolicy

from ..mlp import MLP
from .configuration_ppo import PPOConfig


class PPOPolicy(PreTrainedPolicy):
    """Proximal Policy Optimization (PPO) policy.

    Original paper: https://arxiv.org/abs/1707.06347
    """

    config_class = PPOConfig
    name = "ppo"

    # Additional tensors that must be produced during rollouts (in addition to
    # the standard tensors used for all policies)
    rollout_tensor_spec = {
        "value": {"shape": (), "dtype": np.dtype("float32")},
        "action.log_prob": {"shape": (), "dtype": np.dtype("float32")},
        "advantage": {"shape": (), "dtype": np.dtype("float32")},
        "return": {"shape": (), "dtype": np.dtype("float32")},
    }

    def __init__(
        self,
        config: PPOConfig | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default
                instantiation of the configuration class is used.
            dataset_stats (unused): For compliance with policy protocol.
        """
        super().__init__(config)
        self.config = config or PPOConfig()

        self.policy_net = MLP(
            in_channels=self.config.input_shapes["observation.state"][0],
            hidden_channels=[*self.config.hidden_dim, self.config.output_shapes["action"][0]],
            nonlinearity=self.config.nonlinearity,
        )
        self.value_net = MLP(
            in_channels=self.config.input_shapes["observation.state"][0],
            hidden_channels=[*self.config.hidden_dim, 1],
            nonlinearity=self.config.nonlinearity,
        )
        self.log_std = nn.Parameter(torch.zeros(self.config.output_shapes["action"][0]))

    def get_optim_params(self):
        return self.parameters()

    def reset(self):
        """Unused, for compliance with policy protocol."""

    def compute_values(self, obs: Tensor) -> Tensor:
        """Compute the value predictions for the given observations."""
        return self.value_net(obs).squeeze(-1)

    def get_distribution(self, obs: Tensor) -> torch.distributions.Normal:
        """Get the policy distribution from the policy network."""
        mean_action = self.policy_net(obs)
        std = torch.exp(self.log_std).expand_as(mean_action)
        return torch.distributions.Normal(mean_action, std)

    def compute_log_probs(self, dist: torch.distributions.Normal, action: Tensor) -> Tensor:
        """Compute the log probabilities for the given distribution and actions."""
        return dist.log_prob(action).sum(-1)

    def forward(self, batch: dict[str, Tensor]) -> dict:
        """Run the batch through the model and compute the loss for training or validation."""
        dist = self.get_distribution(batch["observation.state"])

        # Compute actor loss
        log_probs = self.compute_log_probs(dist=dist, action=batch["action"])
        ratios = torch.exp(log_probs - batch["action.log_prob"])
        policy_loss_1 = batch["advantage"] * ratios
        policy_loss_2 = batch["advantage"] * torch.clamp(
            ratios, 1 - self.config.clip_range, 1 + self.config.clip_range
        )
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

        # Compute critic loss
        values = self.compute_values(batch["observation.state"])
        value_loss = F.mse_loss(batch["return"], values)

        # Compute entropy loss
        entropy_loss = -dist.entropy().sum(-1).mean()

        # Compute total loss
        total_loss = (
            policy_loss + self.config.entropy_coeff * entropy_loss + self.config.value_loss_coeff * value_loss
        )

        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "loss": total_loss,
        }

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select an action given the current state."""
        dist = self.get_distribution(batch["observation.state"])
        return dist.sample()

    @torch.no_grad()
    def update_from_rollouts(self, batch: dict) -> dict[str, Tensor]:
        """
        Processes a batch of trajectory rollouts and collects additional data, i.e. log probabilities,
        values, advantages, and returns which are used during training.

        Args:
            rollouts (dict[str, Tensor]): A dictionary of tensors representing trajectory rollouts.

        Returns:
            dict[str, Tensor]: A dictionary of additional tensors (if any) to be logged or stored.
        """
        dist = self.get_distribution(batch["observation.state"])
        log_probs = self.compute_log_probs(dist=dist, action=batch["action"])

        values = self.compute_values(batch["observation.state"])
        advantages, returns = compute_advantages(
            values=values,
            rewards=batch["next.reward"],
            dones=batch["next.done"],
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

        return {
            "value": values,
            "action.log_prob": log_probs,
            "advantage": advantages,
            "return": returns,
        }


def compute_advantages(
    values: Tensor,
    rewards: Tensor,
    dones: Tensor,
    gamma: float,
    gae_lambda: float,
    normalize: bool = True,
) -> tuple[Tensor, Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE) and returns.

    Args:
        values (Tensor): Predicted state values
        rewards (Tensor): Rewards received
        dones (Tensor): Done flags
        gamma (float): Discount factor
        lam (float): GAE lambda parameter

    Returns:
        tuple: (advantages, returns)
    """
    batch_size = len(rewards)
    advantages = torch.zeros_like(rewards)

    next_advantage = 0
    next_value = 0

    for t in reversed(range(batch_size)):
        not_done = not dones[t]
        td_error = rewards[t] + gamma * next_value * not_done - values[t]
        advantages[t] = td_error + gamma * gae_lambda * next_advantage * not_done

        next_advantage = advantages[t]
        next_value = values[t]

    returns = advantages + values

    # Normalize advantages. Batch size must have length > 1, otherwise std dev is nan
    if normalize and advantages.shape[0] > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages, returns
