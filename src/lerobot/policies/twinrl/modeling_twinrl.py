#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""TwinRL-VLA policy.

Joint IL+RL training objective (paper Eq. 6):
    L_actor(ψ) = β * L_IL + η * L_Q

where:
    L_IL = MSE(π_mean(s), a_demo)           -- BC/imitation loss
    L_Q  = -E[min_i Q_i(s, π(s))]          -- RL Q-gradient

Critic uses Cal-QL (paper Appendix C, official code calql_critic_loss_fn):
    L_critic = L_TD + α * L_CQL

where L_CQL applies a logsumexp penalty over sampled OOD actions, with
Monte Carlo return lower-bound clipping to prevent Q underestimation.

Paper: https://arxiv.org/abs/2602.09023
"""

import math
from dataclasses import asdict
from typing import Literal

import einops
import torch
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.sac.modeling_sac import (
    MLP,
    CriticEnsemble,
    CriticHead,
    Policy,
    SACObservationEncoder,
)
from lerobot.policies.twinrl.configuration_twinrl import TwinRLConfig
from lerobot.utils.constants import ACTION


class TwinRLPolicy(PreTrainedPolicy):
    """TwinRL policy with joint BC+RL actor loss and Cal-QL critic."""

    config_class = TwinRLConfig
    name = "twinrl"

    def __init__(self, config: TwinRLConfig | None = None):
        super().__init__(config)
        self.config = config
        config.validate_features()

        action_dim = config.output_features[ACTION].shape[0]
        self._init_encoders()
        self._init_critics(action_dim)
        self._init_actor(action_dim)
        self.to(config.device)

    def get_optim_params(self) -> dict:
        return {
            "actor": [
                p
                for n, p in self.actor.named_parameters()
                if not n.startswith("encoder") or not self.shared_encoder
            ],
            "critic": self.critic_ensemble.parameters(),
        }

    def reset(self):
        pass

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Deterministic action selection: return policy mean."""
        _, _, mean_actions = self.actor(batch, None)
        return mean_actions

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        raise NotImplementedError("TwinRLPolicy returns single actions, not chunks.")

    def forward(
        self,
        batch: dict[str, Tensor | dict[str, Tensor]],
        model: Literal["actor", "critic"] = "critic",
    ) -> dict[str, Tensor]:
        """Compute loss for either the actor or critic.

        For "actor":
            batch must contain "state", ACTION (demo actions for BC term),
            and optionally "observation_feature".
        For "critic":
            batch must also contain "reward", "next_state", "done",
            and optionally "complementary_info" with "mc_returns".
        """
        observations: dict[str, Tensor] = batch["state"]
        obs_features: Tensor | None = batch.get("observation_feature")

        if model == "critic":
            return {
                "loss_critic": self.compute_loss_critic(
                    observations=observations,
                    actions=batch[ACTION],
                    rewards=batch["reward"],
                    next_observations=batch["next_state"],
                    done=batch["done"],
                    mc_returns=self._get_mc_returns(batch),
                    obs_features=obs_features,
                    next_obs_features=batch.get("next_observation_feature"),
                )
            }

        if model == "actor":
            return {
                "loss_actor": self.compute_loss_actor(
                    observations=observations,
                    demo_actions=batch[ACTION],
                    obs_features=obs_features,
                )
            }

        raise ValueError(f"Unknown model: {model}")

    # ------------------------------------------------------------------
    # Actor loss: β * L_IL + η * L_Q  (paper Eq. 6)
    # ------------------------------------------------------------------

    def compute_loss_actor(
        self,
        observations: dict[str, Tensor],
        demo_actions: Tensor,
        obs_features: Tensor | None = None,
    ) -> Tensor:
        """Joint BC + RL actor loss."""
        # sampled_actions: reparameterized sample (for Q-gradient)
        # mean_actions: policy mean (for IL regression)
        sampled_actions, _, mean_actions = self.actor(observations, obs_features)

        # IL term: MSE between policy mean and demo action (no noise bias)
        il_loss = torch.nn.functional.mse_loss(mean_actions, demo_actions)

        # RL term: maximise mean Q over ensemble (use sample for RL gradient)
        q_values = self._critic_forward(observations, sampled_actions, obs_features)
        mean_q = q_values.mean(dim=0)
        q_loss = -mean_q.mean()

        return self.config.bc_weight * il_loss + self.config.q_weight * q_loss

    # ------------------------------------------------------------------
    # Critic loss: L_TD + α * L_CQL  (Cal-QL from official TwinRL code)
    # ------------------------------------------------------------------

    def compute_loss_critic(
        self,
        observations: dict[str, Tensor],
        actions: Tensor,
        rewards: Tensor,
        next_observations: dict[str, Tensor],
        done: Tensor,
        mc_returns: Tensor | None = None,
        obs_features: Tensor | None = None,
        next_obs_features: Tensor | None = None,
    ) -> Tensor:
        # --- TD target (standard Bellman) ---
        with torch.no_grad():
            next_actions, _, _ = self.actor(next_observations, next_obs_features)
            next_q = self._critic_forward(next_observations, next_actions, next_obs_features, target=True)
            if self.config.num_subsample_critics is not None:
                idx = torch.randperm(self.config.num_critics)[: self.config.num_subsample_critics]
                next_q = next_q[idx]
            min_next_q = next_q.min(dim=0)[0]
            td_target = rewards + (1 - done) * self.config.discount * min_next_q

        q_preds = self._critic_forward(observations, actions, obs_features)
        td_target_exp = einops.repeat(td_target, "b -> e b", e=q_preds.shape[0])
        td_loss = torch.nn.functional.mse_loss(q_preds, td_target_exp, reduction="none").mean(dim=1).sum()

        if not self.config.use_calql or self.config.cql_alpha == 0.0:
            return td_loss

        # --- CQL penalty with optional Cal-QL MC lower bound ---
        cql_diff = self._compute_cql_loss(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            mc_returns=mc_returns,
            obs_features=obs_features,
            next_obs_features=next_obs_features,
        )
        cql_loss = cql_diff.clamp(self.config.cql_clip_diff_min, self.config.cql_clip_diff_max).mean()
        return td_loss + self.config.cql_alpha * cql_loss

    def _compute_cql_loss(
        self,
        observations: dict[str, Tensor],
        actions: Tensor,
        next_observations: dict[str, Tensor],
        mc_returns: Tensor | None,
        obs_features: Tensor | None,
        next_obs_features: Tensor | None,
    ) -> Tensor:
        """Cal-QL: logsumexp over OOD actions - Q(s, a_demo), lower-bounded by MC returns."""
        batch_size, action_dim = actions.shape
        n = self.config.cql_n_actions
        device = actions.device

        # Sample OOD actions: uniform random + policy at s + policy at s'
        random_actions = torch.rand(batch_size, n, action_dim, device=device) * 2 - 1
        with torch.no_grad():
            policy_actions_s, _, _ = self.actor(observations, obs_features)
            policy_actions_s_next, _, _ = self.actor(next_observations, next_obs_features)
        policy_actions_s = policy_actions_s.unsqueeze(1).expand(-1, n, -1)
        policy_actions_s_next = policy_actions_s_next.unsqueeze(1).expand(-1, n, -1)

        # Concatenate: (b, 3*n, action_dim)  — order: [random, current, next]
        all_actions = torch.cat([random_actions, policy_actions_s, policy_actions_s_next], dim=1)

        # Q values for all sampled actions: (num_critics, b, 3*n)
        q_sampled = self._critic_forward_multiple(observations, all_actions, obs_features)

        # Q on demo actions: (num_critics, b)
        q_demo = self._critic_forward(observations, actions, obs_features)

        # Cal-QL: clip sampled Q values to be >= MC return lower bound
        if mc_returns is not None:
            mc_lb = mc_returns.view(1, batch_size, 1).expand_as(q_sampled)
            q_sampled = torch.maximum(q_sampled, mc_lb)

        # Append q_demo as an extra column, apply log(K) importance correction, then logsumexp
        # Matches official: cql_q_samples = concat([ood, q_pred], axis=-1); subtract log(K)*temp
        k = q_sampled.shape[-1] + 1  # 3n + 1
        q_extended = torch.cat([q_sampled, q_demo.unsqueeze(-1)], dim=-1)
        q_extended = q_extended - math.log(k) * self.config.cql_temp
        ood_values = (
            torch.logsumexp(q_extended / self.config.cql_temp, dim=-1) * self.config.cql_temp
        )  # (num_critics, b)

        cql_diff = ood_values - q_demo  # (num_critics, b)
        return cql_diff

    # ------------------------------------------------------------------
    # Target network update
    # ------------------------------------------------------------------

    def update_target_networks(self):
        tau = self.config.critic_target_update_weight
        for target_p, p in zip(
            self.critic_target.parameters(), self.critic_ensemble.parameters(), strict=True
        ):
            target_p.data.copy_(p.data * tau + target_p.data * (1.0 - tau))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _critic_forward(
        self,
        observations: dict[str, Tensor],
        actions: Tensor,
        obs_features: Tensor | None = None,
        target: bool = False,
    ) -> Tensor:
        critics = self.critic_target if target else self.critic_ensemble
        return critics(observations, actions, obs_features)

    def _critic_forward_multiple(
        self,
        observations: dict[str, Tensor],
        actions: Tensor,
        obs_features: Tensor | None = None,
    ) -> Tensor:
        """Evaluate critic for a batch of actions per state.

        actions: (b, n_actions, action_dim)
        Returns: (num_critics, b, n_actions)
        """
        b, n, _ = actions.shape
        # Flatten to (b*n, action_dim) and tile observations
        flat_actions = actions.reshape(b * n, -1)
        flat_obs = {
            k: v.unsqueeze(1).expand(-1, n, *v.shape[1:]).reshape(b * n, *v.shape[1:])
            for k, v in observations.items()
        }
        flat_features = None
        if obs_features is not None:
            flat_features = (
                obs_features.unsqueeze(1)
                .expand(-1, n, *obs_features.shape[1:])
                .reshape(b * n, *obs_features.shape[1:])
            )

        q_flat = self.critic_ensemble(flat_obs, flat_actions, flat_features)  # (num_critics, b*n)
        return q_flat.reshape(q_flat.shape[0], b, n)

    @staticmethod
    def _get_mc_returns(batch: dict) -> Tensor | None:
        comp = batch.get("complementary_info")
        if comp is not None:
            return comp.get("mc_returns")
        return None

    # ------------------------------------------------------------------
    # Initialisation helpers (reuse SAC modules)
    # ------------------------------------------------------------------

    def _init_encoders(self):
        self.shared_encoder = self.config.shared_encoder
        self.encoder_critic = SACObservationEncoder(self.config)
        self.encoder_actor = (
            self.encoder_critic if self.shared_encoder else SACObservationEncoder(self.config)
        )

    def _init_critics(self, action_dim: int):
        def _make_ensemble():
            heads = [
                CriticHead(
                    input_dim=self.encoder_critic.output_dim + action_dim,
                    **asdict(self.config.critic_network_kwargs),
                )
                for _ in range(self.config.num_critics)
            ]
            return CriticEnsemble(encoder=self.encoder_critic, ensemble=heads)

        self.critic_ensemble = _make_ensemble()
        self.critic_target = _make_ensemble()
        self.critic_target.load_state_dict(self.critic_ensemble.state_dict())

        if self.config.use_torch_compile:
            self.critic_ensemble = torch.compile(self.critic_ensemble)
            self.critic_target = torch.compile(self.critic_target)

    def _init_actor(self, action_dim: int):
        self.actor = Policy(
            encoder=self.encoder_actor,
            network=MLP(
                input_dim=self.encoder_actor.output_dim,
                **asdict(self.config.actor_network_kwargs),
            ),
            action_dim=action_dim,
            encoder_is_shared=self.shared_encoder,
            **asdict(self.config.policy_kwargs),
        )
