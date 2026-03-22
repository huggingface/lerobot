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
"""RLT (RL Token) algorithm.

Implements the two-stage training from "RL Token: Bootstrapping Online RL
with Vision-Language-Action Models" (Xu et al., Physical Intelligence, 2026).

Stage 1 (offline): Train RL-token encoder/decoder via reconstruction loss.
Stage 2 (online):  Train actor-critic with chunked TD, BC regularization,
                   reference-action pass-through, and reference-action dropout.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torch.optim import Optimizer

from lerobot.policies.rlt.modeling_rlt import MLP, RLTPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.rl.algorithms.base import (
    BatchType,
    RLAlgorithm,
    TrainingStats,
)
from lerobot.rl.algorithms.rlt.configuration_rlt import RLTAlgorithmConfig
from lerobot.utils.constants import ACTION


class RLTCritic(nn.Module):
    """Q-function over (state, action_chunk) pairs.

    Paper Eq. 3: Q_psi(x, a_{1:C})

    Training-only component — lives on the algorithm side, not in the policy.
    """

    def __init__(self, state_dim: int, action_chunk_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.net = MLP(state_dim + action_chunk_dim, hidden_dims, output_dim=1)

    def forward(self, state: Tensor, action_chunk: Tensor) -> Tensor:
        x = torch.cat([state, action_chunk], dim=-1)
        return self.net(x)


class RLTAlgorithm(RLAlgorithm):
    """RL Token: lightweight actor-critic on frozen VLA features.

    Owns the ``RLTPolicy`` (RL-token encoder/decoder + actor), a critic
    ensemble, and target networks.  All VLA-specific logic (embedding
    extraction, reference actions) lives in ``_prepare_forward_batch``.
    """

    def __init__(self, policy: RLTPolicy, config: RLTAlgorithmConfig):
        self.policy = policy
        self.config = config
        self.optimizers: dict[str, Optimizer] = {}
        self._optimization_step: int = 0
        self._device = get_device_from_parameters(self.policy)
        self._is_online = False

        self._init_critics()
        self._move_to_device()

    # ── Initialization ───────────────────────────────────────────────

    def _init_critics(self) -> None:
        state_dim = self.policy._state_dim
        action_chunk_dim = self.policy._action_chunk_dim
        hidden_dims = self.policy.config.critic.hidden_dims

        self.critics = torch.nn.ModuleList(
            [RLTCritic(state_dim, action_chunk_dim, hidden_dims) for _ in range(self.config.num_critics)]
        )
        self.critic_targets = torch.nn.ModuleList([copy.deepcopy(c) for c in self.critics])
        for ct in self.critic_targets:
            ct.requires_grad_(False)

    def _move_to_device(self) -> None:
        self.critics.to(self._device)
        self.critic_targets.to(self._device)

    # ── Offline phase (Stage 1): RL-token training ───────────────────

    def supports_offline_phase(self) -> bool:
        return True

    def offline_update(self, batch_iterator: Iterator[BatchType]) -> TrainingStats:
        """Train RL-token encoder/decoder on demonstration data.

        Paper Eq. 2: L_ro = E[ sum_i || h(d([z_rl, z_bar_{1:i-1}]))_i - z_bar_i ||^2 ]
        """
        batch = next(batch_iterator)

        vla_embeddings = batch["state"]["observation.vla_embeddings"].to(self._device)
        z_vla = vla_embeddings.detach()  # stop-gradient on VLA embeddings

        z_rl = self.policy.rl_token_encoder(z_vla)
        z_reconstructed = self.policy.rl_token_decoder(z_rl, z_vla)

        loss_ro = F.mse_loss(z_reconstructed, z_vla)

        self.optimizers["rl_token"].zero_grad()
        loss_ro.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy.rl_token_encoder.parameters()) + list(self.policy.rl_token_decoder.parameters()),
            max_norm=self.config.clip_grad_norm,
        )
        self.optimizers["rl_token"].step()

        self._optimization_step += 1
        return TrainingStats(losses={"loss_rl_token": loss_ro.item()})

    def transition_to_online(self) -> None:
        """Freeze RL-token modules; rebuild optimizers for actor-critic only."""
        self.policy.rl_token_encoder.requires_grad_(False)
        self.policy.rl_token_decoder.requires_grad_(False)
        self._is_online = True

        self.optimizers = {
            "actor": torch.optim.Adam(self.policy.actor.parameters(), lr=self.config.actor_lr),
            "critic": torch.optim.Adam(self.critics.parameters(), lr=self.config.critic_lr),
        }
        self._optimization_step = 0

    # ── Online phase (Stage 2): Actor-Critic ─────────────────────────

    def update(self, batch_iterator: Iterator[BatchType]) -> TrainingStats:
        """One full RLT update step with UTD critic warm-up.

        Pulls ``utd_ratio`` batches. First ``utd_ratio - 1`` are critic-only;
        the last batch also updates the actor (every ``policy_update_freq`` steps).
        """
        for _ in range(self.config.utd_ratio - 1):
            batch = next(batch_iterator)
            fb = self._prepare_forward_batch(batch)
            self._critic_step(fb)
            self._update_target_networks()

        batch = next(batch_iterator)
        fb = self._prepare_forward_batch(batch)
        critic_loss = self._critic_step(fb)

        stats = TrainingStats(losses={"loss_critic": critic_loss})

        if self._optimization_step % self.config.policy_update_freq == 0:
            actor_loss, bc_loss, q_val = self._actor_step(fb)
            stats.losses["loss_actor"] = actor_loss
            stats.extra["bc_loss"] = bc_loss
            stats.extra["q_value_mean"] = q_val

        self._update_target_networks()
        self._optimization_step += 1
        return stats

    def _prepare_forward_batch(self, batch: BatchType) -> dict[str, Any]:
        """Convert a replay batch into algorithm-ready tensors.

        Extracts RL-token from VLA embeddings, builds RL state, reads
        reference action from complementary_info.
        """
        obs = batch["state"]
        next_obs = batch["next_state"]
        device = self._device

        vla_emb = obs["observation.vla_embeddings"].to(device)
        next_vla_emb = next_obs["observation.vla_embeddings"].to(device)

        with torch.no_grad():
            z_rl = self.policy.rl_token_encoder(vla_emb)
            z_rl_next = self.policy.rl_token_encoder(next_vla_emb)

        parts = [z_rl]
        next_parts = [z_rl_next]
        if "observation.state" in obs and self.policy._proprioception_dim > 0:
            prop = obs["observation.state"].to(device)
            next_prop = next_obs["observation.state"].to(device)
            parts.append(prop)
            next_parts.append(next_prop)

        state = torch.cat(parts, dim=-1)
        next_state = torch.cat(next_parts, dim=-1)

        action = batch[ACTION].to(device)
        reward = batch["reward"].to(device)
        done = batch["done"].to(device)

        ref_action = None
        comp_info = batch.get("complementary_info")
        if comp_info is not None and "reference_action" in comp_info:
            ref_action = comp_info["reference_action"].to(device)

        return {
            "state": state,
            "next_state": next_state,
            "action": action,
            "reward": reward,
            "done": done,
            "reference_action": ref_action,
        }

    def _critic_step(self, fb: dict[str, Any]) -> float:
        """Paper Eq. 3: chunked TD with clipped double-Q target."""
        state = fb["state"]
        next_state = fb["next_state"]
        action = fb["action"]
        reward = fb["reward"]
        done = fb["done"]

        with torch.no_grad():
            ref = fb.get("reference_action")
            if ref is None:
                ref = torch.zeros_like(action)
            next_action = self.policy.actor(next_state, ref)

            target_qs = [ct(next_state, next_action) for ct in self.critic_targets]
            min_target_q = torch.min(torch.cat(target_qs, dim=-1), dim=-1, keepdim=True).values

            discount_chunk = self.config.discount**self.config.chunk_size
            td_target = reward.unsqueeze(-1) + (1 - done.unsqueeze(-1)) * discount_chunk * min_target_q

        q_preds = [c(state, action) for c in self.critics]
        loss = sum(F.mse_loss(q, td_target) for q in q_preds)

        self.optimizers["critic"].zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critics.parameters(), max_norm=self.config.clip_grad_norm)
        self.optimizers["critic"].step()
        return loss.item()

    def _actor_step(self, fb: dict[str, Any]) -> tuple[float, float, float]:
        """Paper Eq. 5: maximize Q while staying near VLA reference.

        L_pi(theta) = E[ -Q(x, a) + beta * ||a - a_tilde||^2 ]
        With reference-action dropout applied to the actor's ref input.
        """
        state = fb["state"]
        ref = fb.get("reference_action")
        if ref is None:
            ref = torch.zeros(state.shape[0], self.policy._action_chunk_dim, device=self._device)

        # Reference-action dropout (paper Section IV-B)
        mask = (torch.rand(ref.shape[0], 1, device=self._device) > self.config.ref_dropout).float()
        ref_input = ref * mask

        action = self.policy.actor(state, ref_input)

        q_value = self.critics[0](state, action)

        bc_loss = F.mse_loss(action, ref)

        loss = -q_value.mean() + self.config.bc_reg_coeff * bc_loss

        self.optimizers["actor"].zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), max_norm=self.config.clip_grad_norm)
        self.optimizers["actor"].step()

        return loss.item(), bc_loss.item(), q_value.mean().item()

    def _update_target_networks(self) -> None:
        tau = self.config.tau
        for critic, target in zip(self.critics, self.critic_targets, strict=True):
            for p, tp in zip(critic.parameters(), target.parameters(), strict=True):
                tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

    # ── Optimizer management ─────────────────────────────────────────

    def make_optimizers(self) -> dict[str, Optimizer]:
        """Create optimizers. Initially for RL-token (Stage 1)."""
        self.optimizers = {
            "rl_token": torch.optim.Adam(
                list(self.policy.rl_token_encoder.parameters())
                + list(self.policy.rl_token_decoder.parameters()),
                lr=self.config.rl_token_lr,
            ),
            "actor": torch.optim.Adam(self.policy.actor.parameters(), lr=self.config.actor_lr),
            "critic": torch.optim.Adam(self.critics.parameters(), lr=self.config.critic_lr),
        }
        return self.optimizers

    def get_optimizers(self) -> dict[str, Optimizer]:
        return self.optimizers

    # ── Weight sync ──────────────────────────────────────────────────

    def get_weights(self) -> dict[str, Any]:
        """Push actor + RL-token encoder to actors (small footprint)."""
        weights = {
            "actor": self.policy.actor.state_dict(),
            "rl_token_encoder": self.policy.rl_token_encoder.state_dict(),
        }
        return {k: {kk: vv.cpu() for kk, vv in v.items()} for k, v in weights.items()}

    def load_weights(self, weights: dict[str, Any], device: str | torch.device = "cpu") -> None:
        if "actor" in weights:
            self.policy.actor.load_state_dict({k: v.to(device) for k, v in weights["actor"].items()})
        if "rl_token_encoder" in weights:
            self.policy.rl_token_encoder.load_state_dict(
                {k: v.to(device) for k, v in weights["rl_token_encoder"].items()}
            )
