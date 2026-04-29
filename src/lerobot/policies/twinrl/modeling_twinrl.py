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
    L_IL = recon_loss (ConRFT Karras-weighted) or MSE(π_mean, a_demo) for Gaussian
    L_Q  = -E[mean_i Q_i(s, π(s))]         -- RL Q-gradient

Critic uses Cal-QL (paper Appendix C, official code calql_critic_loss_fn):
    L_critic = L_TD + α * L_CQL

where L_CQL applies a logsumexp penalty over sampled OOD actions (3n), with
Monte Carlo return lower-bound clipping to prevent Q underestimation.

Paper: https://arxiv.org/abs/2602.09023
"""

import math
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict
from functools import partial
from typing import Literal

import einops
import torch
import torch.nn as nn
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


class OctoActorEncoder(nn.Module):
    """Frozen OctoTransformer used as actor feature extractor.

    Loads weights from HuggingFace (requires octo-pytorch package).
    Outputs the mean-pooled readout_action token: (b, token_embedding_size).
    The transformer weights are frozen; only downstream layers are trained.
    """

    def __init__(self, config):
        super().__init__()
        from octo_pytorch.model.modeling_octo import OctoModel

        octo = OctoModel.from_pretrained(config.octo_model_name)
        self.transformer = octo.octo_transformer
        self.text_processor = octo.text_processor
        self._output_dim = self.transformer.token_embedding_size
        for param in self.transformer.parameters():
            param.requires_grad = False

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        observations: dict[str, Tensor],
        obs_features: Tensor | None = None,
        cache: Tensor | None = None,
        detach: bool = False,
    ) -> Tensor:
        """Return (b, token_embedding_size) readout feature vector."""
        features = obs_features if obs_features is not None else cache
        if features is not None:
            return features if not detach else features.detach()

        device = next(self.transformer.parameters()).device
        img = observations.get("image_primary", observations.get("observation.image"))
        if img is None:
            raise ValueError("OctoActorEncoder requires 'image_primary' in observations.")

        if img.ndim == 4:
            img = img.unsqueeze(1)

        b, horizon, c, h, w = img.shape
        # Octo expects (256, 256) for primary and (128, 128) for wrist
        if h != 256 or w != 256:
            img = torch.nn.functional.interpolate(
                img.view(b * horizon, c, h, w), size=(256, 256), mode="bilinear", align_corners=False
            ).view(b, horizon, c, 256, 256)

        # Scale to [0, 255] if input is [0, 1]
        if img.max() <= 1.1:
            img = img * 255.0

        obs_octo = {
            "image_primary": img.permute(0, 1, 3, 4, 2),
            "pad_mask_dict": {"image_primary": torch.ones(b, horizon, dtype=torch.bool, device=device)},
        }

        if "image_wrist" in observations:
            w_img = observations["image_wrist"]
            if w_img.ndim == 4:
                w_img = w_img.unsqueeze(1)

            bw, hw, cw, hh, ww = w_img.shape
            if hh != 128 or ww != 128:
                w_img = torch.nn.functional.interpolate(
                    w_img.view(bw * hw, cw, hh, ww), size=(128, 128), mode="bilinear", align_corners=False
                ).view(bw, hw, cw, 128, 128)

            if w_img.max() <= 1.1:
                w_img = w_img * 255.0

            obs_octo["image_wrist"] = w_img.permute(0, 1, 3, 4, 2)
            obs_octo["pad_mask_dict"]["image_wrist"] = torch.ones(bw, hw, dtype=torch.bool, device=device)

        tasks = {
            "language_instruction": self.text_processor.encode([""] * b),
            "pad_mask_dict": {"language_instruction": torch.ones(b, dtype=torch.bool, device=device)},
        }
        for k, v in tasks["language_instruction"].items():
            tasks["language_instruction"][k] = v.to(device)

        window = obs_octo["image_primary"].shape[1]
        pad_mask = torch.ones(b, window, dtype=torch.bool, device=device)

        with torch.no_grad():
            outputs = self.transformer(obs_octo, tasks, pad_mask)

        # tokens: (b, window, n_tokens, embed) → mean over tokens → last window step
        feats = outputs["readout_action"].tokens.mean(dim=-2)[:, -1, :]  # (b, embed)
        return feats


class ConsistencyActorHead(nn.Module):
    """Karras consistency model actor (ConRFT-style, matches JAX ConsistencyPolicy_octo).

    base_network(x_t, sigma, obs_enc) predicts x_0 via boundary conditions:
        c_skip * x_t + c_out * network([c_in * x_t, t_embed, obs_enc])

    Requires octo-pytorch package for FourierFeatures and MLPResNet primitives.
    """

    def __init__(self, encoder: nn.Module, obs_enc_dim: int, action_dim: int, config):
        super().__init__()
        from octo_pytorch.model.components.diffusion import FourierFeatures, MLPResNet

        self.encoder = encoder
        self.action_dim = action_dim
        self.sigma_min = config.sigma_min
        self.sigma_max = config.sigma_max
        self.sigma_data = config.sigma_data

        t_dim = config.consistency_t_dim
        hidden_dim = config.actor_network_kwargs.hidden_dims[0]
        num_blocks = len(config.actor_network_kwargs.hidden_dims)

        self.fourier = FourierFeatures(t_dim, learnable=True)
        self.t_proj = nn.Sequential(nn.Linear(t_dim, 2 * t_dim), nn.SiLU(), nn.Linear(2 * t_dim, t_dim))
        net_in_dim = action_dim + t_dim + obs_enc_dim
        self.network = MLPResNet(
            num_blocks=num_blocks,
            out_dim=hidden_dim,
            in_dim=net_in_dim,
            dropout_rate=None,
            hidden_dim=hidden_dim,
            use_layer_norm=True,
        )
        self.output_proj = nn.Linear(hidden_dim, action_dim)

        sigmas = self._karras_sigmas(config.num_scales, config.sigma_min, config.sigma_max, config.rho)
        self.register_buffer("karras_sigmas", sigmas)

    @staticmethod
    def _karras_sigmas(n: int, sigma_min: float, sigma_max: float, rho: float) -> Tensor:
        ramp = torch.linspace(0, 1, n)
        min_inv = sigma_min ** (1 / rho)
        max_inv = sigma_max ** (1 / rho)
        sigmas = (max_inv + ramp * (min_inv - max_inv)) ** rho
        return torch.cat([sigmas, sigmas.new_zeros(1)])  # append 0 sentinel

    def _scalings(self, sigma: Tensor):
        c_skip = self.sigma_data**2 / ((sigma - self.sigma_min) ** 2 + self.sigma_data**2)
        c_out = (sigma - self.sigma_min) * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1.0 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def base_network(self, x_t: Tensor, sigma: Tensor, obs_enc: Tensor) -> Tensor:
        """Predict x_0 from noisy x_t at noise level sigma. sigma: (b,)."""
        c_skip, c_out, c_in = self._scalings(sigma)
        c_skip = c_skip.unsqueeze(-1)
        c_out = c_out.unsqueeze(-1)
        c_in = c_in.unsqueeze(-1)
        rescaled_t = 1000.0 * 0.25 * torch.log(sigma + 1e-44)
        t_embed = self.t_proj(self.fourier(rescaled_t.unsqueeze(-1)))  # (b, t_dim)
        net_in = torch.cat([c_in * x_t, t_embed, obs_enc], dim=-1)
        features = self.network(net_in)
        denoised = self.output_proj(features)
        return c_out * denoised + c_skip * x_t

    def forward(self, observations: dict[str, Tensor], obs_features: Tensor | None = None):
        """Single-step inference: denoise from x_t ~ N(0, sigma_max²).
        Returns (x_0, None, x_0) to match Policy interface.
        """
        obs_enc = self.encoder(observations, obs_features)
        b, device = obs_enc.shape[0], obs_enc.device
        x_t = torch.randn(b, self.action_dim, device=device) * self.sigma_max
        sigma = self.karras_sigmas[0].expand(b)
        x_0 = self.base_network(x_t, sigma, obs_enc).clamp(-1, 1)
        return x_0, None, x_0


class TwinRLPolicy(PreTrainedPolicy):
    """TwinRL policy with joint BC+RL actor loss and Cal-QL critic."""

    config_class = TwinRLConfig
    name = "twinrl"

    def __init__(self, config: TwinRLConfig | None = None, **kwargs):
        super().__init__(config)
        self.config = config
        config.validate_features()

        action_dim = config.output_features[ACTION].shape[0]
        self._init_encoders()
        self._init_critics(action_dim)
        self._init_actor(action_dim)
        self.register_buffer("_offline_update_step", torch.zeros((), dtype=torch.long))
        self.to(config.device)

    def get_optim_params(self) -> list[dict]:
        return [
            {
                "params": [
                    p
                    for n, p in self.actor.named_parameters()
                    if not n.startswith("encoder") or not self.shared_encoder
                ],
                "lr": self.config.actor_lr,
            },
            {
                "params": list(self.critic_ensemble.parameters()),
                "lr": self.config.critic_lr,
            },
        ]

    def reset(self):
        pass

    def update(self):
        self.update_target_networks()
        self._offline_update_step.add_(1)

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
        model: Literal["actor", "critic"] | None = None,
        reduction: str = "mean",
    ) -> dict[str, Tensor] | tuple[Tensor, dict[str, Tensor]]:
        """Compute loss for either the actor or critic, or both.

        If model is None, it returns (loss, output_dict) for compatibility with lerobot_train.py.
        Otherwise, it returns a dict for compatibility with learner.py.

        For "actor":
            batch must contain observations, ACTION (demo actions for BC term),
            and optionally "observation_feature".
        For "critic":
            batch must also contain "reward", next observations, "done",
            and optionally "complementary_info" with "mc_returns".
        """
        # --- Batch mapping ---
        # Handle flat batches from lerobot_train.py or nested batches from learner.py
        if "state" in batch and isinstance(batch["state"], dict):
            observations = batch["state"]
        else:
            # Assume flat batch; filter keys starting with 'observation.'
            observations = {k: v for k, v in batch.items() if k.startswith("observation.")}

        obs_features: Tensor | None = batch.get("observation_feature")

        train_batch = self._prepare_training_batch(batch, observations, obs_features)

        # Handle specific model requests (learner.py style)
        if model == "critic":
            if not train_batch["has_critic_data"]:
                raise ValueError("TwinRL critic update requires reward and next observations.")

            _, critic_info = self.calql_critic_loss_fn(batch, train_batch)
            return critic_info

        if model == "actor":
            _, actor_info = self.policy_loss_fn(batch, train_batch)
            return actor_info

        # --- Offline training (lerobot_train.py style) ---
        # Mirrors examples/train_offline.py:
        #   for _ in range(config.cta_ratio - 1): update_calql(..., {"critic"})
        #   update_calql(..., {"critic", "actor"})
        networks_to_update = self._lerobot_train_networks_to_update(train_batch)
        return self.update_calql(batch, train_batch, networks_to_update=networks_to_update)

    def _should_update_actor(self) -> bool:
        cta_ratio = max(1, self.config.cta_ratio)
        return int(self._offline_update_step.item()) % cta_ratio == cta_ratio - 1

    def _lerobot_train_networks_to_update(self, train_batch: dict) -> frozenset[str]:
        train_critic_networks_to_update = frozenset({"critic"})
        train_actor_networks_to_update = frozenset({"actor"})
        train_networks_to_update = frozenset({"critic", "actor"})

        if not train_batch["has_critic_data"]:
            return train_actor_networks_to_update
        if self._should_update_actor():
            return train_networks_to_update
        return train_critic_networks_to_update

    def policy_loss_fn(self, batch: dict, train_batch: dict) -> tuple[Tensor, dict[str, Tensor]]:
        actor_loss = self.compute_loss_actor(
            observations=train_batch["actor_observations"],
            demo_actions=train_batch["actions"],
            obs_features=train_batch["obs_features"],
        )
        return actor_loss, {"loss_actor": actor_loss}

    def critic_loss_fn(self, batch: dict, train_batch: dict) -> tuple[Tensor, dict[str, Tensor]]:
        critic_loss = self.compute_loss_critic(
            observations=train_batch["critic_observations"],
            actions=train_batch["actions"],
            rewards=train_batch["rewards"],
            next_observations=train_batch["next_observations"],
            done=train_batch["done"],
            mc_returns=None,
            obs_features=train_batch["obs_features"],
            next_obs_features=batch.get("next_observation_feature"),
            use_calql=False,
        )
        return critic_loss, {"loss_critic": critic_loss}

    def calql_critic_loss_fn(self, batch: dict, train_batch: dict) -> tuple[Tensor, dict[str, Tensor]]:
        critic_loss = self.compute_loss_critic(
            observations=train_batch["critic_observations"],
            actions=train_batch["actions"],
            rewards=train_batch["rewards"],
            next_observations=train_batch["next_observations"],
            done=train_batch["done"],
            mc_returns=self._get_mc_returns(batch),
            obs_features=train_batch["obs_features"],
            next_obs_features=batch.get("next_observation_feature"),
        )
        return critic_loss, {"loss_critic": critic_loss}

    def calql_loss_fns(
        self, batch: dict, train_batch: dict
    ) -> dict[str, Callable[[], tuple[Tensor, dict[str, Tensor]]]]:
        return {
            "actor": partial(self.policy_loss_fn, batch, train_batch),
            "critic": partial(self.calql_critic_loss_fn, batch, train_batch),
        }

    def loss_fns(
        self, batch: dict, train_batch: dict
    ) -> dict[str, Callable[[], tuple[Tensor, dict[str, Tensor]]]]:
        return {
            "actor": partial(self.policy_loss_fn, batch, train_batch),
            "critic": partial(self.critic_loss_fn, batch, train_batch),
        }

    def update_calql(
        self,
        batch: dict,
        train_batch: dict,
        *,
        networks_to_update: frozenset[str] = frozenset({"actor", "critic"}),
    ) -> tuple[Tensor, dict[str, Tensor]]:
        calql_loss_fns = self.calql_loss_fns(batch, train_batch)
        if not networks_to_update.issubset(calql_loss_fns.keys()):
            raise ValueError(f"Invalid gradient steps: {networks_to_update}")

        # Only compute gradients for specified steps. The generic LeRobot
        # trainer performs the backward call after this function returns.
        loss_dict = {}
        total_loss = None
        for name in ("critic", "actor"):
            if name not in networks_to_update:
                continue

            loss, info = calql_loss_fns[name]()
            loss_dict.update(info)
            total_loss = loss if total_loss is None else total_loss + loss

        if total_loss is None:
            total_loss = train_batch["actions"].sum() * 0.0

        loss_dict["loss"] = total_loss
        return total_loss, loss_dict

    def _prepare_training_batch(
        self,
        batch: dict[str, Tensor | dict[str, Tensor]],
        observations: dict[str, Tensor],
        obs_features: Tensor | None,
    ) -> dict:
        # LeRobotDataset with delta_timestamps returns (b, horizon, ...).
        has_sequence = False
        horizon = 1
        for key, value in observations.items():
            if key.startswith("observation.image") and value.ndim == 5:
                has_sequence = True
                horizon = value.shape[1]
                break
            if key.startswith("observation.state") and value.ndim == 3:
                has_sequence = True
                horizon = value.shape[1]
                break

        if has_sequence and horizon == 2:
            actor_observations = {key: value[:, 0] for key, value in observations.items()}
            next_obs_from_seq = {key: value[:, 1] for key, value in observations.items()}
        elif has_sequence:
            actor_observations = observations
            next_obs_from_seq = None
        else:
            actor_observations = observations
            next_obs_from_seq = None

        actions = batch[ACTION]
        if actions.ndim == 3 and actions.shape[1] == 1:
            actions = actions.squeeze(1)

        next_observations = None
        if next_obs_from_seq is not None:
            next_observations = next_obs_from_seq
        elif "next_state" in batch and isinstance(batch["next_state"], dict):
            next_observations = batch["next_state"]
        elif any(key.startswith("next.observation.") for key in batch):
            next_observations = {
                key.replace("next.", ""): value
                for key, value in batch.items()
                if key.startswith("next.observation.")
            }

        rewards = batch.get("reward")
        if rewards is not None and rewards.ndim == 2 and rewards.shape[1] == 1:
            rewards = rewards.squeeze(1)

        done = None
        if rewards is not None:
            done = batch.get("done", batch.get("next.done", torch.zeros_like(rewards)))
            if done.ndim == 2 and done.shape[1] == 1:
                done = done.squeeze(1)

        critic_observations = actor_observations
        if has_sequence and horizon > 2:
            critic_observations = {key: value[:, -1] for key, value in actor_observations.items()}

        return {
            "actions": actions,
            "actor_observations": actor_observations,
            "critic_observations": critic_observations,
            "done": done,
            "has_critic_data": rewards is not None and next_observations is not None,
            "next_observations": next_observations,
            "obs_features": obs_features,
            "rewards": rewards,
        }

    # ------------------------------------------------------------------
    # Actor loss: β * L_IL + η * L_Q  (paper Eq. 6)
    # ------------------------------------------------------------------

    def compute_loss_actor(
        self,
        observations: dict[str, Tensor],
        demo_actions: Tensor,
        obs_features: Tensor | None = None,
    ) -> Tensor:
        """Joint BC + RL actor loss.

        ConRFT path (use_consistency_policy=True):
            IL  = Karras-weighted consistency reconstruction loss (matches JAX policy_loss_fn)
            Q   = -mean_Q on single-step denoised action

        Gaussian path (use_consistency_policy=False):
            IL  = MSE(policy_mean, demo_action)
            Q   = -mean_Q on reparameterized sample
        """
        if self.config.use_consistency_policy:
            b, device = demo_actions.shape[0], demo_actions.device

            # Octo encoder is frozen; detach matches JAX stop_gradient=True
            obs_enc = self.actor.encoder(observations, obs_features).detach()

            # --- ConRFT recon loss (JAX policy_loss_fn lines 327-345) ---
            indices = torch.randint(0, self.config.num_scales - 1, (b,), device=device)
            sigma = self.actor.karras_sigmas[indices]  # (b,)
            noise = torch.randn_like(demo_actions)
            x_t = demo_actions + noise * sigma.unsqueeze(-1)
            distiller = self.actor.base_network(x_t, sigma, obs_enc)
            snrs = sigma**-2
            weights = snrs + 1.0 / self.config.sigma_data**2  # Karras weighting
            il_loss = ((distiller - demo_actions) ** 2).mean(dim=-1).mul(weights).mean()

            # --- Q-gradient: single-step denoised action (JAX lines 347-353) ---
            x_t = torch.randn(b, demo_actions.shape[-1], device=device) * self.config.sigma_max
            sigma_top = self.actor.karras_sigmas[0].expand(b)
            new_actions = self.actor.base_network(x_t, sigma_top, obs_enc).clamp(-1, 1)
            with self._critic_grad_disabled():
                q_values = self._critic_forward(observations, new_actions, obs_features)
            q_loss = -q_values.mean(dim=0).mean()
        else:
            sampled_actions, _, mean_actions = self.actor(observations, obs_features)
            il_loss = torch.nn.functional.mse_loss(mean_actions, demo_actions)
            with self._critic_grad_disabled():
                q_values = self._critic_forward(observations, sampled_actions, obs_features)
            q_loss = -q_values.mean(dim=0).mean()

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
        use_calql: bool = True,
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

        if not use_calql or not self.config.use_calql or self.config.cql_alpha == 0.0:
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

    @contextmanager
    def _critic_grad_disabled(self):
        params = list(self.critic_ensemble.parameters())
        previous = [param.requires_grad for param in params]
        for param in params:
            param.requires_grad_(False)
        try:
            yield
        finally:
            for param, requires_grad in zip(params, previous, strict=True):
                param.requires_grad_(requires_grad)

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
        self.encoder_critic = SACObservationEncoder(self.config)
        if self.config.actor_encoder_type == "octo":
            # Octo actor encoder is always separate from the SAC critic encoder
            self.encoder_actor = OctoActorEncoder(self.config)
            self.shared_encoder = False
        else:
            self.shared_encoder = self.config.shared_encoder
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
        if self.config.use_consistency_policy:
            self.actor = ConsistencyActorHead(
                encoder=self.encoder_actor,
                obs_enc_dim=self.encoder_actor.output_dim,
                action_dim=action_dim,
                config=self.config,
            )
        else:
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
