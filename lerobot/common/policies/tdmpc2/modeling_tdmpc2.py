#!/usr/bin/env python

# Copyright 2024 Nicklas Hansen and The HuggingFace Inc. team.
# All rights reserved.
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
"""Implementation of TD-MPC2: Scalable, Robust World Models for Continuous Control

We refer to the main paper and codebase:
    TD-MPC2 paper: (https://arxiv.org/abs/2310.16828)
    TD-MPC2 code:  (https://github.com/nicklashansen/tdmpc2)
"""

# ruff: noqa: N806

from collections import deque
from copy import deepcopy
from functools import partial
from typing import Callable

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor

from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.tdmpc2.configuration_tdmpc2 import TDMPC2Config
from lerobot.common.policies.tdmpc2.tdmpc2_utils import (
    NormedLinear,
    SimNorm,
    gaussian_logprob,
    soft_cross_entropy,
    squash,
    two_hot_inv,
)
from lerobot.common.policies.utils import get_device_from_parameters, populate_queues


class TDMPC2Policy(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="lerobot",
    repo_url="https://github.com/huggingface/lerobot",
    tags=["robotics", "tdmpc2"],
):
    """Implementation of TD-MPC2 learning + inference."""

    name = "tdmpc2"

    def __init__(
        self, config: TDMPC2Config | None = None, dataset_stats: dict[str, dict[str, Tensor]] | None = None
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__()

        if config is None:
            config = TDMPC2Config()
        self.config = config
        self.model = TDMPC2WorldModel(config)
        # TODO (michel-aractingi) temp fix for gpu
        self.model = self.model.to("cuda:0")

        if config.input_normalization_modes is not None:
            self.normalize_inputs = Normalize(
                config.input_shapes, config.input_normalization_modes, dataset_stats
            )
        else:
            self.normalize_inputs = nn.Identity()
        self.normalize_targets = Normalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )

        image_keys = [k for k in config.input_shapes if k.startswith("observation.image")]
        # Note: This check is covered in the post-init of the config but have a sanity check just in case.
        self._use_image = False
        self._use_env_state = False
        if len(image_keys) > 0:
            assert len(image_keys) == 1
            self._use_image = True
            self.input_image_key = image_keys[0]
        if "observation.environment_state" in config.input_shapes:
            self._use_env_state = True

        self.scale = RunningScale(self.config.target_model_momentum)
        self.discount = (
            self.config.discount
        )  # TODO (michel-aractingi) downscale discount according to episode length

        self.reset()

    def reset(self):
        """
        Clear observation and action queues. Clear previous means for warm starting of MPPI/CEM. Should be
        called on `env.reset()`
        """
        self._queues = {
            "observation.state": deque(maxlen=1),
            "action": deque(maxlen=max(self.config.n_action_steps, self.config.n_action_repeats)),
        }
        if self._use_image:
            self._queues["observation.image"] = deque(maxlen=1)
        if self._use_env_state:
            self._queues["observation.environment_state"] = deque(maxlen=1)
        # Previous mean obtained from the cross-entropy method (CEM) used during MPC. It is used to warm start
        # CEM for the next step.
        self._prev_mean: torch.Tensor | None = None

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        batch = self.normalize_inputs(batch)
        if self._use_image:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.image"] = batch[self.input_image_key]

        self._queues = populate_queues(self._queues, batch)

        # When the action queue is depleted, populate it again by querying the policy.
        if len(self._queues["action"]) == 0:
            batch = {key: torch.stack(list(self._queues[key]), dim=1) for key in batch}

            # Remove the time dimensions as it is not handled yet.
            for key in batch:
                assert batch[key].shape[1] == 1
                batch[key] = batch[key][:, 0]

            # NOTE: Order of observations matters here.
            encode_keys = []
            if self._use_image:
                encode_keys.append("observation.image")
            if self._use_env_state:
                encode_keys.append("observation.environment_state")
            encode_keys.append("observation.state")
            z = self.model.encode({k: batch[k] for k in encode_keys})
            if self.config.use_mpc:  # noqa: SIM108
                actions = self.plan(z)  # (horizon, batch, action_dim)
            else:
                # Plan with the policy (π) alone. This always returns one action so unsqueeze to get a
                # sequence dimension like in the MPC branch.
                actions = self.model.pi(z)[0].unsqueeze(0)

            actions = torch.clamp(actions, -1, +1)

            actions = self.unnormalize_outputs({"action": actions})["action"]

            if self.config.n_action_repeats > 1:
                for _ in range(self.config.n_action_repeats):
                    self._queues["action"].append(actions[0])
            else:
                # Action queue is (n_action_steps, batch_size, action_dim), so we transpose the action.
                self._queues["action"].extend(actions[: self.config.n_action_steps])

        action = self._queues["action"].popleft()
        return action

    @torch.no_grad()
    def plan(self, z: Tensor) -> Tensor:
        """Plan sequence of actions using TD-MPC inference.

        Args:
            z: (batch, latent_dim,) tensor for the initial state.
        Returns:
            (horizon, batch, action_dim,) tensor for the planned trajectory of actions.
        """
        device = get_device_from_parameters(self)

        batch_size = z.shape[0]

        # Sample Nπ trajectories from the policy.
        pi_actions = torch.empty(
            self.config.horizon,
            self.config.n_pi_samples,
            batch_size,
            self.config.output_shapes["action"][0],
            device=device,
        )
        if self.config.n_pi_samples > 0:
            _z = einops.repeat(z, "b d -> n b d", n=self.config.n_pi_samples)
            for t in range(self.config.horizon):
                # Note: Adding a small amount of noise here doesn't hurt during inference and may even be
                # helpful for CEM.
                pi_actions[t] = self.model.pi(_z)[0]
                _z = self.model.latent_dynamics(_z, pi_actions[t])

        # In the CEM loop we will need this for a call to estimate_value with the gaussian sampled
        # trajectories.
        z = einops.repeat(z, "b d -> n b d", n=self.config.n_gaussian_samples + self.config.n_pi_samples)

        # Model Predictive Path Integral (MPPI) with the cross-entropy method (CEM) as the optimization
        # algorithm.
        # The initial mean and standard deviation for the cross-entropy method (CEM).
        mean = torch.zeros(
            self.config.horizon, batch_size, self.config.output_shapes["action"][0], device=device
        )
        # Maybe warm start CEM with the mean from the previous step.
        if self._prev_mean is not None:
            mean[:-1] = self._prev_mean[1:]
        std = self.config.max_std * torch.ones_like(mean)

        for _ in range(self.config.cem_iterations):
            # Randomly sample action trajectories for the gaussian distribution.
            std_normal_noise = torch.randn(
                self.config.horizon,
                self.config.n_gaussian_samples,
                batch_size,
                self.config.output_shapes["action"][0],
                device=std.device,
            )
            gaussian_actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * std_normal_noise, -1, 1)

            # Compute elite actions.
            actions = torch.cat([gaussian_actions, pi_actions], dim=1)
            value = self.estimate_value(z, actions).nan_to_num_(0).squeeze()
            elite_idxs = torch.topk(value, self.config.n_elites, dim=0).indices  # (n_elites, batch)
            elite_value = value.take_along_dim(elite_idxs, dim=0)  # (n_elites, batch)
            # (horizon, n_elites, batch, action_dim)
            elite_actions = actions.take_along_dim(einops.rearrange(elite_idxs, "n b -> 1 n b 1"), dim=1)

            # Update gaussian PDF parameters to be the (weighted) mean and standard deviation of the elites.
            max_value = elite_value.max(0, keepdim=True)[0]  # (1, batch)
            # The weighting is a softmax over trajectory values. Note that this is not the same as the usage
            # of Ω in eqn 4 of the TD-MPC paper. Instead it is the normalized version of it: s = Ω/ΣΩ. This
            # makes the equations: μ = Σ(s⋅Γ), σ = Σ(s⋅(Γ-μ)²).
            score = torch.exp(self.config.elite_weighting_temperature * (elite_value - max_value))
            score /= score.sum(axis=0, keepdim=True)
            # (horizon, batch, action_dim)
            mean = torch.sum(einops.rearrange(score, "n b -> n b 1") * elite_actions, dim=1) / (
                einops.rearrange(score.sum(0), "b -> 1 b 1") + 1e-9
            )
            std = torch.sqrt(
                torch.sum(
                    einops.rearrange(score, "n b -> n b 1")
                    * (elite_actions - einops.rearrange(mean, "h b d -> h 1 b d")) ** 2,
                    dim=1,
                )
                / (einops.rearrange(score.sum(0), "b -> 1 b 1") + 1e-9)
            ).clamp_(self.config.min_std, self.config.max_std)

        # Keep track of the mean for warm-starting subsequent steps.
        self._prev_mean = mean

        # Randomly select one of the elite actions from the last iteration of MPPI/CEM using the softmax
        # scores from the last iteration.
        actions = elite_actions[:, torch.multinomial(score.T, 1).squeeze(), torch.arange(batch_size)]
        return actions

    @torch.no_grad()
    def estimate_value(self, z: Tensor, actions: Tensor):
        """Estimates the value of a trajectory as per eqn 4 of the FOWM paper.

        Args:
            z: (batch, latent_dim) tensor of initial latent states.
            actions: (horizon, batch, action_dim) tensor of action trajectories.
        Returns:
            (batch,) tensor of values.
        """
        # Initialize return and running discount factor.
        G, running_discount = 0, 1
        # Iterate over the actions in the trajectory to simulate the trajectory using the latent dynamics
        # model. Keep track of return.
        for t in range(actions.shape[0]):
            # Estimate the next state (latent) and reward.
            z, reward = self.model.latent_dynamics_and_reward(z, actions[t], discretize_reward=True)
            # Update the return and running discount.
            G += running_discount * reward
            running_discount *= self.config.discount

        # next_action = self.model.pi(z)[0]  # (batch, action_dim)
        # terminal_values = self.model.Qs(z, next_action, return_type="avg")  # (ensemble, batch)

        return G + running_discount * self.model.Qs(z, self.model.pi(z)[0], return_type="avg")

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor | float]:
        """Run the batch through the model and compute the loss.

        Returns a dictionary with loss as a tensor, and other information as native floats.
        """
        device = get_device_from_parameters(self)

        batch = self.normalize_inputs(batch)
        if self._use_image:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.image"] = batch[self.input_image_key]
        batch = self.normalize_targets(batch)

        info = {}

        # (b, t) -> (t, b)
        for key in batch:
            if batch[key].ndim > 1:
                batch[key] = batch[key].transpose(1, 0)

        action = batch["action"]  # (t, b, action_dim)
        reward = batch["next.reward"]  # (t, b)
        observations = {k: v for k, v in batch.items() if k.startswith("observation.")}

        # Apply random image augmentations.
        if self._use_image and self.config.max_random_shift_ratio > 0:
            observations["observation.image"] = flatten_forward_unflatten(
                partial(random_shifts_aug, max_random_shift_ratio=self.config.max_random_shift_ratio),
                observations["observation.image"],
            )

        # Get the current observation for predicting trajectories, and all future observations for use in
        # the latent consistency loss and TD loss.
        current_observation, next_observations = {}, {}
        for k in observations:
            current_observation[k] = observations[k][0]
            next_observations[k] = observations[k][1:]
        horizon, batch_size = next_observations[
            "observation.image" if self._use_image else "observation.environment_state"
        ].shape[:2]

        # Run latent rollout using the latent dynamics model and policy model.
        # Note this has shape `horizon+1` because there are `horizon` actions and a current `z`. Each action
        # gives us a next `z`.
        batch_size = batch["index"].shape[0]
        z_preds = torch.empty(horizon + 1, batch_size, self.config.latent_dim, device=device)
        z_preds[0] = self.model.encode(current_observation)
        reward_preds = torch.empty(horizon, batch_size, self.config.num_bins, device=device)
        for t in range(horizon):
            z_preds[t + 1], reward_preds[t] = self.model.latent_dynamics_and_reward(z_preds[t], action[t])

        # Compute Q value predictions based on the latent rollout.
        q_preds_ensemble = self.model.Qs(
            z_preds[:-1], action, return_type="all"
        )  # (ensemble, horizon, batch)
        info.update({"Q": q_preds_ensemble.mean().item()})

        # Compute various targets with stopgrad.
        with torch.no_grad():
            # Latent state consistency targets for consistency loss.
            z_targets = self.model.encode(next_observations)

            # Compute the TD-target from a reward and the next observation
            pi = self.model.pi(z_targets)[0]
            td_targets = (
                reward
                + self.config.discount
                * self.model.Qs(z_targets, pi, return_type="min", target=True).squeeze()
            )

        # Compute losses.
        # Exponentially decay the loss weight with respect to the timestep. Steps that are more distant in the
        # future have less impact on the loss. Note: unsqueeze will let us broadcast to (seq, batch).
        temporal_loss_coeffs = torch.pow(
            self.config.temporal_decay_coeff, torch.arange(horizon, device=device)
        ).unsqueeze(-1)

        # Compute consistency loss as MSE loss between latents predicted from the rollout and latents
        # predicted from the (target model's) observation encoder.
        consistency_loss = (
            (
                temporal_loss_coeffs
                * F.mse_loss(z_preds[1:], z_targets, reduction="none").mean(dim=-1)
                # `z_preds` depends on the current observation and the actions.
                * ~batch["observation.state_is_pad"][0]
                * ~batch["action_is_pad"]
                # `z_targets` depends on the next observation.
                * ~batch["observation.state_is_pad"][1:]
            )
            .sum(0)
            .mean()
        )
        # Compute the reward loss as MSE loss between rewards predicted from the rollout and the dataset
        # rewards.
        reward_loss = (
            (
                temporal_loss_coeffs
                * soft_cross_entropy(reward_preds, reward, self.config).mean(1)
                * ~batch["next.reward_is_pad"]
                * ~batch["observation.state_is_pad"][0]
                * ~batch["action_is_pad"]
            )
            .sum(0)
            .mean()
        )

        # Compute state-action value loss (TD loss) for all of the Q functions in the ensemble.
        ce_value_loss = 0.0
        for i in range(self.config.q_ensemble_size):
            ce_value_loss += soft_cross_entropy(q_preds_ensemble[i], td_targets, self.config).mean(1)

        q_value_loss = (
            (
                temporal_loss_coeffs
                * ce_value_loss
                # `q_preds_ensemble` depends on the first observation and the actions.
                * ~batch["observation.state_is_pad"][0]
                * ~batch["action_is_pad"]
                # q_targets depends on the reward and the next observations.
                * ~batch["next.reward_is_pad"]
                * ~batch["observation.state_is_pad"][1:]
            )
            .sum(0)
            .mean()
        )

        # Calculate the advantage weighted regression loss for π as detailed in FOWM 3.1.
        # We won't need these gradients again so detach.
        z_preds = z_preds.detach()
        action_preds, _, log_pis, _ = self.model.pi(z_preds[:-1])

        with torch.no_grad():
            # avoid unnessecary computation of the gradients during policy optimization
            # TODO (michel-aractingi): the same logic should be extended when adding task embeddings
            qs = self.model.Qs(z_preds[:-1], action_preds, return_type="avg")
            self.scale.update(qs[0])
            qs = self.scale(qs)

        pi_loss = (
            (self.config.entropy_coef * log_pis - qs).mean(dim=2)
            * temporal_loss_coeffs
            # `action_preds` depends on the first observation and the actions.
            * ~batch["observation.state_is_pad"][0]
            * ~batch["action_is_pad"]
        ).mean()

        loss = (
            self.config.consistency_coeff * consistency_loss
            + self.config.reward_coeff * reward_loss
            + self.config.value_coeff * q_value_loss
            + pi_loss
        )

        info.update(
            {
                "consistency_loss": consistency_loss.item(),
                "reward_loss": reward_loss.item(),
                "Q_value_loss": q_value_loss.item(),
                "pi_loss": pi_loss.item(),
                "loss": loss,
                "sum_loss": loss.item() * self.config.horizon,
                "pi_scale": float(self.scale.value),
            }
        )

        # Undo (b, t) -> (t, b).
        for key in batch:
            if batch[key].ndim > 1:
                batch[key] = batch[key].transpose(1, 0)

        return info

    def update(self):
        """Update the target model's using polyak averaging."""
        self.model.update_target_Q()


class TDMPC2WorldModel(nn.Module):
    """Latent dynamics model used in TD-MPC2."""

    def __init__(self, config: TDMPC2Config):
        super().__init__()
        self.config = config

        self._encoder = TDMPC2ObservationEncoder(config)

        # Define latent dynamics head
        self._dynamics = nn.Sequential(
            NormedLinear(config.latent_dim + config.output_shapes["action"][0], config.mlp_dim),
            NormedLinear(config.mlp_dim, config.mlp_dim),
            NormedLinear(config.mlp_dim, config.latent_dim, act=SimNorm(config.simnorm_dim)),
        )

        # Define reward head
        self._reward = nn.Sequential(
            NormedLinear(config.latent_dim + config.output_shapes["action"][0], config.mlp_dim),
            NormedLinear(config.mlp_dim, config.mlp_dim),
            nn.Linear(config.mlp_dim, max(config.num_bins, 1)),
        )

        # Define policy head
        self._pi = nn.Sequential(
            NormedLinear(config.latent_dim, config.mlp_dim),
            NormedLinear(config.mlp_dim, config.mlp_dim),
            nn.Linear(config.mlp_dim, 2 * config.output_shapes["action"][0]),
        )

        # Define ensemble of Q functions
        self._Qs = nn.ModuleList(
            [
                nn.Sequential(
                    NormedLinear(
                        config.latent_dim + config.output_shapes["action"][0],
                        config.mlp_dim,
                        dropout=config.dropout,
                    ),
                    NormedLinear(config.mlp_dim, config.mlp_dim),
                    nn.Linear(config.mlp_dim, max(config.num_bins, 1)),
                )
                for _ in range(config.q_ensemble_size)
            ]
        )

        self._init_weights()

        self._target_Qs = deepcopy(self._Qs).requires_grad_(False)

        self.log_std_min = torch.tensor(config.log_std_min)
        self.log_std_dif = torch.tensor(config.log_std_max) - self.log_std_min

        self.bins = torch.linspace(config.vmin, config.vmax, config.num_bins)
        self.config.bin_size = (config.vmax - config.vmin) / (config.num_bins - 1)

    def _init_weights(self):
        """Initialize model weights.
        Custom weight initializations proposed in TD-MPC2.

        """

        def _apply_fn(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ParameterList):
                for i, p in enumerate(m):
                    if p.dim() == 3:  # Linear
                        nn.init.trunc_normal_(p, std=0.02)  # Weight
                        nn.init.constant_(m[i + 1], 0)  # Bias

        self.apply(_apply_fn)

        # initialize parameters of the
        for m in [self._reward, *self._Qs]:
            assert isinstance(
                m[-1], nn.Linear
            ), "Sanity check. The last linear layer needs 0 initialization on weights."
            nn.init.zeros_(m[-1].weight)

    def to(self, *args, **kwargs):
        """
        Overriding `to` method to also move additional tensors to device.
        """
        super().to(*args, **kwargs)
        self.log_std_min = self.log_std_min.to(*args, **kwargs)
        self.log_std_dif = self.log_std_dif.to(*args, **kwargs)
        self.bins = self.bins.to(*args, **kwargs)
        return self

    def train(self, mode):
        super().train(mode)
        self._target_Qs.train(False)
        return self

    def encode(self, obs: dict[str, Tensor]) -> Tensor:
        """Encodes an observation into its latent representation."""
        return self._encoder(obs)

    def latent_dynamics_and_reward(
        self, z: Tensor, a: Tensor, discretize_reward: bool = False
    ) -> tuple[Tensor, Tensor, bool]:
        """Predict the next state's latent representation and the reward given a current latent and action.

        Args:
            z: (*, latent_dim) tensor for the current state's latent representation.
            a: (*, action_dim) tensor for the action to be applied.
        Returns:
            A tuple containing:
                - (*, latent_dim) tensor for the next state's latent representation.
                - (*,) tensor for the estimated reward.
        """
        x = torch.cat([z, a], dim=-1)
        reward = self._reward(x).squeeze(-1)
        if discretize_reward:
            reward = two_hot_inv(reward, self.bins)
        return self._dynamics(x), reward

    def latent_dynamics(self, z: Tensor, a: Tensor) -> Tensor:
        """Predict the next state's latent representation given a current latent and action.

        Args:
            z: (*, latent_dim) tensor for the current state's latent representation.
            a: (*, action_dim) tensor for the action to be applied.
        Returns:
            (*, latent_dim) tensor for the next state's latent representation.
        """
        x = torch.cat([z, a], dim=-1)
        return self._dynamics(x)

    def pi(self, z: Tensor) -> Tensor:
        """Samples an action from the learned policy.

        The policy can also have added (truncated) Gaussian noise injected for encouraging exploration when
        generating rollouts for online training.

        Args:
            z: (*, latent_dim) tensor for the current state's latent representation.
            std: The standard deviation of the injected noise.
        Returns:
            (*, action_dim) tensor for the sampled action.
        """
        mu, log_std = self._pi(z).chunk(2, dim=-1)
        log_std = self.log_std_min + 0.5 * self.log_std_dif * (torch.tanh(log_std) + 1)
        eps = torch.randn_like(mu)

        log_pi = gaussian_logprob(eps, log_std)
        pi = mu + eps * log_std.exp()
        mu, pi, log_pi = squash(mu, pi, log_pi)

        return pi, mu, log_pi, log_std

    def Qs(self, z: Tensor, a: Tensor, return_type: str = "min", target=False) -> Tensor:  # noqa: N802
        """Predict state-action value for all of the learned Q functions.

        Args:
            z: (*, latent_dim) tensor for the current state's latent representation.
            a: (*, action_dim) tensor for the action to be applied.
            return_type: either 'min' or 'all' otherwise the average is returned
        Returns:
            (q_ensemble, *) tensor for the value predictions of each learned Q function in the ensemble or the average or min
        """
        x = torch.cat([z, a], dim=-1)

        if target:
            out = torch.stack([q(x).squeeze(-1) for q in self._target_Qs], dim=0)
        else:
            out = torch.stack([q(x).squeeze(-1) for q in self._Qs], dim=0)

        if return_type == "all":
            return out

        Q1, Q2 = out[np.random.choice(len(self._Qs), size=2, replace=False)]
        Q1, Q2 = two_hot_inv(Q1, self.bins), two_hot_inv(Q2, self.bins)
        return torch.min(Q1, Q2) if return_type == "min" else (Q1 + Q2) / 2

    def update_target_Q(self):
        """
        Soft-update target Q-networks using Polyak averaging.
        """
        with torch.no_grad():
            for p, p_target in zip(self._Qs.parameters(), self._target_Qs.parameters(), strict=False):
                p_target.data.lerp_(p.data, self.config.target_model_momentum)


class TDMPC2ObservationEncoder(nn.Module):
    """Encode image and/or state vector observations."""

    def __init__(self, config: TDMPC2Config):
        """
        Creates encoders for pixel and/or state modalities.
        TODO(alexander-soare): The original work allows for multiple images by concatenating them along the
            channel dimension. Re-implement this capability.
        """
        super().__init__()
        self.config = config

        # Define the observation encoder whether its pixels or states
        encoder_dict = {}
        for obs_key in config.input_shapes:
            if "observation.image" in config.input_shapes:
                encoder_module = nn.Sequential(
                    nn.Conv2d(config.input_shapes[obs_key][0], config.image_encoder_hidden_dim, 7, stride=2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(config.image_encoder_hidden_dim, config.image_encoder_hidden_dim, 5, stride=2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(config.image_encoder_hidden_dim, config.image_encoder_hidden_dim, 3, stride=2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(config.image_encoder_hidden_dim, config.image_encoder_hidden_dim, 3, stride=1),
                )
                dummy_batch = torch.zeros(1, *config.input_shapes[obs_key])
                with torch.inference_mode():
                    out_shape = encoder_module(dummy_batch).shape[1:]
                encoder_module.extend(
                    nn.Sequential(
                        nn.Flatten(),
                        NormedLinear(np.prod(out_shape), config.latent_dim, act=SimNorm(config.simnorm_dim)),
                    )
                )

            elif (
                "observation.state" in config.input_shapes
                or "observation.environment_state" in config.input_shapes
            ):
                encoder_module = nn.ModuleList()
                encoder_module.append(
                    NormedLinear(config.input_shapes[obs_key][0], config.state_encoder_hidden_dim)
                )
                assert config.num_enc_layers > 0
                for _ in range(config.num_enc_layers - 1):
                    encoder_module.append(
                        NormedLinear(config.state_encoder_hidden_dim, config.state_encoder_hidden_dim)
                    )
                encoder_module.append(
                    NormedLinear(
                        config.state_encoder_hidden_dim, config.latent_dim, act=SimNorm(config.simnorm_dim)
                    )
                )
                encoder_module = nn.Sequential(*encoder_module)

            else:
                raise NotImplementedError(f"No corresponding encoder module for key {obs_key}.")

            encoder_dict[obs_key.replace(".", "")] = encoder_module

        self.encoder = nn.ModuleDict(encoder_dict)

    def forward(self, obs_dict: dict[str, Tensor]) -> Tensor:
        """Encode the image and/or state vector.

        Each modality is encoded into a feature vector of size (latent_dim,) and then a uniform mean is taken
        over all features.
        """
        feat = []
        for obs_key in self.config.input_shapes:
            if "observation.image" in obs_key:
                feat.append(
                    flatten_forward_unflatten(self.encoder[obs_key.replace(".", "")], obs_dict[obs_key])
                )
            else:
                feat.append(self.encoder[obs_key.replace(".", "")](obs_dict[obs_key]))
        return torch.stack(feat, dim=0).mean(0)


def random_shifts_aug(x: Tensor, max_random_shift_ratio: float) -> Tensor:
    """Randomly shifts images horizontally and vertically.

    Adapted from https://github.com/facebookresearch/drqv2
    """
    b, _, h, w = x.size()
    assert h == w, "non-square images not handled yet"
    pad = int(round(max_random_shift_ratio * h))
    x = F.pad(x, tuple([pad] * 4), "replicate")
    eps = 1.0 / (h + 2 * pad)
    arange = torch.linspace(
        -1.0 + eps,
        1.0 - eps,
        h + 2 * pad,
        device=x.device,
        dtype=torch.float32,
    )[:h]
    arange = einops.repeat(arange, "w -> h w 1", h=h)
    base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
    base_grid = einops.repeat(base_grid, "h w c -> b h w c", b=b)
    # A random shift in units of pixels and within the boundaries of the padding.
    shift = torch.randint(
        0,
        2 * pad + 1,
        size=(b, 1, 1, 2),
        device=x.device,
        dtype=torch.float32,
    )
    shift *= 2.0 / (h + 2 * pad)
    grid = base_grid + shift
    return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


def flatten_forward_unflatten(fn: Callable[[Tensor], Tensor], image_tensor: Tensor) -> Tensor:
    """Helper to temporarily flatten extra dims at the start of the image tensor.

    Args:
        fn: Callable that the image tensor will be passed to. It should accept (B, C, H, W) and return
            (B, *), where * is any number of dimensions.
        image_tensor: An image tensor of shape (**, C, H, W), where ** is any number of dimensions, generally
            different from *.
    Returns:
        A return value from the callable reshaped to (**, *).
    """
    if image_tensor.ndim == 4:
        return fn(image_tensor)
    start_dims = image_tensor.shape[:-3]
    inp = torch.flatten(image_tensor, end_dim=-4)
    flat_out = fn(inp)
    return torch.reshape(flat_out, (*start_dims, *flat_out.shape[1:]))


class RunningScale:
    """Running trimmed scale estimator."""

    def __init__(self, tau):
        self.tau = tau
        self._value = torch.ones(1, dtype=torch.float32, device=torch.device("cuda"))
        self._percentiles = torch.tensor([5, 95], dtype=torch.float32, device=torch.device("cuda"))

    def state_dict(self):
        return dict(value=self._value, percentiles=self._percentiles)

    def load_state_dict(self, state_dict):
        self._value.data.copy_(state_dict["value"])
        self._percentiles.data.copy_(state_dict["percentiles"])

    @property
    def value(self):
        return self._value.cpu().item()

    def _percentile(self, x):
        x_dtype, x_shape = x.dtype, x.shape
        x = x.view(x.shape[0], -1)
        in_sorted, _ = torch.sort(x, dim=0)
        positions = self._percentiles * (x.shape[0] - 1) / 100
        floored = torch.floor(positions)
        ceiled = floored + 1
        ceiled[ceiled > x.shape[0] - 1] = x.shape[0] - 1
        weight_ceiled = positions - floored
        weight_floored = 1.0 - weight_ceiled
        d0 = in_sorted[floored.long(), :] * weight_floored[:, None]
        d1 = in_sorted[ceiled.long(), :] * weight_ceiled[:, None]
        return (d0 + d1).view(-1, *x_shape[1:]).type(x_dtype)

    def update(self, x):
        percentiles = self._percentile(x.detach())
        value = torch.clamp(percentiles[1] - percentiles[0], min=1.0)
        self._value.data.lerp_(value, self.tau)

    def __call__(self, x, update=False):
        if update:
            self.update(x)
        return x * (1 / self.value)

    def __repr__(self):
        return f"RunningScale(S: {self.value})"
