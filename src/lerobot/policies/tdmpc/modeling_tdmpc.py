#!/usr/bin/env python

# Copyright 2024 Nicklas Hansen, Xiaolong Wang, Hao Su,
# and The HuggingFace Inc. team. All rights reserved.
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
"""Implementation of Finetuning Offline World Models in the Real World.

The comments in this code may sometimes refer to these references:
    TD-MPC paper: Temporal Difference Learning for Model Predictive Control (https://huggingface.co/papers/2203.04955)
    FOWM paper: Finetuning Offline World Models in the Real World (https://huggingface.co/papers/2310.16029)
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
from torch import Tensor

from lerobot.constants import ACTION, OBS_ENV_STATE, OBS_IMAGE, OBS_STATE, REWARD
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.tdmpc.configuration_tdmpc import TDMPCConfig
from lerobot.policies.utils import get_device_from_parameters, get_output_shape, populate_queues


class TDMPCPolicy(PreTrainedPolicy):
    """Implementation of TD-MPC learning + inference.

    Please note several warnings for this policy.
        - Evaluation of pretrained weights created with the original FOWM code
            (https://github.com/fyhMer/fowm) works as expected. To be precise: we trained and evaluated a
            model with the FOWM code for the xarm_lift_medium_replay dataset. We ported the weights across
            to LeRobot, and were able to evaluate with the same success metric. BUT, we had to use inter-
            process communication to use the xarm environment from FOWM. This is because our xarm
            environment uses newer dependencies and does not match the environment in FOWM. See
            https://github.com/huggingface/lerobot/pull/103 for implementation details.
        - We have NOT checked that training on LeRobot reproduces the results from FOWM.
        - Nevertheless, we have verified that we can train TD-MPC for PushT. See
          `lerobot/configs/policy/tdmpc_pusht_keypoints.yaml`.
        - Our current xarm datasets were generated using the environment from FOWM. Therefore they do not
          match our xarm environment.
    """

    config_class = TDMPCConfig
    name = "tdmpc"

    def __init__(self, config: TDMPCConfig, dataset_stats: dict[str, dict[str, Tensor]] | None = None):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        self.model = TDMPCTOLD(config)
        self.model_target = deepcopy(self.model)
        for param in self.model_target.parameters():
            param.requires_grad = False

        self.reset()

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        """
        Clear observation and action queues. Clear previous means for warm starting of MPPI/CEM. Should be
        called on `env.reset()`
        """
        self._queues = {
            "observation.state": deque(maxlen=1),
            "action": deque(maxlen=max(self.config.n_action_steps, self.config.n_action_repeats)),
        }
        if self.config.image_features:
            self._queues["observation.image"] = deque(maxlen=1)
        if self.config.env_state_feature:
            self._queues["observation.environment_state"] = deque(maxlen=1)
        # Previous mean obtained from the cross-entropy method (CEM) used during MPC. It is used to warm start
        # CEM for the next step.
        self._prev_mean: torch.Tensor | None = None

    @torch.no_grad
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        batch = {key: torch.stack(list(self._queues[key]), dim=1) for key in batch if key in self._queues}

        # Remove the time dimensions as it is not handled yet.
        for key in batch:
            assert batch[key].shape[1] == 1
            batch[key] = batch[key][:, 0]

        # NOTE: Order of observations matters here.
        encode_keys = []
        if self.config.image_features:
            encode_keys.append(OBS_IMAGE)
        if self.config.env_state_feature:
            encode_keys.append(OBS_ENV_STATE)
        encode_keys.append(OBS_STATE)
        z = self.model.encode({k: batch[k] for k in encode_keys})
        if self.config.use_mpc:  # noqa: SIM108
            actions = self.plan(z)  # (horizon, batch, action_dim)
        else:
            # Plan with the policy (π) alone. This always returns one action so unsqueeze to get a
            # sequence dimension like in the MPC branch.
            actions = self.model.pi(z).unsqueeze(0)

        actions = torch.clamp(actions, -1, +1)

        actions = self.unnormalize_outputs({ACTION: actions})[ACTION]
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGE] = batch[next(iter(self.config.image_features))]

        self._queues = populate_queues(self._queues, batch)

        # When the action queue is depleted, populate it again by querying the policy.
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)

            if self.config.n_action_repeats > 1:
                for _ in range(self.config.n_action_repeats):
                    self._queues[ACTION].append(actions[0])
            else:
                # Action queue is (n_action_steps, batch_size, action_dim), so we transpose the action.
                self._queues[ACTION].extend(actions[: self.config.n_action_steps])

        action = self._queues[ACTION].popleft()
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
            self.config.action_feature.shape[0],
            device=device,
        )
        if self.config.n_pi_samples > 0:
            _z = einops.repeat(z, "b d -> n b d", n=self.config.n_pi_samples)
            for t in range(self.config.horizon):
                # Note: Adding a small amount of noise here doesn't hurt during inference and may even be
                # helpful for CEM.
                pi_actions[t] = self.model.pi(_z, self.config.min_std)
                _z = self.model.latent_dynamics(_z, pi_actions[t])

        # In the CEM loop we will need this for a call to estimate_value with the gaussian sampled
        # trajectories.
        z = einops.repeat(z, "b d -> n b d", n=self.config.n_gaussian_samples + self.config.n_pi_samples)

        # Model Predictive Path Integral (MPPI) with the cross-entropy method (CEM) as the optimization
        # algorithm.
        # The initial mean and standard deviation for the cross-entropy method (CEM).
        mean = torch.zeros(
            self.config.horizon, batch_size, self.config.action_feature.shape[0], device=device
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
                self.config.action_feature.shape[0],
                device=std.device,
            )
            gaussian_actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * std_normal_noise, -1, 1)

            # Compute elite actions.
            actions = torch.cat([gaussian_actions, pi_actions], dim=1)
            value = self.estimate_value(z, actions).nan_to_num_(0)
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
            _mean = torch.sum(einops.rearrange(score, "n b -> n b 1") * elite_actions, dim=1)
            _std = torch.sqrt(
                torch.sum(
                    einops.rearrange(score, "n b -> n b 1")
                    * (elite_actions - einops.rearrange(_mean, "h b d -> h 1 b d")) ** 2,
                    dim=1,
                )
            )
            # Update mean with an exponential moving average, and std with a direct replacement.
            mean = (
                self.config.gaussian_mean_momentum * mean + (1 - self.config.gaussian_mean_momentum) * _mean
            )
            std = _std.clamp_(self.config.min_std, self.config.max_std)

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
            # We will compute the reward in a moment. First compute the uncertainty regularizer from eqn 4
            # of the FOWM paper.
            if self.config.uncertainty_regularizer_coeff > 0:
                regularization = -(
                    self.config.uncertainty_regularizer_coeff * self.model.Qs(z, actions[t]).std(0)
                )
            else:
                regularization = 0
            # Estimate the next state (latent) and reward.
            z, reward = self.model.latent_dynamics_and_reward(z, actions[t])
            # Update the return and running discount.
            G += running_discount * (reward + regularization)
            running_discount *= self.config.discount
        # Add the estimated value of the final state (using the minimum for a conservative estimate).
        # Do so by predicting the next action, then taking a minimum over the ensemble of state-action value
        # estimators.
        # Note: This small amount of added noise seems to help a bit at inference time as observed by success
        # metrics over 50 episodes of xarm_lift_medium_replay.
        next_action = self.model.pi(z, self.config.min_std)  # (batch, action_dim)
        terminal_values = self.model.Qs(z, next_action)  # (ensemble, batch)
        # Randomly choose 2 of the Qs for terminal value estimation (as in App C. of the FOWM paper).
        if self.config.q_ensemble_size > 2:
            G += (
                running_discount
                * torch.min(terminal_values[torch.randint(0, self.config.q_ensemble_size, size=(2,))], dim=0)[
                    0
                ]
            )
        else:
            G += running_discount * torch.min(terminal_values, dim=0)[0]
        # Finally, also regularize the terminal value.
        if self.config.uncertainty_regularizer_coeff > 0:
            G -= running_discount * self.config.uncertainty_regularizer_coeff * terminal_values.std(0)
        return G

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss.

        Returns a dictionary with loss as a tensor, and other information as native floats.
        """
        device = get_device_from_parameters(self)

        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGE] = batch[next(iter(self.config.image_features))]
        batch = self.normalize_targets(batch)

        info = {}

        # (b, t) -> (t, b)
        for key in batch:
            if isinstance(batch[key], torch.Tensor) and batch[key].ndim > 1:
                batch[key] = batch[key].transpose(1, 0)

        action = batch[ACTION]  # (t, b, action_dim)
        reward = batch[REWARD]  # (t, b)
        observations = {k: v for k, v in batch.items() if k.startswith("observation.")}

        # Apply random image augmentations.
        if self.config.image_features and self.config.max_random_shift_ratio > 0:
            observations[OBS_IMAGE] = flatten_forward_unflatten(
                partial(random_shifts_aug, max_random_shift_ratio=self.config.max_random_shift_ratio),
                observations[OBS_IMAGE],
            )

        # Get the current observation for predicting trajectories, and all future observations for use in
        # the latent consistency loss and TD loss.
        current_observation, next_observations = {}, {}
        for k in observations:
            current_observation[k] = observations[k][0]
            next_observations[k] = observations[k][1:]
        horizon, batch_size = next_observations[
            OBS_IMAGE if self.config.image_features else OBS_ENV_STATE
        ].shape[:2]

        # Run latent rollout using the latent dynamics model and policy model.
        # Note this has shape `horizon+1` because there are `horizon` actions and a current `z`. Each action
        # gives us a next `z`.
        batch_size = batch["index"].shape[0]
        z_preds = torch.empty(horizon + 1, batch_size, self.config.latent_dim, device=device)
        z_preds[0] = self.model.encode(current_observation)
        reward_preds = torch.empty_like(reward, device=device)
        for t in range(horizon):
            z_preds[t + 1], reward_preds[t] = self.model.latent_dynamics_and_reward(z_preds[t], action[t])

        # Compute Q and V value predictions based on the latent rollout.
        q_preds_ensemble = self.model.Qs(z_preds[:-1], action)  # (ensemble, horizon, batch)
        v_preds = self.model.V(z_preds[:-1])
        info.update({"Q": q_preds_ensemble.mean().item(), "V": v_preds.mean().item()})

        # Compute various targets with stopgrad.
        with torch.no_grad():
            # Latent state consistency targets.
            z_targets = self.model_target.encode(next_observations)
            # State-action value targets (or TD targets) as in eqn 3 of the FOWM. Unlike TD-MPC which uses the
            # learned state-action value function in conjunction with the learned policy: Q(z, π(z)), FOWM
            # uses a learned state value function: V(z). This means the TD targets only depend on in-sample
            # actions (not actions estimated by π).
            # Note: Here we do not use self.model_target, but self.model. This is to follow the original code
            # and the FOWM paper.
            q_targets = reward + self.config.discount * self.model.V(self.model.encode(next_observations))
            # From eqn 3 of FOWM. These appear as Q(z, a). Here we call them v_targets to emphasize that we
            # are using them to compute loss for V.
            v_targets = self.model_target.Qs(z_preds[:-1].detach(), action, return_min=True)

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
                * F.mse_loss(reward_preds, reward, reduction="none")
                * ~batch["next.reward_is_pad"]
                # `reward_preds` depends on the current observation and the actions.
                * ~batch["observation.state_is_pad"][0]
                * ~batch["action_is_pad"]
            )
            .sum(0)
            .mean()
        )
        # Compute state-action value loss (TD loss) for all of the Q functions in the ensemble.
        q_value_loss = (
            (
                temporal_loss_coeffs
                * F.mse_loss(
                    q_preds_ensemble,
                    einops.repeat(q_targets, "t b -> e t b", e=q_preds_ensemble.shape[0]),
                    reduction="none",
                ).sum(0)  # sum over ensemble
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
        # Compute state value loss as in eqn 3 of FOWM.
        diff = v_targets - v_preds
        # Expectile loss penalizes:
        #   - `v_preds <  v_targets` with weighting `expectile_weight`
        #   - `v_preds >= v_targets` with weighting `1 - expectile_weight`
        raw_v_value_loss = torch.where(
            diff > 0, self.config.expectile_weight, (1 - self.config.expectile_weight)
        ) * (diff**2)
        v_value_loss = (
            (
                temporal_loss_coeffs
                * raw_v_value_loss
                # `v_targets` depends on the first observation and the actions, as does `v_preds`.
                * ~batch["observation.state_is_pad"][0]
                * ~batch["action_is_pad"]
            )
            .sum(0)
            .mean()
        )

        # Calculate the advantage weighted regression loss for π as detailed in FOWM 3.1.
        # We won't need these gradients again so detach.
        z_preds = z_preds.detach()
        # Use stopgrad for the advantage calculation.
        with torch.no_grad():
            advantage = self.model_target.Qs(z_preds[:-1], action, return_min=True) - self.model.V(
                z_preds[:-1]
            )
            info["advantage"] = advantage[0]
            # (t, b)
            exp_advantage = torch.clamp(torch.exp(advantage * self.config.advantage_scaling), max=100.0)
        action_preds = self.model.pi(z_preds[:-1])  # (t, b, a)
        # Calculate the MSE between the actions and the action predictions.
        # Note: FOWM's original code calculates the log probability (wrt to a unit standard deviation
        # gaussian) and sums over the action dimension. Computing the (negative) log probability amounts to
        # multiplying the MSE by 0.5 and adding a constant offset (the log(2*pi)/2 term, times the action
        # dimension). Here we drop the constant offset as it doesn't change the optimization step, and we drop
        # the 0.5 as we instead make a configuration parameter for it (see below where we compute the total
        # loss).
        mse = F.mse_loss(action_preds, action, reduction="none").sum(-1)  # (t, b)
        # NOTE: The original implementation does not take the sum over the temporal dimension like with the
        # other losses.
        # TODO(alexander-soare): Take the sum over the temporal dimension and check that training still works
        # as well as expected.
        pi_loss = (
            exp_advantage
            * mse
            * temporal_loss_coeffs
            # `action_preds` depends on the first observation and the actions.
            * ~batch["observation.state_is_pad"][0]
            * ~batch["action_is_pad"]
        ).mean()

        loss = (
            self.config.consistency_coeff * consistency_loss
            + self.config.reward_coeff * reward_loss
            + self.config.value_coeff * q_value_loss
            + self.config.value_coeff * v_value_loss
            + self.config.pi_coeff * pi_loss
        )

        info.update(
            {
                "consistency_loss": consistency_loss.item(),
                "reward_loss": reward_loss.item(),
                "Q_value_loss": q_value_loss.item(),
                "V_value_loss": v_value_loss.item(),
                "pi_loss": pi_loss.item(),
                "sum_loss": loss.item() * self.config.horizon,
            }
        )

        # Undo (b, t) -> (t, b).
        for key in batch:
            if isinstance(batch[key], torch.Tensor) and batch[key].ndim > 1:
                batch[key] = batch[key].transpose(1, 0)

        return loss, info

    def update(self):
        """Update the target model's parameters with an EMA step."""
        # Note a minor variation with respect to the original FOWM code. Here they do this based on an EMA
        # update frequency parameter which is set to 2 (every 2 steps an update is done). To simplify the code
        # we update every step and adjust the decay parameter `alpha` accordingly (0.99 -> 0.995)
        update_ema_parameters(self.model_target, self.model, self.config.target_model_momentum)


class TDMPCTOLD(nn.Module):
    """Task-Oriented Latent Dynamics (TOLD) model used in TD-MPC."""

    def __init__(self, config: TDMPCConfig):
        super().__init__()
        self.config = config
        self._encoder = TDMPCObservationEncoder(config)
        self._dynamics = nn.Sequential(
            nn.Linear(config.latent_dim + config.action_feature.shape[0], config.mlp_dim),
            nn.LayerNorm(config.mlp_dim),
            nn.Mish(),
            nn.Linear(config.mlp_dim, config.mlp_dim),
            nn.LayerNorm(config.mlp_dim),
            nn.Mish(),
            nn.Linear(config.mlp_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
            nn.Sigmoid(),
        )
        self._reward = nn.Sequential(
            nn.Linear(config.latent_dim + config.action_feature.shape[0], config.mlp_dim),
            nn.LayerNorm(config.mlp_dim),
            nn.Mish(),
            nn.Linear(config.mlp_dim, config.mlp_dim),
            nn.LayerNorm(config.mlp_dim),
            nn.Mish(),
            nn.Linear(config.mlp_dim, 1),
        )
        self._pi = nn.Sequential(
            nn.Linear(config.latent_dim, config.mlp_dim),
            nn.LayerNorm(config.mlp_dim),
            nn.Mish(),
            nn.Linear(config.mlp_dim, config.mlp_dim),
            nn.LayerNorm(config.mlp_dim),
            nn.Mish(),
            nn.Linear(config.mlp_dim, config.action_feature.shape[0]),
        )
        self._Qs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.latent_dim + config.action_feature.shape[0], config.mlp_dim),
                    nn.LayerNorm(config.mlp_dim),
                    nn.Tanh(),
                    nn.Linear(config.mlp_dim, config.mlp_dim),
                    nn.ELU(),
                    nn.Linear(config.mlp_dim, 1),
                )
                for _ in range(config.q_ensemble_size)
            ]
        )
        self._V = nn.Sequential(
            nn.Linear(config.latent_dim, config.mlp_dim),
            nn.LayerNorm(config.mlp_dim),
            nn.Tanh(),
            nn.Linear(config.mlp_dim, config.mlp_dim),
            nn.ELU(),
            nn.Linear(config.mlp_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights.

        Orthogonal initialization for all linear and convolutional layers' weights (apart from final layers
        of reward network and Q networks which get zero initialization).
        Zero initialization for all linear and convolutional layers' biases.
        """

        def _apply_fn(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                gain = nn.init.calculate_gain("relu")
                nn.init.orthogonal_(m.weight.data, gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_apply_fn)
        for m in [self._reward, *self._Qs]:
            assert isinstance(m[-1], nn.Linear), (
                "Sanity check. The last linear layer needs 0 initialization on weights."
            )
            nn.init.zeros_(m[-1].weight)
            nn.init.zeros_(m[-1].bias)  # this has already been done, but keep this line here for good measure

    def encode(self, obs: dict[str, Tensor]) -> Tensor:
        """Encodes an observation into its latent representation."""
        return self._encoder(obs)

    def latent_dynamics_and_reward(self, z: Tensor, a: Tensor) -> tuple[Tensor, Tensor]:
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
        return self._dynamics(x), self._reward(x).squeeze(-1)

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

    def pi(self, z: Tensor, std: float = 0.0) -> Tensor:
        """Samples an action from the learned policy.

        The policy can also have added (truncated) Gaussian noise injected for encouraging exploration when
        generating rollouts for online training.

        Args:
            z: (*, latent_dim) tensor for the current state's latent representation.
            std: The standard deviation of the injected noise.
        Returns:
            (*, action_dim) tensor for the sampled action.
        """
        action = torch.tanh(self._pi(z))
        if std > 0:
            std = torch.ones_like(action) * std
            action += torch.randn_like(action) * std
        return action

    def V(self, z: Tensor) -> Tensor:  # noqa: N802
        """Predict state value (V).

        Args:
            z: (*, latent_dim) tensor for the current state's latent representation.
        Returns:
            (*,) tensor of estimated state values.
        """
        return self._V(z).squeeze(-1)

    def Qs(self, z: Tensor, a: Tensor, return_min: bool = False) -> Tensor:  # noqa: N802
        """Predict state-action value for all of the learned Q functions.

        Args:
            z: (*, latent_dim) tensor for the current state's latent representation.
            a: (*, action_dim) tensor for the action to be applied.
            return_min: Set to true for implementing the detail in App. C of the FOWM paper: randomly select
                2 of the Qs and return the minimum
        Returns:
            (q_ensemble, *) tensor for the value predictions of each learned Q function in the ensemble OR
            (*,) tensor if return_min=True.
        """
        x = torch.cat([z, a], dim=-1)
        if not return_min:
            return torch.stack([q(x).squeeze(-1) for q in self._Qs], dim=0)
        else:
            if len(self._Qs) > 2:  # noqa: SIM108
                Qs = [self._Qs[i] for i in np.random.choice(len(self._Qs), size=2)]
            else:
                Qs = self._Qs
            return torch.stack([q(x).squeeze(-1) for q in Qs], dim=0).min(dim=0)[0]


class TDMPCObservationEncoder(nn.Module):
    """Encode image and/or state vector observations."""

    def __init__(self, config: TDMPCConfig):
        """
        Creates encoders for pixel and/or state modalities.
        TODO(alexander-soare): The original work allows for multiple images by concatenating them along the
            channel dimension. Re-implement this capability.
        """
        super().__init__()
        self.config = config

        if config.image_features:
            self.image_enc_layers = nn.Sequential(
                nn.Conv2d(
                    next(iter(config.image_features.values())).shape[0],
                    config.image_encoder_hidden_dim,
                    7,
                    stride=2,
                ),
                nn.ReLU(),
                nn.Conv2d(config.image_encoder_hidden_dim, config.image_encoder_hidden_dim, 5, stride=2),
                nn.ReLU(),
                nn.Conv2d(config.image_encoder_hidden_dim, config.image_encoder_hidden_dim, 3, stride=2),
                nn.ReLU(),
                nn.Conv2d(config.image_encoder_hidden_dim, config.image_encoder_hidden_dim, 3, stride=2),
                nn.ReLU(),
            )
            dummy_shape = (1, *next(iter(config.image_features.values())).shape)
            out_shape = get_output_shape(self.image_enc_layers, dummy_shape)[1:]
            self.image_enc_layers.extend(
                nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(np.prod(out_shape), config.latent_dim),
                    nn.LayerNorm(config.latent_dim),
                    nn.Sigmoid(),
                )
            )

        if config.robot_state_feature:
            self.state_enc_layers = nn.Sequential(
                nn.Linear(config.robot_state_feature.shape[0], config.state_encoder_hidden_dim),
                nn.ELU(),
                nn.Linear(config.state_encoder_hidden_dim, config.latent_dim),
                nn.LayerNorm(config.latent_dim),
                nn.Sigmoid(),
            )

        if config.env_state_feature:
            self.env_state_enc_layers = nn.Sequential(
                nn.Linear(config.env_state_feature.shape[0], config.state_encoder_hidden_dim),
                nn.ELU(),
                nn.Linear(config.state_encoder_hidden_dim, config.latent_dim),
                nn.LayerNorm(config.latent_dim),
                nn.Sigmoid(),
            )

    def forward(self, obs_dict: dict[str, Tensor]) -> Tensor:
        """Encode the image and/or state vector.

        Each modality is encoded into a feature vector of size (latent_dim,) and then a uniform mean is taken
        over all features.
        """
        feat = []
        # NOTE: Order of observations matters here.
        if self.config.image_features:
            feat.append(
                flatten_forward_unflatten(
                    self.image_enc_layers, obs_dict[next(iter(self.config.image_features))]
                )
            )
        if self.config.env_state_feature:
            feat.append(self.env_state_enc_layers(obs_dict[OBS_ENV_STATE]))
        if self.config.robot_state_feature:
            feat.append(self.state_enc_layers(obs_dict[OBS_STATE]))
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


def update_ema_parameters(ema_net: nn.Module, net: nn.Module, alpha: float):
    """Update EMA parameters in place with ema_param <- alpha * ema_param + (1 - alpha) * param."""
    for ema_module, module in zip(ema_net.modules(), net.modules(), strict=True):
        for (n_p_ema, p_ema), (n_p, p) in zip(
            ema_module.named_parameters(recurse=False), module.named_parameters(recurse=False), strict=True
        ):
            assert n_p_ema == n_p, "Parameter names don't match for EMA model update"
            if isinstance(p, dict):
                raise RuntimeError("Dict parameter not supported")
            if isinstance(module, nn.modules.batchnorm._BatchNorm) or not p.requires_grad:
                # Copy BatchNorm parameters, and non-trainable parameters directly.
                p_ema.copy_(p.to(dtype=p_ema.dtype).data)
            with torch.no_grad():
                p_ema.mul_(alpha)
                p_ema.add_(p.to(dtype=p_ema.dtype).data, alpha=1 - alpha)


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
