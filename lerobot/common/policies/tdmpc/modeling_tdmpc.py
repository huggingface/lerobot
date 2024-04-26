"""

The comments in this code may sometimes refer to these references:
    TD-MPC paper: Temporal Difference Learning for Model Predictive Control (https://arxiv.org/abs/2203.04955)
    FOWM paper: Finetuning Offline World Models in the Real World (https://arxiv.org/abs/2310.16029)
"""

# ruff: noqa: N806

import time
from collections import deque
from copy import deepcopy
from typing import Callable

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

import lerobot.common.policies.tdmpc.helper as h
from lerobot.common.policies.utils import populate_queues
from lerobot.common.utils.utils import get_safe_torch_device


class TDMPCPolicy(nn.Module):
    """Implementation of TD-MPC learning + inference."""

    name = "tdmpc"

    def __init__(self, cfg, n_obs_steps, n_action_steps, device):
        super().__init__()
        self.action_dim = cfg.action_dim

        self.cfg = cfg
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.device = get_safe_torch_device(device)
        self.model = TOLD(cfg)
        self.model.to(self.device)
        self.model_target = deepcopy(self.model)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.model.eval()
        self.model_target.eval()

        self.image_aug = _RandomShiftsAug(cfg)

        self.register_buffer("step", torch.tensor(0))

    def save(self, fp):
        """Save state dict of TOLD model to filepath."""
        torch.save(self.state_dict(), fp)

    def load(self, fp):
        """Load a saved state dict from filepath into current agent."""
        self.load_state_dict(torch.load(fp))

    def reset(self):
        """
        Clear observation and action queues. Clear previous means for warm starting of MPPI/CEM. Should be
        called on `env.reset()`
        """
        self._queues = {
            "observation.image": deque(maxlen=self.n_obs_steps),
            "observation.state": deque(maxlen=self.n_obs_steps),
            "action": deque(maxlen=self.n_action_steps),
        }
        # Previous mean obtained from the cross-entropy method (CEM) used during MPC. It is used to warm start
        # CEM for the next step.
        self._prev_mean: torch.Tensor | None = None

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]):
        """Select a single action given environment observations."""
        assert "observation.image" in batch
        assert "observation.state" in batch
        assert len(batch) == 2

        self.eval()

        self._queues = populate_queues(self._queues, batch)

        # When the action queue is depleted, populate it again by querying the policy.
        if len(self._queues["action"]) == 0:
            batch = {key: torch.stack(list(self._queues[key]), dim=1) for key in batch}

            # TODO(now): this shouldn't be necessary. downstream code should handle this the same as
            # n_obs_step > 1
            if self.n_obs_steps == 1:
                # hack to remove the time dimension
                for key in batch:
                    assert batch[key].shape[1] == 1
                    batch[key] = batch[key][:, 0]

            # Batch processing is not handled internally, so process the batch in a loop.
            # TODO(now): Test that this loop works for batch size > 1.
            action = []  # will be a batch of actions for one step
            batch_size = batch["observation.image"].shape[0]
            for i in range(batch_size):
                # NOTE: Order of observations matters here.
                z = self.model.encode(
                    {k: batch[k][i : i + 1] for k in ["observation.image", "observation.state"]}
                )
                if self.cfg.mpc:  # noqa: SIM108
                    # Note: self.plan does not handle batches, hence the squeeze.
                    a = self.plan(z.squeeze(0))
                else:
                    # TODO(now): in the official implementation self.model.training should evaluate to True during
                    # rollouts generated while training.
                    a = self.model.pi(z, self.cfg.min_std * self.model.training).squeeze(0)
                action.append(a)
            action = torch.stack(action)

            # TODO(now): change this param to action_repeat and constrain n_action_steps to be 1. Add a TODO
            # to be able to do n_action_steps > 1 with action repeat = 0.
            if i in range(self.n_action_steps):
                self._queues["action"].append(action)

        action = self._queues["action"].popleft()
        return action

    @torch.no_grad()
    def plan(self, z: Tensor) -> Tensor:
        """Plan next action using TD-MPC inference.

        TODO(now) Extend this to be able to work with batches.
        TODO(now) Go batch first?
        Args:
            z: (latent_dim,) tensor for the initial state.
        Returns:
            (action_dim,) tensor for the next action.
        """
        # Sample Nπ trajectories from the policy.
        # TODO(now): Be more explicit with these params: num_pi_samples, num_gaussian_samples
        num_pi_trajs = int(self.cfg.mixture_coef * self.cfg.num_samples)
        # Note: if num_pi_trajs is 0 this is fine.
        pi_actions = torch.empty(self.cfg.horizon, num_pi_trajs, self.action_dim, device=self.device)
        if num_pi_trajs > 0:
            _z = einops.repeat(z, "d -> n d", n=num_pi_trajs)
            for t in range(self.cfg.horizon):
                # TODO(now): in the official implementation self.model.training should evaluate to True during
                # rollouts generated while training. Note that in the original impl they don't even use self.model.training here.
                pi_actions[t] = self.model.pi(_z, self.cfg.min_std * self.model.training)
                _z = self.model.latent_dynamics(_z, pi_actions[t])

        z = einops.repeat(z, "d -> n d", n=self.cfg.num_samples + num_pi_trajs)

        # Model Predictive Path Integral (MPPI) with the cross-entropy method (CEM) as the optimization
        # algorithm.
        # The initial mean and standard deviation for the cross-entropy method (CEM).
        # TODO(now): Document somewhere that CEM starts with the prior assumption that the actions centered
        # around 0.
        mean = torch.zeros(self.cfg.horizon, self.action_dim, device=self.device)
        # Maybe warm start CEM with the mean from the previous step.
        if self._prev_mean is not None:
            mean[:-1] = self._prev_mean[1:]
        std = self.cfg.max_std * torch.ones_like(mean)

        for _ in range(self.cfg.iterations):
            # Randomly sample action trajectories for the gaussian distribution.
            # TODO(now): I think this clamping makes assumptions about the input normalization.
            actions = torch.clamp(
                mean.unsqueeze(1)
                + std.unsqueeze(1)
                * torch.randn(self.cfg.horizon, self.cfg.num_samples, self.action_dim, device=std.device),
                -1,
                1,
            )
            # Also include those action trajectories produced by π.
            actions = torch.cat([actions, pi_actions], dim=1)

            # Compute elite actions.
            # TODO(now): It looks like pi_actions never changes in this loop so really we should only estimate
            # its values once.
            # TODO(now): Why would there be a nan? I'm assuming because of the recursive nature of the computation
            # and that at the start of training we have garbage weights?
            value = self.estimate_value(z, actions).nan_to_num_(0)  # shape (N+Nπ,)
            elite_idxs = torch.topk(value, self.cfg.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # Update guassian PDF parameters to be the (weighted) mean and standard deviation of the elites.
            max_value = elite_value.max(0)[0]
            # The weighting is a softmax over trajectory values. Note that this is not the same as the usage
            # of Ω in eqn 4 of the TD-MPC paper. Instead it is the normalized version of it: s = Ω/ΣΩ. This
            # makes the equations: μ = Σ(s⋅Γ), σ = Σ(s⋅(Γ-μ)²).
            score = torch.exp(self.cfg.temperature * (elite_value - max_value))
            score /= score.sum()
            _mean = torch.sum(einops.rearrange(score, "n -> n 1") * elite_actions, dim=1)
            _std = torch.sqrt(
                torch.sum(
                    einops.rearrange(score, "n -> n 1")
                    * (elite_actions - einops.rearrange(_mean, "h d -> h 1 d")) ** 2,
                    dim=1,
                )
            )
            # Update mean with an exponential moving average, and std with a direct replacement.
            mean = self.cfg.momentum * mean + (1 - self.cfg.momentum) * _mean
            std = _std.clamp_(self.cfg.min_std, self.cfg.max_std)

        # Keep track of the mean for warm-starting subsequent steps.
        self._prev_mean = mean

        # Randomly select one of the elite actions from the last iteration of MPPI/CEM using the softmax
        # scores from the last iteration.
        actions = elite_actions[:, torch.multinomial(score, 1).item()]

        # Select only the first action
        action = actions[0]
        # TODO(now): in the official implementation this should evaluate to True during rollouts generated
        # while training. But should it really? Why add more noise yet again?
        if self.model.training:
            action += std[0] * torch.randn_like(std[0])
        # TODO(now): This clamping makes an assumption about the action space.
        return torch.clamp(action, -1, 1)

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

            # TODO(now): The uncertainty cost is the lambda used in eq 4 of the FOWM paper. Give it a better
            # name.
            if self.cfg.uncertainty_cost > 0:
                regularization = -(self.cfg.uncertainty_cost * self.model.Qs(z, actions[t]).std(0))
            else:
                regularization = 0
            # Estimate the next state (latent) and reward.
            z, reward = self.model.latent_dynamics_and_reward(z, actions[t])
            # Update the return and running discount.
            G += running_discount * (reward + regularization)
            running_discount *= self.cfg.discount
        # Add the estimated value of the final state (using the minimum for a conservative estimate).
        # Do so by predicting the next action (with added noise), then computing the
        # TODO(now): Should there be added noise here at inference time?
        next_action = self.model.pi(z, self.cfg.min_std)  # (batch, action_dim)
        terminal_values = self.model.Qs(z, next_action)
        # Randomly choose 2 of the Qs for terminal value estimation (as in App C. of the FOWM paper).
        if self.cfg.num_q > 2:
            G += running_discount * torch.min(terminal_values[torch.randint(0, self.cfg.num_q, size=(2,))])
        else:
            G += running_discount * torch.min(terminal_values)
        # Finally, also regularize the terminal value.
        if self.cfg.uncertainty_cost > 0:
            G -= running_discount * self.cfg.uncertainty_cost * terminal_values.std(0)
        return G

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run the batch through the model and compute the loss."""
        info = {}

        # TODO(alexander-soare): Refactor TDMPC and make it comply with the policy interface documentation.
        batch_size = batch["index"].shape[0]

        # TODO(now): batch first
        for key in batch:
            if batch[key].ndim > 1:
                batch[key] = batch[key].transpose(1, 0)

        action = batch["action"]
        reward = batch["next.reward"]
        observations = {k: v for k, v in batch.items() if k.startswith("observation.")}

        # TODO(now): What are these?
        done = torch.zeros_like(reward, dtype=torch.bool, device=reward.device)
        mask = torch.ones_like(reward, dtype=torch.bool, device=reward.device)

        # Apply random image augmentations.
        observations["observation.image"] = _flatten_forward_unflatten(
            self.image_aug, observations["observation.image"]
        )

        # Get the current observation for predicting trajectories, and all future observations for use in
        # the latent consistency loss and TD loss.
        current_observation, next_observations = {}, {}
        for k in observations:
            current_observation[k] = observations[k][0]
            next_observations[k] = observations[k][1:]
        horizon = next_observations["observation.image"].shape[0]

        # Run latent rollout using the latent dynamics model and policy model.
        # Note this has shape `horizon+1` because there are `horizon` actions and a current `z`. Each action
        # gives us a next `z`.
        z_preds = torch.empty(horizon + 1, batch_size, self.cfg.latent_dim, device=self.device)
        z_preds[0] = self.model.encode(current_observation)
        reward_preds = torch.empty_like(reward, device=self.device)
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
            q_targets = reward + self.cfg.discount * mask * self.model.V(self.model.encode(next_observations))
            # From eqn 3 of FOWM. These appear as Q(z, a). Here we call them v_targets to emphasize that we
            # are using them to compute loss for V.
            v_targets = self.model_target.Qs(z_preds[:-1].detach(), action, return_min=True)

        # Compute losses.
        # TODO(now): use is_pad
        loss_mask = torch.ones_like(mask, device=self.device)
        for t in range(1, horizon):
            loss_mask[t] = loss_mask[t - 1] * (~done[t - 1])
        # Exponentially decay the loss weight with respect to the timestep. Steps that are more distant in the
        # future have less impact on the loss. Note: unsqueeze will let us broadcast to (seq, batch).
        rho = torch.pow(self.cfg.rho, torch.arange(horizon, device=self.device)).unsqueeze(-1)
        # Compute consistency loss as MSE loss between latents predicted from the rollout and latents
        # predicted from the (target model's) observation encoder.
        consistency_loss = (
            rho * F.mse_loss(z_preds[1:], z_targets, reduction="none").mean(dim=-1) * loss_mask
        ).mean()
        # Compute the reward loss as MSE loss between rewards predicted from the rollout and the dataset
        # rewards.
        reward_loss = (rho * F.mse_loss(reward_preds, reward, reduction="none") * loss_mask).mean()
        # Compute state-action value loss (TD loss) for all of the Q functions in the ensemble.
        q_value_loss = (
            F.mse_loss(
                q_preds_ensemble,
                einops.repeat(q_targets, "t b -> e t b", e=q_preds_ensemble.shape[0]),
                reduction="none",
            )
            .sum(0)  # sum over ensemble
            .mean()
        )
        # Compute state value loss as in eqn 3 of FOWM.
        diff = v_targets - v_preds
        # Expectile loss penalizes:
        #   - `v_preds <  v_targets` with weighting `expectile`
        #   - `v_preds >= v_targets` with weighting `1 - expectile`
        raw_v_value_loss = torch.where(diff > 0, self.cfg.expectile, (1 - self.cfg.expectile)) * (diff**2)
        v_value_loss = (rho * raw_v_value_loss * loss_mask).mean()

        # Calculate the advantage weighted regression loss for π as detailed in FOWM 3.1.
        # We won't need these gradients again so detach.
        z_preds = z_preds.detach()
        # Use stopgrad for the advantage calculation.
        with torch.no_grad():
            advantage = self.model_target.Qs(z_preds[:-1], action, return_min=True) - self.model.V(
                z_preds[:-1]
            )
            exp_advantage = torch.clamp(torch.exp(advantage * self.cfg.A_scaling), max=100.0)
        action_preds = self.model.pi(z_preds[:-1])
        # Calculate the MSE between the actions and the action predictions.
        # Note: FOWM's original code calculates the log probability (wrt to a unit standard deviation
        # gaussian) and sums over the action dimension. Computing the log probability amounts to multiplying
        # the MSE by 0.5 and adding a constant offset (the log(2*pi) term) . Here we drop the constant offset
        # as it doesn't change the optimization step, and we drop the 0.5 as we instead make a configuration
        # parameter for it (see below where we compute the total loss).
        mse = F.mse_loss(action_preds, action, reduction="none").sum(-1).mean()
        rho = torch.pow(self.cfg.rho, torch.arange(len(action), device=self.device))
        pi_loss = ((exp_advantage * mse).mean(dim=-1) * rho).mean()
        info["advantage"] = advantage[0]

        loss = (
            self.cfg.consistency_coef * consistency_loss
            + self.cfg.reward_coef * reward_loss
            + self.cfg.value_coef * q_value_loss
            + self.cfg.value_coef * v_value_loss
            + self.cfg.pi_coef * pi_loss
        )

        info.update(
            {
                "consistency_loss": consistency_loss,
                "reward_loss": reward_loss,
                "Q_value_loss": q_value_loss,
                "V_value_loss": v_value_loss,
                "pi_loss": pi_loss,
                "loss": loss,
                "sum_loss": loss * self.cfg.horizon,
            }
        )

        return info

    def update(self, batch: dict[str, Tensor]) -> dict[str, float]:
        """Run the model in train mode, compute the loss, and do an optimization step."""
        start_time = time.time()

        self.optim.zero_grad(set_to_none=True)
        self.model.train()

        fwd_info = self.forward(batch)
        loss = fwd_info["loss"]

        if torch.isnan(loss).item():
            raise RuntimeError("loss has nan")

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False
        )
        self.optim.step()

        info = {
            "consistency_loss": float(fwd_info["consistency_loss"].item()),
            "reward_loss": float(fwd_info["reward_loss"].item()),
            "Q_value_loss": float(fwd_info["Q_value_loss"].item()),
            "V_value_loss": float(fwd_info["V_value_loss"].item()),
            "pi_loss": float(fwd_info["pi_loss"].item()),
            "loss": float(loss.item()),
            "sum_loss": float(fwd_info["sum_loss"].item()),
            "grad_norm": float(grad_norm.item()),
            "lr": self.cfg.lr,
            "update_s": time.time() - start_time,
        }

        # Finalize update step by incrementing the step buffer and updating the ema model weights.
        # TODO(now): remove
        self.step += 1

        # Note a minor variation with respect to the original FOWM code. Here they do this based on an EMA
        # update frequency parameter which is set to 2 (every 2 steps an update is done). To simplify the code
        # we update every step and adjust the decay parameter `alpha` accordingly (0.99 -> 0.995)
        _update_ema_parameters(self.model_target, self.model, self.cfg.ema_alpha)

        return info


class TOLD(nn.Module):
    """Task-Oriented Latent Dynamics (TOLD) model used in TD-MPC."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._encoder = _ObservationEncoder(cfg)
        self._dynamics = nn.Sequential(
            nn.Linear(cfg.latent_dim + cfg.action_dim, cfg.mlp_dim),
            nn.LayerNorm(cfg.mlp_dim),
            nn.Mish(),
            nn.Linear(cfg.mlp_dim, cfg.mlp_dim),
            nn.LayerNorm(cfg.mlp_dim),
            nn.Mish(),
            nn.Linear(cfg.mlp_dim, cfg.latent_dim),
            nn.LayerNorm(cfg.latent_dim),
            nn.Sigmoid(),
        )
        self._reward = nn.Sequential(
            nn.Linear(cfg.latent_dim + cfg.action_dim, cfg.mlp_dim),
            nn.LayerNorm(cfg.mlp_dim),
            nn.Mish(),
            nn.Linear(cfg.mlp_dim, cfg.mlp_dim),
            nn.LayerNorm(cfg.mlp_dim),
            nn.Mish(),
            nn.Linear(cfg.mlp_dim, 1),
        )
        self._pi = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.mlp_dim),
            nn.LayerNorm(cfg.mlp_dim),
            nn.Mish(),
            nn.Linear(cfg.mlp_dim, cfg.mlp_dim),
            nn.LayerNorm(cfg.mlp_dim),
            nn.Mish(),
            nn.Linear(cfg.mlp_dim, cfg.action_dim),
        )
        self._Qs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(cfg.latent_dim + cfg.action_dim, cfg.mlp_dim),
                    nn.LayerNorm(cfg.mlp_dim),
                    nn.Tanh(),
                    nn.Linear(cfg.mlp_dim, cfg.mlp_dim),
                    nn.ELU(),
                    nn.Linear(cfg.mlp_dim, 1),
                )
                for _ in range(cfg.num_q)
            ]
        )
        self._V = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.mlp_dim),
            nn.LayerNorm(cfg.mlp_dim),
            nn.Tanh(),
            nn.Linear(cfg.mlp_dim, cfg.mlp_dim),
            nn.ELU(),
            nn.Linear(cfg.mlp_dim, 1),
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
            assert isinstance(
                m[-1], nn.Linear
            ), "Sanity check. The last linear layer needs 0 initialization on weights."
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
            # TODO(now): Understand why this has gradient pass-through internally but not for the clip
            # parameter.
            return h.TruncatedNormal(action, std).sample(clip=0.3)
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


class _ObservationEncoder(nn.Module):
    """Encode image and/or state vector observations."""

    def __init__(self, cfg):
        """
        Creates encoders for pixel and/or state modalities.
        TODO(now): Consolidate this into just working with a dict even if there is just one modality.
        TODO(alexander-soare): The original work allows for multiple images by concatenating them along the
            channel dimension. Re-implement this capability.
        """
        super().__init__()
        self.cfg = cfg

        # TODO(now): Should this handle single channel images?
        n_img_channels = 3

        if cfg.modality in ["pixels", "all"]:
            self.image_enc_layers = nn.Sequential(
                nn.Conv2d(n_img_channels, cfg.num_channels, 7, stride=2),
                nn.ReLU(),
                nn.Conv2d(cfg.num_channels, cfg.num_channels, 5, stride=2),
                nn.ReLU(),
                nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2),
                nn.ReLU(),
                nn.Conv2d(cfg.num_channels, cfg.num_channels, 3, stride=2),
                nn.ReLU(),
            )
            dummy_batch = torch.zeros(1, n_img_channels, cfg.img_size, cfg.img_size)
            with torch.inference_mode():
                out_shape = self.image_enc_layers(dummy_batch).shape[1:]
            self.image_enc_layers.extend(
                nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(np.prod(out_shape), cfg.latent_dim),
                    nn.LayerNorm(cfg.latent_dim),
                    nn.Sigmoid(),
                )
            )
        if cfg.modality in {"state", "all"}:
            self.state_enc_layers = nn.Sequential(
                nn.Linear(cfg.state_dim, cfg.enc_dim),
                nn.ELU(),
                nn.Linear(cfg.enc_dim, cfg.latent_dim),
                nn.LayerNorm(cfg.latent_dim),
                nn.Sigmoid(),
            )

    def forward(self, obs_dict: dict[str, Tensor]) -> Tensor:
        """Encode the image and/or state vector.

        Each modality is encoded into a feature vector of size (latent_dim,) and then a uniform mean is taken
        over all features.
        """
        feat = []
        if self.cfg.modality in {"pixels", "all"}:
            feat.append(_flatten_forward_unflatten(self.image_enc_layers, obs_dict["observation.image"]))
        if self.cfg.modality in {"state", "all"}:
            feat.append(self.state_enc_layers(obs_dict["observation.state"]))
        return torch.stack(feat, dim=0).mean(0)


class _RandomShiftsAug(nn.Module):
    """
    # TODO(now)
    Random shift image augmentation.
    Adapted from https://github.com/facebookresearch/drqv2
    """

    def __init__(self, cfg):
        super().__init__()
        assert cfg.modality in {"pixels", "all"}
        self.pad = int(cfg.img_size / 21)

    def forward(self, x):
        n, _, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps,
            1.0 - eps,
            h + 2 * self.pad,
            device=x.device,
            dtype=torch.float32,
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(
            0,
            2 * self.pad + 1,
            size=(n, 1, 1, 2),
            device=x.device,
            dtype=torch.float32,
        )
        shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


def _update_ema_parameters(ema_net: nn.Module, net: nn.Module, alpha: float):
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


def _flatten_forward_unflatten(fn: Callable[[Tensor], Tensor], image_tensor: Tensor) -> Tensor:
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
