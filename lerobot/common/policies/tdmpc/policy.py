# ruff: noqa: N806

import time
from collections import deque
from copy import deepcopy

import einops
import numpy as np
import torch
import torch.nn as nn

import lerobot.common.policies.tdmpc.helper as h
from lerobot.common.policies.utils import populate_queues
from lerobot.common.utils import get_safe_torch_device

FIRST_FRAME = 0


class TOLD(nn.Module):
    """Task-Oriented Latent Dynamics (TOLD) model used in TD-MPC."""

    def __init__(self, cfg):
        super().__init__()
        action_dim = cfg.action_dim

        self.cfg = cfg
        self._encoder = h.enc(cfg)
        self._dynamics = h.dynamics(cfg.latent_dim + action_dim, cfg.mlp_dim, cfg.latent_dim)
        self._reward = h.mlp(cfg.latent_dim + action_dim, cfg.mlp_dim, 1)
        self._pi = h.mlp(cfg.latent_dim, cfg.mlp_dim, action_dim)
        self._Qs = nn.ModuleList([h.q(cfg) for _ in range(cfg.num_q)])
        self._V = h.v(cfg)
        self.apply(h.orthogonal_init)
        for m in [self._reward, *self._Qs]:
            m[-1].weight.data.fill_(0)
            m[-1].bias.data.fill_(0)

    def track_q_grad(self, enable=True):
        """Utility function. Enables/disables gradient tracking of Q-networks."""
        for m in self._Qs:
            h.set_requires_grad(m, enable)

    def track_v_grad(self, enable=True):
        """Utility function. Enables/disables gradient tracking of Q-networks."""
        if hasattr(self, "_V"):
            h.set_requires_grad(self._V, enable)

    def encode(self, obs):
        """Encodes an observation into its latent representation."""
        out = self._encoder(obs)
        if isinstance(obs, dict):
            # fusion
            out = torch.stack([v for k, v in out.items()]).mean(dim=0)
        return out

    def next(self, z, a):
        """Predicts next latent state (d) and single-step reward (R)."""
        x = torch.cat([z, a], dim=-1)
        return self._dynamics(x), self._reward(x)

    def next_dynamics(self, z, a):
        """Predicts next latent state (d)."""
        x = torch.cat([z, a], dim=-1)
        return self._dynamics(x)

    def pi(self, z, std=0):
        """Samples an action from the learned policy (pi)."""
        mu = torch.tanh(self._pi(z))
        if std > 0:
            std = torch.ones_like(mu) * std
            return h.TruncatedNormal(mu, std).sample(clip=0.3)
        return mu

    def V(self, z):  # noqa: N802
        """Predict state value (V)."""
        return self._V(z)

    def Q(self, z, a, return_type):  # noqa: N802
        """Predict state-action value (Q)."""
        assert return_type in {"min", "avg", "all"}
        x = torch.cat([z, a], dim=-1)

        if return_type == "all":
            return torch.stack([q(x) for q in self._Qs], dim=0)

        idxs = np.random.choice(self.cfg.num_q, 2, replace=False)
        Q1, Q2 = self._Qs[idxs[0]](x), self._Qs[idxs[1]](x)
        return torch.min(Q1, Q2) if return_type == "min" else (Q1 + Q2) / 2


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
        self.std = h.linear_schedule(cfg.std_schedule, 0)
        self.model = TOLD(cfg)
        self.model.to(self.device)
        self.model_target = deepcopy(self.model)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr)
        # self.bc_optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.model.eval()
        self.model_target.eval()

        self.register_buffer("step", torch.zeros(1))

    def state_dict(self):
        """Retrieve state dict of TOLD model, including slow-moving target network."""
        return {
            "model": self.model.state_dict(),
            "model_target": self.model_target.state_dict(),
        }

    def save(self, fp):
        """Save state dict of TOLD model to filepath."""
        torch.save(self.state_dict(), fp)

    def load(self, fp):
        """Load a saved state dict from filepath into current agent."""
        d = torch.load(fp)
        self.model.load_state_dict(d["model"])
        self.model_target.load_state_dict(d["model_target"])

    def reset(self):
        """
        Clear observation and action queues. Should be called on `env.reset()`
        """
        self._queues = {
            "observation.image": deque(maxlen=self.n_obs_steps),
            "observation.state": deque(maxlen=self.n_obs_steps),
            "action": deque(maxlen=self.n_action_steps),
        }

    @torch.no_grad()
    def select_action(self, batch, step):
        assert "observation.image" in batch
        assert "observation.state" in batch
        assert len(batch) == 2

        self._queues = populate_queues(self._queues, batch)

        t0 = step == 0

        if len(self._queues["action"]) == 0:
            batch = {key: torch.stack(list(self._queues[key]), dim=1) for key in batch}

            if self.n_obs_steps == 1:
                # hack to remove the time dimension
                for key in batch:
                    assert batch[key].shape[1] == 1
                    batch[key] = batch[key][:, 0]

            actions = []
            batch_size = batch["observation.image"].shape[0]
            for i in range(batch_size):
                obs = {
                    "rgb": batch["observation.image"][[i]],
                    "state": batch["observation.state"][[i]],
                }
                # Note: unsqueeze needed because `act` still uses non-batch logic.
                action = self.act(obs, t0=t0, step=self.step)
                actions.append(action)
            action = torch.stack(actions)

            # self.act returns an action for 1 timestep only, so we copy it over `n_action_steps` time
            if i in range(self.n_action_steps):
                self._queues["action"].append(action)

        action = self._queues["action"].popleft()
        return action

    @torch.no_grad()
    def act(self, obs, t0=False, step=None):
        """Take an action. Uses either MPC or the learned policy, depending on the self.cfg.mpc flag."""
        obs = {k: o.detach() for k, o in obs.items()} if isinstance(obs, dict) else obs.detach()
        z = self.model.encode(obs)
        if self.cfg.mpc:
            a = self.plan(z, t0=t0, step=step)
        else:
            a = self.model.pi(z, self.cfg.min_std * self.model.training).squeeze(0)
        return a

    @torch.no_grad()
    def estimate_value(self, z, actions, horizon):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1
        for t in range(horizon):
            if self.cfg.uncertainty_cost > 0:
                G -= (
                    discount
                    * self.cfg.uncertainty_cost
                    * self.model.Q(z, actions[t], return_type="all").std(dim=0)
                )
            z, reward = self.model.next(z, actions[t])
            G += discount * reward
            discount *= self.cfg.discount
        pi = self.model.pi(z, self.cfg.min_std)
        G += discount * self.model.Q(z, pi, return_type="min")
        if self.cfg.uncertainty_cost > 0:
            G -= discount * self.cfg.uncertainty_cost * self.model.Q(z, pi, return_type="all").std(dim=0)
        return G

    @torch.no_grad()
    def plan(self, z, step=None, t0=True):
        """
        Plan next action using TD-MPC inference.
        z: latent state.
        step: current time step. determines e.g. planning horizon.
        t0: whether current step is the first step of an episode.
        """
        # during eval: eval_mode: uniform sampling and action noise is disabled during evaluation.

        assert step is not None
        # Seed steps
        if step < self.cfg.seed_steps and self.model.training:
            return torch.empty(self.action_dim, dtype=torch.float32, device=self.device).uniform_(-1, 1)

        # Sample policy trajectories
        horizon = int(min(self.cfg.horizon, h.linear_schedule(self.cfg.horizon_schedule, step)))
        num_pi_trajs = int(self.cfg.mixture_coef * self.cfg.num_samples)
        if num_pi_trajs > 0:
            pi_actions = torch.empty(horizon, num_pi_trajs, self.action_dim, device=self.device)
            _z = z.repeat(num_pi_trajs, 1)
            for t in range(horizon):
                pi_actions[t] = self.model.pi(_z, self.cfg.min_std)
                _z = self.model.next_dynamics(_z, pi_actions[t])

        # Initialize state and parameters
        z = z.repeat(self.cfg.num_samples + num_pi_trajs, 1)
        mean = torch.zeros(horizon, self.action_dim, device=self.device)
        std = self.cfg.max_std * torch.ones(horizon, self.action_dim, device=self.device)
        if not t0 and hasattr(self, "_prev_mean"):
            mean[:-1] = self._prev_mean[1:]

        # Iterate CEM
        for _ in range(self.cfg.iterations):
            actions = torch.clamp(
                mean.unsqueeze(1)
                + std.unsqueeze(1)
                * torch.randn(horizon, self.cfg.num_samples, self.action_dim, device=std.device),
                -1,
                1,
            )
            if num_pi_trajs > 0:
                actions = torch.cat([actions, pi_actions], dim=1)

            # Compute elite actions
            value = self.estimate_value(z, actions, horizon).nan_to_num_(0)
            elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # Update parameters
            max_value = elite_value.max(0)[0]
            score = torch.exp(self.cfg.temperature * (elite_value - max_value))
            score /= score.sum(0)
            _mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
            _std = torch.sqrt(
                torch.sum(
                    score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(1)) ** 2,
                    dim=1,
                )
                / (score.sum(0) + 1e-9)
            )
            _std = _std.clamp_(self.std, self.cfg.max_std)
            mean, std = self.cfg.momentum * mean + (1 - self.cfg.momentum) * _mean, _std

        # Outputs
        # TODO(rcadene): remove numpy with
        # # Convert score tensor to probabilities using softmax
        # probabilities = torch.softmax(score, dim=0)
        # # Generate a random sample index based on the probabilities
        # sample_index = torch.multinomial(probabilities, 1).item()
        score = score.squeeze(1).cpu().numpy()
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        self._prev_mean = mean
        mean, std = actions[0], _std[0]
        a = mean
        if self.model.training:
            a += std * torch.randn(self.action_dim, device=std.device)
        return torch.clamp(a, -1, 1)

    def update_pi(self, zs, acts=None):
        """Update policy using a sequence of latent states."""
        self.pi_optim.zero_grad(set_to_none=True)
        self.model.track_q_grad(False)
        self.model.track_v_grad(False)

        info = {}
        # Advantage Weighted Regression
        assert acts is not None
        vs = self.model.V(zs)
        qs = self.model_target.Q(zs, acts, return_type="min")
        adv = qs - vs
        exp_a = torch.exp(adv * self.cfg.A_scaling)
        exp_a = torch.clamp(exp_a, max=100.0)
        log_probs = h.gaussian_logprob(self.model.pi(zs) - acts, 0)
        rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
        pi_loss = -((exp_a * log_probs).mean(dim=(1, 2)) * rho).mean()
        info["adv"] = adv[0]

        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model._pi.parameters(),
            self.cfg.grad_clip_norm,
            error_if_nonfinite=False,
        )
        self.pi_optim.step()
        self.model.track_q_grad(True)
        self.model.track_v_grad(True)

        info["pi_loss"] = pi_loss.item()
        return pi_loss.item(), info

    @torch.no_grad()
    def _td_target(self, next_z, reward, mask):
        """Compute the TD-target from a reward and the observation at the following time step."""
        next_v = self.model.V(next_z)
        td_target = reward + self.cfg.discount * mask * next_v.squeeze(2)
        return td_target

    def forward(self, batch, step):
        """Main update function. Corresponds to one iteration of the model learning."""
        start_time = time.time()

        # num_slices = self.cfg.batch_size
        # batch_size = self.cfg.horizon * num_slices

        # if demo_buffer is None:
        #     demo_batch_size = 0
        # else:
        #     # Update oversampling ratio
        #     demo_pc_batch = h.linear_schedule(self.cfg.demo_schedule, step)
        #     demo_num_slices = int(demo_pc_batch * self.batch_size)
        #     demo_batch_size = self.cfg.horizon * demo_num_slices
        #     batch_size -= demo_batch_size
        #     num_slices -= demo_num_slices
        #     replay_buffer._sampler.num_slices = num_slices
        #     demo_buffer._sampler.num_slices = demo_num_slices

        #     assert demo_batch_size % self.cfg.horizon == 0
        #     assert demo_batch_size % demo_num_slices == 0

        # assert batch_size % self.cfg.horizon == 0
        # assert batch_size % num_slices == 0

        # # Sample from interaction dataset

        # def process_batch(batch, horizon, num_slices):
        #     # trajectory t = 256, horizon h = 5
        #     # (t h) ... -> h t ...
        #     batch = batch.reshape(num_slices, horizon).transpose(1, 0).contiguous()

        #     obs = {
        #         "rgb": batch["observation", "image"][FIRST_FRAME].to(self.device, non_blocking=True),
        #         "state": batch["observation", "state"][FIRST_FRAME].to(self.device, non_blocking=True),
        #     }
        #     action = batch["action"].to(self.device, non_blocking=True)
        #     next_obses = {
        #         "rgb": batch["next", "observation", "image"].to(self.device, non_blocking=True),
        #         "state": batch["next", "observation", "state"].to(self.device, non_blocking=True),
        #     }
        #     reward = batch["next", "reward"].to(self.device, non_blocking=True)

        #     idxs = batch["index"][FIRST_FRAME].to(self.device, non_blocking=True)
        #     weights = batch["_weight"][FIRST_FRAME, :, None].to(self.device, non_blocking=True)

        #     # TODO(rcadene): rearrange directly in offline dataset
        #     if reward.ndim == 2:
        #         reward = einops.rearrange(reward, "h t -> h t 1")

        #     assert reward.ndim == 3
        #     assert reward.shape == (horizon, num_slices, 1)
        #     # We dont use `batch["next", "done"]` since it only indicates the end of an
        #     # episode, but not the end of the trajectory of an episode.
        #     # Neither does `batch["next", "terminated"]`
        #     done = torch.zeros_like(reward, dtype=torch.bool, device=reward.device)
        #     mask = torch.ones_like(reward, dtype=torch.bool, device=reward.device)
        #     return obs, action, next_obses, reward, mask, done, idxs, weights

        # batch = replay_buffer.sample(batch_size) if self.cfg.balanced_sampling else replay_buffer.sample()

        # obs, action, next_obses, reward, mask, done, idxs, weights = process_batch(
        #     batch, self.cfg.horizon, num_slices
        # )

        # Sample from demonstration dataset
        # if demo_batch_size > 0:
        #     demo_batch = demo_buffer.sample(demo_batch_size)
        #     (
        #         demo_obs,
        #         demo_action,
        #         demo_next_obses,
        #         demo_reward,
        #         demo_mask,
        #         demo_done,
        #         demo_idxs,
        #         demo_weights,
        #     ) = process_batch(demo_batch, self.cfg.horizon, demo_num_slices)

        #     if isinstance(obs, dict):
        #         obs = {k: torch.cat([obs[k], demo_obs[k]]) for k in obs}
        #         next_obses = {k: torch.cat([next_obses[k], demo_next_obses[k]], dim=1) for k in next_obses}
        #     else:
        #         obs = torch.cat([obs, demo_obs])
        #         next_obses = torch.cat([next_obses, demo_next_obses], dim=1)
        #     action = torch.cat([action, demo_action], dim=1)
        #     reward = torch.cat([reward, demo_reward], dim=1)
        #     mask = torch.cat([mask, demo_mask], dim=1)
        #     done = torch.cat([done, demo_done], dim=1)
        #     idxs = torch.cat([idxs, demo_idxs])
        #     weights = torch.cat([weights, demo_weights])

        batch_size = batch["index"].shape[0]

        # TODO(rcadene): convert tdmpc with (batch size, time/horizon, channels)
        # instead of currently (time/horizon, batch size, channels) which is not the pytorch convention
        # batch size b = 256, time/horizon t = 5
        # b t ... -> t b ...
        for key in batch:
            if batch[key].ndim > 1:
                batch[key] = batch[key].transpose(1, 0)

        action = batch["action"]
        reward = batch["next.reward"]
        #  idxs = batch["index"]  # TODO(rcadene): use idxs to update sampling weights
        done = torch.zeros_like(reward, dtype=torch.bool, device=reward.device)
        mask = torch.ones_like(reward, dtype=torch.bool, device=reward.device)
        weights = torch.ones(batch_size, dtype=torch.bool, device=reward.device)

        obses = {
            "rgb": batch["observation.image"],
            "state": batch["observation.state"],
        }

        shapes = {}
        for k in obses:
            shapes[k] = obses[k].shape
            obses[k] = einops.rearrange(obses[k], "t b ... -> (t b) ... ")

        # Apply augmentations
        aug_tf = h.aug(self.cfg)
        obses = aug_tf(obses)

        for k in obses:
            t, b = shapes[k][:2]
            obses[k] = einops.rearrange(obses[k], "(t b) ... -> t b ... ", b=b, t=t)

        obs, next_obses = {}, {}
        for k in obses:
            obs[k] = obses[k][0]
            next_obses[k] = obses[k][1:].clone()

        horizon = next_obses["rgb"].shape[0]
        loss_mask = torch.ones_like(mask, device=self.device)
        for t in range(1, horizon):
            loss_mask[t] = loss_mask[t - 1] * (~done[t - 1])

        self.optim.zero_grad(set_to_none=True)
        self.std = h.linear_schedule(self.cfg.std_schedule, step)
        self.model.train()

        data_s = time.time() - start_time

        # Compute targets
        with torch.no_grad():
            next_z = self.model.encode(next_obses)
            z_targets = self.model_target.encode(next_obses)
            td_targets = self._td_target(next_z, reward, mask)

        # Latent rollout
        zs = torch.empty(horizon + 1, batch_size, self.cfg.latent_dim, device=self.device)
        reward_preds = torch.empty_like(reward, device=self.device)
        assert reward.shape[0] == horizon
        z = self.model.encode(obs)
        zs[0] = z
        value_info = {"Q": 0.0, "V": 0.0}
        for t in range(horizon):
            z, reward_pred = self.model.next(z, action[t])
            zs[t + 1] = z
            reward_preds[t] = reward_pred.squeeze(1)

        with torch.no_grad():
            v_target = self.model_target.Q(zs[:-1].detach(), action, return_type="min")

        # Predictions
        qs = self.model.Q(zs[:-1], action, return_type="all")
        qs = qs.squeeze(3)
        value_info["Q"] = qs.mean().item()
        v = self.model.V(zs[:-1])
        value_info["V"] = v.mean().item()

        # Losses
        rho = torch.pow(self.cfg.rho, torch.arange(horizon, device=self.device)).view(-1, 1)
        consistency_loss = (rho * torch.mean(h.mse(zs[1:], z_targets), dim=2) * loss_mask).sum(dim=0)
        reward_loss = (rho * h.mse(reward_preds, reward) * loss_mask).sum(dim=0)
        q_value_loss, priority_loss = 0, 0
        for q in range(self.cfg.num_q):
            q_value_loss += (rho * h.mse(qs[q], td_targets) * loss_mask).sum(dim=0)
            priority_loss += (rho * h.l1(qs[q], td_targets) * loss_mask).sum(dim=0)

        expectile = h.linear_schedule(self.cfg.expectile, step)
        v_value_loss = (rho * h.l2_expectile(v_target - v, expectile=expectile).squeeze(2) * loss_mask).sum(
            dim=0
        )

        total_loss = (
            self.cfg.consistency_coef * consistency_loss
            + self.cfg.reward_coef * reward_loss
            + self.cfg.value_coef * q_value_loss
            + self.cfg.value_coef * v_value_loss
        )

        weighted_loss = (total_loss * weights).mean()
        weighted_loss.register_hook(lambda grad: grad * (1 / self.cfg.horizon))
        has_nan = torch.isnan(weighted_loss).item()
        if has_nan:
            print(f"weighted_loss has nan: {total_loss=} {weights=}")
        else:
            weighted_loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.grad_clip_norm, error_if_nonfinite=False
        )
        self.optim.step()

        # if self.cfg.per:
        #     # Update priorities
        #     priorities = priority_loss.clamp(max=1e4).detach()
        #     has_nan = torch.isnan(priorities).any().item()
        #     if has_nan:
        #         print(f"priorities has nan: {priorities=}")
        #     else:
        #         replay_buffer.update_priority(
        #             idxs[:num_slices],
        #             priorities[:num_slices],
        #         )
        #         if demo_batch_size > 0:
        #             demo_buffer.update_priority(demo_idxs, priorities[num_slices:])

        # Update policy + target network
        _, pi_update_info = self.update_pi(zs[:-1].detach(), acts=action)

        if step % self.cfg.update_freq == 0:
            h.ema(self.model._encoder, self.model_target._encoder, self.cfg.tau)
            h.ema(self.model._Qs, self.model_target._Qs, self.cfg.tau)

        self.model.eval()

        info = {
            "consistency_loss": float(consistency_loss.mean().item()),
            "reward_loss": float(reward_loss.mean().item()),
            "Q_value_loss": float(q_value_loss.mean().item()),
            "V_value_loss": float(v_value_loss.mean().item()),
            "sum_loss": float(total_loss.mean().item()),
            "loss": float(weighted_loss.mean().item()),
            "grad_norm": float(grad_norm),
            "lr": self.cfg.lr,
            "data_s": data_s,
            "update_s": time.time() - start_time,
        }
        # info["demo_batch_size"] = demo_batch_size
        info["expectile"] = expectile
        info.update(value_info)
        info.update(pi_update_info)

        self.step[0] = step
        return info
