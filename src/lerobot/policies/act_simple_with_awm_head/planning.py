"""Test-time planners for latent state planning in ACT + AWM policy."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class PlanningConfig:
    """Configuration for test-time latent state planning.

    Shared fields (used by both MPPI and GCP):
        algorithm: Planning algorithm — "mppi" or "gcp".
        n_samples: Number of candidate trajectories sampled per iteration.
        n_iters: Maximum number of refinement iterations.
        noise_std: Initial noise standard deviation for trajectory perturbation.
        action_cost_coef: Optional coefficient for action magnitude regularization.
        noise_decay: Multiplicative decay applied to noise_std each iteration.

    MPPI-specific:
        temperature: Softmax temperature for importance weight computation.

    GCP-specific:
        lr: Gradient descent step size.
        lr_decay: Multiplicative decay applied to lr each iteration.
        convergence_tol: Early-stop when max absolute action change < this value.
        antithetic: Use antithetic perturbation pairs for variance reduction.
    """

    algorithm: str = "mppi"
    n_samples: int = 64
    n_iters: int = 5
    noise_std: float = 0.3
    action_cost_coef: float = 0.0
    noise_decay: float = 0.5

    # MPPI
    temperature: float = 1.0

    # GCP
    lr: float = 0.1
    lr_decay: float = 1.0
    convergence_tol: float = 1e-3
    antithetic: bool = True


class BasePlanner(ABC):
    """Abstract base class for test-time action chunk optimizers."""

    @abstractmethod
    def optimize(
        self,
        z_start: Tensor,
        encoder_pos: Tensor,
        z_goal: Tensor,
        initial_actions: Tensor,
        wm_predict_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    ) -> Tensor:
        """Optimize an action chunk to minimize latent distance to goal.

        Args:
            z_start: (S, dim_model) — current obs encoder input tokens.
            encoder_pos: (S, 1, dim_model) — positional embeddings from encoder.
            z_goal: (S, dim_model) — goal obs encoder input tokens.
            initial_actions: (T, action_dim) — BC warm start.
            wm_predict_fn: Callable[(S,N,D), (S,1,D), (T,N,A)] -> (S,N,D).
                Given batched encoder_in, encoder_pos, and N action sequences,
                returns N predicted next latents.

        Returns:
            Optimized action chunk of shape (T, action_dim).
        """
        ...


class MPPIPlanner(BasePlanner):
    """Model-Predictive Path Integral planner in world-model latent space.

    Iteratively refines a mean action trajectory by:
      1. Sampling K noised trajectories around the current mean.
      2. Rolling out the world model for each trajectory to get z_pred.
      3. Computing cost = 1 - cosine_similarity(z_goal, z_pred).mean(tokens).
      4. Re-weighting with softmax(-cost / temperature) and updating the mean.
    """

    def __init__(self, config: PlanningConfig):
        self.config = config

    def optimize(
        self,
        z_start: Tensor,
        encoder_pos: Tensor,
        z_goal: Tensor,
        initial_actions: Tensor,
        wm_predict_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    ) -> Tensor:
        T, action_dim = initial_actions.shape
        K = self.config.n_samples
        S, D = z_start.shape
        device = initial_actions.device
        dtype = initial_actions.dtype

        mean = initial_actions.clone()  # (T, action_dim)
        noise_std = self.config.noise_std

        # Pre-expand static tensors to avoid repeated allocation.
        encoder_in_k = z_start.unsqueeze(1).expand(S, K, D)  # (S, K, D)
        z_goal_k = z_goal.unsqueeze(1).expand(S, K, D)       # (S, K, D)

        for _ in range(self.config.n_iters):
            # Sample K perturbed trajectories around current mean.
            noise = torch.randn(T, K, action_dim, device=device, dtype=dtype) * noise_std
            actions_k = mean.unsqueeze(1) + noise  # (T, K, action_dim)

            # Predict future latent for all K trajectories.
            z_pred = wm_predict_fn(encoder_in_k, encoder_pos, actions_k)  # (S, K, D)

            # Cosine similarity cost, averaged over encoder tokens.
            cos_sim = F.cosine_similarity(z_goal_k, z_pred, dim=-1).mean(dim=0)  # (K,)
            cost = 1.0 - cos_sim

            if self.config.action_cost_coef > 0:
                action_cost = actions_k.pow(2).mean(dim=(0, 2))  # (K,)
                cost = cost + self.config.action_cost_coef * action_cost

            # MPPI importance weights.
            weights = F.softmax(-cost / self.config.temperature, dim=0)  # (K,)

            # Update mean as weighted sum of trajectories.
            mean = (actions_k * weights.view(1, K, 1)).sum(dim=1)  # (T, action_dim)
            noise_std = noise_std * self.config.noise_decay

        return mean  # (T, action_dim)


class GCPPlanner(BasePlanner):
    """Monte Carlo gradient planner in world-model latent space.

    Estimates first-order gradients of the planning objective via the
    score-function (REINFORCE) trick and performs gradient descent on the
    mean action trajectory.  Antithetic perturbation pairs halve variance
    for the same sample budget.  Iteration stops early when the action
    sequence has converged.

    Per iteration:
      1. Draw half_K perturbation vectors ε ~ N(0, noise_std² I).
         If antithetic=True, mirror them to get K = 2*half_K samples.
      2. Evaluate WM cost for all K perturbed trajectories.
      3. Estimate gradient: g ≈ (1 / (K * noise_std)) * Σ_k cost_k * ε_k
      4. Update mean:  mean ← mean - lr * g
      5. Early-stop if max(|mean_new - mean_old|) < convergence_tol.
      6. Decay noise_std and lr.
    """

    def __init__(self, config: PlanningConfig):
        self.config = config

    def optimize(
        self,
        z_start: Tensor,
        encoder_pos: Tensor,
        z_goal: Tensor,
        initial_actions: Tensor,
        wm_predict_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    ) -> Tensor:
        T, action_dim = initial_actions.shape
        S, D = z_start.shape
        device = initial_actions.device
        dtype = initial_actions.dtype

        cfg = self.config
        half_K = cfg.n_samples // 2
        K = half_K * 2 if cfg.antithetic else half_K

        mean = initial_actions.clone()  # (T, action_dim)
        noise_std = cfg.noise_std
        lr = cfg.lr

        encoder_in_k = z_start.unsqueeze(1).expand(S, K, D)  # (S, K, D)
        z_goal_k = z_goal.unsqueeze(1).expand(S, K, D)       # (S, K, D)

        for _ in range(cfg.n_iters):
            # --- sample perturbations ---
            eps = torch.randn(T, half_K, action_dim, device=device, dtype=dtype) * noise_std
            noise = torch.cat([eps, -eps], dim=1) if cfg.antithetic else eps  # (T, K, A)

            actions_k = mean.unsqueeze(1) + noise  # (T, K, action_dim)

            # --- world-model rollout and cost ---
            z_pred = wm_predict_fn(encoder_in_k, encoder_pos, actions_k)  # (S, K, D)
            cos_sim = F.cosine_similarity(z_goal_k, z_pred, dim=-1).mean(dim=0)  # (K,)
            cost = 1.0 - cos_sim

            if cfg.action_cost_coef > 0:
                cost = cost + cfg.action_cost_coef * actions_k.pow(2).mean(dim=(0, 2))

            # --- score-function gradient estimate ---
            # g ≈ E[cost * ε] / noise_std  (shape: T, action_dim)
            grad = (noise * cost.view(1, K, 1)).mean(dim=1) / noise_std

            # --- gradient descent step with early stopping ---
            new_mean = mean - lr * grad
            delta = (new_mean - mean).abs().max()
            mean = new_mean

            if delta < cfg.convergence_tol:
                break

            noise_std = noise_std * cfg.noise_decay
            lr = lr * cfg.lr_decay

        return mean  # (T, action_dim)


def make_planner(config: PlanningConfig) -> BasePlanner:
    """Instantiate a planner from a PlanningConfig."""
    if config.algorithm == "mppi":
        return MPPIPlanner(config)
    if config.algorithm == "gcp":
        return GCPPlanner(config)
    raise ValueError(f"Unknown planning algorithm: {config.algorithm!r}. Choose 'mppi' or 'gcp'.")
