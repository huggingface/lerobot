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

    Shared fields (used by both MPPI and GBP):
        algorithm: Planning algorithm — "mppi" or "gbp".
        n_samples: Number of candidate trajectories sampled per iteration.
        n_iters: Maximum number of refinement iterations.
        noise_std: Initial noise standard deviation for trajectory perturbation.
        action_cost_coef: Optional coefficient for action magnitude regularization.
        noise_decay: Multiplicative decay applied to noise_std each iteration.

    MPPI-specific:
        temperature: Softmax temperature for importance weight computation.

    GBP-specific:
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

    # GBP
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

        with torch.no_grad():
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


class GBPPlanner(BasePlanner):
    """Gradient-based planning through the world model.

    Differentiates the latent cosine-similarity cost directly w.r.t. the
    action trajectory using autograd through the WM decoder only.
    The encoder tokens passed in are already detached, so no gradient
    reaches the image encoder or action head — only the WM decoder
    parameters are part of the computation graph.

    Per iteration:
      1. Mark the current action sequence as a differentiable leaf.
      2. Run one WM decoder forward pass (N=1, exact, no sampling).
      3. Compute cost = 1 - cosine_similarity(z_goal, z_pred).mean(tokens).
      4. Backpropagate through the WM decoder to get ∂cost/∂actions.
      5. Gradient descent step: mean ← mean - lr * grad.
      6. Early-stop if max(|Δmean|) < convergence_tol.
      7. Decay lr.
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
        cfg = self.config
        S, D = z_start.shape

        # Single-sample encoder input — gradient flows through WM only;
        # z_start is already detached (computed under torch.no_grad in _plan_action_chunk).
        encoder_in_1 = z_start.unsqueeze(1)  # (S, 1, D)

        mean = initial_actions.clone().detach()  # (T, action_dim)
        lr = cfg.lr

        for _ in range(cfg.n_iters):
            # Fresh leaf tensor each iteration — gradient w.r.t. this step's actions only.
            actions_opt = mean.requires_grad_(True)

            # enable_grad overrides any outer torch.no_grad() context (e.g. from the eval loop)
            # so that the WM forward+backward builds a computation graph we can differentiate.
            with torch.enable_grad():
                # WM forward: (S, 1, D).  Gradient flows through wm_decoder → actions_opt.
                z_pred = wm_predict_fn(encoder_in_1, encoder_pos, actions_opt.unsqueeze(1))

                cos_sim = F.cosine_similarity(z_goal, z_pred.squeeze(1), dim=-1).mean()
                cost = 1.0 - cos_sim

                if cfg.action_cost_coef > 0:
                    cost = cost + cfg.action_cost_coef * actions_opt.pow(2).mean()

                # Compute gradient w.r.t. actions_opt only — torch.autograd.grad does not
                # accumulate .grad on WM parameters, keeping the model weights untouched.
                if cost.grad_fn is None:
                    raise RuntimeError(
                        "GBPPlanner: cost has no grad_fn — wm_predict_fn output is fully detached "
                        "from the computation graph. Check that the WM decoder's output depends "
                        "on the action input."
                    )
                (grad,) = torch.autograd.grad(cost, actions_opt, allow_unused=True)
                if grad is None:
                    raise RuntimeError(
                        "GBPPlanner: gradient of cost w.r.t. actions is None — the WM output does "
                        "not depend on the action input. Check that wm_predict_fn uses the actions "
                        "argument in its computation."
                    )

            new_mean = mean - lr * grad
            delta = (new_mean - mean).abs().max()
            mean = new_mean.detach()

            if delta < cfg.convergence_tol:
                break

            lr = lr * cfg.lr_decay

        return mean  # (T, action_dim)


def make_planner(config: PlanningConfig) -> BasePlanner:
    """Instantiate a planner from a PlanningConfig."""
    if config.algorithm == "mppi":
        return MPPIPlanner(config)
    if config.algorithm == "gbp":
        return GBPPlanner(config)
    raise ValueError(f"Unknown planning algorithm: {config.algorithm!r}. Choose 'mppi' or 'gbp'.")
