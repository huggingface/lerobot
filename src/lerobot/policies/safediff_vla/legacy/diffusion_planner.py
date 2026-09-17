import math

import torch
from torch import Tensor, nn


def timestep_embedding(timesteps: Tensor, dim: int) -> Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10_000) * torch.arange(half, device=timesteps.device) / max(half - 1, 1)
    )
    embedding = timesteps.float()[:, None] * frequencies[None]
    embedding = torch.cat((embedding.sin(), embedding.cos()), dim=-1)
    return torch.nn.functional.pad(embedding, (0, dim - embedding.shape[-1]))


class ConditionalDiffusionPlanner(nn.Module):
    """Temporal-conv MLP predicting noise for a complete action chunk.

    Conditions on the pooled backbone latent, the backbone's own nominal action chunk, the
    diffusion timestep, the *current proprioceptive state* (unlike the pooled `latent`, which
    mixes vision/language/state and is heavily diluted by mean-pooling, this is unpooled and
    sharp — a real signal for the fine, near-contact corrections gripper aperture etc. need), and
    the predicted *subgoal state* (where the state-predictor head expects the next pick/place
    event to happen, see `state_predictor.py`) — i.e. the planner computes a path from the current
    state towards that target, rather than having to re-derive both "where am I" and "where am I
    going" from the pooled latent alone.

    Earlier versions of this class projected each horizon position through a *shared but
    per-position-independent* MLP (plain `nn.Linear` on a `[B, H, D]` tensor only ever transforms
    the last axis; the horizon axis `H` is just another batch dimension to it). That gives the
    network zero receptive field across time: every one of the chunk's timesteps is denoised in
    total isolation from its neighbors. Per-step noise-MSE can still go down in that regime (each
    step individually is a well-posed regression target), but nothing forces the *sequence* of
    predictions to cohere into a smooth trajectory — this is the same failure mode diffusion
    trajectory models are generally built to avoid (see Diffusion Policy, Chi et al. 2023, and its
    use of a temporal U-Net for exactly this reason). The 1D convolutions below give each position
    a real receptive field over its neighbors, stacked (`num_temporal_layers`) to widen it further;
    the residual connection around them means the mixer starts as a small correction on top of the
    old per-position pathway rather than replacing it outright, for a stable training start.
    """

    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        latent_dim: int,
        hidden_dim: int,
        time_dim: int,
        temporal_kernel_size: int = 5,
        num_temporal_layers: int = 2,
    ) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.input_projection = nn.Linear(action_dim * 2 + state_dim * 2 + latent_dim + time_dim, hidden_dim)
        self.input_activation = nn.SiLU()
        padding = temporal_kernel_size // 2
        temporal_layers = []
        for _ in range(num_temporal_layers):
            temporal_layers += [
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=temporal_kernel_size, padding=padding),
                nn.SiLU(),
            ]
        self.temporal_mixer = nn.Sequential(*temporal_layers)
        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(
        self,
        noisy_actions: Tensor,
        timesteps: Tensor,
        latent: Tensor,
        nominal: Tensor,
        state: Tensor,
        subgoal: Tensor,
    ) -> Tensor:
        horizon = noisy_actions.shape[-2]
        condition = torch.cat((latent, timestep_embedding(timesteps, self.time_dim)), dim=-1)
        condition = condition[:, None].expand(-1, horizon, -1)
        state = state[:, None].expand(-1, horizon, -1)
        subgoal = subgoal[:, None].expand(-1, horizon, -1)
        hidden = self.input_activation(
            self.input_projection(torch.cat((noisy_actions, nominal, state, subgoal, condition), dim=-1))
        )
        # Conv1d expects [B, channels, length]; hidden is [B, horizon, hidden_dim].
        mixed = self.temporal_mixer(hidden.transpose(1, 2)).transpose(1, 2)
        return self.head(hidden + mixed)
