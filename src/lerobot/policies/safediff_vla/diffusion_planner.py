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
    """Residual temporal MLP predicting noise for a complete action chunk.

    Conditions on the pooled backbone latent, the backbone's own nominal action chunk, the
    diffusion timestep, and (unlike the pooled `latent`, which mixes vision/language/state and
    is heavily diluted by mean-pooling) the *current proprioceptive state* directly and
    unpooled — so the planner always has a sharp signal for the fine, near-contact corrections
    (e.g. gripper aperture) that the pooled latent tends to wash out.
    """

    def __init__(
        self, action_dim: int, state_dim: int, latent_dim: int, hidden_dim: int, time_dim: int
    ) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.input_projection = nn.Linear(action_dim * 2 + state_dim + latent_dim + time_dim, hidden_dim)
        self.network = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, action_dim)
        )

    def forward(
        self, noisy_actions: Tensor, timesteps: Tensor, latent: Tensor, nominal: Tensor, state: Tensor
    ) -> Tensor:
        horizon = noisy_actions.shape[-2]
        condition = torch.cat((latent, timestep_embedding(timesteps, self.time_dim)), dim=-1)
        condition = condition[:, None].expand(-1, horizon, -1)
        state = state[:, None].expand(-1, horizon, -1)
        return self.network(
            self.input_projection(torch.cat((noisy_actions, nominal, state, condition), dim=-1))
        )
