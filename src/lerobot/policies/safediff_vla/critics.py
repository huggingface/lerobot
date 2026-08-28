import torch
from torch import Tensor, nn


class TrajectoryCritic(nn.Module):
    def __init__(self, action_dim: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, latent: Tensor, actions: Tensor) -> Tensor:
        """Return logits with shape ``actions.shape[:-2]``."""
        encoded = self.trajectory_encoder(actions).mean(dim=-2)
        while latent.ndim < encoded.ndim:
            latent = latent.unsqueeze(1)
        latent = latent.expand(*encoded.shape[:-1], latent.shape[-1])
        return self.head(torch.cat((encoded, latent), dim=-1)).squeeze(-1)


def score_candidates(
    task_logits: Tensor,
    risk_logits: Tensor,
    candidates: Tensor,
    nominal: Tensor,
    lambda_risk: float,
    lambda_prior: float,
) -> tuple[Tensor, Tensor]:
    prior_distance = (candidates - nominal[:, None]).square().mean(dim=(-1, -2))
    score = task_logits.sigmoid() - lambda_risk * risk_logits.sigmoid() - lambda_prior * prior_distance
    return score, prior_distance
