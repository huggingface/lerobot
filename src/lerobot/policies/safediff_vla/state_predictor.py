import torch
from torch import Tensor, nn


class StatePredictor(nn.Module):
    """Predicts the proprioceptive state expected `execute_horizon` steps after executing a
    candidate action chunk.

    This replaces the old `TrajectoryCritic` (task/risk success classifiers). Those needed
    `task_success`/`safety_violation` labels that no dataset in this repo actually populates, so
    they never received a real training signal. `observation.state` is, in contrast, already
    present for every frame of every dataset — so this head can be trained purely
    self-supervised (regress against the state actually observed later in the same episode) and
    always has a real reference signal to learn from.
    """

    def __init__(self, action_dim: int, state_dim: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, latent: Tensor, actions: Tensor) -> Tensor:
        """Return predicted state with shape ``(*actions.shape[:-2], state_dim)``."""
        encoded = self.trajectory_encoder(actions).mean(dim=-2)
        latent = latent.expand(*encoded.shape[:-1], latent.shape[-1])
        return self.head(torch.cat((encoded, latent), dim=-1))


def masked_mse_loss(predicted: Tensor, target: Tensor, is_pad: Tensor | None) -> Tensor:
    """MSE over the state-prediction target, ignoring entries padded past the episode end."""
    error = (predicted - target).square()
    if is_pad is None:
        return error.mean()
    valid = (~is_pad).float().unsqueeze(-1)
    denom = valid.sum().clamp_min(1)
    return (error * valid).sum() / denom / error.shape[-1]


def completion_gap(predicted_future_state: Tensor, actual_state: Tensor) -> Tensor:
    """Per-sample normalized L2 gap between a previously predicted future state and what the
    state actually turned out to be. Used at inference to decide whether the sub-goal the last
    chunk aimed for was actually reached."""
    return (predicted_future_state - actual_state).square().mean(dim=-1)
