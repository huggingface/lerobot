import torch
from torch import Tensor, nn


class SubgoalStatePredictor(nn.Module):
    """Predicts the *subgoal state* (the proprioceptive state at the next demonstrated pick/place
    event) from the scene alone: pooled multimodal latent + current state, no action-chunk input.

    Output is a single vector `[B, state_dim]` -- one subgoal per sample, not a per-timestep
    trajectory. `TemporalActionDecoder` broadcasts it across the horizon axis itself as a global
    conditioning signal (see `temporal_decoder.py`'s `subgoal_encoder`); this class does not, and
    must not be read as predicting a `[B, H, state_dim]` future-state trajectory.

    Used by the `temporal_decoder_subgoal` architecture. Unlike `legacy_diffusion` (see
    `legacy/state_predictor.py`'s `StatePredictor`), where a `nominal` action chunk already
    existed by the time the state predictor ran, `TemporalActionDecoder` needs the predicted
    subgoal *as one of its own inputs* -- so nothing resembling a candidate action chunk exists
    yet at the point this has to run, hence the action-free design.
    """

    def __init__(self, state_dim: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, pooled_latent: Tensor, state: Tensor) -> Tensor:
        """`pooled_latent`: [B, latent_dim]. `state`: [B, state_dim]. Returns [B, state_dim]."""
        return self.net(torch.cat((pooled_latent, state), dim=-1))


def completion_gap(predicted_subgoal_state: Tensor, actual_state: Tensor) -> Tensor:
    """Per-sample squared-L2 gap between a previously predicted subgoal state and the state
    actually observed now. Used at inference to decide whether the sub-goal the last committed
    chunk aimed for was actually reached. Shared by the main (`execution.ActionExecutor`) and
    legacy (`legacy/modeling_legacy_diffusion.py`) completion gates."""
    return (predicted_subgoal_state - actual_state).square().mean(dim=-1)
