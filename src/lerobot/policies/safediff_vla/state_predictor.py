import torch
from torch import Tensor, nn


class StatePredictor(nn.Module):
    """Predicts the *subgoal state*: the proprioceptive state at the next pick/place event (the
    frame where the demonstrated gripper next opens or closes — see
    `examples/safediff_vla/compute_subgoal_labels.py` for how that label is derived).

    This replaces the old `TrajectoryCritic` (task/risk success classifiers). Those needed
    `task_success`/`safety_violation` labels that no dataset in this repo actually populates, so
    they never received a real training signal. The subgoal label, in contrast, is derived
    entirely from data every episode already has (`action`'s gripper channel + `observation.state`)
    — no simulator access, no seeds, no new data collection — so this head always has a real
    target to learn from.

    Always called with the backbone's own `nominal` action chunk (never the diffusion-sampled or
    ground-truth clean chunk) so training and inference see the exact same input distribution —
    the previous state-predictor design called this with the ground-truth clean actions during
    training but the diffusion-sampled actions at inference, a train/inference mismatch that
    undermined the completion gate relying on it.
    """

    def __init__(self, action_dim: int, state_dim: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim + state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, latent: Tensor, actions: Tensor, state: Tensor) -> Tensor:
        """Return predicted subgoal state with shape ``(*actions.shape[:-2], state_dim)``.

        Conditions on the *current* state directly (not just the pooled `latent`, which mixes
        it with vision/language and dilutes it) so the head only has to learn the state *delta*
        to the subgoal, rather than re-deriving an absolute target from scratch.
        """
        encoded = self.trajectory_encoder(actions).mean(dim=-2)
        latent = latent.expand(*encoded.shape[:-1], latent.shape[-1])
        state = state.expand(*encoded.shape[:-1], state.shape[-1])
        return self.head(torch.cat((encoded, latent, state), dim=-1))


class FutureStatePredictor(nn.Module):
    """Predicts the subgoal state from the scene alone (pooled multimodal latent + current
    state), with no action-chunk input.

    Used by the `temporal_decoder_future_state` architecture: unlike `legacy_diffusion`, where a
    `nominal` action chunk already existed by the time the state predictor ran, the new
    `TemporalActionDecoder` needs the predicted subgoal *as one of its own inputs* — so nothing
    resembling a candidate action chunk exists yet at the point this has to run. `StatePredictor`
    above therefore doesn't fit here (its `trajectory_encoder` needs actions); this is the
    action-free equivalent.
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
    chunk aimed for was actually reached."""
    return (predicted_subgoal_state - actual_state).square().mean(dim=-1)
