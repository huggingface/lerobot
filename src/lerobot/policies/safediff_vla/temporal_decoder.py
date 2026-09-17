"""Deterministic Temporal Action Decoder -- the main SafeDiff-VLA architecture.

Replaces "diffuse a residual on top of SmolVLA's own nominal action chunk" (the old
`legacy_diffusion` architecture, see `legacy/diffusion_planner.py`): every configuration tried
there (critic-free, state-conditioned, subgoal-conditioned, temporal-conv-mixed) matched or
underperformed simply executing SmolVLA's own nominal chunk unmodified, once the
`action_horizon`/`execute_horizon` mismatch against SmolVLA's native `chunk_size` was fixed --
i.e. refining a nominal action chunk post hoc never once helped. This module instead treats
SmolVLA purely as a multimodal *encoder* (see `SafeDiffVLAPolicy._encode_multimodal_latent`) and
generates the action chunk directly:

    VLM latent tokens + current state (+ optional predicted subgoal state)
        -> Temporal Action Decoder (cross-attention + self-attention across the horizon axis)
        -> action trajectory [B, H, A]

`nominal` (SmolVLA's own action-head output) is not an input here at all -- it's kept around
elsewhere only as an ablation baseline (`architecture: smolvla_nominal`) and an optional
auxiliary-loss target, never as something this decoder edits.

The horizon-axis temporal mixing here is `nn.TransformerDecoder`'s self-attention among the
`horizon` action-query positions -- there is no additional Conv1D mixing layer in this decoder
(the Conv1D `temporal_kernel_size` / `num_temporal_layers` mixer lives only in the *legacy*
`ConditionalDiffusionPlanner`, see `legacy/diffusion_planner.py`).
"""

import math

import torch
from torch import Tensor, nn


def sinusoidal_positions(length: int, dim: int, device: torch.device) -> Tensor:
    """Fixed sin/cos positional embedding for the `length` horizon positions, shape [length, dim]."""
    half = dim // 2
    frequencies = torch.exp(-math.log(10_000) * torch.arange(half, device=device) / max(half - 1, 1))
    positions = torch.arange(length, device=device).float()[:, None] * frequencies[None]
    embedding = torch.cat((positions.sin(), positions.cos()), dim=-1)
    return torch.nn.functional.pad(embedding, (0, dim - embedding.shape[-1]))


class StateEncoder(nn.Module):
    """Small MLP embedding a `[B, state_dim]` (or `[B, H, state_dim]`) state into `hidden_dim`."""

    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state)


class TemporalActionDecoder(nn.Module):
    """Learnable action-query tokens, cross-attending to the VLM's multimodal latent tokens and
    conditioned on robot state, self-attending across the horizon axis (a plain Transformer
    decoder stack) so every position's prediction can depend on its neighbors -- unlike a
    per-position-independent MLP, which by construction cannot make the chunk cohere into a
    single smooth trajectory (see `diffusion_planner.py`'s docstring for the failure mode this
    caused in `legacy_diffusion`).
    """

    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        latent_dim: int,
        hidden_dim: int,
        horizon: int,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        use_subgoal: bool = False,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.use_subgoal = use_subgoal
        self.latent_projection = nn.Linear(latent_dim, hidden_dim)
        self.action_queries = nn.Parameter(torch.randn(horizon, hidden_dim) * 0.02)
        # Fixed (non-learnable) sin/cos position embedding, one per horizon slot. `action_queries`
        # is itself already per-position (so position is not strictly *required* here, the way it
        # would be for a shared/repeated query), but keeping this explicit rather than folding
        # position entirely into the learned query makes the horizon-axis structure legible and
        # is cheap.
        self.register_buffer(
            "horizon_position_embedding", sinusoidal_positions(horizon, hidden_dim, torch.device("cpu"))
        )
        self.state_encoder = StateEncoder(state_dim, hidden_dim)
        if use_subgoal:
            self.subgoal_encoder = StateEncoder(state_dim, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(
        self,
        latent_tokens: Tensor,
        latent_pad_mask: Tensor | None,
        current_state: Tensor,
        subgoal_state: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            latent_tokens: [B, N_tokens, latent_dim] VLM multimodal latent (see
                `SafeDiffVLAPolicy._encode_multimodal_latent`).
            latent_pad_mask: [B, N_tokens] bool, True where `latent_tokens` is real (not padding),
                or None if every token is valid.
            current_state: [B, state_dim].
            subgoal_state: [B, state_dim] predicted subgoal state -- a single target, not a
                per-timestep trajectory (see `state_predictor.py`'s `SubgoalStatePredictor`) --
                added as global conditioning to every query position. Required iff
                `use_subgoal=True`.

        Returns: action trajectory [B, H, action_dim].
        """
        if self.use_subgoal and subgoal_state is None:
            raise ValueError("This decoder was built with use_subgoal=True but got none.")
        batch_size = latent_tokens.shape[0]
        memory = self.latent_projection(latent_tokens)
        memory_key_padding_mask = None if latent_pad_mask is None else ~latent_pad_mask

        queries = (self.action_queries + self.horizon_position_embedding)[None].expand(batch_size, -1, -1)
        queries = queries + self.state_encoder(current_state)[:, None, :]
        if self.use_subgoal:
            queries = queries + self.subgoal_encoder(subgoal_state)[:, None, :]

        decoded = self.transformer(
            tgt=queries, memory=memory, memory_key_padding_mask=memory_key_padding_mask
        )
        return self.action_head(decoded)
