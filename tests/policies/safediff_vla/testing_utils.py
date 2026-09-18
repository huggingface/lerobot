"""Shared test doubles for SafeDiff-VLA's main and legacy test suites. Not collected by pytest
(no `test_` prefix)."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_STATE


class TinyBackbone(nn.Module):
    """Serves both the legacy pooled-hook interface (`extract_safediff_features`, used by
    `LegacySafeDiffVLAPolicy`) and the new token-sequence interface (`encode_multimodal_latent` /
    `multimodal_latent_dim`, used by `temporal_decoder*`), plus a bare `predict_action_chunk` for
    `architecture="smolvla_nominal"` — mirroring the three ways a SafeDiff-VLA policy can talk to
    a real SmolVLA backbone, without needing one."""

    def __init__(
        self,
        horizon: int,
        action_dim: int,
        feature_dim: int = 12,
        num_latent_tokens: int = 6,
        state_dim: int = 5,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.safediff_latent_dim = feature_dim
        self.multimodal_latent_dim = feature_dim
        self.num_latent_tokens = num_latent_tokens
        self.projection = nn.Linear(state_dim, feature_dim)
        self.token_projection = nn.Linear(state_dim, feature_dim)
        self.nominal_head = nn.Linear(state_dim, action_dim)
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def extract_safediff_features(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        # The policy always hands the backbone a plain 2D [B, state_dim] current-state tensor.
        assert batch[OBS_STATE].ndim == 2
        latent = self.projection(batch[OBS_STATE])
        nominal = latent[:, None, : self.action_dim].expand(-1, self.horizon, -1).tanh()
        return nominal, latent

    def encode_multimodal_latent(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        assert batch[OBS_STATE].ndim == 2
        tokens = self.token_projection(batch[OBS_STATE])[:, None, :].expand(-1, self.num_latent_tokens, -1)
        return tokens, None

    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        assert batch[OBS_STATE].ndim == 2
        return self.nominal_head(batch[OBS_STATE])[:, None, :].expand(-1, self.horizon, -1).tanh()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        # Mirrors `SmolVLAPolicy.forward`'s (loss, loss_dict) contract for `architecture=
        # "smolvla_finetune"`: a loss that is differentiable through `self.nominal_head` (a
        # trainable backbone param), so gradient-flow tests can tell frozen from unfrozen.
        assert batch[OBS_STATE].ndim == 2
        pred = self.nominal_head(batch[OBS_STATE])[:, None, :].expand(-1, self.horizon, -1)
        loss = torch.nn.functional.mse_loss(pred, batch[ACTION])
        return loss, {"losses_after_forward": loss.item()}


class NominalCallForbiddenBackbone(TinyBackbone):
    """Same as `TinyBackbone`, except `predict_action_chunk` raises -- used to prove a code path
    never consults the backbone's own nominal action chunk (sanity test G)."""

    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        raise AssertionError("predict_action_chunk() must not be called on this path")
