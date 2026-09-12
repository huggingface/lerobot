#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from .configuration_rl_token import RLTokenConfig

WEIGHTS_FILENAME = "rl_token_model.pt"


def _causal_mask(length: int, device: torch.device) -> Tensor:
    return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)


class RLTokenModel(nn.Module):
    """Compress the final VLA token sequence into one task-specific RL token."""

    def __init__(self, config: RLTokenConfig) -> None:
        super().__init__()
        self.config = config
        width = config.token_dim
        feedforward_dim = int(width * config.mlp_ratio)

        self.encoder_input = nn.Linear(config.vla_dim, width)
        self.rl_embedding = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.encoder_position = nn.Parameter(torch.empty(config.max_tokens + 1, width))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=config.num_heads,
            dim_feedforward=feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.encoder_layers,
            norm=nn.LayerNorm(width),
            enable_nested_tensor=False,
        )

        self.decoder_input = nn.Linear(config.vla_dim, width)
        self.decoder_position = nn.Parameter(torch.empty(config.max_tokens, width))
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=config.num_heads,
            dim_feedforward=feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=config.decoder_layers,
            norm=nn.LayerNorm(width),
            enable_nested_tensor=False,
        )
        self.output_projection = nn.Linear(width, config.vla_dim)

        nn.init.trunc_normal_(self.encoder_position, std=0.02)
        nn.init.trunc_normal_(self.decoder_position, std=0.02)

    def _validate_inputs(self, embeddings: Tensor, mask: Tensor) -> Tensor:
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have shape [B, M, D]")
        if mask.shape != embeddings.shape[:2]:
            raise ValueError("mask must have shape [B, M]")
        if embeddings.shape[-1] != self.config.vla_dim:
            raise ValueError("embedding width does not match RLTokenConfig.vla_dim")
        if embeddings.shape[1] > self.config.max_tokens:
            raise ValueError(
                f"received {embeddings.shape[1]} VLA tokens but max_tokens={self.config.max_tokens}; "
                "increase max_tokens to reconstruct the complete sequence"
            )
        mask = mask.bool()
        if not mask.any(dim=1).all():
            raise ValueError("every sample must contain at least one valid VLA token")
        return mask

    def encode(self, embeddings: Tensor, mask: Tensor | None = None) -> Tensor:
        """Return z_rl, the encoder output at the appended learned token."""
        if mask is None:
            mask = torch.ones(embeddings.shape[:2], dtype=torch.bool, device=embeddings.device)
        mask = self._validate_inputs(embeddings, mask)
        batch_size, length, _ = embeddings.shape

        hidden = self.encoder_input(embeddings)
        rl_token = self.rl_embedding.expand(batch_size, -1, -1)
        hidden = torch.cat([hidden, rl_token], dim=1)
        hidden = hidden + self.encoder_position[: length + 1]
        padding_mask = torch.cat(
            [~mask, torch.zeros(batch_size, 1, dtype=torch.bool, device=mask.device)], dim=1
        )
        return self.encoder(hidden, src_key_padding_mask=padding_mask)[:, -1]

    def reconstruction_loss(self, embeddings: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Autoregressively reconstruct stop-gradient VLA embeddings through z_rl."""
        if mask is None:
            mask = torch.ones(embeddings.shape[:2], dtype=torch.bool, device=embeddings.device)
        targets = embeddings.detach()
        mask = self._validate_inputs(targets, mask)
        batch_size, length, _ = targets.shape

        rl_token = self.encode(targets, mask)
        decoder_inputs = torch.cat([rl_token.unsqueeze(1), self.decoder_input(targets[:, :-1])], dim=1)
        decoder_inputs = decoder_inputs + self.decoder_position[:length]
        input_valid = torch.cat(
            [torch.ones(batch_size, 1, dtype=torch.bool, device=mask.device), mask[:, :-1]], dim=1
        )
        decoded = self.decoder(
            decoder_inputs,
            mask=_causal_mask(length, targets.device),
            src_key_padding_mask=~input_valid,
        )
        predictions = self.output_projection(decoded)
        per_token_mse = (predictions - targets).square().mean(dim=-1)
        weights = mask.to(per_token_mse.dtype)
        loss = (per_token_mse * weights).sum() / weights.sum().clamp_min(1.0)
        return loss, rl_token

    @torch.no_grad()
    def rl_token(self, embeddings: Tensor, mask: Tensor | None = None) -> Tensor:
        return self.encode(embeddings, mask)

    def save_pretrained(self, directory: str | Path) -> None:
        directory = Path(directory)
        self.config.save_pretrained(directory)
        torch.save(self.state_dict(), directory / WEIGHTS_FILENAME)

    @classmethod
    def from_pretrained(
        cls, directory: str | Path, *, map_location: str | torch.device = "cpu"
    ) -> RLTokenModel:
        directory = Path(directory)
        model = cls(RLTokenConfig.from_pretrained(directory))
        state = torch.load(directory / WEIGHTS_FILENAME, map_location=map_location, weights_only=True)
        model.load_state_dict(state)
        return model.to(map_location)
