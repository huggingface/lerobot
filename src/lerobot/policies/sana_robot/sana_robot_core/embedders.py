# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://nvlabs.github.io/Sana/Robot/) at commit bac6da0e4,
# original path sana_robot/policy/embedding/embedders.py. Import paths are rewritten to the LeRobot package.
# Keep this file in sync with the upstream file.

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


def modulate(x, shift, scale):
    """Scale and shift normalized tokens: ``x * (1 + scale) + shift``."""
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """Embed diffusion timesteps with a sinusoidal basis followed by an MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def sinusoidal_embedding(timestep: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Return the sinusoidal embedding [N, dim] of timesteps [N]."""
        half = dim // 2
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=timestep.device)
            / half
        )
        angles = timestep[:, None].float() * frequencies[None]
        embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        embedding = self.sinusoidal_embedding(timestep, self.frequency_embedding_size)
        return self.mlp(embedding.to(self.mlp[0].weight.dtype))


class TextProjection(nn.Module):
    """Two-layer MLP with a tanh-approximated GELU."""

    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class CaptionEmbedder(nn.Module):
    """Project the text encoder's embeddings to the hidden size."""

    def __init__(self, in_channels: int, hidden_size: int) -> None:
        super().__init__()
        self.y_proj = TextProjection(in_channels, hidden_size)

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        return self.y_proj(caption)


class FinalLayer(nn.Module):
    """Output head: a LayerNorm modulated per token by the time embedding, then a linear projection."""

    def __init__(self, hidden_size: int, out_features: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-06)
        self.linear = nn.Linear(hidden_size, out_features, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size**0.5)

    def forward(self, hidden: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        """Project tokens [B, N, hidden] under their time embedding [B, 1, N, hidden]."""
        (shift, scale) = (self.scale_shift_table[None, None] + time_embedding.transpose(1, 2)).unbind(2)
        return self.linear(modulate(self.norm_final(hidden), shift, scale))


class VideoPatchEmbed(nn.Module):
    """Embed video latents [B, C, F, H, W] into tokens [B, N, hidden] with a 3D convolution over patches."""

    def __init__(self, patch_size: tuple, in_channels: int, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels, hidden_size, kernel_size=tuple(patch_size), stride=tuple(patch_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)
