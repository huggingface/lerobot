# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://github.com/NVlabs/Sana/tree/robot) at commit 1e7d6cc51,
# original path sana_robot/train/flow_matching/loss.py. Import paths are rewritten to the LeRobot package;
# the weighting of a rank by the trained chunks of all ranks is not vendored.
# Keep this file in sync with the upstream file.

from __future__ import annotations

import torch


def chunk_video_losses(prediction, target, weights, valid, chunks: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the video loss of every chunk [B, L] and the latent values it averages: the weighted squared error of the chunk's target latents [B, C, L * f, tokens], averaged over the valid ones. ``weights`` and ``valid`` hold one value per target latent frame [B, L * f]."""
    error = (target.float() - prediction.float()).square() * weights.float()[:, None, :, None]
    mask = valid.bool()[:, None, :, None].expand(error.shape)
    per_chunk = (int(chunks), error.shape[2] // int(chunks))
    sums = torch.where(mask, error, torch.zeros_like(error)).unflatten(2, per_chunk).sum(dim=(1, 3, 4))
    counts = mask.unflatten(2, per_chunk).sum(dim=(1, 3, 4))
    return sums / counts.clamp_min(1), counts


def chunk_action_losses(prediction, target, weights, mask, chunks: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the action loss of every chunk [B, L] and the values it averages: the weighted squared error of the chunk's action rows [B, L * A, 80], averaged over the cells of ``mask``. ``weights`` holds one value per action row [B, L * A]."""
    error = (prediction.float() - target.float()).square() * weights.float()[:, :, None]
    mask = mask.bool()
    per_chunk = (int(chunks), error.shape[1] // int(chunks))
    sums = torch.where(mask, error, torch.zeros_like(error)).unflatten(1, per_chunk).sum(dim=(2, 3))
    counts = mask.unflatten(1, per_chunk).sum(dim=(2, 3))
    return sums / counts.clamp_min(1), counts


def mean_over_trained_chunks(losses: torch.Tensor, trained: torch.Tensor) -> torch.Tensor:
    """Return the mean of chunk losses [B, L] over the trained chunks; 0, still in the graph, when there is none."""
    trained = trained.to(device=losses.device, dtype=torch.bool)
    return torch.where(trained, losses, torch.zeros_like(losses)).sum() / trained.sum().clamp_min(1)
