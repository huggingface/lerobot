# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://github.com/NVlabs/Sana/tree/robot) at commit 1e7d6cc51,
# original path sana_robot/train/flow_matching/timesteps.py. Import paths are rewritten to the LeRobot package.
# Keep this file in sync with the upstream file.

from __future__ import annotations

import torch

TRAIN_TIMESTEPS = 1000
VIDEO_FLOW_SHIFT = 5.0
ACTION_FLOW_SHIFT = 1.0


def sample_chunk_timesteps(batch: int, chunks: int, device=None) -> torch.Tensor:
    """Return one raw training timestep per chunk [B, L], uniform over 0 .. 999 and at least 1: every chunk of a window is noised on its own."""
    timesteps = (torch.rand(batch, chunks, device=device) * TRAIN_TIMESTEPS).long()
    return timesteps.clamp_min(1)


def shifted_timesteps(timesteps: torch.Tensor, flow_shift: float) -> torch.Tensor:
    """Return raw timesteps moved by a flow shift, on the 0 .. 1000 scale: s = t / 1000 becomes shift * s / (1 + (shift - 1) * s)."""
    base = timesteps.to(torch.float64) / TRAIN_TIMESTEPS
    return (flow_shift * base / (1 + (flow_shift - 1) * base) * TRAIN_TIMESTEPS).to(torch.float32)


def shifted_sigmas(timesteps: torch.Tensor, flow_shift: float) -> torch.Tensor:
    """Return the noise level in [0, 1] of raw timesteps under a flow shift."""
    return (shifted_timesteps(timesteps, flow_shift) / TRAIN_TIMESTEPS).clamp(0.0, 1.0)


def model_timesteps(timesteps: torch.Tensor, flow_shift: float) -> torch.Tensor:
    """Return the timestep the policy is given for raw timesteps: the shifted timestep cut to a whole number."""
    return torch.trunc(shifted_timesteps(timesteps, flow_shift))
