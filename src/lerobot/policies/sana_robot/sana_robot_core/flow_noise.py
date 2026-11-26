# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://github.com/NVlabs/Sana/tree/robot) at commit 1e7d6cc51,
# original path sana_robot/train/flow_matching/noise.py. Import paths are rewritten to the LeRobot package.
# Keep this file in sync with the upstream file.

from __future__ import annotations

from typing import NamedTuple

import torch


class FlowNoise(NamedTuple):
    """The random draws of a training step: one raw timestep per sample and chunk [B, L], and the noise of the target latents [B, C, L * f, tokens] and of the action rows [B, L * A, 80]."""

    timesteps: torch.Tensor
    video: torch.Tensor
    action: torch.Tensor


def add_noise(
    clean: torch.Tensor, noise: torch.Tensor, sigmas: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the noised sample (1 - sigma) * clean + sigma * noise and the velocity target noise - clean; ``sigmas`` broadcasts against ``clean``."""
    return (1 - sigmas) * clean + sigmas * noise, noise - clean
