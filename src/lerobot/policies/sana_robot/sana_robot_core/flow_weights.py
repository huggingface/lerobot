# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://github.com/NVlabs/Sana/tree/robot) at commit 1e7d6cc51,
# original path sana_robot/train/flow_matching/weights.py. Import paths are rewritten to the LeRobot package.
# Keep this file in sync with the upstream file.

from __future__ import annotations

import torch

from lerobot.policies.sana_robot.sana_robot_core.flow_timesteps import TRAIN_TIMESTEPS, shifted_sigmas

WEIGHT_FLOOR = 0.1


def timestep_weight_table(flow_shift: float, device=None) -> torch.Tensor:
    """Return the loss weight of every raw timestep [1000] under a flow shift: a bell around the middle noise level, zero at its lowest point, mean 1 over the timesteps, and at least ``WEIGHT_FLOOR`` on the noisier half."""
    levels = shifted_sigmas(torch.arange(TRAIN_TIMESTEPS, device=device), flow_shift)
    weights = torch.exp(-2 * (levels - 0.5).square())
    weights = weights - weights.min()
    weights = weights / weights.mean()
    return torch.where(levels > 0.5, weights.clamp_min(WEIGHT_FLOOR), weights)


def timestep_weights(timesteps: torch.Tensor, flow_shift: float) -> torch.Tensor:
    """Return the loss weight of raw timesteps under a flow shift, with the shape of ``timesteps``."""
    return timestep_weight_table(flow_shift, timesteps.device)[timesteps.long()]
