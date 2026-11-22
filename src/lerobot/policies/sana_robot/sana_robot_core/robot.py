# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://nvlabs.github.io/Sana/Robot/) at commit bac6da0e4,
# original path sana_robot/policy/embedding/robot.py. Import paths are rewritten to the LeRobot package.
# Keep this file in sync with the upstream file.

from __future__ import annotations

import torch
import torch.nn as nn

from lerobot.policies.sana_robot.sana_robot_core.embedders import FinalLayer

ROBOT_DIM = 80


class MaskedProjector(nn.Module):
    """Embed a robot state or action row [.., 80] and the mask of its used dimensions into one token; starts at zero."""

    def __init__(self, input_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(2 * input_dim, hidden_size)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        value = value.masked_fill(~mask, 0)
        return self.proj(torch.cat((value, mask.to(value.dtype)), dim=-1))


class ActionFinalLayer(FinalLayer):
    """Output head of the action rows, [B, A, 80]; starts at zero."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__(hidden_size, ROBOT_DIM)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
