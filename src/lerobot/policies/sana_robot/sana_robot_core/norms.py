# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://github.com/NVlabs/Sana/tree/robot) at commit 1e7d6cc51,
# original path sana_robot/policy/backbone/norms.py. Import paths are rewritten to the LeRobot package.
# Keep this file in sync with the upstream file.

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root-mean-square normalization over one dimension with a learnable per-channel scale, computed in float32."""

    def __init__(self, dim: int, scale_factor: float = 1.0, eps: float = 1e-06, norm_dim: int = -1) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim) * scale_factor)
        self.norm_dim = norm_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_shape = [1] * x.dim()
        weight_shape[self.norm_dim] = -1
        values = x.float()
        normalized = values * torch.rsqrt(values.pow(2).mean(self.norm_dim, keepdim=True) + self.eps)
        return (self.weight.view(*weight_shape) * normalized).type_as(x)


class DepthRMSNorm(nn.Module):
    """Root-mean-square normalization over the last dimension without a learnable scale, for the AttnRes keys."""

    def __init__(self, eps: float = 1e-06) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
