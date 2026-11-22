# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://nvlabs.github.io/Sana/Robot/) at commit bac6da0e4,
# original path sana_robot/policy/embedding/rope.py. Import paths are rewritten to the LeRobot package.
# Keep this file in sync with the upstream file.

from __future__ import annotations

import torch


class PhysicalTimeRotary:
    """Rotary positions over (time, height, width) on a physical clock: latent frame ``j`` of a video at ``fps`` sits at time ``base_fps * j / fps``, so videos of every frame rate share one time axis."""

    def __init__(self, attention_head_dim: int, base_fps: float = 16.0, theta: float = 10000.0) -> None:
        spatial_dim = 2 * (attention_head_dim // 6)
        self.axis_dims = (attention_head_dim - 2 * spatial_dim, spatial_dim, spatial_dim)
        self.base_fps = float(base_fps)
        self.theta = float(theta)

    def from_position_ids(self, position_ids: torch.Tensor) -> torch.Tensor:
        """Return the complex rotary phases [B, 1, N, head_dim / 2] of (time, y, x) position ids [B, N, 3]."""
        phases = []
        for axis, dim in enumerate(self.axis_dims):
            exponent = torch.arange(0, dim, 2, device=position_ids.device, dtype=torch.float64) / dim
            inverse_frequency = self.theta ** (-exponent)
            phases.append(position_ids[..., axis : axis + 1].to(torch.float64) * inverse_frequency)
        phase = torch.cat(phases, dim=-1)
        return torch.polar(torch.ones_like(phase), phase).unsqueeze(1)

    def grid_position_ids(
        self, frames: int, height: int, width: int, fps: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Return the (time, y, x) position ids [B, frames * height * width, 3] of one view's latent grid at ``fps`` [B]."""
        batch_size = fps.numel()
        frame_ids = torch.arange(frames, device=device, dtype=torch.float64)
        time_ids = self.base_fps * frame_ids[None] / fps[:, None]
        time_ids = time_ids[:, :, None, None].expand(batch_size, frames, height, width)
        height_ids = torch.arange(height, device=device, dtype=torch.float64)
        height_ids = height_ids[None, None, :, None].expand_as(time_ids)
        width_ids = torch.arange(width, device=device, dtype=torch.float64)
        width_ids = width_ids[None, None, None, :].expand_as(time_ids)
        return torch.stack((time_ids, height_ids, width_ids), dim=-1).reshape(
            batch_size, frames * height * width, 3
        )
