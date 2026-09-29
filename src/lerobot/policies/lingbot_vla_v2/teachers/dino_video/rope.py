# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""3D RoPE for the first-party DINO-video runtime.

The published teacher checkpoint ships two buffers that define the frequency
layout (both restored by :mod:`.checkpoint`):

- ``rope_embed.periods`` — spatial periods, ``head_dim // 4`` values;
- ``rope_embed.periods_t`` — temporal periods, ``head_dim // 8`` values.

Angle layout per head (``head_dim`` channels, e.g. 64 for ViT-L/16 heads):

- channels ``[0, head_dim/2)`` carry row angles, ``[head_dim/2, head_dim)``
  carry column angles; the spatial angle is ``2*pi * coord / periods[f]``
  where ``coord = 2 * (i + 0.5) / grid - 1`` (``normalize_coords="separate"``
  divides rows by the row count and columns by the column count);
- channels whose index satisfies ``idx % 4 == 3`` are overwritten with the
  temporal angle ``t / periods_t[f]`` — note there is **no** ``2*pi`` factor,
  so adjacent frames keep a fixed angular difference. The temporal coordinate
  is ``t = frame_index * base_fps / fps`` (physical time measured in units of
  ``1 / base_fps`` seconds); ``fps=None`` means unit steps;
- CLS / storage prefix tokens get identity rotation (angle 0) on spatial
  channels and, when ``prefix_temporal`` is set, their frame's temporal angle
  on temporal channels.

Rotation is the half-split convention: channel ``i`` pairs with
``i + head_dim/2`` via ``out = x * cos + rotate_half(x) * sin`` where
``rotate_half`` negates the second half. On temporal channels both members of
a pair share the same temporal frequency, so the temporal part is a proper 2D
rotation; spatial channels couple a row angle with a column angle, which is
exactly what the published weights were trained with.

Hard constraint: position buffers stay fp32 even when activations are bf16 —
they must not participate in module-level ``.to(dtype=...)`` casts
(:meth:`FirstPartyDinoVideoBackbone.cast_for_inference
<.backbone.FirstPartyDinoVideoBackbone.cast_for_inference>` restores them).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from .backbone import PackedVideoTokens

_TWO_PI = 2.0 * math.pi


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Negate-and-swap the two halves of the last dimension."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` by the precomputed tables (broadcast over heads)."""
    return x * cos + _rotate_half(x) * sin


class VideoRoPE(nn.Module):
    """Rotary embedding over (frame, row, col) token coordinates."""

    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float = 100.0,
        temporal_base: float = 10000.0,
        normalize_coords: str = "separate",
        prefix_temporal: bool = False,
        base_fps: float = 24.0,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if embed_dim % (4 * num_heads) != 0:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by 4 * num_heads {num_heads}.")
        if normalize_coords not in {"min", "max", "separate"}:
            raise ValueError(f"unknown normalize_coords {normalize_coords!r}.")
        self.head_dim = embed_dim // num_heads
        self.spatial_base = float(base)
        self.temporal_base = float(temporal_base)
        self.normalize_coords = normalize_coords
        self.prefix_temporal = bool(prefix_temporal)
        self.base_fps = float(base_fps)
        self.register_buffer(
            "periods", torch.empty(self.head_dim // 4, dtype=dtype, device=device), persistent=True
        )
        self.register_buffer(
            "periods_t", torch.empty(self.head_dim // 8, dtype=dtype, device=device), persistent=True
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Fill both period buffers with the published default frequencies.

        Exact values come from the checkpoint (strict restore); these defaults
        only make the module self-consistent before loading.
        """
        dd = {"device": self.periods.device, "dtype": self.periods.dtype}
        n_spatial = self.head_dim // 4
        self.periods.data = self.spatial_base ** (2 * torch.arange(n_spatial, **dd) / (self.head_dim // 2))
        n_temporal = self.head_dim // 8
        self.periods_t.data = self.temporal_base ** (2 * torch.arange(n_temporal, **dd) / (2 * n_temporal))

    @property
    def temporal_channel_mask(self) -> torch.Tensor:
        """Bool mask over head_dim that is True on temporal channels (idx % 4 == 3)."""
        return torch.arange(self.head_dim, device=self.periods.device) % 4 == 3

    def position_tables(
        self,
        coordinates: PackedVideoTokens,
        *,
        fps: float | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(sin, cos)`` of shape ``[batch or 1, 1, tokens, head_dim]``.

        ``batch`` is 1 for scalar/None ``fps`` and the clip batch size for a
        per-sample ``fps`` tensor; the singleton head axis lets the tables
        broadcast against ``[B, heads, tokens, head_dim]`` queries/keys.
        """
        device = self.periods.device
        grid_h, grid_w = _grid_size(coordinates)
        frame_index = coordinates.frame_index.to(device=device, dtype=self.periods.dtype)
        row = coordinates.row.to(device=device)
        col = coordinates.col.to(device=device)
        is_patch = row >= 0

        coords_r = self._spatial_coords(row, own_grid=grid_h, other_grid=grid_w)
        coords_c = self._spatial_coords(col, own_grid=grid_w, other_grid=grid_h)
        # Prefix tokens rotate by identity on spatial channels (angle 0).
        coords_r = torch.where(is_patch, coords_r, torch.zeros_like(coords_r))
        coords_c = torch.where(is_patch, coords_c, torch.zeros_like(coords_c))

        step = self._temporal_step(fps, batch_size=coordinates.flat.shape[0])
        if isinstance(step, torch.Tensor):
            t_coord = frame_index.unsqueeze(0) * step.unsqueeze(1)  # [batch, tokens]
        else:
            t_coord = (frame_index * step).unsqueeze(0)  # [1, tokens]
        batch = t_coord.shape[0]

        # Spatial angles: [tokens, head_dim // 2] per axis, duplicated frequencies.
        # The multiply-then-divide order matches the published teacher exactly.
        angles_r = (_TWO_PI * coords_r.unsqueeze(-1) / self.periods).repeat(1, 2)  # [tokens, hd // 2]
        angles_c = (_TWO_PI * coords_c.unsqueeze(-1) / self.periods).repeat(1, 2)
        angles = torch.cat((angles_r, angles_c), dim=-1)  # [tokens, head_dim]
        angles = angles.unsqueeze(0).expand(batch, -1, -1).clone()  # [batch, tokens, head_dim]

        # Temporal angles (no 2*pi factor), gated to 0 on prefix tokens unless
        # prefix_temporal is set.
        angles_t = (t_coord.unsqueeze(-1) / self.periods_t).repeat(1, 1, 2)  # [batch, tokens, head_dim // 4]
        temporal_gate = (is_patch | self.prefix_temporal).unsqueeze(-1)  # [tokens, 1]
        angles_t = torch.where(temporal_gate, angles_t, torch.zeros_like(angles_t))

        angles[..., self.temporal_channel_mask] = angles_t
        sin = torch.sin(angles).unsqueeze(1)  # [batch, 1, tokens, head_dim]
        cos = torch.cos(angles).unsqueeze(1)
        return sin, cos

    def forward(
        self,
        q: torch.Tensor,  # [B, heads, tokens, head_dim]
        k: torch.Tensor,  # [B, heads, tokens, head_dim]
        coordinates: PackedVideoTokens,
        *,
        fps: float | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate ``q``/``k`` in fp32 and return them in their input dtype."""
        sin, cos = self.position_tables(coordinates, fps=fps)
        q_dtype, k_dtype = q.dtype, k.dtype
        q = _apply_rope(q.to(sin.dtype), sin, cos).to(q_dtype)
        k = _apply_rope(k.to(sin.dtype), sin, cos).to(k_dtype)
        return q, k

    def _spatial_coords(self, index: torch.Tensor, *, own_grid: int, other_grid: int) -> torch.Tensor:
        """Normalized (i + 0.5) / grid coordinates mapped to (-1, 1) as fp32.

        ``own_grid`` is the extent along the axis of ``index`` (row count for
        rows, column count for columns); ``"separate"`` divides each axis by
        its own extent while ``"min"``/``"max"`` share one divisor.
        """
        if self.normalize_coords == "max":
            divisor = max(own_grid, other_grid)
        elif self.normalize_coords == "min":
            divisor = min(own_grid, other_grid)
        else:
            divisor = own_grid
        index_fp = index.to(dtype=self.periods.dtype)
        return 2.0 * ((index_fp + 0.5) / divisor) - 1.0

    def _temporal_step(self, fps: float | torch.Tensor | None, *, batch_size: int) -> float | torch.Tensor:
        """Frame step in ``1 / base_fps``-second units (``base_fps / fps``)."""
        if fps is None:
            return 1.0
        if isinstance(fps, torch.Tensor):
            fps_tensor = fps.to(device=self.periods.device, dtype=torch.float32)
            if fps_tensor.ndim == 0:
                fps_value = float(fps_tensor.item())
                _check_positive_fps(fps_value)
                return self.base_fps / fps_value
            if fps_tensor.ndim != 1 or fps_tensor.shape[0] != batch_size:
                raise ValueError(
                    f"fps tensor must be scalar or shape ({batch_size},), got {tuple(fps_tensor.shape)}."
                )
            if bool((fps_tensor <= 0).any()):
                raise ValueError("fps must be positive everywhere.")
            return self.base_fps / fps_tensor
        _check_positive_fps(float(fps))
        return self.base_fps / float(fps)


def _check_positive_fps(fps: float) -> None:
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}.")


def _grid_size(coordinates: PackedVideoTokens) -> tuple[int, int]:
    """Patch-grid size from the packed layout metadata."""
    grid_hw = coordinates.extras.get("grid_hw")
    if grid_hw is not None:
        return int(grid_hw[0]), int(grid_hw[1])
    rows = coordinates.row[coordinates.row >= 0]
    cols = coordinates.col[coordinates.col >= 0]
    if rows.numel() == 0:
        raise ValueError("packed layout carries no patch tokens; cannot infer the RoPE grid.")
    return int(rows.max().item()) + 1, int(cols.max().item()) + 1
