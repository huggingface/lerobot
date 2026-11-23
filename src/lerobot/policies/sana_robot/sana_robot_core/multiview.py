# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://github.com/NVlabs/Sana/tree/robot) at commit 1e7d6cc51,
# original path sana_robot/policy/layout/multiview.py. Import paths are rewritten to the LeRobot package.
# Keep this file in sync with the upstream file.

from __future__ import annotations

from collections.abc import Sequence

import torch

SPATIAL_LAYOUT_2X2 = "2x2"
SPATIAL_LAYOUT_T_SHAPE = "t_shape"
TILE_SHAPE_2X2 = (8, 16)
T_SHAPE_HEAD_COLUMN = 3


def tile_2x2_origins(view_shapes: Sequence[tuple[int, int]], view_slot_ids: Sequence[int]) -> list:
    """Return the (y, x) origin of every view in the 2x2 layout: slot 0, 1, 2, 3 is the top-left, top-right, bottom-left, bottom-right tile of ``TILE_SHAPE_2X2`` latents, and a view sits centred in its tile."""
    (tile_height, tile_width) = TILE_SHAPE_2X2
    origins = []
    for (height, width), slot in zip(view_shapes, view_slot_ids, strict=True):
        (row, column) = divmod(int(slot), 2)
        origins.append(
            (
                row * tile_height + (tile_height - height) / 2.0,
                column * tile_width + (tile_width - width) / 2.0,
            )
        )
    return origins


def t_shape_origins(view_shapes: Sequence[tuple[int, int]], view_slot_ids: Sequence[int]) -> list:
    """Return the (y, x) origin of every view in the T layout: the view in slot 0 (the head camera) on the left, the other views (the wrist cameras) stacked in one column right next to it, from the top in slot order."""
    slots = [int(slot) for slot in view_slot_ids]
    head = slots.index(0)
    column = T_SHAPE_HEAD_COLUMN + view_shapes[head][1]
    origins = {head: (0, T_SHAPE_HEAD_COLUMN)}
    row = 0
    for _, view in sorted((slot, view) for (view, slot) in enumerate(slots) if view != head):
        origins[view] = (row, column)
        row += view_shapes[view][0]
    return [origins[view] for view in range(len(slots))]


def view_position_ids(
    *,
    frames: int,
    view_shapes: Sequence[tuple[int, int]],
    origins: Sequence[tuple[float, float]],
    fps: torch.Tensor,
    base_fps: float,
    device: torch.device,
) -> torch.Tensor:
    """Return the (time, y, x) position ids [B, N, 3] of the video tokens of every view in turn: latent frame j at time ``base_fps * j / fps`` and every view on its own grid, which starts at its (y, x) origin."""
    fps = torch.as_tensor(fps, device=device, dtype=torch.float64).reshape(-1)
    frame_ids = torch.arange(int(frames), device=device, dtype=torch.float64)
    time = float(base_fps) * frame_ids[None] / fps[:, None]
    position_ids = []
    for (height, width), (y_origin, x_origin) in zip(view_shapes, origins, strict=True):
        view_time = time[:, :, None, None].expand(-1, -1, height, width)
        view_y = (torch.arange(height, device=device, dtype=torch.float64) + y_origin)[
            None, None, :, None
        ].expand_as(view_time)
        view_x = (torch.arange(width, device=device, dtype=torch.float64) + x_origin)[
            None, None, None, :
        ].expand_as(view_time)
        position_ids.append(
            torch.stack((view_time, view_y, view_x), dim=-1).reshape(
                fps.numel(), int(frames) * height * width, 3
            )
        )
    return torch.cat(position_ids, dim=1)


def strip_to_view_tokens(
    tokens: torch.Tensor, frames: int, shapes: Sequence[tuple[int, int]]
) -> torch.Tensor:
    """Reorder tokens from frame-first (every latent frame holding its views side by side) to view-first (every view holding its frames)."""
    areas = [int(height) * int(width) for (height, width) in shapes]
    tokens = tokens.unflatten(1, (int(frames), sum(areas)))
    return torch.cat([view.flatten(1, 2) for view in tokens.split(areas, dim=2)], dim=1)


def view_to_strip_tokens(
    tokens: torch.Tensor, frames: int, shapes: Sequence[tuple[int, int]]
) -> torch.Tensor:
    """Reorder tokens from view-first back to frame-first; the inverse of ``strip_to_view_tokens``."""
    areas = [int(height) * int(width) for (height, width) in shapes]
    views = [
        view.unflatten(1, (int(frames), area))
        for (view, area) in zip(
            tokens.split([int(frames) * area for area in areas], dim=1), areas, strict=True
        )
    ]
    return torch.cat(views, dim=2).flatten(1, 2)


def batch_view_count(data_info: dict) -> int:
    """Return the number of camera views of the batch."""
    counts = torch.as_tensor(data_info["view_count"]).reshape(-1)
    return int(counts[0].item())


def batch_view_shapes(data_info: dict) -> tuple[tuple[int, int], ...]:
    """Return the latent (height, width) of every view of the batch."""
    value = data_info["view_latent_shape"]
    if isinstance(value, torch.Tensor):
        if value.ndim == 3:
            value = value[0]
        value = value.tolist()
    return tuple((int(height), int(width)) for (height, width) in value)


def batch_view_slot_ids(data_info: dict, view_count: int) -> tuple[int, ...]:
    """Return the tile slot (0 to 3) of every view of the batch."""
    value = data_info.get("view_slot_ids")
    value = torch.as_tensor(value)
    if value.ndim == 2:
        value = value[0]
    slots = tuple(int(slot) for slot in value.tolist())
    return slots
