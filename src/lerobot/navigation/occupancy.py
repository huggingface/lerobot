#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""2D occupancy projection of the voxel map + A* path planning.

Ported from the dyna360 research stack. Derived, not maintained: every
call to :func:`project_voxel_map_to_grid` rebuilds the 3-class grid from
a fresh ``VoxelMap.snapshot()``, so the projection reflects whatever the
keyframe loop most recently carved or added — no separate obstacle
structure to keep in sync.

Coordinate convention: OpenCV (X right, Y *down*, Z forward), matching
the navigation world frame. "Up" is the −Y direction. The top-down grid
indexes the XZ plane; cell ``(iz, ix)`` covers world rectangle
``[origin_x + ix·cell, origin_x + (ix+1)·cell]`` ×
``[origin_z + iz·cell, origin_z + (iz+1)·cell]``.

Classes:
  - ``UNOBSERVED`` (0): no voxel projects here. The base must not plan
    through it (might be an unseen obstacle), but explorers treat it as
    the goal class.
  - ``NAVIGABLE`` (1): observed ground / open space.
  - ``OBSTACLE`` (2): at least one voxel in the robot-height band
    projects here.
"""

# ruff: noqa: N806  — H, W, D are conventional array-dimension names (and appear verbatim in error strings)
from __future__ import annotations

import heapq
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from lerobot.navigation.voxel_map import VoxelMap

LOG = logging.getLogger(__name__)

# Class constants — picked so a colormap can index directly.
UNOBSERVED = np.int8(0)
NAVIGABLE = np.int8(1)
OBSTACLE = np.int8(2)


@dataclass(frozen=True)
class OccupancyGrid:
    """3-class top-down grid plus its world↔cell mapping."""

    classes: np.ndarray  # (H, W) int8 — H = z-extent, W = x-extent
    cell_size: float  # m per cell
    origin_x: float  # world x of the LEFT edge of column 0
    origin_z: float  # world z of the TOP edge of row 0
    ground_y: float  # world y of the (auto-estimated or given) ground plane

    @property
    def shape(self) -> tuple[int, int]:
        return self.classes.shape  # (H, W)

    def world_to_cell(self, x: float, z: float) -> tuple[int, int]:
        """Return ``(iz, ix)``, clipped to grid extents."""
        ix = int(np.clip(math.floor((x - self.origin_x) / self.cell_size), 0, self.shape[1] - 1))
        iz = int(np.clip(math.floor((z - self.origin_z) / self.cell_size), 0, self.shape[0] - 1))
        return iz, ix

    def cell_to_world(self, iz: int, ix: int) -> tuple[float, float]:
        """Return ``(x, z)`` at the *centre* of cell ``(iz, ix)``."""
        x = self.origin_x + (ix + 0.5) * self.cell_size
        z = self.origin_z + (iz + 0.5) * self.cell_size
        return x, z

    def is_navigable(self, iz: int, ix: int) -> bool:
        H, W = self.shape
        return 0 <= iz < H and 0 <= ix < W and self.classes[iz, ix] == NAVIGABLE

    def is_obstacle(self, iz: int, ix: int) -> bool:
        """Whether cell ``(iz, ix)`` is a known obstacle. Out-of-bounds is
        not an obstacle (it is simply unobservable) — used by
        ``SafeBaseController``'s occupancy gate."""
        H, W = self.shape
        return 0 <= iz < H and 0 <= ix < W and self.classes[iz, ix] == OBSTACLE

    def is_in_bounds(self, iz: int, ix: int) -> bool:
        H, W = self.shape
        return 0 <= iz < H and 0 <= ix < W

    def nearest_navigable_cell(self, iz: int, ix: int, max_radius: int = 50) -> tuple[int, int] | None:
        """BFS outward until a navigable cell is found, or give up."""
        if self.is_navigable(iz, ix):
            return iz, ix
        for r in range(1, max_radius + 1):
            for diz in range(-r, r + 1):
                for dix in range(-r, r + 1):
                    if max(abs(diz), abs(dix)) != r:
                        continue  # ring only, not the interior
                    if self.is_navigable(iz + diz, ix + dix):
                        return iz + diz, ix + dix
        return None


def estimate_ground_y(xyz: np.ndarray, percentile: float = 95.0) -> float:
    """Estimate the world-frame y of the ground plane.

    Y is down (OpenCV), so the ground is at the LARGEST y values. Using a
    high percentile (default 95) is robust to outliers below the ground.
    """
    if xyz.size == 0:
        return 0.0
    return float(np.percentile(xyz[:, 1], percentile))


def project_voxel_map_to_grid(
    voxel_map: VoxelMap,
    *,
    cell_size: float = 0.1,
    ground_y: float | None = None,
    obstacle_y_range: tuple[float, float] = (-2.0, -0.1),
    bbox: tuple[float, float, float, float] | None = None,
    bbox_pad: float = 1.0,
    inflate_cells: int = 0,
) -> OccupancyGrid:
    """Snapshot the voxel map and project it into a 2D occupancy grid.

    ``obstacle_y_range`` is interpreted *relative* to ``ground_y`` with the
    Y-down convention, so the default ``(-2.0, -0.1)`` means "voxels
    between 2.0 m and 0.1 m above the ground are obstacles". Anything above
    the ceiling band or below ground level is silently ignored.

    ``inflate_cells`` dilates the OBSTACLE class by N cells of clearance
    (square morphology) — a body-radius safety margin for the base without
    resampling the voxel map.
    """
    snap = voxel_map.snapshot()
    xyz = snap.xyz

    if xyz.size == 0:
        # Empty map → a 1×1 grid of UNOBSERVED at world origin.
        return OccupancyGrid(
            classes=np.zeros((1, 1), dtype=np.int8),
            cell_size=float(cell_size),
            origin_x=0.0,
            origin_z=0.0,
            ground_y=ground_y if ground_y is not None else 0.0,
        )

    if ground_y is None:
        ground_y = estimate_ground_y(xyz)

    # Promote to float64 — VoxelMap snapshots are float32, and naive
    # ``(float32_array <= float64_scalar)`` lets numpy downcast the scalar
    # back to float32, which causes edge-case bugs (e.g. 1.0 <= 0.999999999
    # becomes True because the threshold rounds up to 1.0 in float32).
    x_arr = xyz[:, 0].astype(np.float64)
    y_arr = xyz[:, 1].astype(np.float64)
    z_arr = xyz[:, 2].astype(np.float64)

    abs_y_top = ground_y + obstacle_y_range[0]  # most-negative y (highest above ground)
    abs_y_bottom = ground_y + obstacle_y_range[1]  # closer to ground
    is_obstacle = (y_arr >= abs_y_top) & (y_arr <= abs_y_bottom)

    if bbox is None:
        x_min = float(x_arr.min()) - bbox_pad
        x_max = float(x_arr.max()) + bbox_pad
        z_min = float(z_arr.min()) - bbox_pad
        z_max = float(z_arr.max()) + bbox_pad
    else:
        x_min, z_min, x_max, z_max = bbox

    W = max(1, int(math.ceil((x_max - x_min) / cell_size)))
    H = max(1, int(math.ceil((z_max - z_min) / cell_size)))
    classes = np.zeros((H, W), dtype=np.int8)  # default UNOBSERVED

    # eps absorbs float32→float64 representation drift so points that should
    # land exactly on a cell boundary aren't randomly bumped into the
    # previous cell. 1e-3 of a cell width is well above float32's ~1e-7
    # relative precision and well below the 0.5-cell misclassification
    # threshold.
    eps = cell_size * 1e-3
    ix = np.clip(np.floor((x_arr - x_min) / cell_size + eps).astype(np.int32), 0, W - 1)
    iz = np.clip(np.floor((z_arr - z_min) / cell_size + eps).astype(np.int32), 0, H - 1)

    # Two-pass labelling: any voxel makes a cell observed (-> NAVIGABLE);
    # obstacle voxels then upgrade those cells to OBSTACLE.
    classes[iz, ix] = NAVIGABLE
    obs_iz = iz[is_obstacle]
    obs_ix = ix[is_obstacle]
    classes[obs_iz, obs_ix] = OBSTACLE

    if inflate_cells > 0:
        classes = _inflate_obstacles(classes, inflate_cells)

    return OccupancyGrid(
        classes=classes,
        cell_size=float(cell_size),
        origin_x=x_min,
        origin_z=z_min,
        ground_y=float(ground_y),
    )


def _inflate_obstacles(classes: np.ndarray, radius: int) -> np.ndarray:
    """Dilate OBSTACLE cells by `radius` cells (Chebyshev). Pure-numpy
    morphological dilation — fine for our grid sizes."""
    out = classes.copy()
    obs = classes == OBSTACLE
    H, W = classes.shape
    for diz in range(-radius, radius + 1):
        for dix in range(-radius, radius + 1):
            if diz == 0 and dix == 0:
                continue
            sl_src_iz = slice(max(0, -diz), H - max(0, diz))
            sl_src_ix = slice(max(0, -dix), W - max(0, dix))
            sl_dst_iz = slice(max(0, diz), H - max(0, -diz))
            sl_dst_ix = slice(max(0, dix), W - max(0, -dix))
            inflated = obs[sl_src_iz, sl_src_ix]
            # Only upgrade NAVIGABLE → OBSTACLE; never overwrite UNOBSERVED
            # so the frontier (NAVIGABLE↔UNOBSERVED boundary) survives.
            target = out[sl_dst_iz, sl_dst_ix]
            promote = inflated & (target == NAVIGABLE)
            target[promote] = OBSTACLE
            out[sl_dst_iz, sl_dst_ix] = target
    return out


# --------------------------------------------------------------------- A*

_DIAG_COST = math.sqrt(2.0)
_NEIGHBOURS_ORTHO = ((-1, 0), (1, 0), (0, -1), (0, 1))
_NEIGHBOURS_DIAG = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def astar(
    grid: OccupancyGrid,
    start_world: tuple[float, float],
    goal_world: tuple[float, float],
    *,
    allow_unobserved_goal: bool = True,
) -> list[tuple[float, float]] | None:
    """Plan a path from ``start_world`` to ``goal_world`` in (x, z) world m.

    Returns a list of world ``(x, z)`` waypoints, or ``None`` if no path
    exists. Start/goal are snapped to the nearest navigable cell.
    """
    H, W = grid.shape
    if H == 0 or W == 0:
        return None

    s_iz, s_ix = grid.world_to_cell(*start_world)
    g_iz, g_ix = grid.world_to_cell(*goal_world)

    if not grid.is_navigable(s_iz, s_ix):
        snapped = grid.nearest_navigable_cell(s_iz, s_ix)
        if snapped is None:
            return None
        s_iz, s_ix = snapped
    if not grid.is_navigable(g_iz, g_ix):
        if not allow_unobserved_goal:
            return None
        snapped = grid.nearest_navigable_cell(g_iz, g_ix)
        if snapped is None:
            return None
        g_iz, g_ix = snapped

    def heuristic(iz: int, ix: int) -> float:
        d_iz = abs(iz - g_iz)
        d_ix = abs(ix - g_ix)
        return (max(d_iz, d_ix) - min(d_iz, d_ix)) + _DIAG_COST * min(d_iz, d_ix)

    open_heap: list[tuple[float, int, tuple[int, int]]] = []
    counter = 0  # tiebreaker so heapq doesn't compare tuples on ties
    heapq.heappush(open_heap, (0.0, counter, (s_iz, s_ix)))
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], float] = {(s_iz, s_ix): 0.0}

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current == (g_iz, g_ix):
            return _reconstruct_path(came_from, current, grid)

        cur_iz, cur_ix = current
        cur_g = g_score[current]

        for diz, dix in _NEIGHBOURS_ORTHO:
            n = (cur_iz + diz, cur_ix + dix)
            if not grid.is_navigable(*n):
                continue
            tentative = cur_g + 1.0
            if tentative < g_score.get(n, float("inf")):
                came_from[n] = current
                g_score[n] = tentative
                counter += 1
                heapq.heappush(open_heap, (tentative + heuristic(*n), counter, n))

        for diz, dix in _NEIGHBOURS_DIAG:
            n = (cur_iz + diz, cur_ix + dix)
            if not grid.is_navigable(*n):
                continue
            # Prevent corner-cutting: both perpendicular neighbours must be
            # navigable, or we'd squeeze through an obstacle's diagonal.
            if not grid.is_navigable(cur_iz + diz, cur_ix):
                continue
            if not grid.is_navigable(cur_iz, cur_ix + dix):
                continue
            tentative = cur_g + _DIAG_COST
            if tentative < g_score.get(n, float("inf")):
                came_from[n] = current
                g_score[n] = tentative
                counter += 1
                heapq.heappush(open_heap, (tentative + heuristic(*n), counter, n))

    return None


def _reconstruct_path(
    came_from: dict[tuple[int, int], tuple[int, int]],
    end: tuple[int, int],
    grid: OccupancyGrid,
) -> list[tuple[float, float]]:
    cells = [end]
    while cells[-1] in came_from:
        cells.append(came_from[cells[-1]])
    cells.reverse()
    return [grid.cell_to_world(iz, ix) for iz, ix in cells]


# ----------------------------------------------------------------- frontier


def find_frontier_cells(grid: OccupancyGrid) -> np.ndarray:
    """Return cells on the NAVIGABLE↔UNOBSERVED boundary, as ``(K, 2)`` int.

    These are the cells exploration aims for: places we already know we
    can stand at, but with unknown adjacent territory worth visiting.
    """
    nav = grid.classes == NAVIGABLE
    unobs = grid.classes == UNOBSERVED
    if not nav.any() or not unobs.any():
        return np.zeros((0, 2), dtype=np.int32)

    boundary = np.zeros_like(nav)
    boundary[1:, :] |= nav[1:, :] & unobs[:-1, :]
    boundary[:-1, :] |= nav[:-1, :] & unobs[1:, :]
    boundary[:, 1:] |= nav[:, 1:] & unobs[:, :-1]
    boundary[:, :-1] |= nav[:, :-1] & unobs[:, 1:]
    iz, ix = np.where(boundary)
    return np.stack([iz, ix], axis=-1).astype(np.int32)


def occupancy_to_rgb(grid: OccupancyGrid) -> np.ndarray:
    """Render the 3-class grid as an (H, W, 3) uint8 image."""
    img = np.zeros((*grid.shape, 3), dtype=np.uint8)
    img[grid.classes == UNOBSERVED] = (40, 40, 50)
    img[grid.classes == NAVIGABLE] = (200, 200, 200)
    img[grid.classes == OBSTACLE] = (220, 60, 60)
    return img
