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

"""Unit tests for the B1 occupancy projection + A*."""

from __future__ import annotations

import numpy as np

from lerobot.navigation.occupancy import (
    NAVIGABLE,
    OBSTACLE,
    UNOBSERVED,
    OccupancyGrid,
    astar,
    estimate_ground_y,
    find_frontier_cells,
    occupancy_to_rgb,
    project_voxel_map_to_grid,
)
from lerobot.navigation.voxel_map import VoxelMap


def _vm_from_points(pts: list[tuple[float, float, float]], voxel_size: float = 0.1) -> VoxelMap:
    """Build a VoxelMap from a list of world-XYZ points (all at conf=1.0)."""
    vm = VoxelMap(voxel_size=voxel_size)
    if not pts:
        return vm
    arr = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 3)
    rgb = np.full((len(pts), 1, 3), 200, dtype=np.uint8)
    conf = np.ones((len(pts), 1), dtype=np.float32)
    vm.add(arr, rgb, conf, frame=0, t=0.0)
    return vm


def test_estimate_ground_y_picks_high_percentile():
    # +y is down in OpenCV; ground = largest y. Numpy's default percentile
    # interpolation lands at 0.9 here (95% of way from 0.5 to 1.0 of the
    # last interval = 0.5 + 0.4·1.0 = 0.9), so we just check the estimate
    # is at least that high and clearly above the median.
    xyz = np.array(
        [[0, -1, 0], [0, -0.5, 0], [0, 0.0, 0], [0, 0.5, 0], [0, 1.0, 0]],
        dtype=np.float32,
    )
    g = estimate_ground_y(xyz)
    assert g >= 0.8
    median_y = float(np.median(xyz[:, 1]))
    assert g > median_y


def test_projection_makes_ground_navigable_and_obstacles_red():
    # Ground row at y=1.0 across z=0..1; an obstacle at y=0.2 (above ground).
    pts: list[tuple[float, float, float]] = []
    for x in np.linspace(-0.5, 0.5, 11):
        for z in np.linspace(0.0, 1.0, 11):
            pts.append((float(x), 1.0, float(z)))  # floor
    # Standing obstacle column at (x=0, z=0.5)
    for y in np.linspace(0.2, 0.9, 8):
        pts.append((0.0, float(y), 0.5))
    vm = _vm_from_points(pts, voxel_size=0.05)
    grid = project_voxel_map_to_grid(vm, cell_size=0.1, obstacle_y_range=(-2.0, -0.1))
    # The obstacle column should yield at least one OBSTACLE cell.
    assert (grid.classes == OBSTACLE).sum() >= 1
    # The floor away from the obstacle should be NAVIGABLE.
    iz, ix = grid.world_to_cell(-0.4, 0.0)
    assert grid.classes[iz, ix] == NAVIGABLE
    # Everywhere outside the observed area should still be UNOBSERVED.
    assert (grid.classes == UNOBSERVED).any()


def test_empty_voxelmap_projects_to_single_unobserved_cell():
    vm = VoxelMap()
    grid = project_voxel_map_to_grid(vm)
    assert grid.shape == (1, 1)
    assert grid.classes[0, 0] == UNOBSERVED


def test_world_to_cell_and_back_roundtrip():
    classes = np.full((4, 6), NAVIGABLE, dtype=np.int8)
    grid = OccupancyGrid(classes=classes, cell_size=0.5, origin_x=-1.0, origin_z=2.0, ground_y=0.0)
    iz, ix = grid.world_to_cell(0.25, 3.4)
    x, z = grid.cell_to_world(iz, ix)
    # The recovered (x, z) should land inside the same cell.
    iz2, ix2 = grid.world_to_cell(x, z)
    assert (iz, ix) == (iz2, ix2)


def test_astar_finds_straight_path_when_unblocked():
    classes = np.full((10, 10), NAVIGABLE, dtype=np.int8)
    grid = OccupancyGrid(classes=classes, cell_size=0.1, origin_x=0.0, origin_z=0.0, ground_y=0.0)
    path = astar(grid, (0.05, 0.05), (0.85, 0.85))
    assert path is not None
    assert len(path) >= 2
    # Should land near the goal.
    assert abs(path[-1][0] - 0.85) < 0.1 and abs(path[-1][1] - 0.85) < 0.1


def test_astar_routes_around_obstacle_wall():
    # Vertical wall at column 5, rows 1..8.
    classes = np.full((10, 10), NAVIGABLE, dtype=np.int8)
    classes[1:9, 5] = OBSTACLE
    grid = OccupancyGrid(classes=classes, cell_size=0.1, origin_x=0.0, origin_z=0.0, ground_y=0.0)
    path = astar(grid, (0.05, 0.45), (0.95, 0.45))
    assert path is not None
    # The path must not pass through any obstacle cell.
    for x, z in path:
        iz, ix = grid.world_to_cell(x, z)
        assert classes[iz, ix] != OBSTACLE


def test_astar_returns_none_when_no_path():
    # Wall that completely separates start from goal.
    classes = np.full((10, 10), NAVIGABLE, dtype=np.int8)
    classes[:, 5] = OBSTACLE
    grid = OccupancyGrid(classes=classes, cell_size=0.1, origin_x=0.0, origin_z=0.0, ground_y=0.0)
    path = astar(grid, (0.05, 0.5), (0.95, 0.5))
    assert path is None


def test_astar_no_corner_cutting_through_obstacles():
    # Two diagonal obstacle cells that would let a naive A* squeeze through.
    classes = np.full((4, 4), NAVIGABLE, dtype=np.int8)
    classes[1, 2] = OBSTACLE
    classes[2, 1] = OBSTACLE
    grid = OccupancyGrid(classes=classes, cell_size=1.0, origin_x=0.0, origin_z=0.0, ground_y=0.0)
    path = astar(grid, (0.5, 0.5), (2.5, 2.5))
    assert path is not None
    # The path should NOT step (1, 1) → (2, 2) since that diagonal cuts the
    # obstacle corner. Verify by checking no two consecutive cells are a
    # diagonal move with both perpendicular cells blocked.
    cells = [grid.world_to_cell(x, z) for x, z in path]
    for prev, nxt in zip(cells, cells[1:], strict=False):
        diz = nxt[0] - prev[0]
        dix = nxt[1] - prev[1]
        if abs(diz) == 1 and abs(dix) == 1:
            assert grid.is_navigable(prev[0] + diz, prev[1]) and grid.is_navigable(prev[0], prev[1] + dix), (
                f"Corner cut detected at step {prev}->{nxt}"
            )


def test_obstacle_inflation_grows_obstacle_class():
    classes = np.full((5, 5), NAVIGABLE, dtype=np.int8)
    classes[2, 2] = OBSTACLE
    # Test the private helper directly — we don't need a voxel map for this.
    from lerobot.navigation.occupancy import _inflate_obstacles

    inflated = _inflate_obstacles(classes, radius=1)
    # 3x3 block now obstacle (around the single original cell).
    assert (inflated[1:4, 1:4] == OBSTACLE).all()
    # Far corner stays navigable.
    assert inflated[0, 0] == NAVIGABLE


def test_find_frontier_cells_are_navigable_next_to_unobserved():
    classes = np.full((5, 5), UNOBSERVED, dtype=np.int8)
    classes[1:4, 1:4] = NAVIGABLE
    grid = OccupancyGrid(classes=classes, cell_size=0.1, origin_x=0.0, origin_z=0.0, ground_y=0.0)
    cells = find_frontier_cells(grid)
    # Every frontier cell is NAVIGABLE itself.
    for iz, ix in cells:
        assert classes[iz, ix] == NAVIGABLE
    # And every frontier cell has at least one UNOBSERVED 4-neighbour.
    for iz, ix in cells:
        adj_unobs = (
            (iz > 0 and classes[iz - 1, ix] == UNOBSERVED)
            or (iz < 4 and classes[iz + 1, ix] == UNOBSERVED)
            or (ix > 0 and classes[iz, ix - 1] == UNOBSERVED)
            or (ix < 4 and classes[iz, ix + 1] == UNOBSERVED)
        )
        assert adj_unobs


def test_find_frontier_empty_when_no_unobserved():
    classes = np.full((5, 5), NAVIGABLE, dtype=np.int8)
    grid = OccupancyGrid(classes=classes, cell_size=0.1, origin_x=0.0, origin_z=0.0, ground_y=0.0)
    cells = find_frontier_cells(grid)
    assert cells.shape == (0, 2)


def test_occupancy_to_rgb_returns_uint8_image():
    classes = np.array([[UNOBSERVED, NAVIGABLE, OBSTACLE]], dtype=np.int8)
    grid = OccupancyGrid(classes=classes, cell_size=0.1, origin_x=0.0, origin_z=0.0, ground_y=0.0)
    img = occupancy_to_rgb(grid)
    assert img.shape == (1, 3, 3)
    assert img.dtype == np.uint8
    # Obstacle cell is reddish.
    assert img[0, 2, 0] > img[0, 2, 1] and img[0, 2, 0] > img[0, 2, 2]


def test_carving_makes_obstacle_vanish_on_next_projection():
    """End-to-end: an object voxel becomes an obstacle; after we delete it
    from the VoxelMap (simulating carving), the next projection no longer
    flags that cell as OBSTACLE — that's what makes the obstacle map a
    cheap derived view."""
    pts = []
    for x in np.linspace(-0.5, 0.5, 11):
        for z in np.linspace(0.0, 1.0, 11):
            pts.append((float(x), 1.0, float(z)))  # floor
    for y in np.linspace(0.2, 0.9, 8):
        pts.append((0.0, float(y), 0.5))  # obstacle column
    vm = _vm_from_points(pts, voxel_size=0.05)
    grid1 = project_voxel_map_to_grid(vm, cell_size=0.1)
    assert (grid1.classes == OBSTACLE).sum() >= 1

    # Surgically drop every voxel in the obstacle band.
    cnt = vm._count.astype(np.float64).reshape(-1, 1)  # noqa: SLF001
    means = vm._xyz_sum / cnt  # noqa: SLF001
    keep = (means[:, 1] >= 0.95) | (means[:, 1] <= 0.05)
    vm._idx = vm._idx[keep]  # noqa: SLF001
    vm._count = vm._count[keep]  # noqa: SLF001
    vm._xyz_sum = vm._xyz_sum[keep]  # noqa: SLF001
    vm._rgb_sum = vm._rgb_sum[keep]  # noqa: SLF001
    vm._last_frame = vm._last_frame[keep]  # noqa: SLF001
    vm._last_time = vm._last_time[keep]  # noqa: SLF001
    vm._lookup = {  # noqa: SLF001
        (int(vm._idx[i, 0]), int(vm._idx[i, 1]), int(vm._idx[i, 2])): i  # noqa: SLF001
        for i in range(len(vm._idx))  # noqa: SLF001
    }

    grid2 = project_voxel_map_to_grid(vm, cell_size=0.1)
    assert (grid2.classes == OBSTACLE).sum() == 0


def test_astar_works_after_inflation():
    # Without inflation, robot can hug a 1-cell-wide wall; with inflation,
    # it has to take a wider detour.
    classes = np.full((11, 11), NAVIGABLE, dtype=np.int8)
    classes[5, 1:9] = OBSTACLE  # horizontal wall
    grid_raw = OccupancyGrid(classes=classes.copy(), cell_size=0.1, origin_x=0.0, origin_z=0.0, ground_y=0.0)
    path_raw = astar(grid_raw, (0.45, 0.0), (0.45, 1.0))
    assert path_raw is not None  # the wall has an opening at the edges
    from lerobot.navigation.occupancy import _inflate_obstacles

    inflated = _inflate_obstacles(classes, radius=1)
    grid_inf = OccupancyGrid(classes=inflated, cell_size=0.1, origin_x=0.0, origin_z=0.0, ground_y=0.0)
    # After inflation the path is at least as long (often longer).
    path_inf = astar(grid_inf, (0.45, 0.0), (0.45, 1.0))
    if path_inf is not None:
        assert len(path_inf) >= len(path_raw)


def test_obstacle_range_outside_voxel_y_produces_only_navigable():
    """A range well above the actual voxel y values should classify the floor
    as NAVIGABLE only — no obstacles. (Replaces the prior sub-float32-epsilon
    test which exercised numpy's float-downcast quirk rather than the
    occupancy semantics.)"""
    pts = [(float(x), 1.0, float(z)) for x in [0, 0.1] for z in [0, 0.1]]
    vm = _vm_from_points(pts, voxel_size=0.05)
    # ground_y will be 1.0; this range looks for obstacles 5 m–10 m above
    # the floor, where there are none.
    grid = project_voxel_map_to_grid(vm, obstacle_y_range=(-10.0, -5.0))
    assert (grid.classes == OBSTACLE).sum() == 0
    assert (grid.classes == NAVIGABLE).sum() >= 1
