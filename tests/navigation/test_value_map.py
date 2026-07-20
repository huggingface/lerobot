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

"""Unit tests for the B2-full value-map exploration."""

# ruff: noqa: N803, N806  — D: conventional feature-dimension name
from __future__ import annotations

import math

import numpy as np

from lerobot.navigation.occupancy import (
    NAVIGABLE,
    UNOBSERVED,
    OccupancyGrid,
    project_voxel_map_to_grid,
)
from lerobot.navigation.value_map import (
    ValueMapConfig,
    compute_value_maps,
    pick_best_frontier_cell,
)
from lerobot.navigation.voxel_map import VoxelMap


def _vm_from(points, *, voxel_size=0.1, t0=0.0, dt=0.0, features=None):
    vm = VoxelMap(voxel_size=voxel_size)
    rgb = np.full((1, 1, 3), 200, dtype=np.uint8)
    conf = np.ones((1, 1), dtype=np.float32)
    for i, p in enumerate(points):
        pt = np.array([[[p[0], p[1], p[2]]]], dtype=np.float64)
        feat = None
        if features is not None:
            feat = features[i].reshape(1, 1, -1).astype(np.float16)
        vm.add(pt, rgb, conf, frame=i, t=t0 + i * dt, feat_map=feat)
    return vm


def _grid_around(vm: VoxelMap, cell_size: float = 0.5) -> OccupancyGrid:
    return project_voxel_map_to_grid(vm, cell_size=cell_size, inflate_cells=0)


def test_recency_high_for_unobserved_cells():
    """Cells with no voxel projection should default to unknown_value."""
    vm = _vm_from([(0.0, 1.0, 0.0)])
    grid = _grid_around(vm)
    cfg = ValueMapConfig(unknown_value=0.95)
    vm_values = compute_value_maps(vm, grid, cfg=cfg)
    # The voxel only fills one cell; the rest should be unknown.
    unobs = grid.classes == UNOBSERVED
    assert unobs.any()
    np.testing.assert_allclose(vm_values.recency[unobs], 0.95, atol=1e-6)


def test_recency_drops_for_recent_observation():
    """A freshly-observed cell scores LOW on V_T (recency)."""
    vm = _vm_from([(0.0, 1.0, 0.0)], t0=100.0)
    grid = _grid_around(vm)
    cfg = ValueMapConfig(recency_mid_s=10.0, recency_scale_s=3.0, unknown_value=1.0)
    # now_t == t0 → age = 0 → sigmoid((0 - 10) / 3) ≈ 0.04
    values = compute_value_maps(vm, grid, now_t=100.0, cfg=cfg)
    # Find the cell that received the voxel.
    obs_mask = values.last_time > -math.inf
    assert obs_mask.any()
    assert values.recency[obs_mask].max() < 0.1


def test_recency_grows_with_age():
    vm = _vm_from([(0.0, 1.0, 0.0)], t0=0.0)
    grid = _grid_around(vm)
    cfg = ValueMapConfig(recency_mid_s=10.0, recency_scale_s=3.0)
    # 30 seconds later — V_T should be near 1.
    values = compute_value_maps(vm, grid, now_t=30.0, cfg=cfg)
    obs_mask = values.last_time > -math.inf
    assert values.recency[obs_mask].max() > 0.9


def test_similarity_high_for_matching_query():
    D = 8
    feat_couch = np.eye(D)[0]
    vm = _vm_from(
        [(0.0, 1.0, 0.0)],
        features=[feat_couch],
    )
    grid = _grid_around(vm)
    text_emb = np.eye(D)[0]  # same direction as couch
    cfg = ValueMapConfig(similarity_mid=0.15, similarity_scale=0.05)
    values = compute_value_maps(vm, grid, text_emb=text_emb, cfg=cfg)
    assert values.similarity is not None
    assert values.similarity.max() > 0.95


def test_similarity_low_for_orthogonal_query():
    D = 8
    feat_couch = np.eye(D)[0]
    vm = _vm_from([(0.0, 1.0, 0.0)], features=[feat_couch])
    grid = _grid_around(vm)
    text_emb = np.eye(D)[3]  # orthogonal
    values = compute_value_maps(vm, grid, text_emb=text_emb)
    assert values.similarity is not None
    # Cells with content but no match: low V_S.
    has_voxel = values.last_time > -math.inf
    assert values.similarity[has_voxel].max() < 0.1


def test_similarity_is_none_when_no_query():
    vm = _vm_from([(0.0, 1.0, 0.0)])
    grid = _grid_around(vm)
    values = compute_value_maps(vm, grid)
    assert values.similarity is None
    np.testing.assert_array_equal(values.combined, values.recency)


def test_combined_balances_recency_and_similarity():
    D = 8
    # Two voxels with different features: one matches query, one doesn't.
    feats = [np.eye(D)[0], np.eye(D)[3]]
    vm = _vm_from(
        [(0.0, 1.0, 0.0), (2.0, 1.0, 0.0)],
        features=feats,
        t0=0.0,
    )
    grid = _grid_around(vm, cell_size=0.5)
    cfg = ValueMapConfig(alpha_similarity=0.7, recency_mid_s=5.0, recency_scale_s=2.0)
    text_emb = np.eye(D)[0]
    values = compute_value_maps(vm, grid, text_emb=text_emb, now_t=0.0, cfg=cfg)

    # Cell with matching feature should have HIGHER combined value than the
    # non-matching observed cell at the same age.
    snap_xyz = vm.snapshot().xyz
    iz_m = int((snap_xyz[0, 2] - grid.origin_z) / grid.cell_size)
    ix_m = int((snap_xyz[0, 0] - grid.origin_x) / grid.cell_size)
    iz_n = int((snap_xyz[1, 2] - grid.origin_z) / grid.cell_size)
    ix_n = int((snap_xyz[1, 0] - grid.origin_x) / grid.cell_size)
    assert values.combined[iz_m, ix_m] > values.combined[iz_n, ix_n]


def test_pick_best_frontier_prefers_high_value_cell():
    classes = np.full((6, 6), UNOBSERVED, dtype=np.int8)
    classes[1:5, 1:5] = NAVIGABLE
    grid = OccupancyGrid(classes=classes, cell_size=0.5, origin_x=0.0, origin_z=0.0, ground_y=0.0)
    frontier_cells = np.array([[1, 1], [4, 4]], dtype=np.int32)
    from lerobot.navigation.value_map import ValueMaps

    # Make cell (4, 4) more valuable than (1, 1).
    combined = np.zeros((6, 6), dtype=np.float32)
    combined[1, 1] = 0.2
    combined[4, 4] = 0.9
    values = ValueMaps(
        last_time=np.full((6, 6), -math.inf),
        recency=combined.copy(),
        similarity=None,
        combined=combined,
    )
    best_idx, (x, z), d, score = pick_best_frontier_cell(
        grid,
        frontier_cells,
        values,
        robot_position_xz=(0.0, 0.0),
        cfg=ValueMapConfig(distance_discount_per_meter=0.0),
    )
    assert best_idx == 1
    assert score > 0.8


def test_distance_discount_prefers_closer_when_values_equal():
    classes = np.full((6, 6), UNOBSERVED, dtype=np.int8)
    classes[0:6, 0:6] = NAVIGABLE
    grid = OccupancyGrid(classes=classes, cell_size=1.0, origin_x=0.0, origin_z=0.0, ground_y=0.0)
    frontier_cells = np.array([[0, 0], [5, 5]], dtype=np.int32)
    from lerobot.navigation.value_map import ValueMaps

    same = np.ones((6, 6), dtype=np.float32)
    values = ValueMaps(
        last_time=np.full((6, 6), -math.inf),
        recency=same.copy(),
        similarity=None,
        combined=same,
    )
    best_idx, _, _, _ = pick_best_frontier_cell(
        grid,
        frontier_cells,
        values,
        robot_position_xz=(0.0, 0.0),
        cfg=ValueMapConfig(distance_discount_per_meter=0.5),
    )
    # Robot at origin → (0, 0) is closer than (5, 5).
    assert best_idx == 0


def test_compute_value_maps_with_empty_voxel_map_returns_unknown():
    vm = VoxelMap()
    classes = np.full((4, 4), UNOBSERVED, dtype=np.int8)
    grid = OccupancyGrid(classes=classes, cell_size=1.0, origin_x=0.0, origin_z=0.0, ground_y=0.0)
    values = compute_value_maps(vm, grid)
    np.testing.assert_allclose(values.recency, 1.0)
    assert values.similarity is None
