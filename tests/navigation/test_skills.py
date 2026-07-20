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

"""Unit tests for the unified ``SpatialSkills`` API (B1+B2+B3+B4)."""

# ruff: noqa: N803, N806  — D: conventional feature-dimension name
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from lerobot.navigation.base_controller import StubBaseController
from lerobot.navigation.skills import SkillsConfig, SpatialSkills
from lerobot.navigation.voxel_map import VoxelMap

# ----- fakes / fixtures ---------------------------------------------------


@dataclass
class FakeSiglip:
    """Tiny stand-in for SiglipFeatureExtractor — text → fixed vector."""

    text_to_vec: dict[str, np.ndarray]
    feature_dim: int = 4

    def encode_text(self, text: str) -> np.ndarray:
        v = self.text_to_vec.get(text)
        if v is None:
            # Default: random-but-deterministic vector
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            v = rng.normal(size=self.feature_dim).astype(np.float32)
        v = v.astype(np.float32)
        v = v / max(np.linalg.norm(v), 1e-6)
        return v


def _vm_with_couch_and_chair(D: int = 4) -> VoxelMap:
    """Two spatially-separated clusters with distinct unit feature vectors."""
    vm = VoxelMap(voxel_size=0.1)
    rgb = np.full((1, 1, 3), 200, dtype=np.uint8)
    conf = np.ones((1, 1), dtype=np.float32)
    couch_vec = np.eye(D)[0].astype(np.float16).reshape(1, 1, D)
    chair_vec = np.eye(D)[1].astype(np.float16).reshape(1, 1, D)
    # Couch cluster around (5, 1, 3)
    for x in (4.9, 5.0, 5.1):
        for z in (2.9, 3.0, 3.1):
            pts = np.array([[[x, 1.0, z]]], dtype=np.float32)
            vm.add(pts, rgb, conf, frame=0, t=0.0, feat_map=couch_vec)
    # Chair cluster around (-3, 1, 1)
    for x in (-3.1, -3.0, -2.9):
        for z in (0.9, 1.0, 1.1):
            pts = np.array([[[x, 1.0, z]]], dtype=np.float32)
            vm.add(pts, rgb, conf, frame=0, t=0.0, feat_map=chair_vec)
    return vm


# ----- locate() -----------------------------------------------------------


def test_locate_returns_centroid_for_matching_query():
    vm = _vm_with_couch_and_chair()
    base = StubBaseController()
    siglip = FakeSiglip(text_to_vec={"couch": np.array([1, 0, 0, 0], dtype=np.float32)})
    skills = SpatialSkills(vm, base, siglip, SkillsConfig(locate_threshold=0.3))
    result = skills.locate("couch")
    assert result.found is True
    assert result.xyz is not None
    # Centroid should land near (5, 1, 3).
    assert abs(result.xyz[0] - 5.0) < 0.2
    assert abs(result.xyz[2] - 3.0) < 0.2
    assert result.confidence > 0.5


def test_locate_abstains_below_threshold():
    """Threshold tuned high enough that an unaligned query returns NOT_FOUND
    rather than picking a "best of the bad" cluster."""
    vm = _vm_with_couch_and_chair()
    base = StubBaseController()
    # Query embedding orthogonal to both clusters' vectors.
    siglip = FakeSiglip(text_to_vec={"banana": np.array([0, 0, 1, 0], dtype=np.float32)})
    skills = SpatialSkills(
        vm,
        base,
        siglip,
        SkillsConfig(locate_threshold=0.5),
    )
    result = skills.locate("banana")
    assert result.found is False
    assert result.xyz is None


def test_locate_distinguishes_two_clusters():
    """red-cup / blue-cup style: two clusters present, the query should pick
    the right one rather than averaging across both."""
    vm = _vm_with_couch_and_chair()
    base = StubBaseController()
    siglip = FakeSiglip(
        text_to_vec={
            "couch": np.array([1, 0, 0, 0], dtype=np.float32),
            "chair": np.array([0, 1, 0, 0], dtype=np.float32),
        }
    )
    skills = SpatialSkills(vm, base, siglip, SkillsConfig(locate_threshold=0.3))
    couch = skills.locate("couch")
    chair = skills.locate("chair")
    assert couch.found and chair.found
    assert abs(couch.xyz[0] - 5.0) < 0.3
    assert abs(chair.xyz[0] - (-3.0)) < 0.3


def test_locate_returns_not_found_without_siglip():
    vm = _vm_with_couch_and_chair()
    skills = SpatialSkills(vm, StubBaseController(), siglip=None)
    assert skills.locate("anything").found is False


def test_locate_returns_not_found_without_features():
    vm = VoxelMap()
    rgb = np.full((1, 1, 3), 200, dtype=np.uint8)
    vm.add(np.zeros((1, 1, 3), dtype=np.float32), rgb, np.ones((1, 1), dtype=np.float32), frame=0, t=0.0)
    skills = SpatialSkills(vm, StubBaseController(), siglip=FakeSiglip({}))
    assert skills.locate("anything").found is False


# ----- goto() -------------------------------------------------------------


def _floor_vm(extent: float = 4.0, y_floor: float = 1.0, voxel_size: float = 0.1) -> VoxelMap:
    """A clear floor of NAVIGABLE cells spanning [-extent, extent] in both x and z.

    Inputs are float64 to avoid float32 precision drift colliding adjacent
    voxels at exact cell boundaries (Pi3X-shaped outputs are continuous and
    don't hit this in practice; this fixture deliberately puts points AT
    voxel boundaries so we'd quietly merge ~25% of them in float32)."""
    vm = VoxelMap(voxel_size=voxel_size)
    pts = []
    # Offset placement by half a voxel so each xz lands at a cell *centre*,
    # robust to small float drift.
    half = voxel_size / 2.0
    for x in np.arange(-extent + half, extent + half, voxel_size):
        for z in np.arange(-extent + half, extent + half, voxel_size):
            pts.append((float(x), y_floor, float(z)))
    arr = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 3)
    rgb_arr = np.full((len(pts), 1, 3), 200, dtype=np.uint8)
    conf_arr = np.ones((len(pts), 1), dtype=np.float32)
    vm.add(arr, rgb_arr, conf_arr, frame=0, t=0.0)
    return vm


def test_goto_reaches_static_goal():
    vm = _floor_vm()
    base = StubBaseController()
    skills = SpatialSkills(
        vm,
        base,
        cfg=SkillsConfig(
            cell_size=0.1,
            obstacle_inflate_cells=0,
            goto_threshold=0.3,
            goto_max_steps=400,
            goto_step_size=0.1,
        ),
    )
    result = skills.goto((2.0, 1.0, 2.0))
    assert result.reached, f"goto did not reach: {result}"
    assert result.distance_to_target < 0.3
    # Should have logged the executed path.
    assert len(result.path_xyz) > 0


def test_goto_blocked_with_wall():
    """A floor with a wall of obstacle voxels splitting the navigable space.
    The wall extends past the floor on both ends so there is no corner
    detour — A* must report no-path."""
    vm = _floor_vm(extent=2.0)
    # Vertical wall along x=0 at obstacle height, spanning more z than the
    # floor so neither end of the wall has a navigable bypass cell.
    wall_pts = [
        (0.0, float(y), float(z)) for y in np.arange(0.2, 0.9, 0.1) for z in np.arange(-3.0, 3.0, 0.1)
    ]
    arr = np.asarray(wall_pts, dtype=np.float64).reshape(-1, 1, 3)
    rgb_arr = np.full((len(wall_pts), 1, 3), 200, dtype=np.uint8)
    conf_arr = np.ones((len(wall_pts), 1), dtype=np.float32)
    vm.add(arr, rgb_arr, conf_arr, frame=0, t=0.0)

    init = np.eye(4)
    init[0, 3] = -1.0
    base = StubBaseController(initial_pose=init)
    skills = SpatialSkills(
        vm,
        base,
        cfg=SkillsConfig(
            cell_size=0.1,
            obstacle_inflate_cells=0,
            goto_threshold=0.2,
            goto_max_steps=200,
        ),
    )
    result = skills.goto((1.0, 1.0, 0.0))
    assert result.reached is False
    assert result.reason == "no path"


def test_goto_stops_when_already_at_goal():
    vm = _floor_vm()
    init = np.eye(4)
    init[0, 3] = 0.5
    base = StubBaseController(initial_pose=init)
    skills = SpatialSkills(vm, base, cfg=SkillsConfig(goto_threshold=0.5))
    result = skills.goto((0.5, 0.0, 0.0))
    assert result.reached and result.n_steps == 0


# ----- explore() ----------------------------------------------------------


def test_explore_returns_a_frontier_when_one_exists():
    # Build a small floor and let project_voxel_map_to_grid pad the bbox so
    # there's UNOBSERVED space around it.
    vm = _floor_vm(extent=1.0)
    base = StubBaseController()
    skills = SpatialSkills(
        vm,
        base,
        cfg=SkillsConfig(
            cell_size=0.1,
            obstacle_inflate_cells=0,
        ),
    )
    result = skills.explore()
    assert result.found_frontier
    assert result.target_xyz is not None


def test_explore_reports_no_frontier_on_empty_voxelmap():
    vm = VoxelMap()
    skills = SpatialSkills(vm, StubBaseController(), cfg=SkillsConfig())
    result = skills.explore()
    assert result.found_frontier is False
    assert result.target_xyz is None


def test_explore_target_distance_matches_pose():
    vm = _floor_vm(extent=1.0)
    init = np.eye(4)
    init[0, 3] = 0.3
    init[2, 3] = -0.4
    base = StubBaseController(initial_pose=init)
    skills = SpatialSkills(vm, base, cfg=SkillsConfig(cell_size=0.1, obstacle_inflate_cells=0))
    result = skills.explore()
    if result.target_xyz is not None:
        d = math.hypot(result.target_xyz[0] - 0.3, result.target_xyz[2] - (-0.4))
        assert abs(d - result.distance_to_target) < 1e-3
