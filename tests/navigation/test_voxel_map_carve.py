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

"""Unit tests for ``VoxelMap.carve`` — DynaMem-style free-space removal (M4)."""

# ruff: noqa: N803, N806  — H, W, D: conventional array-dimension names
from __future__ import annotations

import math

import numpy as np
import pytest

from lerobot.navigation.voxel_map import CarveResult, VoxelMap


def _identity_pose() -> np.ndarray:
    """Camera-to-world identity (camera == world)."""
    return np.eye(4, dtype=np.float64)


def _shifted_pose(tx: float = 0.0, ty: float = 0.0, tz: float = 0.0) -> np.ndarray:
    p = np.eye(4, dtype=np.float64)
    p[:3, 3] = (tx, ty, tz)
    return p


def _solid_depth_view(H: int, W: int, depth: float, conf: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """A synthetic Pi3X view: every pixel sees a surface at ``depth`` meters."""
    fov_deg = 90.0
    focal = W / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    # local_points[v,u] = depth * (K^-1 @ [u,v,1]); z = depth.
    x = (us - cx) * depth / focal
    y = (vs - cy) * depth / focal
    z = np.full_like(x, depth, dtype=np.float64)
    local = np.stack([x, y, z], axis=-1).astype(np.float32)
    return local, np.full((H, W), conf, dtype=np.float32), focal


def _seed_voxel(vm: VoxelMap, xyz: tuple[float, float, float], frame: int = 0) -> None:
    pts = np.asarray([xyz], dtype=np.float32)
    rgb = np.full((1, 3), 200, dtype=np.uint8)
    conf = np.ones(1, dtype=np.float32)
    vm.add(pts, rgb, conf, frame=frame, t=float(frame) * 0.5)


# ---------------------------------------------------------------------------


def test_empty_map_returns_zero():
    vm = VoxelMap(voxel_size=0.1)
    local, conf, focal = _solid_depth_view(32, 32, depth=5.0)
    result = vm.carve(local, conf, _identity_pose(), focal_px=focal, frame=0, t=0.0)
    assert isinstance(result, CarveResult)
    assert result.n_removed == 0
    assert result.removed_xyz.shape == (0, 3)


def test_voxel_in_front_of_surface_is_removed():
    """Voxel at z=2 m, observed surface at z=5 m, margin 0.05: free space."""
    vm = VoxelMap(voxel_size=0.1)
    _seed_voxel(vm, (0.0, 0.0, 2.0))
    assert len(vm) == 1
    local, conf, focal = _solid_depth_view(32, 32, depth=5.0)
    result = vm.carve(local, conf, _identity_pose(), focal_px=focal, frame=1, t=0.5)
    assert result.n_removed == 1
    assert len(vm) == 0
    assert result.removed_xyz.shape == (1, 3)


def test_voxel_at_surface_is_kept():
    """Voxel at z = 5.0 m, surface at 5.0 m — d is NOT < D - margin."""
    vm = VoxelMap(voxel_size=0.1)
    _seed_voxel(vm, (0.0, 0.0, 5.0))
    local, conf, focal = _solid_depth_view(32, 32, depth=5.0)
    result = vm.carve(local, conf, _identity_pose(), focal_px=focal, frame=1, t=0.5, margin=0.05)
    assert result.n_removed == 0
    assert len(vm) == 1


def test_voxel_behind_surface_is_kept():
    """Voxel at z=10 m, surface at z=5 m: voxel is occluded, NOT free space."""
    vm = VoxelMap(voxel_size=0.1)
    _seed_voxel(vm, (0.0, 0.0, 10.0))
    local, conf, focal = _solid_depth_view(32, 32, depth=5.0)
    result = vm.carve(local, conf, _identity_pose(), focal_px=focal, frame=1, t=0.5)
    assert result.n_removed == 0


def test_voxel_behind_camera_is_kept():
    """A voxel with z<=0 in the camera frame can't be seen — must not be carved."""
    vm = VoxelMap(voxel_size=0.1)
    _seed_voxel(vm, (0.0, 0.0, -1.0))
    local, conf, focal = _solid_depth_view(32, 32, depth=5.0)
    result = vm.carve(local, conf, _identity_pose(), focal_px=focal, frame=1, t=0.5)
    assert result.n_removed == 0


def test_voxel_out_of_view_is_kept():
    """A voxel inside no pixel's frustum should be left alone."""
    vm = VoxelMap(voxel_size=0.1)
    # Way off to the side — projects outside the 32x32 image at 90° HFOV.
    _seed_voxel(vm, (50.0, 0.0, 2.0))
    local, conf, focal = _solid_depth_view(32, 32, depth=5.0)
    result = vm.carve(local, conf, _identity_pose(), focal_px=focal, frame=1, t=0.5)
    assert result.n_removed == 0


def test_low_confidence_blocks_carve():
    """If conf at the projected pixel is below threshold, don't remove."""
    vm = VoxelMap(voxel_size=0.1)
    _seed_voxel(vm, (0.0, 0.0, 2.0))
    local, conf, focal = _solid_depth_view(32, 32, depth=5.0, conf=0.2)
    result = vm.carve(local, conf, _identity_pose(), focal_px=focal, frame=1, t=0.5, conf_thresh=0.5)
    assert result.n_removed == 0


def test_invalid_depth_blocks_carve():
    """NaN / non-positive depth at the projected pixel must not trigger carve."""
    vm = VoxelMap(voxel_size=0.1)
    _seed_voxel(vm, (0.0, 0.0, 2.0))
    local, conf, focal = _solid_depth_view(32, 32, depth=5.0)
    local[:, :, 2] = np.nan
    result = vm.carve(local, conf, _identity_pose(), focal_px=focal, frame=1, t=0.5)
    assert result.n_removed == 0


def test_margin_protects_near_surface():
    """A voxel 3 cm in front of a 5 m surface, margin 5 cm: NOT carved."""
    vm = VoxelMap(voxel_size=0.1)
    _seed_voxel(vm, (0.0, 0.0, 5.0 - 0.03))
    local, conf, focal = _solid_depth_view(32, 32, depth=5.0)
    result = vm.carve(local, conf, _identity_pose(), focal_px=focal, frame=1, t=0.5, margin=0.05)
    assert result.n_removed == 0


def test_margin_zero_carves_near_surface():
    """Same setup, margin 0: now the 3 cm gap counts as free space."""
    vm = VoxelMap(voxel_size=0.1)
    _seed_voxel(vm, (0.0, 0.0, 5.0 - 0.03))
    local, conf, focal = _solid_depth_view(32, 32, depth=5.0)
    result = vm.carve(local, conf, _identity_pose(), focal_px=focal, frame=1, t=0.5, margin=0.0)
    assert result.n_removed == 1


def test_carve_compacts_arrays_and_lookup():
    """After carving some but not all voxels, internal storage stays consistent."""
    vm = VoxelMap(voxel_size=0.1)
    # Three voxels: in front (will be carved), at surface (kept), to the side (kept).
    _seed_voxel(vm, (0.0, 0.0, 2.0))
    _seed_voxel(vm, (0.0, 0.0, 5.0))
    _seed_voxel(vm, (50.0, 0.0, 5.0))
    assert len(vm) == 3
    local, conf, focal = _solid_depth_view(32, 32, depth=5.0)
    result = vm.carve(local, conf, _identity_pose(), focal_px=focal, frame=1, t=0.5)
    assert result.n_removed == 1
    assert len(vm) == 2
    # All internal arrays must agree on the new length.
    assert vm._idx.shape == (2, 3)  # noqa: SLF001
    assert vm._count.shape == (2,)  # noqa: SLF001
    assert len(vm._lookup) == 2  # noqa: SLF001
    # Lookup rows must point at valid indices in the new arrays.
    for row in vm._lookup.values():  # noqa: SLF001
        assert 0 <= row < 2
    # A subsequent `add` to a removed voxel must work cleanly (revives it).
    pts = np.asarray([[0.0, 0.0, 2.0]], dtype=np.float32)
    rgb = np.full((1, 3), 100, dtype=np.uint8)
    conf1 = np.ones(1, dtype=np.float32)
    add_stats = vm.add(pts, rgb, conf1, frame=2, t=1.0)
    assert add_stats.n_added == 1
    assert len(vm) == 3


def test_carve_shape_validation():
    vm = VoxelMap(voxel_size=0.1)
    _seed_voxel(vm, (0.0, 0.0, 2.0))
    bad_local = np.zeros((32, 32, 2), dtype=np.float32)
    conf = np.ones((32, 32), dtype=np.float32)
    with pytest.raises(ValueError, match=r"H, W, 3"):
        vm.carve(bad_local, conf, _identity_pose(), focal_px=16.0, frame=0, t=0.0)

    local = np.zeros((32, 32, 3), dtype=np.float32)
    bad_conf = np.ones((16, 32), dtype=np.float32)
    with pytest.raises(ValueError, match=r"conf shape"):
        vm.carve(local, bad_conf, _identity_pose(), focal_px=16.0, frame=0, t=0.0)

    bad_pose = np.eye(3)
    with pytest.raises(ValueError, match=r"pose must be"):
        vm.carve(local, conf, bad_pose, focal_px=16.0, frame=0, t=0.0)


def test_carve_with_translated_camera():
    """If the camera has moved, world-space voxels must be transformed correctly
    before the free-space test."""
    vm = VoxelMap(voxel_size=0.1)
    # Voxel at world position (10, 0, 2). With camera at world (10, 0, 0),
    # the voxel is 2 m in front of the camera's local +Z; surface at 5 m → carve.
    _seed_voxel(vm, (10.0, 0.0, 2.0))
    pose = _shifted_pose(tx=10.0, ty=0.0, tz=0.0)
    local, conf, focal = _solid_depth_view(32, 32, depth=5.0)
    result = vm.carve(local, conf, pose, focal_px=focal, frame=1, t=0.5)
    assert result.n_removed == 1
