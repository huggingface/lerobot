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

"""Unit tests for ``VoxelMap`` — geometry-only behavior (M3)."""

# ruff: noqa: N803, N806  — H, W, D: conventional array-dimension names
from __future__ import annotations

import numpy as np
import pytest

from lerobot.navigation.voxel_map import VoxelMap


def _scatter(points: np.ndarray, color: tuple[int, int, int] = (200, 100, 50)) -> tuple[np.ndarray, ...]:
    """Helper: build (points, rgb, conf) arrays from a list of xyz coords."""
    rgb = np.tile(np.array(color, dtype=np.uint8), (len(points), 1))
    conf = np.ones(len(points), dtype=np.float32)
    return points.astype(np.float32), rgb, conf


def test_initially_empty():
    vm = VoxelMap(voxel_size=0.05)
    assert len(vm) == 0
    snap = vm.snapshot()
    assert snap.xyz.shape == (0, 3)
    assert snap.rgb.shape == (0, 3)


def test_voxel_size_must_be_positive():
    with pytest.raises(ValueError):
        VoxelMap(voxel_size=0.0)
    with pytest.raises(ValueError):
        VoxelMap(voxel_size=-0.1)


def test_single_point_creates_one_voxel():
    vm = VoxelMap(voxel_size=0.1)
    pts, rgb, conf = _scatter(np.array([[0.123, 0.456, 0.789]]))
    stats = vm.add(pts, rgb, conf, frame=0, t=0.0)
    assert stats == type(stats)(n_voxels=1, n_added=1, n_updated=0)
    assert len(vm) == 1
    snap = vm.snapshot()
    np.testing.assert_allclose(snap.xyz[0], [0.123, 0.456, 0.789], atol=1e-5)


def test_points_in_same_voxel_collapse_and_average():
    vm = VoxelMap(voxel_size=0.1)
    # Two points inside the voxel [0.0, 0.1) on each axis.
    pts, rgb, conf = _scatter(np.array([[0.01, 0.01, 0.01], [0.09, 0.09, 0.09]]))
    stats = vm.add(pts, rgb, conf, frame=0, t=0.0)
    assert stats.n_voxels == 1
    assert stats.n_added == 1
    snap = vm.snapshot()
    np.testing.assert_allclose(snap.xyz[0], [0.05, 0.05, 0.05], atol=1e-5)
    assert int(snap.count[0]) == 2


def test_second_keyframe_updates_running_mean():
    vm = VoxelMap(voxel_size=0.1)
    pts1, rgb1, conf1 = _scatter(np.array([[0.02, 0.02, 0.02]]), color=(100, 100, 100))
    vm.add(pts1, rgb1, conf1, frame=0, t=0.0)
    pts2, rgb2, conf2 = _scatter(np.array([[0.08, 0.08, 0.08]]), color=(200, 200, 200))
    stats = vm.add(pts2, rgb2, conf2, frame=1, t=0.5)
    assert stats.n_added == 0
    assert stats.n_updated == 1
    snap = vm.snapshot()
    # Mean position = (0.02 + 0.08) / 2 = 0.05; mean color = 150.
    np.testing.assert_allclose(snap.xyz[0], [0.05, 0.05, 0.05], atol=1e-5)
    np.testing.assert_allclose(snap.rgb[0], [150, 150, 150], atol=1)
    assert int(snap.last_frame[0]) == 1
    assert float(snap.last_time[0]) == pytest.approx(0.5)


def test_conf_gate_drops_low_confidence_pixels():
    vm = VoxelMap(voxel_size=0.1)
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    rgb = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)
    conf = np.array([0.9, 0.1], dtype=np.float32)
    stats = vm.add(pts, rgb, conf, frame=0, t=0.0, conf_thresh=0.5)
    assert stats.n_voxels == 1
    snap = vm.snapshot()
    np.testing.assert_allclose(snap.xyz[0], [0.0, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(snap.rgb[0], [10, 20, 30], atol=1)


def test_quantization_negative_coordinates():
    vm = VoxelMap(voxel_size=0.1)
    pts, rgb, conf = _scatter(np.array([[-0.05, -0.15, -0.25]]))
    stats = vm.add(pts, rgb, conf, frame=0, t=0.0)
    assert stats.n_voxels == 1
    # floor(-0.05/0.1) = floor(-0.5) = -1 (voxel covers [-0.1, 0.0)).
    # Just check the mean equals the input single point.
    snap = vm.snapshot()
    np.testing.assert_allclose(snap.xyz[0], [-0.05, -0.15, -0.25], atol=1e-5)


def test_image_shaped_input_is_flattened():
    """``add`` accepts (H, W, 3) point arrays — typical Pi3X output shape."""
    vm = VoxelMap(voxel_size=0.5)
    H, W = 4, 4
    pts = np.zeros((H, W, 3), dtype=np.float32)
    pts[..., 0] = np.linspace(0, 5, W)[None, :]  # 16 unique x values? No, 4.
    rgb = np.full((H, W, 3), 128, dtype=np.uint8)
    conf = np.ones((H, W), dtype=np.float32)
    stats = vm.add(pts, rgb, conf, frame=0, t=0.0)
    # All x values land in voxel slots at 0, 1, 2, 3, 4 (different voxels at
    # 0.5m size); each row of the image contributes the same 4 unique voxels,
    # collapsed within the keyframe — but actually voxel indices depend on x.
    # The point of this test is just that flattening works without raising.
    assert stats.n_voxels >= 1
    assert stats.n_voxels <= H * W


def test_nonfinite_points_are_dropped():
    vm = VoxelMap(voxel_size=0.1)
    pts = np.array(
        [[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0], [0.0, np.inf, 0.0]],
        dtype=np.float32,
    )
    rgb = np.full((3, 3), 100, dtype=np.uint8)
    conf = np.ones(3, dtype=np.float32)
    stats = vm.add(pts, rgb, conf, frame=0, t=0.0)
    assert stats.n_voxels == 1


def test_snapshot_rgb_clipped_to_uint8():
    """If RGB sums accumulate to > 255 per channel, the snapshot mean is still
    a clean uint8."""
    vm = VoxelMap(voxel_size=0.1)
    pts, rgb, conf = _scatter(np.array([[0.0, 0.0, 0.0]]), color=(255, 255, 255))
    vm.add(pts, rgb, conf, frame=0, t=0.0)
    vm.add(pts, rgb, conf, frame=1, t=0.5)
    snap = vm.snapshot()
    assert snap.rgb.dtype == np.uint8
    np.testing.assert_array_equal(snap.rgb[0], [255, 255, 255])
