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

"""Unit tests for the C2 scene-mutation helper on VoxelMap."""

from __future__ import annotations

import numpy as np

from lerobot.navigation.voxel_map import VoxelMap


def test_remove_voxels_in_box_zero_when_empty():
    vm = VoxelMap()
    assert vm.remove_voxels_in_box((-1, -1, -1), (1, 1, 1)) == 0


def test_remove_voxels_in_box_only_deletes_inside():
    vm = VoxelMap(voxel_size=0.1)
    pts = np.array(
        [
            [[0.0, 0.0, 0.0]],  # inside
            [[0.1, 0.0, 0.0]],  # inside
            [[2.0, 0.0, 0.0]],  # outside
        ],
        dtype=np.float64,
    )
    rgb = np.full((3, 1, 3), 200, dtype=np.uint8)
    conf = np.ones((3, 1), dtype=np.float32)
    vm.add(pts, rgb, conf, frame=0, t=0.0)
    assert len(vm) == 3

    n = vm.remove_voxels_in_box((-0.05, -0.05, -0.05), (0.15, 0.05, 0.05))
    assert n == 2
    assert len(vm) == 1
    snap = vm.snapshot()
    np.testing.assert_allclose(snap.xyz[0], [2.0, 0.0, 0.0], atol=1e-3)


def test_remove_voxels_keeps_feature_arrays_aligned():
    vm = VoxelMap(voxel_size=0.1)
    pts = np.array([[[0.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]], dtype=np.float64)
    rgb = np.full((2, 1, 3), 200, dtype=np.uint8)
    conf = np.ones((2, 1), dtype=np.float32)
    feat = np.array([[[1.0, 0.0, 0.0, 0.0]], [[0.0, 1.0, 0.0, 0.0]]], dtype=np.float16)
    vm.add(pts, rgb, conf, frame=0, t=0.0, feat_map=feat)
    assert len(vm) == 2
    assert vm._feat_sum.shape == (2, 4)  # noqa: SLF001

    vm.remove_voxels_in_box((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))
    assert len(vm) == 1
    # Feature arrays now have length 1 — matches _count.
    assert vm._feat_sum.shape == (1, 4)  # noqa: SLF001
    snap = vm.snapshot(include_features=True)
    # Surviving voxel had vector [0, 1, 0, 0]; normalized stays the same.
    assert snap.feat is not None
    np.testing.assert_allclose(
        snap.feat[0].astype(np.float32),
        [0.0, 1.0, 0.0, 0.0],
        atol=1e-3,
    )


def test_remove_voxels_compacts_lookup():
    """After deletion the dict→row map must still point to the right rows."""
    vm = VoxelMap(voxel_size=0.1)
    pts = np.array(
        [[[0.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]], [[4.0, 0.0, 0.0]]],
        dtype=np.float64,
    )
    rgb = np.full((3, 1, 3), 200, dtype=np.uint8)
    conf = np.ones((3, 1), dtype=np.float32)
    vm.add(pts, rgb, conf, frame=0, t=0.0)

    vm.remove_voxels_in_box((1.5, -0.5, -0.5), (2.5, 0.5, 0.5))  # delete the middle
    assert len(vm) == 2

    # Adding a new voxel at one of the surviving positions should UPDATE
    # (not append), which only works if the lookup row indices are correct.
    new_pt = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float64)
    stats = vm.add(
        new_pt, np.full((1, 1, 3), 50, dtype=np.uint8), np.ones((1, 1), dtype=np.float32), frame=1, t=1.0
    )
    assert stats.n_added == 0
    assert stats.n_updated == 1
    assert len(vm) == 2
