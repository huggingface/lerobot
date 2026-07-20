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

"""Tests for the viz-free keyframe integration loop.

Uses FakeGeometryRunner output fed through integrate_keyframe into a real
VoxelMap — no models, no viz.
"""

from __future__ import annotations

import numpy as np

from lerobot.navigation.geometry import FakeGeometryRunner
from lerobot.navigation.pipeline import (
    KeyframeContext,
    PipelineConfig,
    integrate_keyframe,
    upsample_features_to_view,
)
from lerobot.navigation.voxel_map import VoxelMap


def _ctx_from_geometry(frame_idx=0, t=0.0, feat_map=None) -> KeyframeContext:
    h = w = 14
    views = np.full((1, h, w, 3), 120, dtype=np.uint8)
    out = FakeGeometryRunner(depth=3.0, focal_px=100.0)(views)
    return KeyframeContext(
        frame_idx=frame_idx,
        t_sec=t,
        rgb_uint8=views[0],
        points_world=out.points[0],
        local_points=out.local_points[0],
        conf=out.conf[0],
        pose=out.camera_poses[0],
        feat_map=feat_map,
    )


def test_integrate_adds_voxels():
    vm = VoxelMap(voxel_size=0.05)
    ctx = _ctx_from_geometry()
    carve, stats = integrate_keyframe(vm, ctx, PipelineConfig(focal_px=100.0))
    assert stats.n_added > 0
    assert len(vm) == stats.n_voxels
    assert carve.n_removed == 0  # nothing to carve on an empty map


def test_integrate_second_frame_updates_not_duplicates():
    vm = VoxelMap(voxel_size=0.05)
    pcfg = PipelineConfig(focal_px=100.0)
    integrate_keyframe(vm, _ctx_from_geometry(frame_idx=0, t=0.0), pcfg)
    n_after_first = len(vm)
    _, stats2 = integrate_keyframe(vm, _ctx_from_geometry(frame_idx=1, t=0.5), pcfg)
    # Same synthetic view → same voxels updated, not a second copy.
    assert stats2.n_added == 0
    assert len(vm) == n_after_first


def test_integrate_with_features_enables_query():
    vm = VoxelMap(voxel_size=0.05)
    h = w = 14
    d = 8
    feat = np.zeros((h, w, d), dtype=np.float16)
    feat[..., 0] = 1.0  # every pixel carries basis vector 0
    ctx = _ctx_from_geometry(feat_map=feat)
    integrate_keyframe(vm, ctx, PipelineConfig(focal_px=100.0))
    assert vm.feature_dim == d
    q = np.zeros(d, dtype=np.float32)
    q[0] = 1.0
    result = vm.query(q, top_k=5)
    assert result.score.size > 0
    assert float(result.score.max()) > 0.9  # basis-0 query matches basis-0 voxels


def test_upsample_features_to_view_shape():
    patch = np.zeros((3, 3, 8), dtype=np.float16)
    up = upsample_features_to_view(patch, view_h=28, view_w=28)
    assert up.shape == (28, 28, 8)
    assert up.dtype == np.float16
