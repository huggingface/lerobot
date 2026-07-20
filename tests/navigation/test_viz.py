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

"""Headless tests for the Rerun map visualizer.

``spawn=False`` buffers to an in-memory recording, so these run without a
display and skip cleanly when rerun-sdk isn't installed.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rerun", reason="rerun-sdk not installed (pip install 'lerobot[viz]')")

from lerobot.navigation.dog_cli import _build_dry_run  # noqa: E402
from lerobot.navigation.viz import MapVisualizer, _recency_colors  # noqa: E402
from lerobot.navigation.voxel_map import VoxelMap  # noqa: E402


def test_recency_colors_recent_vs_old():
    last = np.array([10.0, 0.0])  # one recent, one old
    colors = _recency_colors(last, now=10.0, horizon_s=10.0)
    assert colors.shape == (2, 3)
    assert colors.dtype == np.uint8
    # Recent voxel (age 0) is more green/cyan; old (age 1) is more red.
    assert colors[0, 1] > colors[1, 1]  # green channel higher for recent
    assert colors[1, 0] > colors[0, 0]  # red channel higher for old


def _viz() -> MapVisualizer:
    return MapVisualizer(app_id="test-dog-nav", spawn=False)


def test_log_map_and_dynamics_do_not_raise():
    viz = _viz()
    vm = VoxelMap(voxel_size=0.1)
    pts = np.array([[[0.0, 0.0, 1.0]], [[0.1, 0.0, 1.0]]], dtype=np.float64)
    rgb = np.full((2, 1, 3), 200, dtype=np.uint8)
    conf = np.ones((2, 1), dtype=np.float32)
    vm.add(pts, rgb, conf, frame=0, t=0.0)

    viz.set_time(0.0)
    viz.log_map(vm.snapshot(), now=0.0)
    viz.log_removed(np.array([[0.5, 0.0, 1.0]], dtype=np.float32))  # a carved voxel
    viz.log_robot(np.eye(4))
    viz.log_path([(0.0, 0.0, 0.0), (0.5, 0.0, 0.5)])
    viz.log_target((0.1, 0.0, 1.0))


def test_log_empty_map_clears_cleanly():
    viz = _viz()
    viz.set_time(1.0)
    viz.log_map(VoxelMap().snapshot())  # empty
    viz.log_removed(np.zeros((0, 3), dtype=np.float32))
    viz.log_target(None)


def test_recency_color_mode():
    viz = MapVisualizer(app_id="test-recency", spawn=False, color_mode="recency")
    vm = VoxelMap(voxel_size=0.1)
    pts = np.array([[[0.0, 0.0, 1.0]]], dtype=np.float64)
    vm.add(pts, np.full((1, 1, 3), 100, np.uint8), np.ones((1, 1), np.float32), frame=0, t=5.0)
    viz.set_time(5.0)
    viz.log_map(vm.snapshot(), now=5.0)


def test_dry_run_controller_with_viz_navigates():
    """End-to-end: the dry-run stack with a headless visualizer still reaches
    the couch, and every viz call along the way is exercised."""
    viz = _viz()
    controller = _build_dry_run(viz=viz)
    result = controller.handle_prompt("couch")
    assert result.fully_successful
