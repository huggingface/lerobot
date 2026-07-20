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

"""Tests for geometry runners + the odometry similarity anchor.

Model-free: only the FakeGeometryRunner and the pure-numpy Umeyama fit
are exercised. LingBotMapRunner is checked for its no-SDK error only.
"""

# ruff: noqa: N806  — R, U, S, Vt, D: conventional linear-algebra / array-dimension names
from __future__ import annotations

import numpy as np
import pytest

from lerobot.navigation.geometry import (
    FakeGeometryRunner,
    GeometryOutput,
    GeometryRunner,
    LingBotMapRunner,
    align_trajectory_to_odometry,
    umeyama_similarity,
)


def _views(n=2, h=14, w=14) -> np.ndarray:
    return np.zeros((n, h, w, 3), dtype=np.uint8)


def test_fake_runner_satisfies_protocol():
    assert isinstance(FakeGeometryRunner(), GeometryRunner)


def test_fake_runner_output_shapes():
    out = FakeGeometryRunner(depth=3.0, focal_px=100.0)(_views(2, 14, 14))
    assert isinstance(out, GeometryOutput)
    assert out.points.shape == (2, 14, 14, 3)
    assert out.local_points.shape == (2, 14, 14, 3)
    assert out.conf.shape == (2, 14, 14)
    assert out.camera_poses.shape == (2, 4, 4)


def test_fake_runner_depth_is_constant():
    out = FakeGeometryRunner(depth=2.5)(_views())
    # local_points z channel is the depth everywhere.
    assert np.allclose(out.local_points[..., 2], 2.5)


def test_fake_runner_rejects_bad_shape():
    with pytest.raises(ValueError, match="N, H, W, 3"):
        FakeGeometryRunner()(np.zeros((14, 14, 3), dtype=np.uint8))


def test_lingbot_runner_raises_without_sdk():
    runner = LingBotMapRunner(device="cpu")
    with pytest.raises((RuntimeError, ValueError)):
        # Either the lazy import fails (no lingbot-map) or shape check trips
        # first — both are acceptable "did not silently succeed" outcomes.
        runner(_views())


# ----- Umeyama similarity --------------------------------------------------


def test_umeyama_recovers_known_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(20, 3))
    # Known transform: scale 2.5, a rotation about z by 30°, translation.
    theta = np.deg2rad(30.0)
    c, s = np.cos(theta), np.sin(theta)
    R_true = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    s_true, t_true = 2.5, np.array([1.0, -2.0, 0.5])
    dst = (s_true * (R_true @ src.T)).T + t_true

    s_fit, R_fit, t_fit = umeyama_similarity(src, dst)
    assert s_fit == pytest.approx(s_true, rel=1e-6)
    np.testing.assert_allclose(R_fit, R_true, atol=1e-6)
    np.testing.assert_allclose(t_fit, t_true, atol=1e-6)


def test_umeyama_reconstructs_points():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(10, 3))
    dst = 0.5 * src + np.array([3.0, 0.0, -1.0])
    s, R, t = umeyama_similarity(src, dst)
    recon = (s * (R @ src.T)).T + t
    np.testing.assert_allclose(recon, dst, atol=1e-6)


def test_umeyama_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        umeyama_similarity(np.zeros((5, 3)), np.zeros((4, 3)))


def test_align_requires_three_points():
    with pytest.raises(ValueError, match="at least 3"):
        align_trajectory_to_odometry(np.zeros((2, 3)), np.zeros((2, 3)))


def test_align_scale_anchor_makes_metric():
    """A monocular trajectory at half scale is recovered to metric."""
    odom = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], dtype=np.float64)
    cam = odom * 0.5  # model world is half-scale
    s, R, t = align_trajectory_to_odometry(cam, odom)
    assert s == pytest.approx(2.0, rel=1e-6)
    recon = (s * (R @ cam.T)).T + t
    np.testing.assert_allclose(recon, odom, atol=1e-9)
