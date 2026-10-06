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
"""Tests for point-cloud geometry and the DP3 encoder.

The geometry is checked against facts known independently of the code -- a plane
at a known depth, a pixel exactly one focal length off-axis, a pure translation,
a 90 degree rotation -- rather than against a golden array, because a golden
array would happily encode a transposed rotation forever.
"""

import numpy as np
import pytest
import torch

from lerobot.policies.common.pointcloud import (
    PointCloudEncoder,
    normalize_points,
    sample_points,
    unproject,
)


@pytest.fixture
def pinhole():
    """An odd-sized image so the principal point lands exactly on a pixel."""
    size, focal = 33, 8.0
    centre = (size - 1) // 2
    intrinsics = torch.tensor([[focal, 0.0, float(centre)], [0.0, focal, float(centre)], [0.0, 0.0, 1.0]])
    return size, focal, centre, intrinsics


def test_unproject_plane_is_at_the_stated_depth(pinhole):
    size, _, _, intrinsics = pinhole
    depth = torch.full((size, size), 2.0)
    points = unproject(depth[None], intrinsics[None])[0]
    assert torch.allclose(points[..., 2], torch.full((size, size), 2.0))


def test_unproject_principal_point_is_on_the_optical_axis(pinhole):
    size, _, centre, intrinsics = pinhole
    points = unproject(torch.full((1, size, size), 2.0), intrinsics[None])[0]
    assert torch.allclose(points[centre, centre, :2], torch.zeros(2), atol=1e-6)


def test_unproject_off_axis_pixel_subtends_the_right_angle(pinhole):
    """A pixel one focal length off-axis sits at 45 degrees, so x == z."""
    size, focal, centre, intrinsics = pinhole
    points = unproject(torch.full((1, size, size), 2.0), intrinsics[None])[0]
    off_axis = points[centre, centre + int(focal)]
    assert off_axis[0].item() == pytest.approx(off_axis[2].item(), abs=1e-5)


def test_unproject_applies_translation_exactly(pinhole):
    size, _, _, intrinsics = pinhole
    depth = torch.full((1, size, size), 2.0)
    extrinsics = torch.eye(4)
    extrinsics[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
    camera = unproject(depth, intrinsics[None])[0]
    world = unproject(depth, intrinsics[None], extrinsics[None])[0]
    assert torch.allclose(world - camera, extrinsics[:3, 3].expand_as(camera), atol=1e-5)


def test_unproject_applies_rotation_exactly(pinhole):
    """90 degrees about +z sends (x, y, z) to (-y, x, z)."""
    size, _, _, intrinsics = pinhole
    depth = torch.full((1, size, size), 2.0)
    extrinsics = torch.eye(4)
    extrinsics[:3, :3] = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    camera = unproject(depth, intrinsics[None])[0]
    world = unproject(depth, intrinsics[None], extrinsics[None])[0]
    expected = torch.stack([-camera[..., 1], camera[..., 0], camera[..., 2]], dim=-1)
    assert torch.allclose(world, expected, atol=1e-5)


def test_unproject_is_differentiable_in_the_extrinsics(pinhole):
    """Gradients through calibration are what would let a policy refine it."""
    size, _, _, intrinsics = pinhole
    extrinsics = torch.eye(4).requires_grad_(True)
    unproject(torch.full((1, size, size), 2.0), intrinsics[None], extrinsics[None]).sum().backward()
    assert extrinsics.grad is not None
    assert torch.isfinite(extrinsics.grad).all()


def test_calibration_error_matches_the_documented_rule_of_thumb():
    """The module docstring promises ~9 mm per degree at d = 0.7 m.

    That number is `(pi/4) * d * sin(eps)`, where the pi/4 is the mean sine of
    the angle between a uniformly random rotation axis and the line of sight.
    Checked here by Monte Carlo so the docstring cannot quietly drift.
    """
    distance, eps_deg, rng = 0.7, 1.0, np.random.default_rng(0)
    point = np.array([0.0, 0.0, distance])
    displacements = []
    for _ in range(4000):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        angle = np.deg2rad(eps_deg)
        rotation = np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)
        displacements.append(np.linalg.norm(rotation @ point - point))
    measured_mm = float(np.mean(displacements)) * 1000
    predicted_mm = (np.pi / 4) * distance * np.sin(np.deg2rad(eps_deg)) * 1000
    assert measured_mm == pytest.approx(predicted_mm, abs=0.35)
    assert measured_mm == pytest.approx(9.0, abs=1.0)


def test_sample_points_returns_only_valid_points():
    points = torch.randn(4, 500, 3)
    valid = torch.zeros(4, 500, dtype=torch.bool)
    valid[:, :50] = True
    sampled = sample_points(points, valid, 32)
    assert sampled.shape == (4, 32, 3)
    for b in range(4):
        for i in range(32):
            assert any(torch.allclose(sampled[b, i], points[b, j]) for j in range(50))


def test_sample_points_survives_a_frame_with_no_returns():
    """A dropped depth frame should not kill a training run."""
    points = torch.randn(2, 100, 3)
    sampled = sample_points(points, torch.zeros(2, 100, dtype=torch.bool), 16)
    assert sampled.shape == (2, 16, 3)
    assert torch.all(sampled == 0)


def test_sample_points_pads_when_asked_for_more_than_exist():
    points = torch.randn(2, 40, 3)
    valid = torch.ones(2, 40, dtype=torch.bool)
    assert sample_points(points, valid, 128).shape == (2, 128, 3)


def test_sample_points_never_returns_invalid_points_when_valid_are_scarce():
    """The case that matters: fewer valid returns than requested points.

    Taking the first `num_points` of a valid-first ordering silently emits
    invalid points here -- at the sensor origin, or outside a workspace crop --
    and nothing downstream notices. Sampling must wrap into the valid set
    instead.
    """
    points = torch.randn(4, 500, 3)
    valid = torch.zeros(4, 500, dtype=torch.bool)
    valid[:, :9] = True  # only 9 valid, 32 requested
    sampled = sample_points(points, valid, 32)
    assert sampled.shape == (4, 32, 3)
    for b in range(4):
        for i in range(32):
            assert any(torch.allclose(sampled[b, i], points[b, j]) for j in range(9))


@pytest.mark.parametrize("in_channels", [3, 6])
def test_encoder_shapes_and_feature_dim(in_channels):
    encoder = PointCloudEncoder(in_channels=in_channels, out_features=128)
    out = encoder(torch.randn(3, 256, in_channels))
    assert out.shape == (3, 128)
    assert encoder.feature_dim == 128


def test_encoder_is_permutation_invariant():
    """A point cloud is a set. If shuffling changes the embedding, it is not."""
    encoder = PointCloudEncoder(out_features=64).eval()
    points = torch.randn(2, 128, 3)
    shuffled = points[:, torch.randperm(128)]
    with torch.no_grad():
        assert torch.allclose(encoder(points), encoder(shuffled), atol=1e-5)


def test_encoder_stays_small():
    """DP3's finding is that a *simple* encoder wins; guard against creep."""
    n_params = sum(p.numel() for p in PointCloudEncoder().parameters())
    assert n_params < 300_000


def test_encoder_rejects_wrong_channel_count():
    with pytest.raises(ValueError):
        PointCloudEncoder(in_channels=3)(torch.randn(2, 64, 6))
    with pytest.raises(ValueError):
        PointCloudEncoder(in_channels=4)


def test_normalize_points_maps_the_workspace_cube_to_unit_range():
    centre = torch.tensor([0.0, 0.0, 0.2])
    corner = centre + torch.tensor([0.3, 0.3, 0.3])
    assert torch.allclose(normalize_points(corner, centre, 0.6), torch.ones(3))
    assert torch.allclose(normalize_points(centre, centre, 0.6), torch.zeros(3))
