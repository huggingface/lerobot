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
"""Tests for LIBERO's depth conversions.

These cover the two transforms between robosuite's depth buffer and something a
3D policy can unproject. Both were wrong at some point in development and
neither failure was visible downstream: a point cloud built from a mis-scaled or
mirrored depth map is still a perfectly plausible point cloud.

They run without MuJoCo or a GPU, so they guard the arithmetic in CI. The
geometry itself is verified separately against simulator ground truth, by
projecting known object positions into the image and checking the depth sampled
there lands just in front of each object.
"""

import numpy as np
import pytest

# `lerobot.envs.libero` imports the `libero` package at module scope, so these
# skip where it is absent. They are pure arithmetic and run in milliseconds
# wherever the [libero] extra is installed.
pytest.importorskip("libero", reason="requires the libero extra")

from lerobot.envs.libero import (  # noqa: E402
    DEPTH_IMAGE_ORIGIN,
    depth_to_metres,
    orient_depth,
)


class FakeSim:
    """The three numbers `depth_to_metres` needs, with LIBERO's real values."""

    class model:  # noqa: N801
        class stat:  # noqa: N801
            extent = 1.0619

        class vis:  # noqa: N801
            class map:  # noqa: N801,A003
                znear = 0.01
                zfar = 499.5


def test_depth_conversion_is_metric_and_monotonic():
    sim = FakeSim()
    near = sim.model.vis.map.znear * sim.model.stat.extent
    far = sim.model.vis.map.zfar * sim.model.stat.extent

    # The buffer's endpoints must land on the near and far planes.
    assert depth_to_metres(np.array([0.0]), sim)[0] == pytest.approx(near, rel=1e-6)
    assert depth_to_metres(np.array([1.0]), sim)[0] == pytest.approx(far, rel=1e-6)

    raw = np.linspace(0.0, 1.0, 64)
    metres = depth_to_metres(raw, sim)
    assert np.all(np.diff(metres) > 0), "depth must increase with the raw buffer"


def test_depth_conversion_matches_the_observed_range():
    """A tabletop scene sits in the top 1.6% of LIBERO's buffer.

    LIBERO's far plane is about 530 m for a scene barely 3 m deep, so raw values
    crowd near 1.0. Measured on libero_spatial the buffer spans 0.9843 to
    0.9966, which must come back as roughly 0.68 to 3.07 m. If someone changes
    the conversion, this is the number that moves.
    """
    metres = depth_to_metres(np.array([0.9843, 0.9966]), FakeSim())
    assert metres[0] == pytest.approx(0.677, abs=0.02)
    assert metres[1] == pytest.approx(3.066, abs=0.05)


def test_conversion_precision_is_not_lost_to_the_far_plane():
    """The crowding near 1.0 looks like it must cost precision. It does not.

    dz/dd is about z^2 / near, so at 1 m one float32 step of the raw buffer is
    a few micrometres. Worth pinning, because the obvious reaction to a 530 m
    far plane is to 'fix' it and change the rendering.
    """
    sim = FakeSim()
    near = sim.model.vis.map.znear * sim.model.stat.extent
    d = np.float32(0.99)
    step = np.spacing(d)
    delta = abs(float(depth_to_metres(np.array([d + step]), sim)[0] - depth_to_metres(np.array([d]), sim)[0]))
    z = float(depth_to_metres(np.array([d]), sim)[0])
    assert delta < 1e-4, f"{delta * 1e6:.1f} um per float32 step at {z:.2f} m"
    assert delta == pytest.approx(z**2 / near * float(step), rel=0.05)


def test_orient_depth_flips_height_not_width():
    """The bug this test exists for: robosuite hands back (H, W, 1).

    The natural spelling `depth[..., ::-1, :]` flips rows for a 2D array and
    flips WIDTH for (H, W, 1). Both produce a plausible point cloud, and the
    end-to-end geometry check measured 316 mm against the 308 mm expected for a
    horizontal mirror, which is how it was caught.
    """
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.testing.assert_array_equal(orient_depth(a), a[::-1])
    np.testing.assert_array_equal(orient_depth(a[..., None]), a[::-1])
    assert orient_depth(a[..., None]).shape == (3, 4)


def test_orient_depth_is_an_involution():
    a = np.random.default_rng(0).random((5, 7)).astype(np.float32)
    np.testing.assert_array_equal(orient_depth(orient_depth(a)), a)


def test_origin_constant_is_declared():
    """Stated so a reader knows why the flip exists rather than guessing."""
    assert DEPTH_IMAGE_ORIGIN == "bottom-left"


def test_config_passes_use_depth_through():
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig

    assert LiberoEnvConfig().gym_kwargs["use_depth"] is False
    assert LiberoEnvConfig(use_depth=True).gym_kwargs["use_depth"] is True
