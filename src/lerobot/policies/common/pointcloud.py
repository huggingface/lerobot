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
"""Point-cloud geometry, for policies that consume depth.

This module has no LeRobot imports and no optional dependencies: it is plain
PyTorch, so it works wherever LeRobot works, including aarch64 and CPU-only
machines. That is deliberate. The reference implementations of 3D manipulation
policies generally need PyTorch3D or a custom CUDA extension, and for a lot of
people that is the reason they never try one.

* `unproject` -- depth image plus pinhole intrinsics to 3D points. Optionally
  transformed to a world frame by an extrinsic.
* `sample_points` -- a fixed-size subsample of the valid points.
* `normalize_points` -- map a workspace cube to roughly [-1, 1].

**On frames, which is the decision that matters.** `unproject` will give you
points in the camera's own frame or in a world frame, and the choice has
consequences that are easy to miss:

* **Camera frame** needs only intrinsics. Intrinsics come from the sensor and
  do not drift. The policy has to learn the camera-to-robot relation from data,
  which costs some sample efficiency but nothing else.
* **World frame** needs extrinsics, and inherits their error. A rotational
  calibration error of `eps` displaces every reconstructed point by about
  `(pi/4) * d * sin(eps)`, where `d` is the camera-to-workspace distance and the
  `pi/4` is the mean sine of the angle between a randomly-oriented error axis
  and the line of sight. At `d = 0.7 m` that is roughly 9 mm per degree. Two
  degrees of stale hand-eye calibration moves your point cloud by more than the
  width of many objects worth grasping.

So `frame="camera"` is the default -- but the honest form of that advice is a
threshold, not a preference, and it has been measured. On a keypose
manipulation task with a 20 mm object, comparing a camera-frame policy against
world-frame and canonicalised ones across a sweep of calibration error:

* at **0 degrees** of calibration error the world frame wins by a wide margin,
  3.6 mm against 18.7 mm -- geometry is worth a lot when you have it;
* the two cross **between 1 and 2 degrees**;
* at **5 degrees** the camera frame wins, 18.7 mm against 33.6 mm, and a
  canonicalised policy is by then worse than using no cameras at all.

The crossover is also predictable in advance from `(pi/4) * d * sin(eps)`
alone, which means you can decide before you build: measure your hand-eye
residual, and if it is below about a degree use `frame="world"`, otherwise stay
in `frame="camera"`. Multi-camera fusion needs `"world"` regardless, since
clouds from different cameras are otherwise in different frames.
"""

from __future__ import annotations

import torch
from torch import Tensor


def unproject(
    depth: Tensor,
    intrinsics: Tensor,
    extrinsics: Tensor | None = None,
) -> Tensor:
    """Depth images to 3D points.

    Args:
        depth: `(..., H, W)` z-depth in metres -- distance along the optical
            axis, which is what RealSense, Kinect and simulators all report. Not
            ray length.
        intrinsics: `(..., 3, 3)` pinhole matrix, with the same leading
            dimensions as `depth` minus the two spatial ones.
        extrinsics: optional `(..., 4, 4)` **camera-to-world** transform in the
            OpenCV convention (+z forward, +y down). If given, the returned
            points are in the world frame.

    Returns:
        `(..., H, W, 3)` points.

    Differentiable in every argument, including the extrinsics, which is what
    would let a policy refine its own calibration.
    """
    *lead, height, width = depth.shape
    rows, cols = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    cols = cols.expand(*lead, height, width)
    rows = rows.expand(*lead, height, width)

    fx = intrinsics[..., 0, 0][..., None, None]
    fy = intrinsics[..., 1, 1][..., None, None]
    cx = intrinsics[..., 0, 2][..., None, None]
    cy = intrinsics[..., 1, 2][..., None, None]
    points = torch.stack([(cols - cx) / fx * depth, (rows - cy) / fy * depth, depth], dim=-1)

    if extrinsics is not None:
        rotation = extrinsics[..., :3, :3][..., None, None, :, :]
        translation = extrinsics[..., :3, 3][..., None, None, :]
        points = (rotation @ points[..., None]).squeeze(-1) + translation
    return points


def sample_points(
    points: Tensor,
    valid: Tensor,
    num_points: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Take a fixed-size random subset of the valid points, per batch element.

    Args:
        points: `(B, N, C)`.
        valid: `(B, N)` bool. Depth sensors return zeros and holes; those points
            are at the origin and would drag any pooled statistic toward it.
        num_points: how many to keep.

    Returns:
        `(B, num_points, C)`.

    Uniform random rather than farthest-point sampling. FPS is what the original
    DP3 uses and it gives slightly better coverage, but it is O(N * num_points)
    on the GPU per frame and the measured difference is small; uniform keeps this
    dependency-free and fast enough to run in a dataloader. Swap it out if your
    scene has very uneven density.

    Sampling is with replacement out of the valid set, so the output is always
    `num_points` long and always valid, however few returns a frame had. Batch
    elements with no valid points at all come back as zeros rather than raising.
    """
    batch, n, _ = points.shape
    any_valid = valid.any(dim=1)

    # Rank valid points ahead of invalid ones by sorting a random key offset by
    # 1 for invalid points, so the valid ones occupy the prefix of the argsort.
    key = torch.rand(batch, n, device=points.device, generator=generator)
    key = key + (~valid).to(key.dtype)
    order = key.argsort(dim=1)

    # Then wrap the request into that prefix. Taking the first `num_points` of
    # the argsort directly would silently return INVALID points whenever a frame
    # has fewer valid returns than requested -- points at the sensor origin, or
    # outside the workspace crop -- and nothing downstream would notice. Wrapping
    # samples the valid set with replacement instead, which is the standard fix
    # and keeps the output shape fixed.
    n_valid = valid.sum(dim=1, keepdim=True).clamp(min=1)
    position = torch.arange(num_points, device=points.device).expand(batch, num_points)
    order = torch.gather(order, 1, position % n_valid)

    gathered = torch.gather(points, 1, order.unsqueeze(-1).expand(-1, -1, points.shape[-1]))
    # A frame with no valid returns at all yields zeros rather than raising: a
    # dropped depth frame should not kill a training run. Its gathered points are
    # invalid ones, which can be NaN or inf, so they are replaced rather than
    # multiplied by zero (NaN * 0 is NaN).
    return torch.where(any_valid[:, None, None], gathered, torch.zeros_like(gathered))


def normalize_points(points: Tensor, centre: Tensor, extent: float) -> Tensor:
    """Map a workspace cube to roughly [-1, 1] so the encoder sees sane scales.

    Point coordinates arrive in metres and a bare MLP does not like that; the
    LayerNorms absorb some of it but centring on the workspace is cheaper and
    makes the learned features transfer between setups with different origins.
    """
    return (points - centre) / (extent / 2.0)
