# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Focused regression tests for rotation-vector interpolation."""

import pytest
import torch

from lerobot.utils.action_interpolator import ActionInterpolator


def _rotvec_to_quaternion(rotvec: torch.Tensor) -> torch.Tensor:
    """Independent test helper returning an [x, y, z, w] quaternion."""
    angle = torch.linalg.vector_norm(rotvec)
    scale = torch.where(
        angle > 1e-12,
        torch.sin(angle / 2) / angle.clamp_min(1e-12),
        torch.tensor(0.5, dtype=rotvec.dtype),
    )
    return torch.cat((rotvec * scale, torch.cos(angle / 2).reshape(1)))


def _rotation_distance(start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
    """Return the SO(3) geodesic distance between two rotation vectors."""
    start_quat = _rotvec_to_quaternion(start)
    end_quat = _rotvec_to_quaternion(end)
    dot = torch.sum(start_quat * end_quat).abs().clamp(max=1.0)
    return 2 * torch.arccos(dot)


def test_slerp_follows_geodesic_for_issue_rotvecs():
    """Each intermediate orientation follows the shortest SO(3) path from issue #3691."""
    previous = torch.tensor([0.1, 0.2, 0.3, 1.45, -0.14, 2.34, 0.5], dtype=torch.float64)
    current = torch.tensor([0.2, 0.3, 0.4, -1.72, -0.38, -2.31, 0.6], dtype=torch.float64)
    interpolator = ActionInterpolator(multiplier=5, rotation_dims=[3, 4, 5])

    interpolator.add(previous)
    interpolator.get()
    interpolator.add(current)

    endpoint_distance = _rotation_distance(previous[3:6], current[3:6])
    for step in range(1, 6):
        action = interpolator.get()
        assert action is not None
        actual_distance = _rotation_distance(previous[3:6], action[3:6])
        torch.testing.assert_close(actual_distance, endpoint_distance * (step / 5), atol=1e-6, rtol=1e-6)


def test_slerp_preserves_linear_dimensions_and_verbatim_policy_endpoint():
    """Only rotvec triplets use SLERP and the cycle ends with the exact policy tensor."""
    previous = torch.tensor([0.0, 0.0, 0.0, 1.45, -0.14, 2.34, 0.0])
    current = torch.tensor([1.0, 2.0, 3.0, -1.72, -0.38, -2.31, 1.0])
    interpolator = ActionInterpolator(multiplier=2, rotation_dims=[3, 4, 5])

    interpolator.add(previous)
    interpolator.get()
    interpolator.add(current)

    midpoint = interpolator.get()
    endpoint = interpolator.get()

    assert midpoint is not None and endpoint is not None
    torch.testing.assert_close(midpoint[:3], torch.tensor([0.5, 1.0, 1.5]))
    torch.testing.assert_close(midpoint[6:], torch.tensor([0.5]))
    assert torch.equal(endpoint, current)
    assert interpolator.emitted_policy_action


def test_slerp_supports_bimanual_rotation_triplets():
    """Two configured rotvec triplets interpolate independently."""
    previous = torch.tensor([1.45, -0.14, 2.34, 0.0, 0.0, 0.2], dtype=torch.float64)
    current = torch.tensor([-1.72, -0.38, -2.31, 0.2, -0.1, 0.5], dtype=torch.float64)
    interpolator = ActionInterpolator(multiplier=2, rotation_dims=[0, 1, 2, 3, 4, 5])

    interpolator.add(previous)
    interpolator.get()
    interpolator.add(current)
    midpoint = interpolator.get()

    assert midpoint is not None
    for start in (0, 3):
        total = _rotation_distance(previous[start : start + 3], current[start : start + 3])
        halfway = _rotation_distance(previous[start : start + 3], midpoint[start : start + 3])
        torch.testing.assert_close(halfway, total / 2, atol=1e-6, rtol=1e-6)


def test_slerp_handles_identity_rotation_without_nan():
    """A zero rotation vector remains numerically stable."""
    interpolator = ActionInterpolator(multiplier=2, rotation_dims=[0, 1, 2])
    interpolator.add(torch.zeros(3))
    interpolator.get()
    interpolator.add(torch.tensor([0.1, -0.2, 0.3]))

    assert torch.isfinite(interpolator.get()).all()


def test_rotation_dims_requires_complete_triplets():
    """The public configuration rejects incomplete rotation-vector groups."""
    with pytest.raises(ValueError, match="triplets"):
        ActionInterpolator(multiplier=2, rotation_dims=[3, 4])


@pytest.mark.parametrize("rotation_dims", [[-1, 0, 1], [5, 6, 7]])
def test_rotation_dims_rejects_out_of_range_indices(rotation_dims):
    """Invalid action dimensions fail clearly on the first action."""
    interpolator = ActionInterpolator(multiplier=2, rotation_dims=rotation_dims)

    with pytest.raises(ValueError, match="out of range"):
        interpolator.add(torch.zeros(7))
