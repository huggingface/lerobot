#!/usr/bin/env python

# Copyright 2024 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

"""Pure-numpy/scipy unit tests for ``MapXRControllerActionToRobotAction``.

These tests deliberately do NOT import ``isaacteleop`` — the processor is the
pure-math bridge between the XR teleoperator and the closed-loop IK pipeline,
and must be testable without the XR runtime.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lerobot.teleoperators.isaac_teleop.xr_controller_processor import (
    _OPENXR_TO_ROBOT,
    MapXRControllerActionToRobotAction,
    _remap_openxr_to_robot,
)

# Identity quaternion (x, y, z, w).
_IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _make_action(pos, quat=_IDENTITY_QUAT, gripper=1.0, enabled=True):
    return {
        "ee_pos": np.asarray(pos, dtype=np.float32),
        "ee_quat": np.asarray(quat, dtype=np.float32),
        "gripper": float(gripper),
        "enabled": bool(enabled),
    }


def _targets(action):
    return np.array(
        [
            action["target_x"],
            action["target_y"],
            action["target_z"],
            action["target_wx"],
            action["target_wy"],
            action["target_wz"],
        ]
    )


def test_openxr_to_robot_is_proper_rotation():
    # det = +1 and orthonormal -> a proper rotation (no reflection).
    assert np.isclose(np.linalg.det(_OPENXR_TO_ROBOT), 1.0)
    assert np.allclose(_OPENXR_TO_ROBOT @ _OPENXR_TO_ROBOT.T, np.eye(3))


def test_quaternion_change_of_basis_round_trips():
    # A rotation expressed in the robot frame, mapped back to OpenXR and forward
    # again, must be unchanged.
    rng = np.random.default_rng(0)
    quat = Rotation.from_rotvec(rng.normal(size=3)).as_quat().astype(np.float32)
    pos = rng.normal(size=3).astype(np.float32)

    pos_r, quat_r = _remap_openxr_to_robot(pos, quat)
    # Map the robot-frame rotation back to OpenXR via the inverse (transpose).
    r_back = Rotation.from_matrix(
        _OPENXR_TO_ROBOT.T @ Rotation.from_quat(quat_r).as_matrix() @ _OPENXR_TO_ROBOT
    )
    assert np.allclose(r_back.as_matrix(), Rotation.from_quat(quat).as_matrix(), atol=1e-5)
    assert np.allclose(_OPENXR_TO_ROBOT.T @ pos_r, pos, atol=1e-5)


def test_engage_frame_has_zero_delta():
    step = MapXRControllerActionToRobotAction()
    out = step.action(_make_action([1.0, 2.0, 3.0], enabled=True))
    assert out["enabled"] is True
    assert np.allclose(_targets(out), 0.0, atol=1e-6)


def test_delta_accumulates_after_engage():
    step = MapXRControllerActionToRobotAction()
    step.action(_make_action([0.0, 0.0, 0.0], enabled=True))  # captures origin
    out = step.action(_make_action([1.0, 0.0, 0.0], enabled=True))

    # OpenXR (1,0,0) -> robot frame via _OPENXR_TO_ROBOT.
    expected = _OPENXR_TO_ROBOT @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(_targets(out)[:3], expected, atol=1e-5)
    assert not np.allclose(_targets(out)[:3], 0.0)


def test_engage_frame_zero_rotation_with_nonidentity_orientation():
    # Even with a non-identity controller orientation, the engage frame captures
    # that orientation as the origin so the emitted rotvec delta is exactly zero
    # (the arm must not jump in orientation on engage).
    step = MapXRControllerActionToRobotAction()
    quat = Rotation.from_euler("xyz", [30, -45, 60], degrees=True).as_quat().astype(np.float32)
    out = step.action(_make_action([0.5, -0.2, 1.0], quat=quat, enabled=True))
    assert np.allclose(_targets(out)[3:], 0.0, atol=1e-6)


def test_orientation_delta_matches_origin_rot_inv_times_rot():
    # After engage, the emitted rotvec equals (origin_rot_inv * rot) computed in
    # the ROBOT frame (the processor remaps the quaternion before accumulating).
    step = MapXRControllerActionToRobotAction()
    quat0 = Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_quat().astype(np.float32)
    quat1 = Rotation.from_euler("xyz", [40, -15, 5], degrees=True).as_quat().astype(np.float32)

    step.action(_make_action([0.0, 0.0, 0.0], quat=quat0, enabled=True))  # captures origin
    out = step.action(_make_action([0.0, 0.0, 0.0], quat=quat1, enabled=True))

    # Reproduce the expected delta in the robot frame using the same remap.
    _, quat0_r = _remap_openxr_to_robot(np.zeros(3, dtype=np.float32), quat0)
    _, quat1_r = _remap_openxr_to_robot(np.zeros(3, dtype=np.float32), quat1)
    expected = (Rotation.from_quat(quat0_r).inv() * Rotation.from_quat(quat1_r)).as_rotvec()

    assert np.allclose(_targets(out)[3:], expected, atol=1e-5)
    # And it is a genuinely non-zero rotation, so the channel is exercised.
    assert not np.allclose(_targets(out)[3:], 0.0)


def test_disabled_zeroes_targets():
    step = MapXRControllerActionToRobotAction()
    step.action(_make_action([0.0, 0.0, 0.0], enabled=True))
    out = step.action(_make_action([5.0, 5.0, 5.0], enabled=False))
    assert out["enabled"] is False
    assert np.allclose(_targets(out), 0.0, atol=1e-6)


def test_rising_edge_recaptures_origin():
    step = MapXRControllerActionToRobotAction()
    # First engage at origin A.
    step.action(_make_action([0.0, 0.0, 0.0], enabled=True))
    step.action(_make_action([1.0, 0.0, 0.0], enabled=True))
    # Disengage, controller drifts away.
    step.action(_make_action([10.0, 10.0, 10.0], enabled=False))
    # Re-engage at a brand new pose -> this is a rising edge -> zero delta.
    out = step.action(_make_action([10.0, 10.0, 10.0], enabled=True))
    assert np.allclose(_targets(out), 0.0, atol=1e-6)


def test_reset_clears_origin_and_edge():
    step = MapXRControllerActionToRobotAction()
    step.action(_make_action([0.0, 0.0, 0.0], enabled=True))
    step.action(_make_action([2.0, 0.0, 0.0], enabled=True))

    step.reset()
    assert step._origin_pos is None
    assert step._origin_rot_inv is None
    assert step._prev_enabled is False

    # The first engaged frame after reset is a true rising edge -> zero delta,
    # no stale origin carried across the episode boundary.
    out = step.action(_make_action([7.0, 8.0, 9.0], enabled=True))
    assert np.allclose(_targets(out), 0.0, atol=1e-6)


@pytest.mark.parametrize(
    ("gripper", "expected"),
    [(1.0, 2.0), (-1.0, 0.0), (0.0, 2.0)],
)
def test_gripper_state_maps_to_discrete_index(gripper, expected):
    # +1 (open) -> 2, -1 (closed) -> 0, with NO pre-negation (the downstream
    # discrete_gripper mode encodes the SO100 sign).
    step = MapXRControllerActionToRobotAction()
    out = step.action(_make_action([0.0, 0.0, 0.0], gripper=gripper, enabled=True))
    assert out["gripper_vel"] == expected


def test_transform_features_swaps_keys():
    step = MapXRControllerActionToRobotAction()
    from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature

    features = {
        PipelineFeatureType.ACTION: {
            "ee_pos": PolicyFeature(type=FeatureType.ACTION, shape=(3,)),
            "ee_quat": PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
            "gripper": PolicyFeature(type=FeatureType.ACTION, shape=(1,)),
            "enabled": PolicyFeature(type=FeatureType.ACTION, shape=(1,)),
        }
    }
    out = step.transform_features(features)[PipelineFeatureType.ACTION]
    for popped in ["ee_pos", "ee_quat", "gripper"]:
        assert popped not in out
    for added in ["target_x", "target_y", "target_z", "target_wx", "target_wy", "target_wz", "gripper_vel"]:
        assert added in out
