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

"""Pure-numpy unit tests for the Isaac Teleop XR -> SO-101 processor steps.

These tests deliberately do NOT import ``isaacteleop`` — the processor steps are
the pure-math bridge between the XR teleoperator and the closed-loop IK pipeline,
and must be testable without the XR runtime.
"""

import numpy as np
import pytest

from lerobot.teleoperators.isaac_teleop.wrist_roll_processor import OverwriteWristRollFromAngle
from lerobot.teleoperators.isaac_teleop.xr_controller_processor import (
    _OPENXR_TO_ROBOT,
    MapXRControllerActionToRobotAction,
)

# Identity quaternion (x, y, z, w).
_IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


def _make_action(pos, quat=_IDENTITY_QUAT, closedness=1.0, wrist_roll=0.0, enabled=True):
    ee_pose = np.asarray([*pos, *quat], dtype=np.float32)
    return {
        "ee_pose": ee_pose,
        "wrist_roll": float(wrist_roll),
        "closedness": float(closedness),
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


def test_engage_frame_has_zero_delta():
    # On the rising edge of enabled the engage-home is latched, so the first
    # engaged frame emits a zero delta regardless of where the controller is.
    step = MapXRControllerActionToRobotAction()
    out = step.action(_make_action([1.0, 2.0, 3.0], enabled=True))
    assert out["enabled"] is True
    assert np.allclose(_targets(out), 0.0, atol=1e-6)


def test_delta_accumulates_after_engage():
    step = MapXRControllerActionToRobotAction()
    step.action(_make_action([0.0, 0.0, 0.0], enabled=True))  # latches engage-home
    out = step.action(_make_action([1.0, 0.0, 0.0], enabled=True))

    # Base-frame delta = _OPENXR_TO_ROBOT @ (ee_pose[:3] - engage_home).
    expected = _OPENXR_TO_ROBOT @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(_targets(out)[:3], expected, atol=1e-5)
    assert not np.allclose(_targets(out)[:3], 0.0)


def test_orientation_delta_is_always_zero():
    # The orientation channel is intentionally unused on the 5-DOF position-only
    # arm; target_w* must be zero even with a non-identity controller orientation.
    step = MapXRControllerActionToRobotAction()
    quat = (0.1, 0.2, 0.3, 0.927)  # arbitrary non-identity
    step.action(_make_action([0.0, 0.0, 0.0], quat=quat, enabled=True))
    out = step.action(_make_action([1.0, 2.0, 3.0], quat=quat, enabled=True))
    assert np.allclose(_targets(out)[3:], 0.0, atol=1e-12)


def test_disabled_zeroes_targets():
    step = MapXRControllerActionToRobotAction()
    step.action(_make_action([0.0, 0.0, 0.0], enabled=True))
    out = step.action(_make_action([5.0, 5.0, 5.0], enabled=False))
    assert out["enabled"] is False
    assert np.allclose(_targets(out), 0.0, atol=1e-6)


def test_rising_edge_recaptures_engage_home():
    step = MapXRControllerActionToRobotAction()
    # First engage at home A.
    step.action(_make_action([0.0, 0.0, 0.0], enabled=True))
    step.action(_make_action([1.0, 0.0, 0.0], enabled=True))
    # Disengage, controller drifts away.
    step.action(_make_action([10.0, 10.0, 10.0], enabled=False))
    # Re-engage at a brand new pose -> rising edge -> zero delta.
    out = step.action(_make_action([10.0, 10.0, 10.0], enabled=True))
    assert np.allclose(_targets(out), 0.0, atol=1e-6)


def test_reset_clears_engage_home_and_edge():
    step = MapXRControllerActionToRobotAction()
    step.action(_make_action([0.0, 0.0, 0.0], enabled=True))
    step.action(_make_action([2.0, 0.0, 0.0], enabled=True))

    step.reset()
    assert step._engage_home is None
    assert step._prev_enabled is False

    # The first engaged frame after reset is a true rising edge -> zero delta,
    # no stale engage-home carried across the episode boundary.
    out = step.action(_make_action([7.0, 8.0, 9.0], enabled=True))
    assert np.allclose(_targets(out), 0.0, atol=1e-6)


@pytest.mark.parametrize(
    ("closedness", "expected"),
    [(0.0, 0.0), (1.0, 100.0), (0.5, 50.0)],
)
def test_gripper_pos_is_closedness_times_100(closedness, expected):
    step = MapXRControllerActionToRobotAction()
    out = step.action(_make_action([0.0, 0.0, 0.0], closedness=closedness, enabled=True))
    assert out["ee.gripper_pos"] == pytest.approx(expected)


def test_wrist_roll_passes_through_and_gripper_vel_zero():
    step = MapXRControllerActionToRobotAction()
    out = step.action(_make_action([0.0, 0.0, 0.0], wrist_roll=0.7, enabled=True))
    assert out["wrist_roll"] == pytest.approx(0.7)
    assert out["gripper_vel"] == 0.0


def test_wrist_roll_held_when_disabled_after_engage():
    # The clutch gates the whole arm pose: on release the wrist_roll holds the
    # last commanded value (not the live controller roll), so the arm freezes.
    step = MapXRControllerActionToRobotAction()
    step.action(_make_action([0.0, 0.0, 0.0], wrist_roll=0.5, enabled=True))  # commands 0.5
    out = step.action(_make_action([5.0, 5.0, 5.0], wrist_roll=0.9, enabled=False))
    assert out["wrist_roll"] == pytest.approx(0.5)  # held, not 0.9
    assert np.allclose(_targets(out), 0.0, atol=1e-6)


def test_wrist_roll_absent_before_first_engage():
    # Before any engage there is no commanded roll to hold, so the bridge emits no
    # wrist_roll key and the post-IK overwrite leaves the IK-produced joint alone.
    step = MapXRControllerActionToRobotAction()
    out = step.action(_make_action([0.0, 0.0, 0.0], wrist_roll=0.9, enabled=False))
    assert "wrist_roll" not in out


def test_overwrite_wrist_roll_noop_when_absent():
    # No-op (leaves an existing wrist_roll.pos untouched) when no roll command is present.
    step = OverwriteWristRollFromAngle()
    out = step.action({"wrist_roll.pos": 12.0})
    assert out["wrist_roll.pos"] == pytest.approx(12.0)
    assert "wrist_roll" not in out


def test_transform_features_swaps_keys():
    step = MapXRControllerActionToRobotAction()
    from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature

    features = {
        PipelineFeatureType.ACTION: {
            "ee_pose": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
            "wrist_roll": PolicyFeature(type=FeatureType.ACTION, shape=(1,)),
            "closedness": PolicyFeature(type=FeatureType.ACTION, shape=(1,)),
            "enabled": PolicyFeature(type=FeatureType.ACTION, shape=(1,)),
        }
    }
    out = step.transform_features(features)[PipelineFeatureType.ACTION]
    for popped in ["ee_pose", "closedness"]:
        assert popped not in out
    for added in [
        "enabled",
        "target_x",
        "target_y",
        "target_z",
        "target_wx",
        "target_wy",
        "target_wz",
        "gripper_vel",
        "wrist_roll",
        "ee.gripper_pos",
    ]:
        assert added in out


@pytest.mark.parametrize(
    ("wrist_roll_rad", "expected_deg"),
    [(np.pi, 180.0), (0.0, 0.0), (-np.pi / 2, -90.0)],
)
def test_overwrite_wrist_roll_from_angle(wrist_roll_rad, expected_deg):
    step = OverwriteWristRollFromAngle()
    out = step.action({"wrist_roll": float(wrist_roll_rad)})
    assert out["wrist_roll.pos"] == pytest.approx(expected_deg)
    assert "wrist_roll" not in out


def test_overwrite_wrist_roll_transform_features():
    step = OverwriteWristRollFromAngle()
    from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature

    features = {
        PipelineFeatureType.ACTION: {
            "wrist_roll": PolicyFeature(type=FeatureType.ACTION, shape=(1,)),
        }
    }
    out = step.transform_features(features)[PipelineFeatureType.ACTION]
    assert "wrist_roll" not in out
    assert "wrist_roll.pos" in out
