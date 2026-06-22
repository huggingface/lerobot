#!/usr/bin/env python

# Copyright 2026 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

The clutch (engage latch, no-teleport, hold while disengaged) now lives in the
owning loop (``examples/isaac_teleop_to_so101/teleoperate.py``); it is not part of
these processor steps. These tests cover only the LeRobot-side glue: the stateless
absolute-pose mapper, the static ``base_T_anchor`` config matrix, and the post-IK
wrist-roll overwrite (a standalone step still available for richer pipelines).
"""

import numpy as np
import pytest

from lerobot.teleoperators.isaac_teleop.config_isaac_teleop import (
    _DEFAULT_BASE_T_ANCHOR,
    XRControllerConfig,
)
from lerobot.teleoperators.isaac_teleop.wrist_roll_processor import OverwriteWristRollFromAngle
from lerobot.teleoperators.isaac_teleop.xr_controller_processor import (
    MapXRControllerActionToRobotAction,
)

# Identity quaternion (x, y, z, w).
_IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


def _make_action(pos, quat=_IDENTITY_QUAT, closedness=1.0):
    ee_pose = np.asarray([*pos, *quat], dtype=np.float32)
    return {
        "ee_pose": ee_pose,
        "closedness": float(closedness),
    }


def _ee(action):
    return np.array([action["ee.x"], action["ee.y"], action["ee.z"]])


def _ee_orientation(action):
    return np.array([action["ee.wx"], action["ee.wy"], action["ee.wz"]])


# ----------------------------------------------------------------------------
# MapXRControllerActionToRobotAction: stateless absolute-pose mapper
# ----------------------------------------------------------------------------


def test_mapper_writes_absolute_position_from_ee_pose():
    # The clutch already rebased the pose; the mapper passes ee_pose[:3] straight
    # through to ee.x/y/z (no delta, no per-frame state).
    step = MapXRControllerActionToRobotAction()
    out = step.action(_make_action([1.0, 2.0, 3.0]))
    assert np.allclose(_ee(out), [1.0, 2.0, 3.0])


def test_mapper_is_stateless_across_frames():
    # No engage-home / delta accumulation: each frame's ee.* depends only on that
    # frame's ee_pose (the clutch upstream owns all per-frame state).
    step = MapXRControllerActionToRobotAction()
    step.action(_make_action([5.0, 5.0, 5.0]))
    out = step.action(_make_action([1.0, 2.0, 3.0]))
    assert np.allclose(_ee(out), [1.0, 2.0, 3.0])


def test_mapper_identity_quat_is_zero_rotvec():
    # The identity grip quaternion maps to the zero rotvec orientation target.
    step = MapXRControllerActionToRobotAction()
    out = step.action(_make_action([1.0, 2.0, 3.0]))  # identity quat
    assert np.allclose(_ee_orientation(out), 0.0, atol=1e-12)


def test_mapper_orientation_is_quat_as_rotvec():
    # ee.w* is the rotvec of the ee_pose quaternion. A 90 deg rotation about z
    # (quat [0, 0, sin45, cos45]) maps to the rotvec [0, 0, pi/2].
    step = MapXRControllerActionToRobotAction()
    s, c = np.sin(np.pi / 4), np.cos(np.pi / 4)
    out = step.action(_make_action([0.0, 0.0, 0.0], quat=(0.0, 0.0, s, c)))
    assert np.allclose(_ee_orientation(out), [0.0, 0.0, np.pi / 2], atol=1e-6)


def test_mapper_emits_all_six_ee_components():
    # IK pops all six ee.* + ee.gripper_pos; missing any raises. The mapper must
    # emit the full set every frame.
    step = MapXRControllerActionToRobotAction()
    out = step.action(_make_action([0.0, 0.0, 0.0]))
    for key in ["ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz", "ee.gripper_pos"]:
        assert key in out


@pytest.mark.parametrize(
    ("closedness", "expected"),
    # Inverted polarity: c=0 (open) -> 100, c=1 (closed) -> 0 (SO-101 calibration).
    [(0.0, 100.0), (1.0, 0.0), (0.5, 50.0)],
)
def test_mapper_gripper_pos_inverts_closedness(closedness, expected):
    step = MapXRControllerActionToRobotAction()
    out = step.action(_make_action([0.0, 0.0, 0.0], closedness=closedness))
    assert out["ee.gripper_pos"] == pytest.approx(expected)


def test_mapper_consumes_input_keys():
    # ee_pose / closedness are popped; they must not leak downstream.
    step = MapXRControllerActionToRobotAction()
    out = step.action(_make_action([0.0, 0.0, 0.0]))
    for key in ["ee_pose", "closedness"]:
        assert key not in out


def test_mapper_transform_features_swaps_keys():
    step = MapXRControllerActionToRobotAction()
    from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature

    features = {
        PipelineFeatureType.ACTION: {
            "ee_pose": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
            "closedness": PolicyFeature(type=FeatureType.ACTION, shape=(1,)),
        }
    }
    out = step.transform_features(features)[PipelineFeatureType.ACTION]
    for popped in ["ee_pose", "closedness"]:
        assert popped not in out
    for added in ["ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz", "ee.gripper_pos"]:
        assert added in out


# ----------------------------------------------------------------------------
# base_T_anchor config matrix
# ----------------------------------------------------------------------------


def test_base_t_anchor_default_is_a_proper_rotation():
    # The default base_T_anchor is the OpenXR-anchor -> robot-base rebase. Its 3x3
    # block must be a proper rotation (det = +1, orthonormal); the homogeneous row
    # and zero translation make it a pure rotation.
    config = XRControllerConfig()
    mat = np.asarray(config.base_T_anchor, dtype=np.float32)
    assert mat.shape == (4, 4)

    rot = mat[:3, :3]
    assert np.isclose(np.linalg.det(rot), 1.0)
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-6)

    assert np.allclose(mat[3, :], [0.0, 0.0, 0.0, 1.0])
    assert np.allclose(mat[:3, 3], 0.0)


def test_base_t_anchor_default_matches_module_constant():
    # The config field defaults to the module-level constant (single source of truth).
    assert XRControllerConfig().base_T_anchor == _DEFAULT_BASE_T_ANCHOR


def test_base_t_anchor_axis_mapping():
    # Pin the INTENDED axis mapping of the default base_T_anchor so a future edit to
    # the matrix can't silently flip a sign. It rebases OpenXR anchor axes
    # (X=right, Y=up, Z=back/toward-user) into the robot base (X=forward, Y=left,
    # Z=up). This documents intent for hardware bring-up — confirm it matches the
    # physical setup during hardware bring-up.
    rot = np.asarray(XRControllerConfig().base_T_anchor, dtype=float)[:3, :3]
    # controller right (anchor +X) -> robot -Y (robot +Y is left)
    assert np.allclose(rot @ [1.0, 0.0, 0.0], [0.0, -1.0, 0.0])
    # controller up (anchor +Y) -> robot +Z (up)
    assert np.allclose(rot @ [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    # controller toward-user (anchor +Z) -> robot -X (robot +X is forward)
    assert np.allclose(rot @ [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0])


# ----------------------------------------------------------------------------
# OverwriteWristRollFromAngle: post-IK rad -> deg overwrite
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("wrist_roll_rad", "expected_deg"),
    [(np.pi, 180.0), (0.0, 0.0), (-np.pi / 2, -90.0)],
)
def test_overwrite_wrist_roll_from_angle(wrist_roll_rad, expected_deg):
    step = OverwriteWristRollFromAngle()
    out = step.action({"wrist_roll": float(wrist_roll_rad)})
    assert out["wrist_roll.pos"] == pytest.approx(expected_deg)
    assert "wrist_roll" not in out


def test_overwrite_wrist_roll_noop_when_absent():
    # No-op (leaves an existing wrist_roll.pos untouched) when no roll command is present.
    step = OverwriteWristRollFromAngle()
    out = step.action({"wrist_roll.pos": 12.0})
    assert out["wrist_roll.pos"] == pytest.approx(12.0)
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


# ---------------------------------------------------------------------------
# EEBoundsAndSafety.raise_on_jump — the XR pipeline relies on clamp+warn (not a
# crash) so a transient controller tracking glitch cannot abort the teleop loop.
# (Pure numpy; EEBoundsAndSafety does not require isaacteleop or placo to import.)
# ---------------------------------------------------------------------------


def _ee_action(x):
    return {"ee.x": float(x), "ee.y": 0.0, "ee.z": 0.0, "ee.wx": 0.0, "ee.wy": 0.0, "ee.wz": 0.0}


def test_ee_bounds_raises_on_jump_by_default():
    from lerobot.robots.so_follower.robot_kinematic_processor import EEBoundsAndSafety

    step = EEBoundsAndSafety(end_effector_bounds={"min": [-1, -1, -1], "max": [1, 1, 1]}, max_ee_step_m=0.1)
    step.action(_ee_action(0.0))
    with pytest.raises(ValueError, match="EE jump"):
        step.action(_ee_action(0.9))


def test_ee_bounds_clamps_instead_of_raising_when_disabled():
    from lerobot.robots.so_follower.robot_kinematic_processor import EEBoundsAndSafety

    step = EEBoundsAndSafety(
        end_effector_bounds={"min": [-1, -1, -1], "max": [1, 1, 1]},
        max_ee_step_m=0.1,
        raise_on_jump=False,
    )
    step.action(_ee_action(0.0))
    out = step.action(_ee_action(0.9))  # 0.9 jump -> clamped to the 0.1 per-frame limit, no raise
    assert out["ee.x"] == pytest.approx(0.1)
