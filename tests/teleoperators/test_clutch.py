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

"""Pure-numpy unit tests for the Isaac Teleop XR clutch.

Like ``test_xr_controller_processor``, these tests import the clutch from its
submodule (``...isaac_teleop.clutch``) rather than the package top-level, so they
collect and run WITHOUT ``isaacteleop`` / the XR runtime installed.

The clutch is the pure-math heart of the XR -> SO-101 teleop loop: latch an origin
on engage, then track the base-frame delta of the controller onto the EE. The
invariants that matter are (1) engaging produces NO teleport in position or
orientation, (2) the delta is 1:1 in position and base-frame left-composed in
orientation, and (3) a mid-task re-clutch resumes from the last commanded pose.
"""

import numpy as np

from lerobot.teleoperators.isaac_teleop.clutch import Clutch
from lerobot.utils.rotation import Rotation

# Identity quaternion (x, y, z, w).
_IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0])


def _pose_matrix(pos, rotvec=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Build a 4x4 homogeneous base_T_ee from a position and a rotation vector."""
    mat = np.eye(4)
    mat[:3, :3] = Rotation.from_rotvec(np.asarray(rotvec, dtype=float)).as_matrix()
    mat[:3, 3] = np.asarray(pos, dtype=float)
    return mat


# ----------------------------------------------------------------------------
# Seeding: the constructor latches home from the measured startup EE pose.
# ----------------------------------------------------------------------------


def test_engage_frame_returns_seeded_home_position():
    # Engaging at any controller pose and immediately rebasing at that SAME pose
    # returns exactly the seeded home — no teleport on the first squeeze.
    clutch = Clutch(_pose_matrix([0.1, 0.2, 0.3]))
    grip = np.array([5.0, -3.0, 2.0])
    clutch.engage(grip, _IDENTITY_QUAT)
    pos, _ = clutch.rebase(grip, _IDENTITY_QUAT)
    assert np.allclose(pos, [0.1, 0.2, 0.3])


def test_engage_frame_returns_seeded_home_orientation():
    # Same no-teleport guarantee for orientation: the engage-frame output rotation
    # equals the seeded home rotation regardless of the controller origin orientation.
    home_rotvec = [0.0, np.pi / 3, 0.0]
    clutch = Clutch(_pose_matrix([0.0, 0.0, 0.0], rotvec=home_rotvec))
    grip_quat = Rotation.from_rotvec([np.pi / 5, 0.0, 0.0]).as_quat()
    clutch.engage(np.zeros(3), grip_quat)
    _, out_quat = clutch.rebase(np.zeros(3), grip_quat)
    expected = Rotation.from_rotvec(home_rotvec).as_matrix()
    assert np.allclose(Rotation.from_quat(out_quat).as_matrix(), expected, atol=1e-6)


# ----------------------------------------------------------------------------
# Position: the controller -> EE translation is 1:1 from the engage origin.
# ----------------------------------------------------------------------------


def test_position_delta_is_one_to_one():
    clutch = Clutch(_pose_matrix([1.0, 1.0, 1.0]))
    clutch.engage(np.array([0.0, 0.0, 0.0]), _IDENTITY_QUAT)
    # Move the controller +0.3 in x, -0.2 in y from the origin.
    pos, _ = clutch.rebase(np.array([0.3, -0.2, 0.0]), _IDENTITY_QUAT)
    assert np.allclose(pos, [1.3, 0.8, 1.0])


def test_position_is_relative_to_origin_not_absolute():
    # The delta is measured from the engage origin, so a nonzero origin does not
    # leak into the output — only the change from it does.
    clutch = Clutch(_pose_matrix([0.0, 0.0, 0.0]))
    clutch.engage(np.array([10.0, 10.0, 10.0]), _IDENTITY_QUAT)
    pos, _ = clutch.rebase(np.array([10.5, 10.0, 10.0]), _IDENTITY_QUAT)
    assert np.allclose(pos, [0.5, 0.0, 0.0])


# ----------------------------------------------------------------------------
# Orientation: base-frame (left-composed) delta.
# ----------------------------------------------------------------------------


def test_orientation_delta_is_base_frame_left_composed():
    # Home orientation = 90 deg about base X. Rotate the controller 90 deg about
    # base Z from the (identity) origin. The clutch left-composes the delta:
    #   R_out = R_ctrl @ R_home = Rz(90) @ Rx(90)  (NOT the body-frame Rx @ Rz).
    home_rotvec = [np.pi / 2, 0.0, 0.0]
    clutch = Clutch(_pose_matrix([0.0, 0.0, 0.0], rotvec=home_rotvec))
    clutch.engage(np.zeros(3), _IDENTITY_QUAT)
    ctrl_quat = Rotation.from_rotvec([0.0, 0.0, np.pi / 2]).as_quat()
    _, out_quat = clutch.rebase(np.zeros(3), ctrl_quat)

    r_z = Rotation.from_rotvec([0.0, 0.0, np.pi / 2])
    r_x = Rotation.from_rotvec(home_rotvec)
    expected = (r_z * r_x).as_matrix()
    assert np.allclose(Rotation.from_quat(out_quat).as_matrix(), expected, atol=1e-6)


def test_orientation_is_relative_to_origin_orientation():
    # A nonzero controller origin orientation cancels out on the engage frame:
    # rebasing at the origin orientation returns the home orientation unchanged.
    clutch = Clutch(_pose_matrix([0.0, 0.0, 0.0]))
    origin_quat = Rotation.from_rotvec([0.0, 0.0, np.pi / 4]).as_quat()
    clutch.engage(np.zeros(3), origin_quat)
    _, out_quat = clutch.rebase(np.zeros(3), origin_quat)
    assert np.allclose(Rotation.from_quat(out_quat).as_matrix(), np.eye(3), atol=1e-6)


# ----------------------------------------------------------------------------
# Re-clutch: a fresh engage resumes from the LAST COMMANDED pose (no jump when
# the controller has moved away while disengaged).
# ----------------------------------------------------------------------------


def test_reclutch_resumes_from_last_commanded_position():
    clutch = Clutch(_pose_matrix([0.0, 0.0, 0.0]))
    # First engagement: drive the EE to x=1.0, then "release" (stop calling rebase).
    clutch.engage(np.array([0.0, 0.0, 0.0]), _IDENTITY_QUAT)
    clutch.rebase(np.array([1.0, 0.0, 0.0]), _IDENTITY_QUAT)

    # Re-engage with the controller somewhere else entirely. The new home latches to
    # the last commanded pose (x=1.0), so the first rebase does NOT jump to the
    # controller's absolute position.
    clutch.engage(np.array([5.0, 5.0, 5.0]), _IDENTITY_QUAT)
    pos, _ = clutch.rebase(np.array([5.0, 5.0, 5.0]), _IDENTITY_QUAT)
    assert np.allclose(pos, [1.0, 0.0, 0.0])
    # And it tracks the new delta from there.
    pos, _ = clutch.rebase(np.array([6.0, 5.0, 5.0]), _IDENTITY_QUAT)
    assert np.allclose(pos, [2.0, 0.0, 0.0])


def test_reclutch_resumes_from_last_commanded_orientation():
    clutch = Clutch(_pose_matrix([0.0, 0.0, 0.0]))
    # First engagement: rotate the EE 90 deg about base Z, then release.
    clutch.engage(np.zeros(3), _IDENTITY_QUAT)
    ctrl_quat = Rotation.from_rotvec([0.0, 0.0, np.pi / 2]).as_quat()
    clutch.rebase(np.zeros(3), ctrl_quat)

    # Re-engage with the controller at a different orientation. The first rebase
    # returns the last commanded orientation (Rz(90)), not a jump.
    reengage_quat = Rotation.from_rotvec([np.pi / 3, 0.0, 0.0]).as_quat()
    clutch.engage(np.zeros(3), reengage_quat)
    _, out_quat = clutch.rebase(np.zeros(3), reengage_quat)
    expected = Rotation.from_rotvec([0.0, 0.0, np.pi / 2]).as_matrix()
    assert np.allclose(Rotation.from_quat(out_quat).as_matrix(), expected, atol=1e-6)


def test_disengaged_gap_does_not_change_commanded_pose():
    # rebase() is the only thing that advances the last-commanded pose. Between a
    # release and a re-engage the loop calls neither engage() nor rebase(), so the
    # held pose is exactly where the last rebase left it — re-engaging and rebasing
    # at the new origin reproduces that pose.
    clutch = Clutch(_pose_matrix([0.0, 0.0, 0.0]))
    clutch.engage(np.zeros(3), _IDENTITY_QUAT)
    pos_a, quat_a = clutch.rebase(np.array([0.4, 0.1, -0.2]), _IDENTITY_QUAT)

    clutch.engage(np.array([9.0, 9.0, 9.0]), _IDENTITY_QUAT)
    pos_b, quat_b = clutch.rebase(np.array([9.0, 9.0, 9.0]), _IDENTITY_QUAT)
    assert np.allclose(pos_a, pos_b)
    assert np.allclose(
        Rotation.from_quat(quat_a).as_matrix(), Rotation.from_quat(quat_b).as_matrix(), atol=1e-6
    )
