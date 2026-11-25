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

"""Placo kinematics and gravity on a made-up robot: a two-link arm and a one-link arm on one base."""

import numpy as np
import pytest

pytest.importorskip("placo")

from lerobot.model import GravityCompensation, RobotKinematics  # noqa: E402


def _link(name, mass=1.0):
    return (
        f'<link name="{name}"><inertial><origin xyz="0.5 0 0"/><mass value="{mass}"/>'
        '<inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial></link>'
    )


def _joint(name, parent, child, xyz="0 0 0", axis="0 1 0", kind="revolute"):
    limit = '<limit lower="-3" upper="3" effort="10" velocity="10"/>' if kind == "revolute" else ""
    axis_tag = f'<axis xyz="{axis}"/>' if kind == "revolute" else ""
    return (
        f'<joint name="{name}" type="{kind}"><parent link="{parent}"/><child link="{child}"/>'
        f'<origin xyz="{xyz}"/>{axis_tag}{limit}</joint>'
    )


@pytest.fixture(scope="module")
def urdf(tmp_path_factory):
    """Arm "a" pitches in the x-z plane (joints a1, a2); arm "b" yaws in the x-y plane (joint b1)."""
    parts = [
        _link("base"),
        _link("a_link1"),
        _link("a_link2"),
        _link("a_tip", mass=0.001),
        _link("b_link1"),
        _link("b_tip", mass=0.001),
        _joint("a1", "base", "a_link1"),
        _joint("a2", "a_link1", "a_link2", xyz="1 0 0"),
        _joint("a_tip_joint", "a_link2", "a_tip", xyz="1 0 0", kind="fixed"),
        _joint("b1", "base", "b_link1", xyz="0 1 0", axis="0 0 1"),
        _joint("b_tip_joint", "b_link1", "b_tip", xyz="1 0 0", kind="fixed"),
    ]
    path = tmp_path_factory.mktemp("urdf") / "robot.urdf"
    path.write_text(f'<robot name="test">{"".join(parts)}</robot>')
    return path


def test_two_frames_are_solved_together(urdf):
    kinematics = RobotKinematics(urdf, ("a_tip", "b_tip"), ["a1", "a2", "b1"])
    goal = np.array([20.0, 40.0, 30.0])
    targets = {frame: kinematics.forward_kinematics(goal, frame) for frame in ("a_tip", "b_tip")}
    solution = goal
    for _ in range(3):
        solution = kinematics.inverse_kinematics(np.zeros(3), targets, orientation_weight=0.0, max_iters=20)
    for frame, target in targets.items():
        reached = kinematics.forward_kinematics(solution, frame)
        np.testing.assert_allclose(reached[:3, 3], target[:3, 3], atol=1e-3)


def test_joints_left_out_keep_their_position(urdf):
    kinematics = RobotKinematics(urdf, ("a_tip", "b_tip"), ["a1", "b1"])
    targets = {"a_tip": np.eye(4), "b_tip": np.eye(4)}
    targets["a_tip"][:3, 3] = [1.0, 0.0, -1.0]  # reachable only by bending a2
    kinematics.inverse_kinematics(np.zeros(2), targets)
    assert kinematics.robot.get_joint("a2") == 0.0


def test_single_frame_api_is_unchanged_and_checks_its_targets(urdf):
    kinematics = RobotKinematics(urdf, "b_tip", ["b1"])
    target = kinematics.forward_kinematics(np.array([45.0]))
    np.testing.assert_allclose(target[:3, 3], [np.sqrt(0.5), 1 + np.sqrt(0.5), 0], atol=1e-9)
    # A trailing gripper value passes through, as for the SO arms.
    solution = kinematics.inverse_kinematics(np.array([0.0, 7.0]), target)
    assert solution == pytest.approx([45.0, 7.0], abs=0.5)
    with pytest.raises(ValueError, match="Expected target poses"):
        kinematics.inverse_kinematics(np.zeros(1), {"a_tip": target})


def test_posture_weight_picks_the_solution_closest_to_zero(urdf):
    # a_tip at [1, 0, -1] is reached with a1 = 0 and a2 = 90 deg, or with a1 = 90 deg and a2 = -90 deg.
    target = np.eye(4)
    target[:3, 3] = [1.0, 0.0, -1.0]
    plain = RobotKinematics(urdf, "a_tip", ["a1", "a2"])
    posture = RobotKinematics(urdf, "a_tip", ["a1", "a2"], posture_weight=1e-3)
    start = np.array([80.0, -80.0])
    for _ in range(20):
        start_plain = plain.inverse_kinematics(start, target, orientation_weight=0.0)
        start = posture.inverse_kinematics(start, target, orientation_weight=0.0)
    assert np.linalg.norm(start) < np.linalg.norm(start_plain)


def test_gravity_torques_are_scaled_and_capped(urdf):
    plain = GravityCompensation(urdf, ["b1", "a1"], base_frame="base")
    torques = plain.torques(np.zeros(2), {"a2": 0.0})
    assert torques[0] == pytest.approx(0.0, abs=1e-9)  # b1 turns about the vertical axis
    # a1 holds two 1 kg links whose centres sit 0.5 m and 1.5 m out, and a 1 g tip 2.5 m out.
    assert abs(torques[1]) == pytest.approx((0.5 + 1.5 + 0.001 * 2.5) * 9.81, rel=1e-3)
    scaled = GravityCompensation(urdf, ["b1", "a1"], base_frame="base", scale=[1.0, 0.5], max_torque=5.0)
    np.testing.assert_allclose(scaled.torques(np.zeros(2)), [0.0, np.sign(torques[1]) * 5.0], atol=1e-9)
    with pytest.raises(ValueError, match="one value per joint"):
        GravityCompensation(urdf, ["a1"], base_frame="base", scale=[1.0, 2.0])
