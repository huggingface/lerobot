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

"""G1 arm IK and gravity wiring, without downloading the G1 models."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from lerobot.envs.configs import G1EndEffector
from lerobot.robots.unitree_g1 import g1_kinematics
from lerobot.robots.unitree_g1.g1_utils import G1_29_JointArmIndex


@pytest.fixture
def assets(monkeypatch):
    monkeypatch.setattr(g1_kinematics, "g1_assets_dir", lambda: Path("/g1/assets"))


def test_arm_joints_follow_the_g1_motor_order():
    assert len(g1_kinematics.G1_29_ARM_JOINTS) == len(G1_29_JointArmIndex)
    assert g1_kinematics.G1_29_ARM_JOINTS[0] == "left_shoulder_pitch_joint"
    assert g1_kinematics.G1_29_ARM_JOINTS[10] == "right_elbow_joint"


@pytest.mark.parametrize(
    ("end_effector", "urdf"),
    [(G1EndEffector.DEX1, "g1_29dof_with_dex1_1.urdf"), (G1EndEffector.DEX3, "g1_body29_hand14.urdf")],
)
def test_gravity_model_matches_the_end_effector(assets, monkeypatch, end_effector, urdf):
    model = MagicMock()
    monkeypatch.setattr(g1_kinematics, "GravityCompensation", model)
    g1_kinematics.make_g1_gravity_compensation(end_effector)
    model.assert_called_once_with(
        Path("/g1/assets") / urdf, g1_kinematics.G1_29_ARM_JOINTS, base_frame="pelvis"
    )


def test_gravity_model_needs_an_end_effector_model(assets):
    with pytest.raises(ValueError, match="no G1 model"):
        g1_kinematics.make_g1_gravity_compensation(G1EndEffector.DUMMY)


def test_smoothing_weights_the_newest_solution_most():
    smoothing = g1_kinematics.WeightedMovingFilter([0.4, 0.3, 0.2, 0.1], 1)
    for value in (1.0, 2.0, 3.0):
        smoothing.add_data(np.array([value]))
    assert smoothing.filtered_data == pytest.approx([3.0])  # passes through until the window fills
    smoothing.add_data(np.array([4.0]))
    assert smoothing.filtered_data == pytest.approx([0.4 * 4 + 0.3 * 3 + 0.2 * 2 + 0.1 * 1])
    smoothing.add_data(np.array([4.0]))  # a repeated sample is skipped
    assert smoothing.filtered_data == pytest.approx([3.0])


class FakeKinematics:
    """Solves every IK target with fixed joint angles, and places the wrists at the identity."""

    def __init__(self, urdf_path, target_frame_name, joint_names, posture_weight):
        self.frames, self.joint_names = target_frame_name, joint_names
        self.targets = None

    def inverse_kinematics(self, current_joint_pos, desired_ee_pose, orientation_weight):
        self.targets = desired_ee_pose
        return np.full(len(self.joint_names), 10.0)

    def forward_kinematics(self, joint_pos_deg, frame_name):
        return np.eye(4)


def test_ik_targets_the_wrists_and_keeps_frozen_joints_at_zero(assets, monkeypatch):
    monkeypatch.setattr(g1_kinematics, "RobotKinematics", FakeKinematics)
    ik = g1_kinematics.G1ArmKinematics(frozen_joints=["left_wrist_yaw_joint"])
    assert "left_wrist_yaw_joint" not in ik.kinematics.joint_names
    left_ee, _ = ik.forward_kinematics(np.zeros(14))
    assert left_ee[0, 3] == pytest.approx(0.05)  # the end effector sits 5 cm past the wrist

    q = ik.solve_ik(np.eye(4), np.eye(4))
    assert q[6] == 0.0
    np.testing.assert_allclose(np.delete(q, 6), np.deg2rad(10.0))
    np.testing.assert_allclose(ik.kinematics.targets["left_wrist_yaw_link"][:3, 3], [-0.05, 0, 0])
