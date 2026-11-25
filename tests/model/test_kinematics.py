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

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from lerobot.model import kinematics as kinematics_mod
from lerobot.model.kinematics import RobotKinematics


class _FakeRobotWrapper:
    def __init__(self, _urdf_path: str):
        self.joints: dict[str, float] = {"j1": 0.0, "j2": 0.0, "j3": 0.0}

    def joint_names(self) -> list[str]:
        return list(self.joints)

    def set_joint(self, name: str, value_rad: float) -> None:
        self.joints[name] = float(value_rad)

    def get_joint(self, name: str) -> float:
        return self.joints[name]

    def update_kinematics(self) -> None:
        pass

    def get_T_world_frame(self, _frame_name: str) -> np.ndarray:  # noqa: N802
        pose = np.eye(4)
        pose[:3, 3] = list(self.joints.values())
        return pose


class _FakeKinematicsSolver:
    def __init__(self, robot: _FakeRobotWrapper):
        self.robot = robot
        self.initial_guess_rad: dict[str, float] = {}

    def mask_fbase(self, _mask: bool) -> None:
        pass

    def add_frame_task(self, _frame_name: str, _pose: np.ndarray) -> MagicMock:
        return MagicMock()

    def solve(self, _enable: bool) -> None:
        if not self.initial_guess_rad:
            self.initial_guess_rad = dict(self.robot.joints)
        for name, deg in zip(("j1", "j2", "j3"), (15.0, -30.0, 45.0), strict=True):
            self.robot.set_joint(name, np.deg2rad(deg))


@pytest.fixture(autouse=True)
def _mock_placo(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_placo = SimpleNamespace(
        RobotWrapper=_FakeRobotWrapper,
        KinematicsSolver=_FakeKinematicsSolver,
    )
    monkeypatch.setattr(kinematics_mod, "placo", fake_placo)
    monkeypatch.setattr(kinematics_mod, "_placo_runtime_error", None)
    monkeypatch.setattr(kinematics_mod, "require_package", lambda *args, **kwargs: None)


def test_default_identity_mapping() -> None:
    kin = RobotKinematics("dummy.urdf", joint_names=["j1", "j2", "j3"])
    q_motor = np.array([10.0, -20.0, 30.0])

    np.testing.assert_allclose(kin.motor_to_urdf_deg(q_motor), q_motor)
    np.testing.assert_allclose(kin.urdf_to_motor_deg(q_motor), q_motor)
    np.testing.assert_allclose(kin.forward_kinematics(q_motor)[:3, 3], np.deg2rad(q_motor))

    ik_deg = kin.inverse_kinematics(np.array([0.0, 0.0, 0.0, 77.0]), np.eye(4))
    np.testing.assert_allclose(ik_deg, np.array([15.0, -30.0, 45.0, 77.0]))


def test_fk_and_ik_apply_signs_and_offsets() -> None:
    kin = RobotKinematics(
        "dummy.urdf",
        joint_names=["j1", "j2", "j3"],
        joint_signs=[-1, 1, -1],
        joint_offsets_deg=[10.0, -20.0, 90.0],
    )
    q_motor = np.array([30.0, 50.0, 40.0])
    expected_urdf = np.array([-20.0, 30.0, 50.0])

    # Round-trip conversion and forward kinematics
    np.testing.assert_allclose(kin.motor_to_urdf_deg(q_motor), expected_urdf)
    np.testing.assert_allclose(kin.urdf_to_motor_deg(expected_urdf), q_motor)
    np.testing.assert_allclose(kin.forward_kinematics(q_motor)[:3, 3], np.deg2rad(expected_urdf))

    # Inverse kinematics: initial guess in motor coords + trailing gripper value (88.0)
    solved_motor = kin.inverse_kinematics(np.array([5.0, 20.0, 90.0, 88.0]), np.eye(4))
    np.testing.assert_allclose(list(kin.solver.initial_guess_rad.values()), np.deg2rad([5.0, 0.0, 0.0]))
    np.testing.assert_allclose(solved_motor, np.array([-5.0, -10.0, 45.0, 88.0]))


def test_invalid_joint_mapping_raises_value_error() -> None:
    with pytest.raises(ValueError, match="joint_signs must have 3 values of \\+1 or -1"):
        RobotKinematics("dummy.urdf", joint_names=["j1", "j2", "j3"], joint_signs=[1, 0, -1])

    with pytest.raises(ValueError, match="joint_signs must have 3 values of \\+1 or -1"):
        RobotKinematics("dummy.urdf", joint_names=["j1", "j2", "j3"], joint_signs=[1, -1])

    with pytest.raises(ValueError, match="joint_offsets_deg must have 3 values"):
        RobotKinematics("dummy.urdf", joint_names=["j1", "j2", "j3"], joint_offsets_deg=[10.0, 20.0])
