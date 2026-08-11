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

"""Hardware-free tests for the Unitree G1 helpers and config.

Everything here is pure numpy/dataclass logic, so these run everywhere: no Unitree SDK,
no onnxruntime, no robot. Tests that need the SDK mocks live in ``test_unitree_g1.py``
and the SONIC decoder tests in ``test_sonic_whole_body.py``.
"""

import numpy as np
import pytest

from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config
from lerobot.robots.unitree_g1.g1_utils import (
    ISAACLAB_TO_MUJOCO,
    MUJOCO_TO_ISAACLAB,
    NUM_MOTORS,
    REMOTE_AXES,
    REMOTE_BUTTONS,
    REMOTE_KEYS,
    G1_29_JointArmIndex,
    G1_29_JointIndex,
    default_remote_input,
    get_gravity_orientation,
    make_robot_controller,
)


class TestJointIndices:
    def test_num_motors(self):
        assert NUM_MOTORS == 29

    def test_joint_index_count(self):
        assert len(G1_29_JointIndex) == 29

    def test_joint_arm_index_count(self):
        assert len(G1_29_JointArmIndex) == 14

    def test_joint_values_are_dense_range(self):
        """Values must be exactly 0..28: they index into SDK motor arrays and numpy vectors."""
        assert sorted(j.value for j in G1_29_JointIndex) == list(range(NUM_MOTORS))

    def test_arm_indices_are_subset_of_full(self):
        full_values = {j.value for j in G1_29_JointIndex}
        arm_values = {j.value for j in G1_29_JointArmIndex}
        assert arm_values.issubset(full_values)

    def test_arm_names_match_full_enum(self):
        """Feature keys are built from ``.name``, so arm names must agree with the full enum."""
        full = {j.name: j.value for j in G1_29_JointIndex}
        for arm_joint in G1_29_JointArmIndex:
            assert arm_joint.name in full
            assert full[arm_joint.name] == arm_joint.value

    def test_arm_indices_start_at_15(self):
        assert min(j.value for j in G1_29_JointArmIndex) == 15
        assert max(j.value for j in G1_29_JointArmIndex) == 28

    def test_enum_naming_consistency(self):
        """Verify all wrist joints use consistent PascalCase naming."""
        wrist_joints = [j for j in G1_29_JointIndex if "Wrist" in j.name]
        for j in wrist_joints:
            # Should be "WristYaw", "WristPitch", "WristRoll" — no lowercase after "Wrist"
            after_wrist = j.name.split("Wrist")[1]
            assert after_wrist[0].isupper(), f"{j.name} has inconsistent casing after 'Wrist'"


class TestJointOrderPermutation:
    """The IsaacLab <-> MuJoCo permutation is load-bearing for the SONIC decoder."""

    def test_shapes_and_dtypes(self):
        assert ISAACLAB_TO_MUJOCO.shape == (NUM_MOTORS,)
        assert MUJOCO_TO_ISAACLAB.shape == (NUM_MOTORS,)
        assert ISAACLAB_TO_MUJOCO.dtype == np.int32
        assert MUJOCO_TO_ISAACLAB.dtype == np.int32

    def test_is_a_permutation(self):
        assert sorted(ISAACLAB_TO_MUJOCO.tolist()) == list(range(NUM_MOTORS))
        assert sorted(MUJOCO_TO_ISAACLAB.tolist()) == list(range(NUM_MOTORS))

    def test_index_maps_are_mutual_inverses(self):
        identity = np.arange(NUM_MOTORS)
        np.testing.assert_array_equal(ISAACLAB_TO_MUJOCO[MUJOCO_TO_ISAACLAB], identity)
        np.testing.assert_array_equal(MUJOCO_TO_ISAACLAB[ISAACLAB_TO_MUJOCO], identity)

    def test_gather_round_trip_restores_vector(self):
        """Reordering a vector one way then the other must return the original."""
        v = np.arange(NUM_MOTORS, dtype=np.float32) * 0.1
        np.testing.assert_array_equal(v[MUJOCO_TO_ISAACLAB][ISAACLAB_TO_MUJOCO], v)
        np.testing.assert_array_equal(v[ISAACLAB_TO_MUJOCO][MUJOCO_TO_ISAACLAB], v)

    def test_permutation_is_not_the_identity(self):
        """Guards against someone 'simplifying' the maps into a no-op."""
        assert not np.array_equal(ISAACLAB_TO_MUJOCO, np.arange(NUM_MOTORS))


class TestRemoteInput:
    def test_remote_keys_structure(self):
        assert len(REMOTE_AXES) == 4
        assert len(REMOTE_BUTTONS) == 16
        assert len(REMOTE_KEYS) == 20
        assert REMOTE_KEYS == REMOTE_AXES + REMOTE_BUTTONS

    def test_remote_keys_are_unique(self):
        assert len(set(REMOTE_KEYS)) == len(REMOTE_KEYS)

    def test_default_remote_input(self):
        d = default_remote_input()
        assert len(d) == 20
        assert all(v == 0.0 for v in d.values())
        assert set(d.keys()) == set(REMOTE_KEYS)

    def test_remote_input_is_fresh_dict(self):
        """Callers mutate this per tick; instances must not share state."""
        first = default_remote_input()
        first["remote.lx"] = 1.0
        assert default_remote_input()["remote.lx"] == 0.0


class TestGravityOrientation:
    def test_identity_points_down(self):
        """Quaternion [1, 0, 0, 0] (no rotation) should give gravity along -z."""
        g = get_gravity_orientation([1.0, 0.0, 0.0, 0.0])
        assert g.shape == (3,)
        assert g.dtype == np.float32
        np.testing.assert_allclose(g, [0.0, 0.0, -1.0], atol=1e-6)

    def test_upside_down_points_up(self):
        """180 deg about x flips the gravity vector to +z."""
        g = get_gravity_orientation([0.0, 1.0, 0.0, 0.0])
        np.testing.assert_allclose(g, [0.0, 0.0, 1.0], atol=1e-6)

    def test_ninety_degrees_about_x(self):
        s = np.sqrt(0.5)
        g = get_gravity_orientation([s, s, 0.0, 0.0])
        np.testing.assert_allclose(g, [0.0, -1.0, 0.0], atol=1e-6)

    def test_accepts_list_and_array(self):
        as_list = get_gravity_orientation([1.0, 0.0, 0.0, 0.0])
        as_array = get_gravity_orientation(np.array([1.0, 0.0, 0.0, 0.0]))
        assert as_array.dtype == np.float32
        np.testing.assert_allclose(as_list, as_array)

    @pytest.mark.parametrize(
        "quat",
        [
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.5, 0.5, 0.5, 0.5),
            (np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0),
        ],
    )
    def test_unit_quaternion_gives_unit_vector(self, quat):
        g = get_gravity_orientation(list(quat))
        np.testing.assert_allclose(np.linalg.norm(g), 1.0, atol=1e-6)


class TestMakeRobotController:
    def test_none_disables_controller(self):
        assert make_robot_controller(None) is None

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown controller") as excinfo:
            make_robot_controller("NotAController")
        # The error should tell the user what they can pick instead.
        assert "SonicWholeBodyController" in str(excinfo.value)


class TestUnitreeG1Config:
    def test_default_config(self):
        cfg = UnitreeG1Config()
        assert len(cfg.kp) == NUM_MOTORS
        assert len(cfg.kd) == NUM_MOTORS
        assert len(cfg.default_positions) == NUM_MOTORS
        assert cfg.is_simulation is True
        assert cfg.controller is None
        assert cfg.gravity_compensation is False

    def test_gains_are_positive(self):
        cfg = UnitreeG1Config()
        assert all(v > 0 for v in cfg.kp)
        assert all(v > 0 for v in cfg.kd)

    def test_config_copies_gains(self):
        """Each config instance should have its own copy of gains."""
        cfg1 = UnitreeG1Config()
        cfg2 = UnitreeG1Config()
        cfg1.kp[0] = 999.0
        assert cfg2.kp[0] != 999.0

    def test_default_positions_are_copied_too(self):
        cfg1 = UnitreeG1Config()
        cfg2 = UnitreeG1Config()
        cfg1.default_positions[0] = 999.0
        assert cfg2.default_positions[0] != 999.0

    def test_control_dt_is_positive(self):
        cfg = UnitreeG1Config()
        assert cfg.control_dt > 0
