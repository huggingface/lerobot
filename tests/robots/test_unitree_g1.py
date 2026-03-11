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

"""Tests for Unitree G1 robot. Meant to be run in an environment where the Unitree SDK is installed."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lerobot.utils.import_utils import _unitree_sdk_available

if not _unitree_sdk_available:
    pytest.skip("Unitree SDK not available", allow_module_level=True)

from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config
from lerobot.robots.unitree_g1.g1_utils import (
    NUM_MOTORS,
    REMOTE_AXES,
    REMOTE_BUTTONS,
    REMOTE_KEYS,
    G1_29_JointArmIndex,
    G1_29_JointIndex,
    default_remote_input,
    get_gravity_orientation,
)

# ---------------------------------------------------------------------------
# Unit tests for g1_utils (no SDK needed)
# ---------------------------------------------------------------------------


class TestG1Utils:
    def test_num_motors(self):
        assert NUM_MOTORS == 29

    def test_joint_index_count(self):
        assert len(G1_29_JointIndex) == 29

    def test_joint_arm_index_count(self):
        assert len(G1_29_JointArmIndex) == 14

    def test_arm_indices_are_subset_of_full(self):
        full_values = {j.value for j in G1_29_JointIndex}
        arm_values = {j.value for j in G1_29_JointArmIndex}
        assert arm_values.issubset(full_values)

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

    def test_remote_keys_structure(self):
        assert len(REMOTE_AXES) == 4
        assert len(REMOTE_BUTTONS) == 16
        assert len(REMOTE_KEYS) == 20
        assert REMOTE_KEYS == REMOTE_AXES + REMOTE_BUTTONS

    def test_default_remote_input(self):
        d = default_remote_input()
        assert len(d) == 20
        assert all(v == 0.0 for v in d.values())
        assert set(d.keys()) == set(REMOTE_KEYS)

    def test_gravity_orientation_identity(self):
        """Quaternion [1, 0, 0, 0] (no rotation) should give gravity along -z."""
        g = get_gravity_orientation([1.0, 0.0, 0.0, 0.0])
        assert g.shape == (3,)
        assert g.dtype == np.float32
        np.testing.assert_allclose(g, [0.0, 0.0, -1.0], atol=1e-6)

    def test_gravity_orientation_dtype(self):
        g = get_gravity_orientation(np.array([1.0, 0.0, 0.0, 0.0]))
        assert g.dtype == np.float32


# ---------------------------------------------------------------------------
# Unit tests for UnitreeG1Config (no SDK needed)
# ---------------------------------------------------------------------------


class TestUnitreeG1Config:
    def test_default_config(self):
        cfg = UnitreeG1Config()
        assert len(cfg.kp) == 29
        assert len(cfg.kd) == 29
        assert len(cfg.default_positions) == 29
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


# ---------------------------------------------------------------------------
# Robot mock and integration tests
# ---------------------------------------------------------------------------


def _make_lowstate_msg_mock():
    """Create a mock that mimics the SDK LowState_ message."""
    msg = MagicMock()
    for i in range(29):
        motor = MagicMock()
        motor.q = float(i) * 0.1
        motor.dq = float(i) * 0.01
        motor.tau_est = float(i) * 0.001
        motor.temperature = 30.0 + i
        msg.motor_state.__getitem__ = lambda self, idx, _motors={}: _motors.setdefault(
            idx, MagicMock(q=idx * 0.1, dq=idx * 0.01, tau_est=idx * 0.001, temperature=30.0 + idx)
        )

    msg.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
    msg.imu_state.gyroscope = [0.1, 0.2, 0.3]
    msg.imu_state.accelerometer = [0.0, 0.0, 9.81]
    msg.imu_state.rpy = [0.0, 0.0, 0.0]
    msg.imu_state.temperature = 25.0
    msg.wireless_remote = b"\x00" * 40
    msg.mode_machine = 0
    return msg


def _make_sdk_mocks():
    """Create mocks for the Unitree SDK modules used by UnitreeG1."""
    lowcmd_default = MagicMock()
    lowcmd_default.mode_pr = 0
    lowcmd_default.motor_cmd = [MagicMock() for _ in range(35)]

    crc_mock = MagicMock()
    crc_mock.Crc.return_value = 0

    lowstate_msg = _make_lowstate_msg_mock()

    subscriber_mock = MagicMock()
    subscriber_mock.Read.return_value = lowstate_msg

    publisher_mock = MagicMock()

    return {
        "lowcmd_default": lowcmd_default,
        "crc_mock": crc_mock,
        "subscriber_mock": subscriber_mock,
        "publisher_mock": publisher_mock,
        "lowstate_msg": lowstate_msg,
    }


@pytest.fixture
def unitree_g1():
    """Create a UnitreeG1 robot with all SDK dependencies mocked."""
    mocks = _make_sdk_mocks()

    mock_channel_init = MagicMock()
    mock_channel_pub = MagicMock(return_value=mocks["publisher_mock"])
    mock_channel_sub = MagicMock(return_value=mocks["subscriber_mock"])

    with (
        patch(
            "lerobot.robots.unitree_g1.unitree_g1.make_cameras_from_configs",
            return_value={},
        ),
        patch(
            "lerobot.robots.unitree_g1.unitree_g1.G1_29_ArmIK",
            return_value=MagicMock(),
        ),
        patch(
            "lerobot.robots.unitree_g1.unitree_g1._SDKChannelFactoryInitialize",
            mock_channel_init,
        ),
        patch(
            "lerobot.robots.unitree_g1.unitree_g1._SDKChannelPublisher",
            mock_channel_pub,
        ),
        patch(
            "lerobot.robots.unitree_g1.unitree_g1._SDKChannelSubscriber",
            mock_channel_sub,
        ),
        patch(
            "lerobot.robots.unitree_g1.unitree_g1.unitree_hg_msg_dds__LowCmd_",
            MagicMock(return_value=mocks["lowcmd_default"]),
        ),
        patch(
            "lerobot.robots.unitree_g1.unitree_g1.hg_LowCmd",
            MagicMock,
        ),
        patch(
            "lerobot.robots.unitree_g1.unitree_g1.hg_LowState",
            MagicMock,
        ),
        patch(
            "lerobot.robots.unitree_g1.unitree_g1.CRC",
            MagicMock(return_value=mocks["crc_mock"]),
        ),
    ):
        from lerobot.robots.unitree_g1.unitree_g1 import UnitreeG1

        cfg = UnitreeG1Config(is_simulation=True, gravity_compensation=False)
        robot = UnitreeG1(cfg)
        yield robot, mocks
        if robot.is_connected:
            robot.disconnect()


def test_init_state(unitree_g1):
    robot, _ = unitree_g1
    assert not robot.is_connected
    assert robot.controller is None


def test_observation_features(unitree_g1):
    robot, _ = unitree_g1
    features = robot.observation_features
    # Should have .q for all 29 joints (no cameras configured)
    assert len(features) == 29
    for joint in G1_29_JointIndex:
        assert f"{joint.name}.q" in features


def test_action_features_no_controller(unitree_g1):
    robot, _ = unitree_g1
    features = robot.action_features
    # Without controller: all 29 joints
    assert len(features) == 29
    for joint in G1_29_JointIndex:
        assert f"{joint.name}.q" in features


def test_get_observation_before_connect(unitree_g1):
    robot, _ = unitree_g1
    obs = robot.get_observation()
    assert obs == {}


def test_disconnect_idempotent(unitree_g1):
    robot, _ = unitree_g1
    # Should not raise even when not connected
    robot.disconnect()
