#!/usr/bin/env python
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

"""Tests for the NexArmLeader teleoperator class.

Exercises lifecycle, get_action, and send_feedback using a mocked
NexArmMotorsBus — no physical hardware required.
"""

from unittest.mock import MagicMock, patch

import pytest

serial = pytest.importorskip("serial", reason="pyserial is required for NexArm tests")

from lerobot.motors import MotorCalibration  # noqa: E402
from lerobot.motors.nexarm.nexarm import JOINT_NAMES  # noqa: E402
from lerobot.teleoperators.nexarm_leader import (  # noqa: E402
    NexArmLeader,
    NexArmLeaderConfig,
)


def _make_bus_mock(positions: dict[str, float] | None = None) -> MagicMock:
    if positions is None:
        positions = dict.fromkeys(JOINT_NAMES, 0.0)

    bus = MagicMock(name="NexArmBusMock")
    bus.is_connected = False
    bus.is_calibrated = True
    bus.motors = {name: MagicMock(id=i + 1) for i, name in enumerate(JOINT_NAMES)}
    bus.calibration = {
        name: MotorCalibration(
            id=i + 1, drive_mode=0, homing_offset=0, range_min=100, range_max=3900
        )
        for i, name in enumerate(JOINT_NAMES)
    }

    def _connect(handshake=True):  # noqa: ARG001
        bus.is_connected = True

    def _disconnect(disable_torque=True):  # noqa: ARG001
        bus.is_connected = False

    bus.connect.side_effect = _connect
    bus.disconnect.side_effect = _disconnect
    bus.sync_read.return_value = dict(positions)
    bus.enable_torque.return_value = None
    bus.disable_torque.return_value = None
    bus.enter_lerobot_mode.return_value = None
    bus.exit_lerobot_mode.return_value = None
    bus.handshake.return_value = None
    return bus


@pytest.fixture
def leader():
    bus_mock = _make_bus_mock()
    with patch(
        "lerobot.teleoperators.nexarm_leader.nexarm_leader.NexArmMotorsBus",
        return_value=bus_mock,
    ):
        cfg = NexArmLeaderConfig(port="/dev/null")
        teleop = NexArmLeader(cfg)
        teleop._bus_mock = bus_mock
        yield teleop
        if teleop.is_connected:
            teleop.disconnect()


@pytest.fixture
def leader_with_positions():
    positions = {name: float(i + 1) * 10 for i, name in enumerate(JOINT_NAMES)}
    bus_mock = _make_bus_mock(positions)
    with patch(
        "lerobot.teleoperators.nexarm_leader.nexarm_leader.NexArmMotorsBus",
        return_value=bus_mock,
    ):
        cfg = NexArmLeaderConfig(port="/dev/null")
        teleop = NexArmLeader(cfg)
        teleop._bus_mock = bus_mock
        teleop._expected_positions = positions
        yield teleop
        if teleop.is_connected:
            teleop.disconnect()


class TestLeaderConnectDisconnect:
    def test_not_connected_initially(self, leader):
        assert not leader.is_connected

    def test_connect(self, leader):
        leader.connect()
        assert leader.is_connected

    def test_connect_enters_lerobot_mode(self, leader):
        leader.connect()
        leader._bus_mock.enter_lerobot_mode.assert_called_once()

    def test_connect_calls_handshake(self, leader):
        leader.connect()
        leader._bus_mock.handshake.assert_called_once()

    def test_connect_disables_torque(self, leader):
        leader.connect()
        leader._bus_mock.disable_torque.assert_called()

    def test_connect_does_not_calibrate_when_already_calibrated(self, leader):
        leader.connect()
        leader._bus_mock.record_ranges_of_motion.assert_not_called()

    def test_disconnect(self, leader):
        leader.connect()
        leader.disconnect()
        assert not leader.is_connected

    def test_connect_idempotent(self, leader):
        from lerobot.utils.errors import DeviceAlreadyConnectedError

        leader.connect()
        with pytest.raises(DeviceAlreadyConnectedError):
            leader.connect()


class TestLeaderGetAction:
    def test_returns_all_joint_keys(self, leader_with_positions):
        leader_with_positions.connect()
        action = leader_with_positions.get_action()
        for name in JOINT_NAMES:
            assert f"{name}.pos" in action

    def test_returns_correct_values(self, leader_with_positions):
        leader_with_positions.connect()
        action = leader_with_positions.get_action()
        for name in JOINT_NAMES:
            assert action[f"{name}.pos"] == leader_with_positions._expected_positions[name]

    def test_calls_sync_read(self, leader):
        leader.connect()
        leader.get_action()
        leader._bus_mock.sync_read.assert_called_with("Present_Position")


class TestLeaderSendFeedback:
    def test_send_feedback_is_noop(self, leader):
        leader.connect()
        result = leader.send_feedback({"some_key": 1.0})
        assert result is None


class TestLeaderConfig:
    def test_default_baudrate(self):
        cfg = NexArmLeaderConfig(port="/dev/null")
        assert cfg.baudrate == 1_000_000

    def test_action_features_match_joints(self):
        with patch(
            "lerobot.teleoperators.nexarm_leader.nexarm_leader.NexArmMotorsBus",
            return_value=_make_bus_mock(),
        ):
            cfg = NexArmLeaderConfig(port="/dev/null")
            teleop = NexArmLeader(cfg)
            assert set(teleop.action_features.keys()) == {f"{n}.pos" for n in JOINT_NAMES}

    def test_feedback_features_empty(self):
        with patch(
            "lerobot.teleoperators.nexarm_leader.nexarm_leader.NexArmMotorsBus",
            return_value=_make_bus_mock(),
        ):
            cfg = NexArmLeaderConfig(port="/dev/null")
            teleop = NexArmLeader(cfg)
            assert teleop.feedback_features == {}
