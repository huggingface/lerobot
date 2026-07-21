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

"""Tests for the NexArmFollower robot class.

Exercises lifecycle, get_observation / send_action and the new normalised
``sync_read`` / ``sync_write`` API using a mocked NexArmMotorsBus — no
physical hardware required.
"""

from unittest.mock import MagicMock, patch

import pytest

serial = pytest.importorskip("serial", reason="pyserial is required for NexArm tests")

from lerobot.motors import MotorCalibration  # noqa: E402
from lerobot.motors.nexarm.nexarm import JOINT_NAMES  # noqa: E402
from lerobot.robots.nexarm_follower import (  # noqa: E402
    NexArmFollower,
    NexArmFollowerConfig,
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
    bus.sync_write.return_value = None
    bus.enable_torque.return_value = None
    bus.disable_torque.return_value = None
    bus.enter_lerobot_mode.return_value = None
    bus.exit_lerobot_mode.return_value = None
    bus.write_motion_params.return_value = None
    bus.handshake.return_value = None
    return bus


@pytest.fixture
def follower():
    bus_mock = _make_bus_mock()
    with patch(
        "lerobot.robots.nexarm_follower.nexarm_follower.NexArmMotorsBus",
        return_value=bus_mock,
    ):
        cfg = NexArmFollowerConfig(port="/dev/null", cameras={})
        robot = NexArmFollower(cfg)
        robot._bus_mock = bus_mock
        yield robot
        if robot.is_connected:
            robot.disconnect()


@pytest.fixture
def follower_with_positions():
    positions = {name: float(i + 1) * 10 for i, name in enumerate(JOINT_NAMES)}
    bus_mock = _make_bus_mock(positions)
    with patch(
        "lerobot.robots.nexarm_follower.nexarm_follower.NexArmMotorsBus",
        return_value=bus_mock,
    ):
        cfg = NexArmFollowerConfig(port="/dev/null", cameras={})
        robot = NexArmFollower(cfg)
        robot._bus_mock = bus_mock
        robot._expected_positions = positions
        yield robot
        if robot.is_connected:
            robot.disconnect()


class TestConnectDisconnect:
    def test_not_connected_initially(self, follower):
        assert not follower.is_connected

    def test_connect(self, follower):
        follower.connect()
        assert follower.is_connected

    def test_connect_enters_lerobot_mode(self, follower):
        follower.connect()
        follower._bus_mock.enter_lerobot_mode.assert_called_once()

    def test_connect_calls_handshake(self, follower):
        follower.connect()
        follower._bus_mock.handshake.assert_called_once()

    def test_connect_does_not_calibrate_when_already_calibrated(self, follower):
        # is_calibrated defaults to True in the fixture.
        follower.connect()
        # No interactive prompt should run; we just call configure.
        follower._bus_mock.write_motion_params.assert_called_once()
        follower._bus_mock.enable_torque.assert_called_once()

    def test_disconnect(self, follower):
        follower.connect()
        follower.disconnect()
        assert not follower.is_connected

    def test_connect_idempotent(self, follower):
        from lerobot.utils.errors import DeviceAlreadyConnectedError

        follower.connect()
        with pytest.raises(DeviceAlreadyConnectedError):
            follower.connect()


class TestGetObservation:
    def test_returns_all_joint_keys(self, follower_with_positions):
        follower_with_positions.connect()
        obs = follower_with_positions.get_observation()
        for name in JOINT_NAMES:
            assert f"{name}.pos" in obs

    def test_returns_correct_values(self, follower_with_positions):
        follower_with_positions.connect()
        obs = follower_with_positions.get_observation()
        for name in JOINT_NAMES:
            assert obs[f"{name}.pos"] == follower_with_positions._expected_positions[name]

    def test_calls_sync_read(self, follower):
        follower.connect()
        follower.get_observation()
        follower._bus_mock.sync_read.assert_called_with("Present_Position")


class TestSendAction:
    def test_calls_sync_write(self, follower):
        follower.connect()
        action = {f"{name}.pos": 0.0 for name in JOINT_NAMES}
        follower.send_action(action)
        # The first sync_write call (after potential present-read) targets Goal_Position.
        assert follower._bus_mock.sync_write.called
        last_call = follower._bus_mock.sync_write.call_args_list[-1]
        assert last_call.args[0] == "Goal_Position"

    def test_strips_pos_suffix(self, follower):
        follower.connect()
        values = {name: float(i + 1) * 5 for i, name in enumerate(JOINT_NAMES)}
        action = {f"{name}.pos": v for name, v in values.items()}
        follower.send_action(action)
        last_call = follower._bus_mock.sync_write.call_args_list[-1]
        assert last_call.args[1] == values

    def test_returns_action_with_pos_suffix(self, follower):
        follower.connect()
        action = {f"{name}.pos": 0.0 for name in JOINT_NAMES}
        out = follower.send_action(action)
        assert set(out.keys()) == {f"{n}.pos" for n in JOINT_NAMES}

    def test_max_relative_target_clamps_step(self, follower):
        # Pretend the arm is at 0 and we ask for +50; cap is 10.
        follower._bus_mock.sync_read.return_value = dict.fromkeys(JOINT_NAMES, 0.0)
        follower.config.max_relative_target = 10.0
        follower.connect()

        action = {f"{name}.pos": 50.0 for name in JOINT_NAMES}
        follower.send_action(action)

        last_call = follower._bus_mock.sync_write.call_args_list[-1]
        for name in JOINT_NAMES:
            assert last_call.args[1][name] == 10.0


class TestConfig:
    def test_default_baudrate(self):
        cfg = NexArmFollowerConfig(port="/dev/null")
        assert cfg.baudrate == 1_000_000

    def test_default_max_relative_target_none(self):
        cfg = NexArmFollowerConfig(port="/dev/null")
        assert cfg.max_relative_target is None

    def test_action_features_match_joints(self):
        with patch(
            "lerobot.robots.nexarm_follower.nexarm_follower.NexArmMotorsBus",
            return_value=_make_bus_mock(),
        ):
            cfg = NexArmFollowerConfig(port="/dev/null", cameras={})
            robot = NexArmFollower(cfg)
            assert set(robot.action_features.keys()) == {f"{n}.pos" for n in JOINT_NAMES}
