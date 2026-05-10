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

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.bi_so_follower import BiSOFollower, BiSOFollowerConfig
from lerobot.robots.so_follower import (
    SO100Follower,
    SO100FollowerConfig,
    SOFollowerConfig,
)


def _make_bus_mock() -> MagicMock:
    """Return a bus mock with just the attributes used by the robot."""
    bus = MagicMock(name="FeetechBusMock")
    bus.is_connected = False

    def _connect():
        bus.is_connected = True

    def _disconnect(_disable=True):
        bus.is_connected = False

    bus.connect.side_effect = _connect
    bus.disconnect.side_effect = _disconnect

    @contextmanager
    def _dummy_cm():
        yield

    bus.torque_disabled.side_effect = _dummy_cm

    return bus


def _configure_bus_mock(bus_mock: MagicMock, *_args, **kwargs) -> MagicMock:
    bus_mock.motors = kwargs["motors"]
    motors_order: list[str] = list(bus_mock.motors)

    bus_mock.sync_read.return_value = {motor: idx for idx, motor in enumerate(motors_order, 1)}
    bus_mock.sync_write.return_value = None
    bus_mock.write.return_value = None
    bus_mock.disable_torque.return_value = None
    bus_mock.enable_torque.return_value = None
    bus_mock.is_calibrated = True
    return bus_mock


@contextmanager
def _patch_bus(bus_mock: MagicMock):
    with patch(
        "lerobot.robots.so_follower.so_follower.FeetechMotorsBus",
        side_effect=lambda *args, **kwargs: _configure_bus_mock(bus_mock, *args, **kwargs),
    ):
        yield


def _write_values(bus_mock: MagicMock, register: str) -> dict[str, int]:
    return {
        motor: value
        for reg, motor, value in (call.args for call in bus_mock.write.call_args_list)
        if reg == register
    }


@pytest.fixture
def follower():
    bus_mock = _make_bus_mock()

    with (
        _patch_bus(bus_mock),
        patch.object(SO100Follower, "configure", lambda self: None),
    ):
        cfg = SO100FollowerConfig(port="/dev/null")
        robot = SO100Follower(cfg)
        yield robot
        if robot.is_connected:
            robot.disconnect()


def test_connect_disconnect(follower):
    assert not follower.is_connected

    follower.connect()
    assert follower.is_connected

    follower.disconnect()
    assert not follower.is_connected


def test_get_observation(follower):
    follower.connect()
    obs = follower.get_observation()

    expected_keys = {f"{m}.pos" for m in follower.bus.motors}
    assert set(obs.keys()) == expected_keys

    for idx, motor in enumerate(follower.bus.motors, 1):
        assert obs[f"{motor}.pos"] == idx


def test_send_action(follower):
    follower.connect()

    action = {f"{m}.pos": i * 10 for i, m in enumerate(follower.bus.motors, 1)}
    returned = follower.send_action(action)

    assert returned == action

    goal_pos = {m: (i + 1) * 10 for i, m in enumerate(follower.bus.motors)}
    follower.bus.sync_write.assert_called_once_with("Goal_Position", goal_pos)


def test_configure_writes_default_position_p_coefficient():
    bus_mock = _make_bus_mock()

    with _patch_bus(bus_mock):
        robot = SO100Follower(SO100FollowerConfig(port="/dev/null"))
        robot.configure()

    motors = set(robot.bus.motors)
    assert _write_values(bus_mock, "P_Coefficient") == dict.fromkeys(motors, 16)
    assert _write_values(bus_mock, "I_Coefficient") == dict.fromkeys(motors, 0)
    assert _write_values(bus_mock, "D_Coefficient") == dict.fromkeys(motors, 32)
    assert _write_values(bus_mock, "Max_Torque_Limit") == {"gripper": 500}
    assert _write_values(bus_mock, "Protection_Current") == {"gripper": 250}
    assert _write_values(bus_mock, "Overload_Torque") == {"gripper": 25}


def test_configure_writes_overridden_position_p_coefficient():
    bus_mock = _make_bus_mock()

    with _patch_bus(bus_mock):
        robot = SO100Follower(SO100FollowerConfig(port="/dev/null", position_p_coefficient=32))
        robot.configure()

    assert _write_values(bus_mock, "P_Coefficient") == dict.fromkeys(robot.bus.motors, 32)


@pytest.mark.parametrize("value", [-1, 256, 1.5, "32", True])
def test_position_p_coefficient_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="position_p_coefficient must be an integer in \\[0, 255\\]"):
        SO100FollowerConfig(port="/dev/null", position_p_coefficient=value)


@pytest.mark.parametrize("value", [0, 16, 32, 255])
def test_position_p_coefficient_accepts_valid_values(value):
    config = SO100FollowerConfig(port="/dev/null", position_p_coefficient=value)

    assert config.position_p_coefficient == value


def test_position_p_coefficient_preserves_existing_positional_arguments():
    config = SO100FollowerConfig("/dev/null", True, None, {}, False)

    assert config.cameras == {}
    assert config.use_degrees is False
    assert config.position_p_coefficient == 16


def test_bi_so_follower_preserves_position_p_coefficients():
    def make_arm(config):
        arm = MagicMock()
        arm.config = config
        arm.cameras = {}
        return arm

    config = BiSOFollowerConfig(
        left_arm_config=SOFollowerConfig(port="/dev/left", position_p_coefficient=16),
        right_arm_config=SOFollowerConfig(port="/dev/right", position_p_coefficient=32),
    )

    with patch(
        "lerobot.robots.bi_so_follower.bi_so_follower.SOFollower", side_effect=make_arm
    ) as follower_cls:
        BiSOFollower(config)

    left_config = follower_cls.call_args_list[0].args[0]
    right_config = follower_cls.call_args_list[1].args[0]

    assert left_config.position_p_coefficient == 16
    assert right_config.position_p_coefficient == 32
