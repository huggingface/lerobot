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

from lerobot.robots.so_follower import (
    MotorStallError,
    SO100Follower,
    SO100FollowerConfig,
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


@pytest.fixture
def follower(tmp_path):
    bus_mock = _make_bus_mock()

    def _bus_side_effect(*_args, **kwargs):
        bus_mock.motors = kwargs["motors"]
        motors_order: list[str] = list(bus_mock.motors)

        bus_mock.sync_read.return_value = {motor: idx for idx, motor in enumerate(motors_order, 1)}
        bus_mock.sync_write.return_value = None
        bus_mock.write.return_value = None
        bus_mock.disable_torque.return_value = None
        bus_mock.enable_torque.return_value = None
        bus_mock.is_calibrated = True
        return bus_mock

    with (
        patch(
            "lerobot.robots.so_follower.so_follower.FeetechMotorsBus",
            side_effect=_bus_side_effect,
        ),
        patch.object(SO100Follower, "configure", lambda self: None),
    ):
        cfg = SO100FollowerConfig(port="/dev/null", calibration_dir=tmp_path)
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


def test_get_observation_uses_read_retries(follower):
    # Feetech buses can intermittently fail a sync_read; the follower should forward the configured
    # retry count so transient failures don't abort the control loop (see #3131).
    follower.config.num_read_retries = 7
    follower.connect()
    follower.get_observation()

    follower.bus.sync_read.assert_called_once_with("Present_Position", num_retry=7)


def test_send_action_uses_read_retries(follower):
    follower.config.max_relative_target = 10.0
    follower.config.num_read_retries = 7
    follower.connect()

    action = {f"{motor}.pos": value * 10 for value, motor in enumerate(follower.bus.motors, 1)}
    follower.send_action(action)

    follower.bus.sync_read.assert_called_once_with("Present_Position", num_retry=7)


def test_send_action(follower):
    follower.connect()

    action = {f"{m}.pos": i * 10 for i, m in enumerate(follower.bus.motors, 1)}
    returned = follower.send_action(action)

    assert returned == action

    goal_pos = {m: (i + 1) * 10 for i, m in enumerate(follower.bus.motors)}
    follower.bus.sync_write.assert_called_once_with("Goal_Position", goal_pos)


def test_configure_writes_position_pid_coefficients():
    bus_mock = _make_bus_mock()
    bus_mock.motors = ["shoulder_pan"]
    robot = MagicMock()
    robot.bus = bus_mock
    robot.config = SO100FollowerConfig(
        port="/dev/null",
        position_p_coefficient=32,
        position_i_coefficient=1,
        position_d_coefficient=16,
    )

    SO100Follower.configure(robot)

    bus_mock.write.assert_any_call("P_Coefficient", "shoulder_pan", 32)
    bus_mock.write.assert_any_call("I_Coefficient", "shoulder_pan", 1)
    bus_mock.write.assert_any_call("D_Coefficient", "shoulder_pan", 16)


def test_stall_protection_disables_torque_after_persistent_current(monkeypatch, follower):
    follower.config.stall_current_thresholds = {"shoulder_pan": 100}
    follower.config.stall_position_tolerance = 5
    follower.config.stall_timeout_s = 0.5
    follower.connect()

    positions = dict.fromkeys(follower.bus.motors, 0)
    currents = dict.fromkeys(follower.bus.motors, 0)
    currents["shoulder_pan"] = 120
    follower.bus.sync_read.side_effect = lambda register, **_: (
        positions if register == "Present_Position" else currents
    )
    times = iter([10.0, 10.6])
    monkeypatch.setattr("lerobot.robots.so_follower.so_follower.time.perf_counter", lambda: next(times))

    action = {f"{motor}.pos": (20 if motor == "shoulder_pan" else 0) for motor in follower.bus.motors}
    follower.send_action(action)
    follower.send_action(action)

    with pytest.raises(MotorStallError, match="disabled torque.*shoulder_pan"):
        follower.send_action(action)

    follower.bus.disable_torque.assert_called_once_with(num_retry=follower.config.num_read_retries)
    assert follower.bus.sync_write.call_count == 2

    with pytest.raises(MotorStallError, match="latched"):
        follower.send_action(action)


def test_stall_protection_requires_current_and_position_error(monkeypatch, follower):
    follower.config.stall_current_thresholds = {"shoulder_pan": 100}
    follower.config.stall_position_tolerance = 5
    follower.config.stall_timeout_s = 0.5
    follower.connect()

    positions = dict.fromkeys(follower.bus.motors, 0)
    currents = dict.fromkeys(follower.bus.motors, 0)
    currents["shoulder_pan"] = 120
    follower.bus.sync_read.side_effect = lambda register, **_: (
        positions if register == "Present_Position" else currents
    )
    times = iter([10.0, 11.0])
    monkeypatch.setattr("lerobot.robots.so_follower.so_follower.time.perf_counter", lambda: next(times))

    action = {f"{motor}.pos": 0 for motor in follower.bus.motors}
    follower.send_action(action)
    follower.send_action(action)
    follower.send_action(action)

    follower.bus.disable_torque.assert_not_called()
    assert follower.bus.sync_write.call_count == 3


def test_stall_protection_requires_a_continuous_high_current_window(monkeypatch, follower):
    follower.config.stall_current_thresholds = {"shoulder_pan": 100}
    follower.config.stall_position_tolerance = 5
    follower.config.stall_timeout_s = 0.5
    follower.connect()

    positions = dict.fromkeys(follower.bus.motors, 0)
    high_current = dict.fromkeys(follower.bus.motors, 0)
    high_current["shoulder_pan"] = 120
    low_current = dict.fromkeys(follower.bus.motors, 0)
    current_samples = iter([high_current, low_current, high_current, high_current])
    follower.bus.sync_read.side_effect = lambda register, **_: (
        positions if register == "Present_Position" else next(current_samples)
    )
    times = iter([10.0, 10.2, 10.4, 11.0])
    monkeypatch.setattr("lerobot.robots.so_follower.so_follower.time.perf_counter", lambda: next(times))

    action = {f"{motor}.pos": (20 if motor == "shoulder_pan" else 0) for motor in follower.bus.motors}
    follower.send_action(action)
    follower.send_action(action)  # Start the first high-current window.
    follower.send_action(action)  # Low current resets it.
    follower.send_action(action)  # Start a new window.

    with pytest.raises(MotorStallError):
        follower.send_action(action)

    follower.bus.disable_torque.assert_called_once()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"stall_current_thresholds": {"shoulder_pan": 0}}, "must be positive"),
        ({"stall_position_tolerance": 0}, "must be positive"),
        ({"stall_timeout_s": 0}, "must be positive"),
    ],
)
def test_stall_protection_config_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SO100FollowerConfig(port="/dev/null", **kwargs)
