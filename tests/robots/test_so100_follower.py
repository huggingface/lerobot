from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from lerobot.common.robots.so100_follower import (
    SO100Follower,
    SO100FollowerConfig,
)

_MOTORS = SO100Follower(SO100FollowerConfig("")).bus.motors


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
    bus.motors = _MOTORS
    bus.is_calibrated = True
    bus.sync_read.return_value = {m: i for i, m in enumerate(_MOTORS, 1)}
    bus.sync_write.return_value = None
    bus.write.return_value = None
    bus.disable_torque.return_value = None
    bus.enable_torque.return_value = None

    @contextmanager
    def _dummy_cm():
        yield

    bus.torque_disabled.side_effect = _dummy_cm

    return bus


@pytest.fixture
def follower():
    with (
        patch(
            "lerobot.common.robots.so100_follower.so100_follower.FeetechMotorsBus",
            return_value=_make_bus_mock(),
        ),
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

    expected_keys = {f"{m}.pos" for m in _MOTORS}
    assert set(obs.keys()) == expected_keys

    for idx, motor in enumerate(_MOTORS, 1):
        assert obs[f"{motor}.pos"] == idx


def test_send_action(follower):
    follower.connect()

    action = {f"{m}.pos": i * 10 for i, m in enumerate(_MOTORS, 1)}
    returned = follower.send_action(action)

    assert returned == action

    goal_pos = {m: (i + 1) * 10 for i, m in enumerate(_MOTORS)}
    follower.bus.sync_write.assert_called_once_with("Goal_Position", goal_pos)
