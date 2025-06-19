from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from lerobot.common.robots import make_robot_from_config
from lerobot.common.robots.multi_arm_follower import (
    MultiArmFollower,
    MultiArmFollowerConfig,
)
from lerobot.common.robots.so100_follower import (
    SO100Follower,
    SO100FollowerConfig,
)
from lerobot.common.robots.so101_follower import (
    SO101Follower,
    SO101FollowerConfig,
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
def mock_multi_arm_follower(multi_arm_config):
    def _bus_side_effect(*_args, **kwargs):
        bus_mock = _make_bus_mock()
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
            "lerobot.common.robots.so100_follower.so100_follower.FeetechMotorsBus",
            side_effect=_bus_side_effect,
        ),
        patch(
            "lerobot.common.robots.so101_follower.so101_follower.FeetechMotorsBus",
            side_effect=_bus_side_effect,
        ),
        patch.object(SO100Follower, "configure", lambda self: None),
        patch.object(SO101Follower, "configure", lambda self: None),
    ):
        cfg = MultiArmFollowerConfig(arms=multi_arm_config)
        robot = MultiArmFollower(cfg)
        yield robot
        if robot.is_connected:
            robot.disconnect()


multi_arm_configs = [
    (
        [
            SO100FollowerConfig(port="port1"),
            SO100FollowerConfig(port="port2"),
        ]
    ),
    (
        [
            SO101FollowerConfig(port="port1"),
            SO101FollowerConfig(port="port2"),
        ]
    ),
    (
        [
            SO100FollowerConfig(port="port1"),
            SO101FollowerConfig(port="port2"),
            SO101FollowerConfig(port="port3"),
        ]
    ),
]


@pytest.mark.parametrize("multi_arm_config", multi_arm_configs)
def test_make_robot_from_config(multi_arm_config):
    robot_config = MultiArmFollowerConfig(arms=multi_arm_config)
    robot = make_robot_from_config(robot_config)
    assert isinstance(robot, MultiArmFollower)
    for robot_arm, arm_config in zip(robot.arms, multi_arm_config, strict=False):
        assert robot_arm.config_class is type(arm_config)


@pytest.mark.parametrize("multi_arm_config", multi_arm_configs)
def test_connect_disconnect(multi_arm_config, mock_multi_arm_follower):
    assert not mock_multi_arm_follower.is_connected

    mock_multi_arm_follower.connect()
    assert mock_multi_arm_follower.is_connected

    mock_multi_arm_follower.disconnect()
    assert not mock_multi_arm_follower.is_connected


@pytest.mark.parametrize("multi_arm_config", multi_arm_configs)
def test_get_observation(multi_arm_config, mock_multi_arm_follower):
    mock_multi_arm_follower.connect()
    obs = mock_multi_arm_follower.get_observation()

    expected_keys = {
        mock_multi_arm_follower._encode_arm_index(f"{m}.pos", i)
        for i, follower in enumerate(mock_multi_arm_follower.arms)
        for m in follower.bus.motors
    }
    assert set(obs.keys()) == expected_keys

    for i, follower in enumerate(mock_multi_arm_follower.arms):
        for idx, motor in enumerate(follower.bus.motors, 1):
            key = mock_multi_arm_follower._encode_arm_index(f"{motor}.pos", i)
            assert obs[key] == idx


@pytest.mark.parametrize("multi_arm_config", multi_arm_configs)
def test_send_action(multi_arm_config, mock_multi_arm_follower):
    mock_multi_arm_follower.connect()

    action = {
        mock_multi_arm_follower._encode_arm_index(f"{m}.pos", index): i * 10
        for index, follower in enumerate(mock_multi_arm_follower.arms)
        for i, m in enumerate(follower.bus.motors, 1)
    }
    returned = mock_multi_arm_follower.send_action(action)

    assert returned == action

    for follower in mock_multi_arm_follower.arms:
        goal_pos = {m: (i + 1) * 10 for i, m in enumerate(follower.bus.motors)}
        follower.bus.sync_write.assert_called_once_with("Goal_Position", goal_pos)
