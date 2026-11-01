from unittest.mock import PropertyMock, patch

from lerobot.motors.motors_bus import SerialMotorsBus
from lerobot.robots.so_follower import (
    SO100FollowerConfig,
    SO101FollowerConfig,
)
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.robots.so_follower.so101_safe_bus import (
    SO101SafeSerialMotorsBus,
)


def _construct_follower(config):
    with (
        patch.object(
            SOFollower,
            "calibration",
            new_callable=PropertyMock,
            return_value={},
            create=True,
        ),
        patch(
            "lerobot.robots.so_follower.so_follower.make_cameras_from_configs",
            return_value={},
        ),
    ):
        return SOFollower(config)


def test_so100_config_has_correct_type():
    config = SO100FollowerConfig(port="/dev/null")
    assert config.type == "so100_follower"


def test_so101_config_has_correct_type():
    config = SO101FollowerConfig(port="/dev/null")
    assert config.type == "so101_follower"


def test_so100_uses_original_serial_motors_bus():
    config = SO100FollowerConfig(port="/dev/null")

    with (
        patch(
            "lerobot.robots.so_follower.so_follower.SerialMotorsBus"
        ) as normal_bus,
        patch(
            "lerobot.robots.so_follower.so_follower.SO101SafeSerialMotorsBus"
        ) as safe_bus,
    ):
        _construct_follower(config)

        normal_bus.assert_called_once()
        safe_bus.assert_not_called()


def test_so101_uses_safe_serial_motors_bus():
    config = SO101FollowerConfig(port="/dev/null")

    with (
        patch(
            "lerobot.robots.so_follower.so_follower.SerialMotorsBus"
        ) as normal_bus,
        patch(
            "lerobot.robots.so_follower.so_follower.SO101SafeSerialMotorsBus"
        ) as safe_bus,
    ):
        _construct_follower(config)

        safe_bus.assert_called_once()
        normal_bus.assert_not_called()


def test_follower_configs_are_distinct_classes():
    assert SO100FollowerConfig is not SO101FollowerConfig
