import sys
from unittest.mock import patch

import pytest

from tests.mocks import mock_dynamixel_sdk


@pytest.fixture(autouse=True)
def patch_dynamixel_sdk():
    with patch.dict(sys.modules, {"dynamixel_sdk": mock_dynamixel_sdk}):
        yield


def test_patch_sdk():
    assert "dynamixel_sdk" in sys.modules  # Should be patched
    assert sys.modules["dynamixel_sdk"] is mock_dynamixel_sdk  # Should match the mock


def test_abc_implementation():
    from lerobot.common.motors.dynamixel import DynamixelMotorsBus

    # Instantiation should raise an error if the class doesn't implements abstract methods/properties
    DynamixelMotorsBus(port="/dev/dummy-port", motors={"dummy": (1, "xl330-m077")})


def test_configure_motors_all_ids_1():
    from lerobot.common.motors.dynamixel import DynamixelMotorsBus

    # see X_SERIES_BAUDRATE_TABLE
    smaller_baudrate = 9_600
    smaller_baudrate_value = 0

    # This test expect the configuration was already correct.
    motors_bus = DynamixelMotorsBus(port="/dev/dummy-port", motors={"dummy": (1, "xl330-m077")})
    motors_bus.connect()
    motors_bus.write("Baud_Rate", [smaller_baudrate_value] * len(motors_bus))

    motors_bus.set_baudrate(smaller_baudrate)
    motors_bus.write("ID", [1] * len(motors_bus))
    del motors_bus

    # Test configure
    motors_bus = DynamixelMotorsBus(port="/dev/dummy-port", motors={"dummy": (1, "xl330-m077")})
    motors_bus.connect()
    assert motors_bus.are_motors_configured()
    del motors_bus
