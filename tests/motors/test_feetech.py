import sys
from unittest.mock import patch

import pytest

from tests.mocks import mock_scservo_sdk


@pytest.fixture(autouse=True)
def patch_scservo_sdk():
    with patch.dict(sys.modules, {"scservo_sdk": mock_scservo_sdk}):
        yield


def test_patch_sdk():
    assert "scservo_sdk" in sys.modules  # Should be patched
    assert sys.modules["scservo_sdk"] is mock_scservo_sdk  # Should match the mock


def test_abc_implementation():
    from lerobot.common.motors.feetech import FeetechMotorsBus

    # Instantiation should raise an error if the class doesn't implements abstract methods/properties
    FeetechMotorsBus(port="/dev/dummy-port", motors={"dummy": (1, "sts3215")})


def test_configure_motors_all_ids_1():
    from lerobot.common.motors.feetech import FeetechMotorsBus

    # see SCS_SERIES_BAUDRATE_TABLE
    smaller_baudrate = 19_200
    smaller_baudrate_value = 7

    # This test expect the configuration was already correct.
    motors_bus = FeetechMotorsBus(port="/dev/dummy-port", motors={"dummy": (1, "sts3215")})
    motors_bus.connect()
    motors_bus.write("Baud_Rate", [smaller_baudrate_value] * len(motors_bus))

    motors_bus.set_baudrate(smaller_baudrate)
    motors_bus.write("ID", [1] * len(motors_bus))
    del motors_bus

    # Test configure
    motors_bus = FeetechMotorsBus(port="/dev/dummy-port", motors={"dummy": (1, "sts3215")})
    motors_bus.connect()
    assert motors_bus.are_motors_configured()
    del motors_bus
