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
    DynamixelMotorsBus(port="/dev/dummy-port", motors={"dummy", (1, "xl330-m077")})
