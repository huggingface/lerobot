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
    FeetechMotorsBus(port="/dev/dummy-port", motors={"dummy", (1, "sts3215")})
