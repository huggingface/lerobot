import sys
from typing import Generator
from unittest.mock import Mock, patch

import pytest
import scservo_sdk as scs

from lerobot.common.motors import CalibrationMode, Motor
from lerobot.common.motors.feetech import FeetechMotorsBus
from tests.mocks.mock_feetech import MockMotors, MockPortHandler


@pytest.fixture(autouse=True)
def patch_port_handler():
    if sys.platform == "darwin":
        with patch.object(scs, "PortHandler", MockPortHandler):
            yield
    else:
        yield


@pytest.fixture
def mock_motors() -> Generator[MockMotors, None, None]:
    motors = MockMotors()
    motors.open()
    yield motors
    motors.close()


@pytest.fixture
def dummy_motors() -> dict[str, Motor]:
    return {
        "dummy_1": Motor(1, "sts3215", CalibrationMode.RANGE_M100_100),
        "dummy_2": Motor(2, "sts3215", CalibrationMode.RANGE_M100_100),
        "dummy_3": Motor(3, "sts3215", CalibrationMode.RANGE_M100_100),
    }


@pytest.mark.skipif(sys.platform != "darwin", reason=f"No patching needed on {sys.platform=}")
def test_autouse_patch():
    """Ensures that the autouse fixture correctly patches scs.PortHandler with MockPortHandler."""
    assert scs.PortHandler is MockPortHandler


@pytest.mark.parametrize(
    "motor_names, read_values",
    [
        (
            ["dummy_1", "dummy_2"],
            [{"dummy_1": 3000}, {"dummy_2": 1000}],
        ),
    ],
    ids=["two-motors"],
)
def test_find_offset(mock_motors, dummy_motors, motor_names, read_values):
    motors_bus = FeetechMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    motors_bus.connect()

    with patch("builtins.input", return_value=""):
        motors_bus.sync_read = Mock(side_effect=read_values)
        motors_bus.motor_names = motor_names
        motors_bus.write = Mock(return_value=None)
        motors_bus.set_offset = Mock()

        motors_bus.find_offset()

        # For each motor, the zero offset is (present_value - 2047).
        expected_calls = []
        for i, name in enumerate(motor_names):
            present_pos = read_values[i][name]
            expected_offset = present_pos - 2047
            # Each call to set_offset(...) is recorded as a tuple of args: (offset, name)
            expected_calls.append(((expected_offset, name),))
        motors_bus.set_offset.assert_has_calls(expected_calls, any_order=False)


def test_find_min_max(mock_motors, dummy_motors):
    motors_bus = FeetechMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    motors_bus.connect()
    motors_bus.motor_names = list(dummy_motors.keys())
    read_side_effect = [
        {"dummy_1": 10, "dummy_2": 20, "dummy_3": 30},
        {"dummy_1": 4000, "dummy_2": 2000, "dummy_3": 100},
        {"dummy_1": 100, "dummy_2": 4050, "dummy_3": 2010},
    ]
    motors_bus.sync_read = Mock(side_effect=read_side_effect)
    motors_bus.set_min_max = Mock()

    select_returns = [
        ([], [], []),
        ([], [], []),
        ([sys.stdin], [], []),  # triggers break from loop
    ]
    with patch("select.select", side_effect=select_returns), patch("sys.stdin.readline", return_value="\n"):
        motors_bus.find_min_max()

    motors_bus.set_min_max.assert_any_call(10, 4000, "dummy_1")
    motors_bus.set_min_max.assert_any_call(
        0, 4095, "dummy_2"
    )  # Difference is between min and max > 4000 thus set 0 and 4095
    motors_bus.set_min_max.assert_any_call(30, 2010, "dummy_3")
    assert motors_bus.set_min_max.call_count == 3


def test_set_offset_clamping(mock_motors, dummy_motors):
    motors_bus = FeetechMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    motors_bus.connect()
    motors_bus.write = Mock()
    # A very large offset should be clamped to +2047.
    motors_bus.set_offset(9999, "dummy_1")

    motors_bus.write.assert_any_call("Lock", "dummy_1", 0)
    motors_bus.write.assert_any_call("Offset", "dummy_1", 2047)
    motors_bus.write.assert_any_call("Lock", "dummy_1", 1)


def test_set_min_max(mock_motors, dummy_motors):
    motors_bus = FeetechMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    motors_bus.connect()
    motors_bus.write = Mock()
    motors_bus.set_min_max(100, 3000, "dummy_1")

    motors_bus.write.assert_any_call("Lock", "dummy_1", 0)
    motors_bus.write.assert_any_call("Min_Angle_Limit", "dummy_1", 100)
    motors_bus.write.assert_any_call("Max_Angle_Limit", "dummy_1", 3000)
    motors_bus.write.assert_any_call("Lock", "dummy_1", 1)
