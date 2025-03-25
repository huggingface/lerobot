import sys
from typing import Generator
from unittest.mock import Mock, patch

import pytest
import scservo_sdk as scs

from lerobot.common.motors import CalibrationMode, Motor
from lerobot.common.motors.calibration import find_min_max, find_offset, set_min_max, set_offset
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
        "wrist_roll": Motor(2, "sts3215", CalibrationMode.RANGE_M100_100),
        "dummy_3": Motor(3, "sts3215", CalibrationMode.RANGE_M100_100),
    }


@pytest.fixture(autouse=True)
def patch_broadcast_ping():
    with patch.object(FeetechMotorsBus, "broadcast_ping", return_value={1: 777, 2: 777, 3: 777}):
        yield


@pytest.mark.skipif(sys.platform != "darwin", reason=f"No patching needed on {sys.platform=}")
def test_autouse_patch():
    """Ensures that the autouse fixture correctly patches scs.PortHandler with MockPortHandler."""
    assert scs.PortHandler is MockPortHandler


@pytest.mark.parametrize(
    "motor_names, read_values",
    [
        (
            ["dummy_1"],
            [{"dummy_1": 3000}],
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
        with patch("lerobot.common.motors.calibration.set_offset") as mock_set_offset:
            find_offset(motors_bus)
            # Compute the expected offset: 3000 - 2047 = 953.
            expected_calls = [((motors_bus, 953, "dummy_1"),)]
            mock_set_offset.assert_has_calls(expected_calls, any_order=False)


def test_find_min_max(mock_motors, dummy_motors):
    motors_bus = FeetechMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    motors_bus.connect()
    motors_bus.motor_names = list(dummy_motors.keys())
    read_side_effect = [
        {"dummy_1": 10, "wrist_roll": 20, "dummy_3": 30},
        {"dummy_1": 4000, "wrist_roll": 2000, "dummy_3": 100},
        {"dummy_1": 100, "wrist_roll": 4050, "dummy_3": 2010},
    ]
    motors_bus.sync_read = Mock(side_effect=read_side_effect)

    select_returns = [
        ([], [], []),  # First iteration: no input.
        ([], [], []),  # Second iteration.
        ([sys.stdin], [], []),  # Third iteration: simulate pressing ENTER.
    ]
    with (
        patch("lerobot.common.motors.calibration.set_min_max") as mock_set_min_max,
        patch("lerobot.common.motors.calibration.select.select", side_effect=select_returns),
        patch("sys.stdin.readline", return_value="\n"),
    ):
        find_min_max(motors_bus)

    mock_set_min_max.assert_any_call(motors_bus, 10, 4000, "dummy_1")
    mock_set_min_max.assert_any_call(motors_bus, 0, 4095, "wrist_roll")  # wrist_roll is forced to [0,4095]
    mock_set_min_max.assert_any_call(motors_bus, 30, 2010, "dummy_3")
    assert mock_set_min_max.call_count == 3


def test_set_offset_clamping(mock_motors, dummy_motors):
    motors_bus = FeetechMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    motors_bus.connect()
    motors_bus.sync_read = Mock(return_value={"dummy_1": 2047})
    motors_bus.write = Mock()
    # A very large offset should be clamped to +2047.
    set_offset(motors_bus, 9999, "dummy_1")
    motors_bus.write.assert_any_call("Offset", "dummy_1", 2047, raw_value=True)


def test_set_min_max(mock_motors, dummy_motors):
    motors_bus = FeetechMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    motors_bus.connect()

    def _sync_read_side_effect(data_name, motors, *, raw_values=False):
        if data_name == "Min_Angle_Limit":
            return {"dummy_1": 100}
        elif data_name == "Max_Angle_Limit":
            return {"dummy_1": 3000}
        return {}

    motors_bus.sync_read = Mock(side_effect=_sync_read_side_effect)

    motors_bus.write = Mock()
    set_min_max(motors_bus, 100, 3000, "dummy_1")
    motors_bus.write.assert_any_call("Min_Angle_Limit", "dummy_1", 100, raw_value=True)
    motors_bus.write.assert_any_call("Max_Angle_Limit", "dummy_1", 3000, raw_value=True)
