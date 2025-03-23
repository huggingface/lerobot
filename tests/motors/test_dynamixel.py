import sys
from typing import Generator
from unittest.mock import patch

import dynamixel_sdk as dxl
import pytest

from lerobot.common.motors import CalibrationMode, Motor
from lerobot.common.motors.dynamixel import DynamixelMotorsBus
from tests.mocks.mock_dynamixel import MockMotors, MockPortHandler


@pytest.fixture(autouse=True)
def patch_port_handler():
    if sys.platform == "darwin":
        with patch.object(dxl, "PortHandler", MockPortHandler):
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
        "dummy_1": Motor(1, "xl430-w250", CalibrationMode.RANGE_M100_100),
        "dummy_2": Motor(2, "xm540-w270", CalibrationMode.RANGE_M100_100),
        "dummy_3": Motor(3, "xl330-m077", CalibrationMode.RANGE_M100_100),
    }


@pytest.mark.skipif(sys.platform != "darwin", reason=f"No patching needed on {sys.platform=}")
def test_autouse_patch():
    """Ensures that the autouse fixture correctly patches dxl.PortHandler with MockPortHandler."""
    assert dxl.PortHandler is MockPortHandler


@pytest.mark.parametrize(
    "value, n_bytes, expected",
    [
        (0x12,       1, [0x12]),
        (0x1234,     2, [0x34, 0x12]),
        (0x12345678, 4, [0x78, 0x56, 0x34, 0x12]),
        (0,          1, [0x00]),
        (0,          2, [0x00, 0x00]),
        (0,          4, [0x00, 0x00, 0x00, 0x00]),
        (255,        1, [0xFF]),
        (65535,      2, [0xFF, 0xFF]),
        (4294967295, 4, [0xFF, 0xFF, 0xFF, 0xFF]),
    ],
    ids=[
        "1 byte",
        "2 bytes",
        "4 bytes",
        "0 with 1 byte",
        "0 with 2 bytes",
        "0 with 4 bytes",
        "max single byte",
        "max two bytes",
        "max four bytes",
    ],
)  # fmt: skip
def test_split_int_bytes(value, n_bytes, expected):
    assert DynamixelMotorsBus.split_int_bytes(value, n_bytes) == expected


def test_split_int_bytes_invalid_n_bytes():
    with pytest.raises(NotImplementedError):
        DynamixelMotorsBus.split_int_bytes(100, 3)


def test_split_int_bytes_negative_numbers():
    with pytest.raises(ValueError):
        neg = DynamixelMotorsBus.split_int_bytes(-1, 1)
        print(neg)


def test_split_int_bytes_large_number():
    with pytest.raises(ValueError):
        DynamixelMotorsBus.split_int_bytes(2**32, 4)  # 4-byte max is 0xFFFFFFFF


def test_abc_implementation(dummy_motors):
    """Instantiation should raise an error if the class doesn't implement abstract methods/properties."""
    DynamixelMotorsBus(port="/dev/dummy-port", motors=dummy_motors)


@pytest.mark.parametrize(
    "idx, model_nb",
    [
        [1, 1190],
        [2, 1200],
        [3, 1120],
    ],
)
def test_ping(idx, model_nb, mock_motors, dummy_motors):
    stub_name = mock_motors.build_ping_stub(idx, model_nb)
    motors_bus = DynamixelMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    motors_bus.connect()

    ping_model_nb = motors_bus.ping(idx)

    assert ping_model_nb == model_nb
    assert mock_motors.stubs[stub_name].called


def test_broadcast_ping(mock_motors, dummy_motors):
    expected_pings = {
        1: [1060, 50],
        2: [1120, 30],
        3: [1190, 10],
    }
    stub_name = mock_motors.build_broadcast_ping_stub(expected_pings)
    motors_bus = DynamixelMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    motors_bus.connect()

    ping_list = motors_bus.broadcast_ping()

    assert ping_list == expected_pings
    assert mock_motors.stubs[stub_name].called


@pytest.mark.parametrize(
    "motors",
    [
        None,
        [1, 2, 3],
        ["dummy_1", "dummy_2", "dummy_3"],
        [1, "dummy_2", 3],
    ],
    ids=["None", "by ids", "by names", "mixed"],
)
def test_sync_read_all_motors(motors, mock_motors, dummy_motors):
    expected_positions = {
        1: 1337,
        2: 42,
        3: 4016,
    }
    stub_name = mock_motors.build_sync_read_stub("Present_Position", expected_positions)
    motors_bus = DynamixelMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    motors_bus.connect()

    positions_read = motors_bus.sync_read("Present_Position", motors=motors)

    motors = ["dummy_1", "dummy_2", "dummy_3"] if motors is None else motors
    assert mock_motors.stubs[stub_name].called
    assert positions_read == dict(zip(motors, expected_positions.values(), strict=True))


@pytest.mark.parametrize(
    "idx, pos",
    [
        [1, 1337],
        [2, 42],
        [3, 4016],
    ],
)
def test_sync_read_single_motor_by_name(idx, pos, mock_motors, dummy_motors):
    expected_position = {idx: pos}
    stub_name = mock_motors.build_sync_read_stub("Present_Position", expected_position)
    motors_bus = DynamixelMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    motors_bus.connect()

    pos_dict = motors_bus.sync_read("Present_Position", f"dummy_{idx}")

    assert mock_motors.stubs[stub_name].called
    assert pos_dict == {f"dummy_{idx}": pos}


@pytest.mark.parametrize(
    "idx, pos",
    [
        [1, 1337],
        [2, 42],
        [3, 4016],
    ],
)
def test_sync_read_single_motor_by_id(idx, pos, mock_motors, dummy_motors):
    expected_position = {idx: pos}
    stub_name = mock_motors.build_sync_read_stub("Present_Position", expected_position)
    motors_bus = DynamixelMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    motors_bus.connect()

    pos_dict = motors_bus.sync_read("Present_Position", idx)

    assert mock_motors.stubs[stub_name].called
    assert pos_dict == {idx: pos}


@pytest.mark.parametrize(
    "num_retry, num_invalid_try, pos",
    [
        [0, 2, 1337],
        [2, 3, 42],
        [3, 2, 4016],
        [2, 1, 999],
    ],
)
def test_sync_read_num_retry(num_retry, num_invalid_try, pos, mock_motors, dummy_motors):
    expected_position = {1: pos}
    stub_name = mock_motors.build_sync_read_stub(
        "Present_Position", expected_position, num_invalid_try=num_invalid_try
    )
    motors_bus = DynamixelMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    motors_bus.connect()

    if num_retry >= num_invalid_try:
        pos_dict = motors_bus.sync_read("Present_Position", 1, num_retry=num_retry)
        assert pos_dict == {1: pos}
    else:
        with pytest.raises(ConnectionError):
            _ = motors_bus.sync_read("Present_Position", 1, num_retry=num_retry)

    expected_calls = min(1 + num_retry, 1 + num_invalid_try)
    assert mock_motors.stubs[stub_name].calls == expected_calls


@pytest.mark.parametrize(
    "motors",
    [
        [1, 2, 3],
        ["dummy_1", "dummy_2", "dummy_3"],
        [1, "dummy_2", 3],
    ],
    ids=["by ids", "by names", "mixed"],
)
def test_sync_write_all_motors(motors, mock_motors, dummy_motors):
    goal_positions = {
        1: 1337,
        2: 42,
        3: 4016,
    }
    stub_name = mock_motors.build_sync_write_stub("Goal_Position", goal_positions)
    motors_bus = DynamixelMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    motors_bus.connect()

    values = dict(zip(motors, goal_positions.values(), strict=True))
    motors_bus.sync_write("Goal_Position", values)

    assert mock_motors.stubs[stub_name].wait_called()


@pytest.mark.parametrize(
    "data_name, value",
    [
        ["Torque_Enable", 0],
        ["Torque_Enable", 1],
    ],
)
def test_sync_write_all_motors_single_value(data_name, value, mock_motors, dummy_motors):
    values = {m.id: value for m in dummy_motors.values()}
    stub_name = mock_motors.build_sync_write_stub(data_name, values)
    motors_bus = DynamixelMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    motors_bus.connect()

    motors_bus.sync_write(data_name, value)

    assert mock_motors.stubs[stub_name].wait_called()
