import sys
from typing import Generator
from unittest.mock import patch

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
    assert FeetechMotorsBus.split_int_bytes(value, n_bytes) == expected


def test_split_int_bytes_invalid_n_bytes():
    with pytest.raises(NotImplementedError):
        FeetechMotorsBus.split_int_bytes(100, 3)


def test_split_int_bytes_negative_numbers():
    with pytest.raises(ValueError):
        neg = FeetechMotorsBus.split_int_bytes(-1, 1)
        print(neg)


def test_split_int_bytes_large_number():
    with pytest.raises(ValueError):
        FeetechMotorsBus.split_int_bytes(2**32, 4)  # 4-byte max is 0xFFFFFFFF


def test_abc_implementation(dummy_motors):
    """Instantiation should raise an error if the class doesn't implement abstract methods/properties."""
    FeetechMotorsBus(port="/dev/dummy-port", motors=dummy_motors)


@pytest.mark.skip("TODO")
@pytest.mark.parametrize(
    "idx, model_nb",
    [
        (1, 1190),
        (2, 1200),
        (3, 1120),
    ],
)
def test_ping(idx, model_nb, mock_motors, dummy_motors):
    stub_name = mock_motors.build_ping_stub(idx, model_nb)
    motors_bus = FeetechMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    motors_bus.connect()

    ping_model_nb = motors_bus.ping(idx)

    assert ping_model_nb == model_nb
    assert mock_motors.stubs[stub_name].called


@pytest.mark.skip("TODO")
def test_broadcast_ping(mock_motors, dummy_motors):
    expected_pings = {
        1: [1060, 50],
        2: [1120, 30],
        3: [1190, 10],
    }
    stub_name = mock_motors.build_broadcast_ping_stub(expected_pings)
    motors_bus = FeetechMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    motors_bus.connect()

    ping_list = motors_bus.broadcast_ping()

    assert ping_list == expected_pings
    assert mock_motors.stubs[stub_name].called


def test_sync_read_none(mock_motors, dummy_motors):
    expected_positions = {
        "dummy_1": 1337,
        "dummy_2": 42,
        "dummy_3": 4016,
    }
    ids_values = dict(zip([1, 2, 3], expected_positions.values(), strict=True))
    stub_name = mock_motors.build_sync_read_stub("Present_Position", ids_values)
    motors_bus = FeetechMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    motors_bus.connect()

    read_positions = motors_bus.sync_read("Present_Position")

    assert mock_motors.stubs[stub_name].called
    assert read_positions == expected_positions


@pytest.mark.parametrize(
    "id_, position",
    [
        (1, 1337),
        (2, 42),
        (3, 4016),
    ],
)
def test_sync_read_by_id(id_, position, mock_motors, dummy_motors):
    expected_position = {id_: position}
    stub_name = mock_motors.build_sync_read_stub("Present_Position", expected_position)
    motors_bus = FeetechMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    motors_bus.connect()

    read_position = motors_bus.sync_read("Present_Position", id_)

    assert mock_motors.stubs[stub_name].called
    assert read_position == expected_position


@pytest.mark.parametrize(
    "ids, positions",
    [
        ([1],       [1337]),
        ([1, 2],    [1337, 42]),
        ([1, 2, 3], [1337, 42, 4016]),
    ],
    ids=["1 motor", "2 motors", "3 motors"],
)  # fmt: skip
def test_sync_read_by_ids(ids, positions, mock_motors, dummy_motors):
    assert len(ids) == len(positions)
    expected_positions = dict(zip(ids, positions, strict=True))
    stub_name = mock_motors.build_sync_read_stub("Present_Position", expected_positions)
    motors_bus = FeetechMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    motors_bus.connect()

    read_positions = motors_bus.sync_read("Present_Position", ids)

    assert mock_motors.stubs[stub_name].called
    assert read_positions == expected_positions


@pytest.mark.parametrize(
    "id_, position",
    [
        (1, 1337),
        (2, 42),
        (3, 4016),
    ],
)
def test_sync_read_by_name(id_, position, mock_motors, dummy_motors):
    expected_position = {f"dummy_{id_}": position}
    stub_name = mock_motors.build_sync_read_stub("Present_Position", {id_: position})
    motors_bus = FeetechMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    motors_bus.connect()

    read_position = motors_bus.sync_read("Present_Position", f"dummy_{id_}")

    assert mock_motors.stubs[stub_name].called
    assert read_position == expected_position


@pytest.mark.parametrize(
    "ids, positions",
    [
        ([1],       [1337]),
        ([1, 2],    [1337, 42]),
        ([1, 2, 3], [1337, 42, 4016]),
    ],
    ids=["1 motor", "2 motors", "3 motors"],
)  # fmt: skip
def test_sync_read_by_names(ids, positions, mock_motors, dummy_motors):
    assert len(ids) == len(positions)
    names = [f"dummy_{dxl_id}" for dxl_id in ids]
    expected_positions = dict(zip(names, positions, strict=True))
    ids_values = dict(zip(ids, positions, strict=True))
    stub_name = mock_motors.build_sync_read_stub("Present_Position", ids_values)
    motors_bus = FeetechMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    motors_bus.connect()

    read_positions = motors_bus.sync_read("Present_Position", names)

    assert mock_motors.stubs[stub_name].called
    assert read_positions == expected_positions


@pytest.mark.parametrize(
    "num_retry, num_invalid_try, pos",
    [
        (0, 2, 1337),
        (2, 3, 42),
        (3, 2, 4016),
        (2, 1, 999),
    ],
)
def test_sync_read_num_retry(num_retry, num_invalid_try, pos, mock_motors, dummy_motors):
    expected_position = {1: pos}
    stub_name = mock_motors.build_sync_read_stub(
        "Present_Position", expected_position, num_invalid_try=num_invalid_try
    )
    motors_bus = FeetechMotorsBus(
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
    "data_name, value",
    [
        ("Torque_Enable", 0),
        ("Torque_Enable", 1),
        ("Goal_Position", 1337),
        ("Goal_Position", 42),
    ],
)
def test_sync_write_single_value(data_name, value, mock_motors, dummy_motors):
    ids_values = {m.id: value for m in dummy_motors.values()}
    stub_name = mock_motors.build_sync_write_stub(data_name, ids_values)
    motors_bus = FeetechMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    motors_bus.connect()

    motors_bus.sync_write(data_name, value)

    assert mock_motors.stubs[stub_name].wait_called()


@pytest.mark.parametrize(
    "id_, position",
    [
        (1, 1337),
        (2, 42),
        (3, 4016),
    ],
)
def test_sync_write_by_id(id_, position, mock_motors, dummy_motors):
    value = {id_: position}
    stub_name = mock_motors.build_sync_write_stub("Goal_Position", value)
    motors_bus = FeetechMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    motors_bus.connect()

    motors_bus.sync_write("Goal_Position", value)

    assert mock_motors.stubs[stub_name].wait_called()


@pytest.mark.parametrize(
    "ids, positions",
    [
        ([1],       [1337]),
        ([1, 2],    [1337, 42]),
        ([1, 2, 3], [1337, 42, 4016]),
    ],
    ids=["1 motor", "2 motors", "3 motors"],
)  # fmt: skip
def test_sync_write_by_ids(ids, positions, mock_motors, dummy_motors):
    assert len(ids) == len(positions)
    values = dict(zip(ids, positions, strict=True))
    stub_name = mock_motors.build_sync_write_stub("Goal_Position", values)
    motors_bus = FeetechMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    motors_bus.connect()

    motors_bus.sync_write("Goal_Position", values)

    assert mock_motors.stubs[stub_name].wait_called()


@pytest.mark.parametrize(
    "id_, position",
    [
        (1, 1337),
        (2, 42),
        (3, 4016),
    ],
)
def test_sync_write_by_name(id_, position, mock_motors, dummy_motors):
    id_value = {id_: position}
    stub_name = mock_motors.build_sync_write_stub("Goal_Position", id_value)
    motors_bus = FeetechMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    motors_bus.connect()

    write_value = {f"dummy_{id_}": position}
    motors_bus.sync_write("Goal_Position", write_value)

    assert mock_motors.stubs[stub_name].wait_called()


@pytest.mark.parametrize(
    "ids, positions",
    [
        ([1],       [1337]),
        ([1, 2],    [1337, 42]),
        ([1, 2, 3], [1337, 42, 4016]),
    ],
    ids=["1 motor", "2 motors", "3 motors"],
)  # fmt: skip
def test_sync_write_by_names(ids, positions, mock_motors, dummy_motors):
    assert len(ids) == len(positions)
    ids_values = dict(zip(ids, positions, strict=True))
    stub_name = mock_motors.build_sync_write_stub("Goal_Position", ids_values)
    motors_bus = FeetechMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    motors_bus.connect()

    write_values = {f"dummy_{id_}": pos for id_, pos in ids_values.items()}
    motors_bus.sync_write("Goal_Position", write_values)

    assert mock_motors.stubs[stub_name].wait_called()


@pytest.mark.parametrize(
    "data_name, dxl_id, value",
    [
        ("Torque_Enable", 1, 0),
        ("Torque_Enable", 1, 1),
        ("Goal_Position", 2, 1337),
        ("Goal_Position", 3, 42),
    ],
)
def test_write_by_id(data_name, dxl_id, value, mock_motors, dummy_motors):
    stub_name = mock_motors.build_write_stub(data_name, dxl_id, value)
    motors_bus = FeetechMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    motors_bus.connect()

    motors_bus.write(data_name, dxl_id, value)

    assert mock_motors.stubs[stub_name].called


@pytest.mark.parametrize(
    "data_name, dxl_id, value",
    [
        ("Torque_Enable", 1, 0),
        ("Torque_Enable", 1, 1),
        ("Goal_Position", 2, 1337),
        ("Goal_Position", 3, 42),
    ],
)
def test_write_by_name(data_name, dxl_id, value, mock_motors, dummy_motors):
    stub_name = mock_motors.build_write_stub(data_name, dxl_id, value)
    motors_bus = FeetechMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    motors_bus.connect()

    motors_bus.write(data_name, f"dummy_{dxl_id}", value)

    assert mock_motors.stubs[stub_name].called
