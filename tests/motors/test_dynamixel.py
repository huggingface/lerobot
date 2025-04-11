import sys
from typing import Generator
from unittest.mock import MagicMock, patch

import dynamixel_sdk as dxl
import pytest

from lerobot.common.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.common.motors.dynamixel import MODEL_NUMBER_TABLE, DynamixelMotorsBus
from lerobot.common.utils.encoding_utils import encode_twos_complement
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
        "dummy_1": Motor(1, "xl430-w250", MotorNormMode.RANGE_M100_100),
        "dummy_2": Motor(2, "xm540-w270", MotorNormMode.RANGE_M100_100),
        "dummy_3": Motor(3, "xl330-m077", MotorNormMode.RANGE_M100_100),
    }


@pytest.fixture
def dummy_calibration(dummy_motors) -> dict[str, MotorCalibration]:
    drive_modes = [0, 1, 0]
    homings = [-709, -2006, 1624]
    mins = [43, 27, 145]
    maxes = [1335, 3608, 3999]
    calibration = {}
    for name, motor in dummy_motors.items():
        calibration[name] = MotorCalibration(
            id=motor.id,
            drive_mode=drive_modes[motor.id - 1],
            homing_offset=homings[motor.id - 1],
            range_min=mins[motor.id - 1],
            range_max=maxes[motor.id - 1],
        )
    return calibration


@pytest.mark.skipif(sys.platform != "darwin", reason=f"No patching needed on {sys.platform=}")
def test_autouse_patch():
    """Ensures that the autouse fixture correctly patches dxl.PortHandler with MockPortHandler."""
    assert dxl.PortHandler is MockPortHandler


@pytest.mark.parametrize(
    "value, length, expected",
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
def test_serialize_data(value, length, expected):
    assert DynamixelMotorsBus._serialize_data(value, length) == expected


def test_serialize_data_invalid_length():
    with pytest.raises(NotImplementedError):
        DynamixelMotorsBus._serialize_data(100, 3)


def test_serialize_data_negative_numbers():
    with pytest.raises(ValueError):
        neg = DynamixelMotorsBus._serialize_data(-1, 1)
        print(neg)


def test_serialize_data_large_number():
    with pytest.raises(ValueError):
        DynamixelMotorsBus._serialize_data(2**32, 4)  # 4-byte max is 0xFFFFFFFF


def test_abc_implementation(dummy_motors):
    """Instantiation should raise an error if the class doesn't implement abstract methods/properties."""
    DynamixelMotorsBus(port="/dev/dummy-port", motors=dummy_motors)


@pytest.mark.parametrize("id_", [1, 2, 3])
def test_ping(id_, mock_motors, dummy_motors):
    expected_model_nb = MODEL_NUMBER_TABLE[dummy_motors[f"dummy_{id_}"].model]
    stub_name = mock_motors.build_ping_stub(id_, expected_model_nb)
    motors_bus = DynamixelMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    motors_bus.connect(assert_motors_exist=False)

    ping_model_nb = motors_bus.ping(id_)

    assert ping_model_nb == expected_model_nb
    assert mock_motors.stubs[stub_name].called


def test_broadcast_ping(mock_motors, dummy_motors):
    models = {m.id: m.model for m in dummy_motors.values()}
    expected_model_nbs = {id_: MODEL_NUMBER_TABLE[model] for id_, model in models.items()}
    stub_name = mock_motors.build_broadcast_ping_stub(expected_model_nbs)
    motors_bus = DynamixelMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    motors_bus.connect(assert_motors_exist=False)

    ping_model_nbs = motors_bus.broadcast_ping()

    assert ping_model_nbs == expected_model_nbs
    assert mock_motors.stubs[stub_name].called


def test_sync_read_none(mock_motors, dummy_motors):
    expected_positions = {
        "dummy_1": 1337,
        "dummy_2": 42,
        "dummy_3": 4016,
    }
    ids_values = dict(zip([1, 2, 3], expected_positions.values(), strict=True))
    stub_name = mock_motors.build_sync_read_stub("Present_Position", ids_values)
    motors_bus = DynamixelMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    motors_bus.connect(assert_motors_exist=False)

    read_positions = motors_bus.sync_read("Present_Position", normalize=False)

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
def test_sync_read_single_value(id_, position, mock_motors, dummy_motors):
    expected_position = {f"dummy_{id_}": position}
    stub_name = mock_motors.build_sync_read_stub("Present_Position", {id_: position})
    motors_bus = DynamixelMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    motors_bus.connect(assert_motors_exist=False)

    read_position = motors_bus.sync_read("Present_Position", f"dummy_{id_}", normalize=False)

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
def test_sync_read(ids, positions, mock_motors, dummy_motors):
    assert len(ids) == len(positions)
    names = [f"dummy_{dxl_id}" for dxl_id in ids]
    expected_positions = dict(zip(names, positions, strict=True))
    ids_values = dict(zip(ids, positions, strict=True))
    stub_name = mock_motors.build_sync_read_stub("Present_Position", ids_values)
    motors_bus = DynamixelMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    motors_bus.connect(assert_motors_exist=False)

    read_positions = motors_bus.sync_read("Present_Position", names, normalize=False)

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
    expected_position = {"dummy_1": pos}
    stub_name = mock_motors.build_sync_read_stub(
        "Present_Position", {1: pos}, num_invalid_try=num_invalid_try
    )
    motors_bus = DynamixelMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    motors_bus.connect(assert_motors_exist=False)

    if num_retry >= num_invalid_try:
        pos_dict = motors_bus.sync_read("Present_Position", "dummy_1", normalize=False, num_retry=num_retry)
        assert pos_dict == expected_position
    else:
        with pytest.raises(ConnectionError):
            _ = motors_bus.sync_read("Present_Position", "dummy_1", normalize=False, num_retry=num_retry)

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
    motors_bus = DynamixelMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    motors_bus.connect(assert_motors_exist=False)

    motors_bus.sync_write(data_name, value, normalize=False)

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
def test_sync_write(ids, positions, mock_motors, dummy_motors):
    assert len(ids) == len(positions)
    ids_values = dict(zip(ids, positions, strict=True))
    stub_name = mock_motors.build_sync_write_stub("Goal_Position", ids_values)
    motors_bus = DynamixelMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    motors_bus.connect(assert_motors_exist=False)

    write_values = {f"dummy_{id_}": pos for id_, pos in ids_values.items()}
    motors_bus.sync_write("Goal_Position", write_values, normalize=False)

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
def test_write(data_name, dxl_id, value, mock_motors, dummy_motors):
    stub_name = mock_motors.build_write_stub(data_name, dxl_id, value)
    motors_bus = DynamixelMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    motors_bus.connect(assert_motors_exist=False)

    motors_bus.write(data_name, f"dummy_{dxl_id}", value, normalize=False)

    assert mock_motors.stubs[stub_name].called


def test_is_calibrated(mock_motors, dummy_motors, dummy_calibration):
    drive_modes = {m.id: m.drive_mode for m in dummy_calibration.values()}
    encoded_homings = {m.id: encode_twos_complement(m.homing_offset, 4) for m in dummy_calibration.values()}
    mins = {m.id: m.range_min for m in dummy_calibration.values()}
    maxes = {m.id: m.range_max for m in dummy_calibration.values()}
    drive_modes_stub = mock_motors.build_sync_read_stub("Drive_Mode", drive_modes)
    offsets_stub = mock_motors.build_sync_read_stub("Homing_Offset", encoded_homings)
    mins_stub = mock_motors.build_sync_read_stub("Min_Position_Limit", mins)
    maxes_stub = mock_motors.build_sync_read_stub("Max_Position_Limit", maxes)
    motors_bus = DynamixelMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
        calibration=dummy_calibration,
    )
    motors_bus.connect(assert_motors_exist=False)

    is_calibrated = motors_bus.is_calibrated

    assert is_calibrated
    assert mock_motors.stubs[drive_modes_stub].called
    assert mock_motors.stubs[offsets_stub].called
    assert mock_motors.stubs[mins_stub].called
    assert mock_motors.stubs[maxes_stub].called


def test_reset_calibration(mock_motors, dummy_motors):
    write_homing_stubs = []
    write_mins_stubs = []
    write_maxes_stubs = []
    for motor in dummy_motors.values():
        write_homing_stubs.append(mock_motors.build_write_stub("Homing_Offset", motor.id, 0))
        write_mins_stubs.append(mock_motors.build_write_stub("Min_Position_Limit", motor.id, 0))
        write_maxes_stubs.append(mock_motors.build_write_stub("Max_Position_Limit", motor.id, 4095))

    motors_bus = DynamixelMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    motors_bus.connect(assert_motors_exist=False)

    motors_bus.reset_calibration()

    assert all(mock_motors.stubs[stub].called for stub in write_homing_stubs)
    assert all(mock_motors.stubs[stub].called for stub in write_mins_stubs)
    assert all(mock_motors.stubs[stub].called for stub in write_maxes_stubs)


def test_set_half_turn_homings(mock_motors, dummy_motors):
    """
    For this test, we assume that the homing offsets are already 0 such that
    Present_Position == Actual_Position
    """
    current_positions = {
        1: 1337,
        2: 42,
        3: 3672,
    }
    expected_homings = {
        1: 710,  # 2047 - 1337
        2: 2005,  # 2047 - 42
        3: -1625,  # 2047 - 3672
    }
    read_pos_stub = mock_motors.build_sync_read_stub("Present_Position", current_positions)
    write_homing_stubs = []
    for id_, homing in expected_homings.items():
        encoded_homing = encode_twos_complement(homing, 4)
        stub = mock_motors.build_write_stub("Homing_Offset", id_, encoded_homing)
        write_homing_stubs.append(stub)

    motors_bus = DynamixelMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    motors_bus.connect(assert_motors_exist=False)
    motors_bus.reset_calibration = MagicMock()

    motors_bus.set_half_turn_homings()

    motors_bus.reset_calibration.assert_called_once()
    assert mock_motors.stubs[read_pos_stub].called
    assert all(mock_motors.stubs[stub].called for stub in write_homing_stubs)


def test_record_ranges_of_motion(mock_motors, dummy_motors):
    positions = {
        1: [351, 42, 1337],
        2: [28, 3600, 2444],
        3: [4002, 2999, 146],
    }
    expected_mins = {
        "dummy_1": 42,
        "dummy_2": 28,
        "dummy_3": 146,
    }
    expected_maxes = {
        "dummy_1": 1337,
        "dummy_2": 3600,
        "dummy_3": 4002,
    }
    read_pos_stub = mock_motors.build_sequential_sync_read_stub("Present_Position", positions)
    with patch("lerobot.common.motors.motors_bus.enter_pressed", side_effect=[False, True]):
        motors_bus = DynamixelMotorsBus(
            port=mock_motors.port,
            motors=dummy_motors,
        )
        motors_bus.connect(assert_motors_exist=False)

        mins, maxes = motors_bus.record_ranges_of_motion(display_values=False)

    assert mock_motors.stubs[read_pos_stub].calls == 3
    assert mins == expected_mins
    assert maxes == expected_maxes
