import re
import sys
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.dynamixel import MODEL_NUMBER_TABLE, DynamixelMotorsBus
from lerobot.motors.dynamixel.tables import X_SERIES_CONTROL_TABLE
from lerobot.utils.encoding_utils import encode_twos_complement

try:
    import dynamixel_sdk as dxl

    from tests.mocks.mock_dynamixel import MockMotors, MockPortHandler
except (ImportError, ModuleNotFoundError):
    pytest.skip("dynamixel_sdk not available", allow_module_level=True)


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
    for motor, m in dummy_motors.items():
        calibration[motor] = MotorCalibration(
            id=m.id,
            drive_mode=drive_modes[m.id - 1],
            homing_offset=homings[m.id - 1],
            range_min=mins[m.id - 1],
            range_max=maxes[m.id - 1],
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
    ],
    ids=[
        "1 byte",
        "2 bytes",
        "4 bytes",
    ],
)  # fmt: skip
def test__split_into_byte_chunks(value, length, expected):
    bus = DynamixelMotorsBus("", {})
    assert bus._split_into_byte_chunks(value, length) == expected


def test_abc_implementation(dummy_motors):
    """Instantiation should raise an error if the class doesn't implement abstract methods/properties."""
    DynamixelMotorsBus(port="/dev/dummy-port", motors=dummy_motors)


@pytest.mark.parametrize("id_", [1, 2, 3])
def test_ping(id_, mock_motors, dummy_motors):
    expected_model_nb = MODEL_NUMBER_TABLE[dummy_motors[f"dummy_{id_}"].model]
    stub = mock_motors.build_ping_stub(id_, expected_model_nb)
    bus = DynamixelMotorsBus(port=mock_motors.port, motors=dummy_motors)
    bus.connect(handshake=False)

    ping_model_nb = bus.ping(id_)

    assert ping_model_nb == expected_model_nb
    assert mock_motors.stubs[stub].called


def test_broadcast_ping(mock_motors, dummy_motors):
    models = {m.id: m.model for m in dummy_motors.values()}
    expected_model_nbs = {id_: MODEL_NUMBER_TABLE[model] for id_, model in models.items()}
    stub = mock_motors.build_broadcast_ping_stub(expected_model_nbs)
    bus = DynamixelMotorsBus(port=mock_motors.port, motors=dummy_motors)
    bus.connect(handshake=False)

    ping_model_nbs = bus.broadcast_ping()

    assert ping_model_nbs == expected_model_nbs
    assert mock_motors.stubs[stub].called


@pytest.mark.parametrize(
    "addr, length, id_, value",
    [
        (0, 1, 1, 2),
        (10, 2, 2, 999),
        (42, 4, 3, 1337),
    ],
)
def test__read(addr, length, id_, value, mock_motors, dummy_motors):
    stub = mock_motors.build_read_stub(addr, length, id_, value)
    bus = DynamixelMotorsBus(port=mock_motors.port, motors=dummy_motors)
    bus.connect(handshake=False)

    read_value, _, _ = bus._read(addr, length, id_)

    assert mock_motors.stubs[stub].called
    assert read_value == value


@pytest.mark.parametrize("raise_on_error", (True, False))
def test__read_error(raise_on_error, mock_motors, dummy_motors):
    addr, length, id_, value, error = (10, 4, 1, 1337, dxl.ERRNUM_DATA_LIMIT)
    stub = mock_motors.build_read_stub(addr, length, id_, value, error=error)
    bus = DynamixelMotorsBus(port=mock_motors.port, motors=dummy_motors)
    bus.connect(handshake=False)

    if raise_on_error:
        with pytest.raises(
            RuntimeError, match=re.escape("[RxPacketError] The data value exceeds the limit value!")
        ):
            bus._read(addr, length, id_, raise_on_error=raise_on_error)
    else:
        _, _, read_error = bus._read(addr, length, id_, raise_on_error=raise_on_error)
        assert read_error == error

    assert mock_motors.stubs[stub].called


@pytest.mark.parametrize("raise_on_error", (True, False))
def test__read_comm(raise_on_error, mock_motors, dummy_motors):
    addr, length, id_, value = (10, 4, 1, 1337)
    stub = mock_motors.build_read_stub(addr, length, id_, value, reply=False)
    bus = DynamixelMotorsBus(port=mock_motors.port, motors=dummy_motors)
    bus.connect(handshake=False)

    if raise_on_error:
        with pytest.raises(ConnectionError, match=re.escape("[TxRxResult] There is no status packet!")):
            bus._read(addr, length, id_, raise_on_error=raise_on_error)
    else:
        _, read_comm, _ = bus._read(addr, length, id_, raise_on_error=raise_on_error)
        assert read_comm == dxl.COMM_RX_TIMEOUT

    assert mock_motors.stubs[stub].called


@pytest.mark.parametrize(
    "addr, length, id_, value",
    [
        (0, 1, 1, 2),
        (10, 2, 2, 999),
        (42, 4, 3, 1337),
    ],
)
def test__write(addr, length, id_, value, mock_motors, dummy_motors):
    stub = mock_motors.build_write_stub(addr, length, id_, value)
    bus = DynamixelMotorsBus(port=mock_motors.port, motors=dummy_motors)
    bus.connect(handshake=False)

    comm, error = bus._write(addr, length, id_, value)

    assert mock_motors.stubs[stub].called
    assert comm == dxl.COMM_SUCCESS
    assert error == 0


@pytest.mark.parametrize("raise_on_error", (True, False))
def test__write_error(raise_on_error, mock_motors, dummy_motors):
    addr, length, id_, value, error = (10, 4, 1, 1337, dxl.ERRNUM_DATA_LIMIT)
    stub = mock_motors.build_write_stub(addr, length, id_, value, error=error)
    bus = DynamixelMotorsBus(port=mock_motors.port, motors=dummy_motors)
    bus.connect(handshake=False)

    if raise_on_error:
        with pytest.raises(
            RuntimeError, match=re.escape("[RxPacketError] The data value exceeds the limit value!")
        ):
            bus._write(addr, length, id_, value, raise_on_error=raise_on_error)
    else:
        _, write_error = bus._write(addr, length, id_, value, raise_on_error=raise_on_error)
        assert write_error == error

    assert mock_motors.stubs[stub].called


@pytest.mark.parametrize("raise_on_error", (True, False))
def test__write_comm(raise_on_error, mock_motors, dummy_motors):
    addr, length, id_, value = (10, 4, 1, 1337)
    stub = mock_motors.build_write_stub(addr, length, id_, value, reply=False)
    bus = DynamixelMotorsBus(port=mock_motors.port, motors=dummy_motors)
    bus.connect(handshake=False)

    if raise_on_error:
        with pytest.raises(ConnectionError, match=re.escape("[TxRxResult] There is no status packet!")):
            bus._write(addr, length, id_, value, raise_on_error=raise_on_error)
    else:
        write_comm, _ = bus._write(addr, length, id_, value, raise_on_error=raise_on_error)
        assert write_comm == dxl.COMM_RX_TIMEOUT

    assert mock_motors.stubs[stub].called


@pytest.mark.parametrize(
    "addr, length, ids_values",
    [
        (0, 1, {1: 4}),
        (10, 2, {1: 1337, 2: 42}),
        (42, 4, {1: 1337, 2: 42, 3: 4016}),
    ],
    ids=["1 motor", "2 motors", "3 motors"],
)
def test__sync_read(addr, length, ids_values, mock_motors, dummy_motors):
    stub = mock_motors.build_sync_read_stub(addr, length, ids_values)
    bus = DynamixelMotorsBus(port=mock_motors.port, motors=dummy_motors)
    bus.connect(handshake=False)

    read_values, _ = bus._sync_read(addr, length, list(ids_values))

    assert mock_motors.stubs[stub].called
    assert read_values == ids_values


@pytest.mark.parametrize("raise_on_error", (True, False))
def test__sync_read_comm(raise_on_error, mock_motors, dummy_motors):
    addr, length, ids_values = (10, 4, {1: 1337})
    stub = mock_motors.build_sync_read_stub(addr, length, ids_values, reply=False)
    bus = DynamixelMotorsBus(port=mock_motors.port, motors=dummy_motors)
    bus.connect(handshake=False)

    if raise_on_error:
        with pytest.raises(ConnectionError, match=re.escape("[TxRxResult] There is no status packet!")):
            bus._sync_read(addr, length, list(ids_values), raise_on_error=raise_on_error)
    else:
        _, read_comm = bus._sync_read(addr, length, list(ids_values), raise_on_error=raise_on_error)
        assert read_comm == dxl.COMM_RX_TIMEOUT

    assert mock_motors.stubs[stub].called


@pytest.mark.parametrize(
    "addr, length, ids_values",
    [
        (0, 1, {1: 4}),
        (10, 2, {1: 1337, 2: 42}),
        (42, 4, {1: 1337, 2: 42, 3: 4016}),
    ],
    ids=["1 motor", "2 motors", "3 motors"],
)
def test__sync_write(addr, length, ids_values, mock_motors, dummy_motors):
    stub = mock_motors.build_sync_write_stub(addr, length, ids_values)
    bus = DynamixelMotorsBus(port=mock_motors.port, motors=dummy_motors)
    bus.connect(handshake=False)

    comm = bus._sync_write(addr, length, ids_values)

    assert mock_motors.stubs[stub].wait_called()
    assert comm == dxl.COMM_SUCCESS


def test_is_calibrated(mock_motors, dummy_motors, dummy_calibration):
    drive_modes = {m.id: m.drive_mode for m in dummy_calibration.values()}
    encoded_homings = {m.id: encode_twos_complement(m.homing_offset, 4) for m in dummy_calibration.values()}
    mins = {m.id: m.range_min for m in dummy_calibration.values()}
    maxes = {m.id: m.range_max for m in dummy_calibration.values()}
    drive_modes_stub = mock_motors.build_sync_read_stub(*X_SERIES_CONTROL_TABLE["Drive_Mode"], drive_modes)
    offsets_stub = mock_motors.build_sync_read_stub(*X_SERIES_CONTROL_TABLE["Homing_Offset"], encoded_homings)
    mins_stub = mock_motors.build_sync_read_stub(*X_SERIES_CONTROL_TABLE["Min_Position_Limit"], mins)
    maxes_stub = mock_motors.build_sync_read_stub(*X_SERIES_CONTROL_TABLE["Max_Position_Limit"], maxes)
    bus = DynamixelMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
        calibration=dummy_calibration,
    )
    bus.connect(handshake=False)

    is_calibrated = bus.is_calibrated

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
        write_homing_stubs.append(
            mock_motors.build_write_stub(*X_SERIES_CONTROL_TABLE["Homing_Offset"], motor.id, 0)
        )
        write_mins_stubs.append(
            mock_motors.build_write_stub(*X_SERIES_CONTROL_TABLE["Min_Position_Limit"], motor.id, 0)
        )
        write_maxes_stubs.append(
            mock_motors.build_write_stub(*X_SERIES_CONTROL_TABLE["Max_Position_Limit"], motor.id, 4095)
        )

    bus = DynamixelMotorsBus(port=mock_motors.port, motors=dummy_motors)
    bus.connect(handshake=False)

    bus.reset_calibration()

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
    read_pos_stub = mock_motors.build_sync_read_stub(
        *X_SERIES_CONTROL_TABLE["Present_Position"], current_positions
    )
    write_homing_stubs = []
    for id_, homing in expected_homings.items():
        encoded_homing = encode_twos_complement(homing, 4)
        stub = mock_motors.build_write_stub(*X_SERIES_CONTROL_TABLE["Homing_Offset"], id_, encoded_homing)
        write_homing_stubs.append(stub)

    bus = DynamixelMotorsBus(port=mock_motors.port, motors=dummy_motors)
    bus.connect(handshake=False)
    bus.reset_calibration = MagicMock()

    bus.set_half_turn_homings()

    bus.reset_calibration.assert_called_once()
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
    read_pos_stub = mock_motors.build_sequential_sync_read_stub(
        *X_SERIES_CONTROL_TABLE["Present_Position"], positions
    )
    with patch("lerobot.motors.motors_bus.enter_pressed", side_effect=[False, True]):
        bus = DynamixelMotorsBus(port=mock_motors.port, motors=dummy_motors)
        bus.connect(handshake=False)

        mins, maxes = bus.record_ranges_of_motion(display_values=False)

    assert mock_motors.stubs[read_pos_stub].calls == 3
    assert mins == expected_mins
    assert maxes == expected_maxes
