import re
import sys
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import MODEL_NUMBER, MODEL_NUMBER_TABLE, FeetechMotorsBus
from lerobot.motors.feetech.tables import STS_SMS_SERIES_CONTROL_TABLE
from lerobot.utils.encoding_utils import encode_sign_magnitude

try:
    import scservo_sdk as scs

    from tests.mocks.mock_feetech import MockMotors, MockPortHandler
except (ImportError, ModuleNotFoundError):
    pytest.skip("scservo_sdk not available", allow_module_level=True)


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
        "dummy_1": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
        "dummy_2": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
        "dummy_3": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
    }


@pytest.fixture
def dummy_calibration(dummy_motors) -> dict[str, MotorCalibration]:
    homings = [-709, -2006, 1624]
    mins = [43, 27, 145]
    maxes = [1335, 3608, 3999]
    calibration = {}
    for motor, m in dummy_motors.items():
        calibration[motor] = MotorCalibration(
            id=m.id,
            drive_mode=0,
            homing_offset=homings[m.id - 1],
            range_min=mins[m.id - 1],
            range_max=maxes[m.id - 1],
        )
    return calibration


@pytest.mark.skipif(sys.platform != "darwin", reason=f"No patching needed on {sys.platform=}")
def test_autouse_patch():
    """Ensures that the autouse fixture correctly patches scs.PortHandler with MockPortHandler."""
    assert scs.PortHandler is MockPortHandler


@pytest.mark.parametrize(
    "protocol, value, length, expected",
    [
        (0, 0x12,       1, [0x12]),
        (1, 0x12,       1, [0x12]),
        (0, 0x1234,     2, [0x34, 0x12]),
        (1, 0x1234,     2, [0x12, 0x34]),
        (0, 0x12345678, 4, [0x78, 0x56, 0x34, 0x12]),
        (1, 0x12345678, 4, [0x56, 0x78, 0x12, 0x34]),
    ],
    ids=[
        "P0: 1 byte",
        "P1: 1 byte",
        "P0: 2 bytes",
        "P1: 2 bytes",
        "P0: 4 bytes",
        "P1: 4 bytes",
    ],
)  # fmt: skip
def test__split_into_byte_chunks(protocol, value, length, expected):
    bus = FeetechMotorsBus("", {}, protocol_version=protocol)
    assert bus._split_into_byte_chunks(value, length) == expected


def test_abc_implementation(dummy_motors):
    """Instantiation should raise an error if the class doesn't implement abstract methods/properties."""
    FeetechMotorsBus(port="/dev/dummy-port", motors=dummy_motors)


@pytest.mark.parametrize("id_", [1, 2, 3])
def test_ping(id_, mock_motors, dummy_motors):
    expected_model_nb = MODEL_NUMBER_TABLE[dummy_motors[f"dummy_{id_}"].model]
    addr, length = MODEL_NUMBER
    ping_stub = mock_motors.build_ping_stub(id_)
    mobel_nb_stub = mock_motors.build_read_stub(addr, length, id_, expected_model_nb)
    bus = FeetechMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    bus.connect(handshake=False)

    ping_model_nb = bus.ping(id_)

    assert ping_model_nb == expected_model_nb
    assert mock_motors.stubs[ping_stub].called
    assert mock_motors.stubs[mobel_nb_stub].called


def test_broadcast_ping(mock_motors, dummy_motors):
    models = {m.id: m.model for m in dummy_motors.values()}
    addr, length = MODEL_NUMBER
    ping_stub = mock_motors.build_broadcast_ping_stub(list(models))
    mobel_nb_stubs = []
    expected_model_nbs = {}
    for id_, model in models.items():
        model_nb = MODEL_NUMBER_TABLE[model]
        stub = mock_motors.build_read_stub(addr, length, id_, model_nb)
        expected_model_nbs[id_] = model_nb
        mobel_nb_stubs.append(stub)
    bus = FeetechMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    bus.connect(handshake=False)

    ping_model_nbs = bus.broadcast_ping()

    assert ping_model_nbs == expected_model_nbs
    assert mock_motors.stubs[ping_stub].called
    assert all(mock_motors.stubs[stub].called for stub in mobel_nb_stubs)


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
    bus = FeetechMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    bus.connect(handshake=False)

    read_value, _, _ = bus._read(addr, length, id_)

    assert mock_motors.stubs[stub].called
    assert read_value == value


@pytest.mark.parametrize("raise_on_error", (True, False))
def test__read_error(raise_on_error, mock_motors, dummy_motors):
    addr, length, id_, value, error = (10, 4, 1, 1337, scs.ERRBIT_VOLTAGE)
    stub = mock_motors.build_read_stub(addr, length, id_, value, error=error)
    bus = FeetechMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    bus.connect(handshake=False)

    if raise_on_error:
        with pytest.raises(RuntimeError, match=re.escape("[RxPacketError] Input voltage error!")):
            bus._read(addr, length, id_, raise_on_error=raise_on_error)
    else:
        _, _, read_error = bus._read(addr, length, id_, raise_on_error=raise_on_error)
        assert read_error == error

    assert mock_motors.stubs[stub].called


@pytest.mark.parametrize("raise_on_error", (True, False))
def test__read_comm(raise_on_error, mock_motors, dummy_motors):
    addr, length, id_, value = (10, 4, 1, 1337)
    stub = mock_motors.build_read_stub(addr, length, id_, value, reply=False)
    bus = FeetechMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    bus.connect(handshake=False)

    if raise_on_error:
        with pytest.raises(ConnectionError, match=re.escape("[TxRxResult] There is no status packet!")):
            bus._read(addr, length, id_, raise_on_error=raise_on_error)
    else:
        _, read_comm, _ = bus._read(addr, length, id_, raise_on_error=raise_on_error)
        assert read_comm == scs.COMM_RX_TIMEOUT

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
    bus = FeetechMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
    )
    bus.connect(handshake=False)

    comm, error = bus._write(addr, length, id_, value)

    assert mock_motors.stubs[stub].called
    assert comm == scs.COMM_SUCCESS
    assert error == 0


@pytest.mark.parametrize("raise_on_error", (True, False))
def test__write_error(raise_on_error, mock_motors, dummy_motors):
    addr, length, id_, value, error = (10, 4, 1, 1337, scs.ERRBIT_VOLTAGE)
    stub = mock_motors.build_write_stub(addr, length, id_, value, error=error)
    bus = FeetechMotorsBus(port=mock_motors.port, motors=dummy_motors)
    bus.connect(handshake=False)

    if raise_on_error:
        with pytest.raises(RuntimeError, match=re.escape("[RxPacketError] Input voltage error!")):
            bus._write(addr, length, id_, value, raise_on_error=raise_on_error)
    else:
        _, write_error = bus._write(addr, length, id_, value, raise_on_error=raise_on_error)
        assert write_error == error

    assert mock_motors.stubs[stub].called


@pytest.mark.parametrize("raise_on_error", (True, False))
def test__write_comm(raise_on_error, mock_motors, dummy_motors):
    addr, length, id_, value = (10, 4, 1, 1337)
    stub = mock_motors.build_write_stub(addr, length, id_, value, reply=False)
    bus = FeetechMotorsBus(port=mock_motors.port, motors=dummy_motors)
    bus.connect(handshake=False)

    if raise_on_error:
        with pytest.raises(ConnectionError, match=re.escape("[TxRxResult] There is no status packet!")):
            bus._write(addr, length, id_, value, raise_on_error=raise_on_error)
    else:
        write_comm, _ = bus._write(addr, length, id_, value, raise_on_error=raise_on_error)
        assert write_comm == scs.COMM_RX_TIMEOUT

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
    bus = FeetechMotorsBus(port=mock_motors.port, motors=dummy_motors)
    bus.connect(handshake=False)

    read_values, _ = bus._sync_read(addr, length, list(ids_values))

    assert mock_motors.stubs[stub].called
    assert read_values == ids_values


@pytest.mark.parametrize("raise_on_error", (True, False))
def test__sync_read_comm(raise_on_error, mock_motors, dummy_motors):
    addr, length, ids_values = (10, 4, {1: 1337})
    stub = mock_motors.build_sync_read_stub(addr, length, ids_values, reply=False)
    bus = FeetechMotorsBus(port=mock_motors.port, motors=dummy_motors)
    bus.connect(handshake=False)

    if raise_on_error:
        with pytest.raises(ConnectionError, match=re.escape("[TxRxResult] There is no status packet!")):
            bus._sync_read(addr, length, list(ids_values), raise_on_error=raise_on_error)
    else:
        _, read_comm = bus._sync_read(addr, length, list(ids_values), raise_on_error=raise_on_error)
        assert read_comm == scs.COMM_RX_TIMEOUT

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
    bus = FeetechMotorsBus(port=mock_motors.port, motors=dummy_motors)
    bus.connect(handshake=False)

    comm = bus._sync_write(addr, length, ids_values)

    assert mock_motors.stubs[stub].wait_called()
    assert comm == scs.COMM_SUCCESS


def test_is_calibrated(mock_motors, dummy_motors, dummy_calibration):
    mins_stubs, maxes_stubs, homings_stubs = [], [], []
    for cal in dummy_calibration.values():
        mins_stubs.append(
            mock_motors.build_read_stub(
                *STS_SMS_SERIES_CONTROL_TABLE["Min_Position_Limit"], cal.id, cal.range_min
            )
        )
        maxes_stubs.append(
            mock_motors.build_read_stub(
                *STS_SMS_SERIES_CONTROL_TABLE["Max_Position_Limit"], cal.id, cal.range_max
            )
        )
        homings_stubs.append(
            mock_motors.build_read_stub(
                *STS_SMS_SERIES_CONTROL_TABLE["Homing_Offset"],
                cal.id,
                encode_sign_magnitude(cal.homing_offset, 11),
            )
        )

    bus = FeetechMotorsBus(
        port=mock_motors.port,
        motors=dummy_motors,
        calibration=dummy_calibration,
    )
    bus.connect(handshake=False)

    is_calibrated = bus.is_calibrated

    assert is_calibrated
    assert all(mock_motors.stubs[stub].called for stub in mins_stubs)
    assert all(mock_motors.stubs[stub].called for stub in maxes_stubs)
    assert all(mock_motors.stubs[stub].called for stub in homings_stubs)


def test_reset_calibration(mock_motors, dummy_motors):
    write_homing_stubs = []
    write_mins_stubs = []
    write_maxes_stubs = []
    for motor in dummy_motors.values():
        write_homing_stubs.append(
            mock_motors.build_write_stub(*STS_SMS_SERIES_CONTROL_TABLE["Homing_Offset"], motor.id, 0)
        )
        write_mins_stubs.append(
            mock_motors.build_write_stub(*STS_SMS_SERIES_CONTROL_TABLE["Min_Position_Limit"], motor.id, 0)
        )
        write_maxes_stubs.append(
            mock_motors.build_write_stub(*STS_SMS_SERIES_CONTROL_TABLE["Max_Position_Limit"], motor.id, 4095)
        )

    bus = FeetechMotorsBus(port=mock_motors.port, motors=dummy_motors)
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
        1: -710,  # 1337 - 2047
        2: -2005,  # 42 - 2047
        3: 1625,  # 3672 - 2047
    }
    read_pos_stub = mock_motors.build_sync_read_stub(
        *STS_SMS_SERIES_CONTROL_TABLE["Present_Position"], current_positions
    )
    write_homing_stubs = []
    for id_, homing in expected_homings.items():
        encoded_homing = encode_sign_magnitude(homing, 11)
        stub = mock_motors.build_write_stub(
            *STS_SMS_SERIES_CONTROL_TABLE["Homing_Offset"], id_, encoded_homing
        )
        write_homing_stubs.append(stub)

    bus = FeetechMotorsBus(port=mock_motors.port, motors=dummy_motors)
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
    stub = mock_motors.build_sequential_sync_read_stub(
        *STS_SMS_SERIES_CONTROL_TABLE["Present_Position"], positions
    )
    with patch("lerobot.motors.motors_bus.enter_pressed", side_effect=[False, True]):
        bus = FeetechMotorsBus(port=mock_motors.port, motors=dummy_motors)
        bus.connect(handshake=False)

        mins, maxes = bus.record_ranges_of_motion(display_values=False)

    assert mock_motors.stubs[stub].calls == 3
    assert mins == expected_mins
    assert maxes == expected_maxes
