#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import sys
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.encoding_utils import encode_sign_magnitude
from lerobot.motors.feetech import MODEL_NUMBER, MODEL_NUMBER_TABLE, FeetechMotorsBus
from lerobot.motors.feetech.tables import STS_SMS_SERIES_CONTROL_TABLE

try:
    import scservo_sdk as scs

    from tests.mocks.mock_feetech import MockMotors, MockPortHandler, _split_into_byte_chunks
except (ImportError, ModuleNotFoundError):
    pytest.skip("scservo_sdk not available", allow_module_level=True)

# Present_Position (56-57), Present_Load (60-61) and Present_Current (69-70) span one 15-byte block on STS
# servos. The registers in between (velocity, voltage, temperature, ...) are part of the block on the wire
# but must be ignored by the reader.
TELEMETRY_NAMES = ["Present_Position", "Present_Load", "Present_Current"]
TELEMETRY_BLOCK_START, TELEMETRY_BLOCK_LENGTH = 56, 15


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

    assert mock_motors.stubs[stub].wait_called()
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


def test__sync_read_retries_after_transient_failure(mock_motors, dummy_motors):
    addr, length, ids_values = (10, 4, {1: 1337})
    stub = mock_motors.build_sync_read_stub(addr, length, ids_values, num_invalid_try=1)
    bus = FeetechMotorsBus(port=mock_motors.port, motors=dummy_motors)
    bus.connect(handshake=False)

    read_values, read_comm = bus._sync_read(addr, length, list(ids_values), num_retry=1)

    assert read_comm == scs.COMM_SUCCESS
    assert read_values == ids_values
    assert mock_motors.stubs[stub].calls == 2


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


def _encode_telemetry(
    positions: dict[int, int], loads: dict[int, int], currents: dict[int, int]
) -> dict[str, dict[int, int]]:
    """Encode signed telemetry values the way an STS servo reports them (Present_Current has no sign bit)."""
    return {
        "Present_Position": {id_: encode_sign_magnitude(val, 15) for id_, val in positions.items()},
        "Present_Load": {id_: encode_sign_magnitude(val, 10) for id_, val in loads.items()},
        "Present_Current": dict(currents),
    }


def _build_telemetry_block_stub(
    mock_motors: MockMotors, encoded: dict[str, dict[int, int]], filler: int = 0xAB
) -> str:
    """Stub one sync read of the whole telemetry block, with `filler` in the bytes no register maps to."""
    ids_data = {}
    for id_ in encoded["Present_Position"]:
        block = [filler] * TELEMETRY_BLOCK_LENGTH
        for data_name in TELEMETRY_NAMES:
            addr, length = STS_SMS_SERIES_CONTROL_TABLE[data_name]
            offset = addr - TELEMETRY_BLOCK_START
            block[offset : offset + length] = _split_into_byte_chunks(encoded[data_name][id_], length)
        ids_data[id_] = block
    return mock_motors.build_sync_read_block_stub(TELEMETRY_BLOCK_START, TELEMETRY_BLOCK_LENGTH, ids_data)


def test_sync_read_block(mock_motors, dummy_motors):
    """A block read decodes each register exactly like one sync_read per register would."""
    positions = {1: -1337, 2: 42, 3: 3672}
    loads = {1: 300, 2: -512, 3: -7}
    currents = {1: 12, 2: 0, 3: 999}
    encoded = _encode_telemetry(positions, loads, currents)
    block_stub = _build_telemetry_block_stub(mock_motors, encoded)
    single_stubs = [
        mock_motors.build_sync_read_stub(*STS_SMS_SERIES_CONTROL_TABLE[name], encoded[name])
        for name in TELEMETRY_NAMES
    ]
    bus = FeetechMotorsBus(port=mock_motors.port, motors=dummy_motors)
    bus.connect(handshake=False)

    block_values = bus.sync_read_block(TELEMETRY_NAMES, normalize=False)
    single_values = {name: bus.sync_read(name, normalize=False) for name in TELEMETRY_NAMES}

    assert block_values == single_values
    assert block_values["Present_Position"] == {"dummy_1": -1337, "dummy_2": 42, "dummy_3": 3672}
    assert block_values["Present_Load"] == {"dummy_1": 300, "dummy_2": -512, "dummy_3": -7}
    assert block_values["Present_Current"] == {"dummy_1": 12, "dummy_2": 0, "dummy_3": 999}
    assert mock_motors.stubs[block_stub].calls == 1
    assert all(mock_motors.stubs[stub].calls == 1 for stub in single_stubs)


def test_sync_read_block_normalize(mock_motors, dummy_motors, dummy_calibration):
    """With normalize=True only registers in `normalized_data` (Present_Position) are scaled."""
    positions = {1: 1000, 2: 2000, 3: 3000}
    loads = {1: 300, 2: -512, 3: -7}
    currents = {1: 12, 2: 0, 3: 999}
    encoded = _encode_telemetry(positions, loads, currents)
    block_stub = _build_telemetry_block_stub(mock_motors, encoded)
    position_stub = mock_motors.build_sync_read_stub(
        *STS_SMS_SERIES_CONTROL_TABLE["Present_Position"], encoded["Present_Position"]
    )
    bus = FeetechMotorsBus(port=mock_motors.port, motors=dummy_motors, calibration=dummy_calibration)
    bus.connect(handshake=False)

    block_values = bus.sync_read_block(TELEMETRY_NAMES)
    normalized_positions = bus.sync_read("Present_Position")

    assert block_values["Present_Position"] == normalized_positions
    assert all(isinstance(val, float) for val in block_values["Present_Position"].values())
    assert block_values["Present_Load"] == {"dummy_1": 300, "dummy_2": -512, "dummy_3": -7}
    assert block_values["Present_Current"] == {"dummy_1": 12, "dummy_2": 0, "dummy_3": 999}
    assert mock_motors.stubs[block_stub].calls == 1
    assert mock_motors.stubs[position_stub].calls == 1


def test_sync_read_block_single_transaction(mock_motors, dummy_motors):
    """Reading a block for a subset of motors sends exactly one sync read packet on the bus."""
    encoded = _encode_telemetry({1: 100, 3: -300}, {1: 1, 3: -3}, {1: 5, 3: 7})
    block_stub = _build_telemetry_block_stub(mock_motors, encoded)
    bus = FeetechMotorsBus(port=mock_motors.port, motors=dummy_motors)
    bus.connect(handshake=False)

    with patch.object(bus.sync_reader, "txRxPacket", wraps=bus.sync_reader.txRxPacket) as mock_txrx:
        block_values = bus.sync_read_block(TELEMETRY_NAMES, ["dummy_1", "dummy_3"], normalize=False)

    assert mock_txrx.call_count == 1
    assert mock_motors.stubs[block_stub].calls == 1
    assert block_values == {
        "Present_Position": {"dummy_1": 100, "dummy_3": -300},
        "Present_Load": {"dummy_1": 1, "dummy_3": -3},
        "Present_Current": {"dummy_1": 5, "dummy_3": 7},
    }


def test_sync_read_block_deduplicates_names(mock_motors, dummy_motors):
    encoded = _encode_telemetry({1: 1, 2: 2, 3: 3}, {1: 4, 2: 5, 3: 6}, {1: 7, 2: 8, 3: 9})
    block_stub = _build_telemetry_block_stub(mock_motors, encoded)
    bus = FeetechMotorsBus(port=mock_motors.port, motors=dummy_motors)
    bus.connect(handshake=False)

    block_values = bus.sync_read_block(
        ["Present_Position", "Present_Current", "Present_Position", "Present_Load"], normalize=False
    )

    assert list(block_values) == ["Present_Position", "Present_Current", "Present_Load"]
    assert mock_motors.stubs[block_stub].calls == 1


@pytest.mark.parametrize(
    "data_names, error, match",
    [
        ([], ValueError, "'data_names' should contain at least one register name."),
        ("Present_Position", TypeError, "'data_names' should be a sequence of register names"),
        (
            ["Present_Position", "Not_A_Register"],
            KeyError,
            "Address for 'Not_A_Register' not found in sts3215 control table.",
        ),
    ],
    ids=["empty", "single_str", "unknown_register"],
)
def test_sync_read_block_invalid_names(data_names, error, match, mock_motors, dummy_motors):
    bus = FeetechMotorsBus(port=mock_motors.port, motors=dummy_motors)
    bus.connect(handshake=False)

    with pytest.raises(error, match=re.escape(match)):
        bus.sync_read_block(data_names)


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

    assert all(mock_motors.stubs[stub].wait_called() for stub in write_homing_stubs)
    assert all(mock_motors.stubs[stub].wait_called() for stub in write_mins_stubs)
    assert all(mock_motors.stubs[stub].wait_called() for stub in write_maxes_stubs)


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
    assert all(mock_motors.stubs[stub].wait_called() for stub in write_homing_stubs)


@pytest.mark.parametrize(
    "initial_phase, expected_phase",
    [
        (0b00010000, 0b00000000),  # bit 4 set - cleared
        (0b11111111, 0b11101111),  # all bits set - bit 4 cleared, others preserved
        (0b00000000, 0b00000000),  # bit 4 already 0 - unchanged
    ],
    ids=["bit4_set", "all_bits_set", "bit4_already_cleared"],
)
def test_configure_motors_clears_sts3215_phase_bit4(initial_phase, expected_phase, mock_motors, dummy_motors):
    """Phase register bit 4 (angle feedback mode) must be cleared for sts3215, other bits preserved."""
    phase_read_stubs = []
    phase_write_stubs = []
    for motor in dummy_motors.values():
        mock_motors.build_write_stub(*STS_SMS_SERIES_CONTROL_TABLE["Return_Delay_Time"], motor.id, 0)
        mock_motors.build_write_stub(*STS_SMS_SERIES_CONTROL_TABLE["Maximum_Acceleration"], motor.id, 254)
        mock_motors.build_write_stub(*STS_SMS_SERIES_CONTROL_TABLE["Acceleration"], motor.id, 254)
        phase_read_stubs.append(
            mock_motors.build_read_stub(*STS_SMS_SERIES_CONTROL_TABLE["Phase"], motor.id, initial_phase)
        )
        if initial_phase != expected_phase:
            phase_write_stubs.append(
                mock_motors.build_write_stub(*STS_SMS_SERIES_CONTROL_TABLE["Phase"], motor.id, expected_phase)
            )

    bus = FeetechMotorsBus(port=mock_motors.port, motors=dummy_motors)
    bus.connect(handshake=False)

    with patch.object(bus, "write", wraps=bus.write) as mock_write:
        bus.configure_motors()

    assert all(mock_motors.stubs[stub].called for stub in phase_read_stubs)
    if initial_phase != expected_phase:  # ensure that phase is written only if it needs to be changed
        assert all(mock_motors.stubs[stub].wait_called() for stub in phase_write_stubs)
    else:  # If no write should be made, ensure that Phase is not written for any motor
        write_data_names = [call.args[0] for call in mock_write.call_args_list]
        assert "Phase" not in write_data_names


def test_configure_motors_skips_phase_for_non_sts3215(mock_motors):
    """Phase register must not be touched for motors other than sts3215."""
    motors = {
        "dummy_1": Motor(1, "sts3250", MotorNormMode.RANGE_M100_100),
        "dummy_2": Motor(2, "sts3250", MotorNormMode.RANGE_M100_100),
        "dummy_3": Motor(3, "sts3250", MotorNormMode.RANGE_M100_100),
    }
    for motor in motors.values():
        mock_motors.build_write_stub(*STS_SMS_SERIES_CONTROL_TABLE["Return_Delay_Time"], motor.id, 0)
        mock_motors.build_write_stub(*STS_SMS_SERIES_CONTROL_TABLE["Maximum_Acceleration"], motor.id, 254)
        mock_motors.build_write_stub(*STS_SMS_SERIES_CONTROL_TABLE["Acceleration"], motor.id, 254)

    bus = FeetechMotorsBus(port=mock_motors.port, motors=motors)
    bus.connect(handshake=False)

    with patch.object(bus, "read", wraps=bus.read) as mock_read:
        bus.configure_motors()
        read_data_names = [call.args[0] for call in mock_read.call_args_list]

    assert "Phase" not in read_data_names


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
    bus = FeetechMotorsBus(port=mock_motors.port, motors=dummy_motors)
    bus.connect(handshake=False)

    with (
        patch("lerobot.motors.motors_bus.enter_pressed", side_effect=[False, True]),
        patch("lerobot.motors.motors_bus.time.sleep") as mock_sleep,
        patch.object(bus, "sync_read", wraps=bus.sync_read) as mock_sync_read,
    ):
        mins, maxes = bus.record_ranges_of_motion(display_values=False)

    assert mock_motors.stubs[stub].calls == 3
    assert all(call.kwargs["num_retry"] == 5 for call in mock_sync_read.call_args_list)
    mock_sleep.assert_called_once_with(0.02)
    assert mins == expected_mins
    assert maxes == expected_maxes
