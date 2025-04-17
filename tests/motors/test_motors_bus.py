# ruff: noqa: N802

import re
from unittest.mock import patch

import pytest

from lerobot.common.motors.motors_bus import (
    Motor,
    MotorNormMode,
    MotorsBus,
    assert_same_address,
    get_address,
    get_ctrl_table,
)

DUMMY_CTRL_TABLE_1 = {
    "Firmware_Version": (0, 1),
    "Model_Number": (1, 2),
    "Present_Position": (3, 4),
    "Goal_Position": (11, 2),
}

DUMMY_CTRL_TABLE_2 = {
    "Model_Number": (0, 2),
    "Firmware_Version": (2, 1),
    "Present_Position": (3, 4),
    "Present_Velocity": (7, 4),
    "Goal_Position": (11, 4),
    "Goal_Velocity": (15, 4),
    "Lock": (19, 1),
}

DUMMY_MODEL_CTRL_TABLE = {
    "model_1": DUMMY_CTRL_TABLE_1,
    "model_2": DUMMY_CTRL_TABLE_2,
    "model_3": DUMMY_CTRL_TABLE_2,
}

DUMMY_BAUDRATE_TABLE = {
    0: 1_000_000,
    1: 500_000,
    2: 250_000,
}

DUMMY_MODEL_BAUDRATE_TABLE = {
    "model_1": DUMMY_BAUDRATE_TABLE,
    "model_2": DUMMY_BAUDRATE_TABLE,
    "model_3": DUMMY_BAUDRATE_TABLE,
}

DUMMY_ENCODING_TABLE = {
    "Present_Position": 8,
    "Goal_Position": 10,
}

DUMMY_MODEL_ENCODING_TABLE = {
    "model_1": DUMMY_ENCODING_TABLE,
    "model_2": DUMMY_ENCODING_TABLE,
    "model_3": DUMMY_ENCODING_TABLE,
}

DUMMY_MODEL_NUMBER_TABLE = {
    "model_1": 1234,
    "model_2": 5678,
    "model_3": 5799,
}

DUMMY_MODEL_RESOLUTION_TABLE = {
    "model_1": 4096,
    "model_2": 1024,
    "model_3": 4096,
}


class MockPortHandler:
    def __init__(self, port_name):
        self.is_open: bool = False
        self.baudrate: int
        self.packet_start_time: float
        self.packet_timeout: float
        self.tx_time_per_byte: float
        self.is_using: bool = False
        self.port_name: str = port_name
        self.ser = None

    def openPort(self):
        self.is_open = True
        return self.is_open

    def closePort(self):
        self.is_open = False

    def clearPort(self): ...
    def setPortName(self, port_name):
        self.port_name = port_name

    def getPortName(self):
        return self.port_name

    def setBaudRate(self, baudrate):
        self.baudrate: baudrate

    def getBaudRate(self):
        return self.baudrate

    def getBytesAvailable(self): ...
    def readPort(self, length): ...
    def writePort(self, packet): ...
    def setPacketTimeout(self, packet_length): ...
    def setPacketTimeoutMillis(self, msec): ...
    def isPacketTimeout(self): ...
    def getCurrentTime(self): ...
    def getTimeSinceStart(self): ...
    def setupPort(self, cflag_baud): ...
    def getCFlagBaud(self, baudrate): ...


class MockMotorsBus(MotorsBus):
    available_baudrates = [500_000, 1_000_000]
    default_timeout = 1000
    model_baudrate_table = DUMMY_MODEL_BAUDRATE_TABLE
    model_ctrl_table = DUMMY_MODEL_CTRL_TABLE
    model_encoding_table = DUMMY_MODEL_ENCODING_TABLE
    model_number_table = DUMMY_MODEL_NUMBER_TABLE
    model_resolution_table = DUMMY_MODEL_RESOLUTION_TABLE
    normalized_data = ["Present_Position", "Goal_Position"]

    def __init__(self, port: str, motors: dict[str, Motor]):
        super().__init__(port, motors)
        self.port_handler = MockPortHandler(port)

    def _assert_protocol_is_compatible(self, instruction_name): ...
    def _handshake(self): ...
    def _find_single_motor(self, motor, initial_baudrate): ...
    def configure_motors(self): ...
    def read_calibration(self): ...
    def write_calibration(self, calibration_dict): ...
    def disable_torque(self, motors, num_retry): ...
    def _disable_torque(self, motor, model, num_retry): ...
    def enable_torque(self, motors, num_retry): ...
    def _get_half_turn_homings(self, positions): ...
    def _encode_sign(self, data_name, ids_values): ...
    def _decode_sign(self, data_name, ids_values): ...
    def _split_into_byte_chunks(self, value, length): ...
    def broadcast_ping(self, num_retry, raise_on_error): ...


@pytest.fixture
def dummy_motors() -> dict[str, Motor]:
    return {
        "dummy_1": Motor(1, "model_2", MotorNormMode.RANGE_M100_100),
        "dummy_2": Motor(2, "model_3", MotorNormMode.RANGE_M100_100),
        "dummy_3": Motor(3, "model_2", MotorNormMode.RANGE_0_100),
    }


def test_get_ctrl_table():
    model = "model_1"
    ctrl_table = get_ctrl_table(DUMMY_MODEL_CTRL_TABLE, model)
    assert ctrl_table == DUMMY_CTRL_TABLE_1


def test_get_ctrl_table_error():
    model = "model_99"
    with pytest.raises(KeyError, match=f"Control table for {model=} not found."):
        get_ctrl_table(DUMMY_MODEL_CTRL_TABLE, model)


def test_get_address():
    addr, n_bytes = get_address(DUMMY_MODEL_CTRL_TABLE, "model_1", "Firmware_Version")
    assert addr == 0
    assert n_bytes == 1


def test_get_address_error():
    model = "model_1"
    data_name = "Lock"
    with pytest.raises(KeyError, match=f"Address for '{data_name}' not found in {model} control table."):
        get_address(DUMMY_MODEL_CTRL_TABLE, "model_1", data_name)


def test_assert_same_address():
    models = ["model_1", "model_2"]
    assert_same_address(DUMMY_MODEL_CTRL_TABLE, models, "Present_Position")


def test_assert_same_length_different_addresses():
    models = ["model_1", "model_2"]
    with pytest.raises(
        NotImplementedError,
        match=re.escape("At least two motor models use a different address"),
    ):
        assert_same_address(DUMMY_MODEL_CTRL_TABLE, models, "Model_Number")


def test_assert_same_address_different_length():
    models = ["model_1", "model_2"]
    with pytest.raises(
        NotImplementedError,
        match=re.escape("At least two motor models use a different bytes representation"),
    ):
        assert_same_address(DUMMY_MODEL_CTRL_TABLE, models, "Goal_Position")


def test__serialize_data_invalid_length():
    bus = MockMotorsBus("", {})
    with pytest.raises(NotImplementedError):
        bus._serialize_data(100, 3)


def test__serialize_data_negative_numbers():
    bus = MockMotorsBus("", {})
    with pytest.raises(ValueError):
        bus._serialize_data(-1, 1)


def test__serialize_data_large_number():
    bus = MockMotorsBus("", {})
    with pytest.raises(ValueError):
        bus._serialize_data(2**32, 4)  # 4-byte max is 0xFFFFFFFF


@pytest.mark.parametrize(
    "data_name, id_, value",
    [
        ("Firmware_Version", 1, 14),
        ("Model_Number", 1, 5678),
        ("Present_Position", 2, 1337),
        ("Present_Velocity", 3, 42),
    ],
)
def test_read(data_name, id_, value, dummy_motors):
    bus = MockMotorsBus("/dev/dummy-port", dummy_motors)
    bus.connect(handshake=False)
    addr, length = DUMMY_CTRL_TABLE_2[data_name]

    with (
        patch.object(MockMotorsBus, "_read", return_value=(value, 0, 0)) as mock__read,
        patch.object(MockMotorsBus, "_decode_sign", return_value={id_: value}) as mock__decode_sign,
        patch.object(MockMotorsBus, "_normalize", return_value={id_: value}) as mock__normalize,
    ):
        returned_value = bus.read(data_name, f"dummy_{id_}")

    assert returned_value == value
    mock__read.assert_called_once_with(
        addr,
        length,
        id_,
        num_retry=0,
        raise_on_error=True,
        err_msg=f"Failed to read '{data_name}' on {id_=} after 1 tries.",
    )
    mock__decode_sign.assert_called_once_with(data_name, {id_: value})
    if data_name in bus.normalized_data:
        mock__normalize.assert_called_once_with(data_name, {id_: value})


@pytest.mark.parametrize(
    "data_name, id_, value",
    [
        ("Goal_Position", 1, 1337),
        ("Goal_Velocity", 2, 3682),
        ("Lock", 3, 1),
    ],
)
def test_write(data_name, id_, value, dummy_motors):
    bus = MockMotorsBus("/dev/dummy-port", dummy_motors)
    bus.connect(handshake=False)
    addr, length = DUMMY_CTRL_TABLE_2[data_name]

    with (
        patch.object(MockMotorsBus, "_write", return_value=(0, 0)) as mock__write,
        patch.object(MockMotorsBus, "_encode_sign", return_value={id_: value}) as mock__encode_sign,
        patch.object(MockMotorsBus, "_unnormalize", return_value={id_: value}) as mock__unnormalize,
    ):
        bus.write(data_name, f"dummy_{id_}", value)

    mock__write.assert_called_once_with(
        addr,
        length,
        id_,
        value,
        num_retry=0,
        raise_on_error=True,
        err_msg=f"Failed to write '{data_name}' on {id_=} with '{value}' after 1 tries.",
    )
    mock__encode_sign.assert_called_once_with(data_name, {id_: value})
    if data_name in bus.normalized_data:
        mock__unnormalize.assert_called_once_with(data_name, {id_: value})


@pytest.mark.parametrize(
    "data_name, id_, value",
    [
        ("Firmware_Version", 1, 14),
        ("Model_Number", 1, 5678),
        ("Present_Position", 2, 1337),
        ("Present_Velocity", 3, 42),
    ],
)
def test_sync_read_by_str(data_name, id_, value, dummy_motors):
    bus = MockMotorsBus("/dev/dummy-port", dummy_motors)
    bus.connect(handshake=False)
    addr, length = DUMMY_CTRL_TABLE_2[data_name]
    ids = [id_]
    expected_value = {f"dummy_{id_}": value}

    with (
        patch.object(MockMotorsBus, "_sync_read", return_value=({id_: value}, 0)) as mock__sync_read,
        patch.object(MockMotorsBus, "_decode_sign", return_value={id_: value}) as mock__decode_sign,
        patch.object(MockMotorsBus, "_normalize", return_value={id_: value}) as mock__normalize,
    ):
        returned_dict = bus.sync_read(data_name, f"dummy_{id_}")

    assert returned_dict == expected_value
    mock__sync_read.assert_called_once_with(
        addr,
        length,
        ids,
        num_retry=0,
        raise_on_error=True,
        err_msg=f"Failed to sync read '{data_name}' on {ids=} after 1 tries.",
    )
    mock__decode_sign.assert_called_once_with(data_name, {id_: value})
    if data_name in bus.normalized_data:
        mock__normalize.assert_called_once_with(data_name, {id_: value})


@pytest.mark.parametrize(
    "data_name, ids_values",
    [
        ("Model_Number", {1: 5678}),
        ("Present_Position", {1: 1337, 2: 42}),
        ("Present_Velocity", {1: 1337, 2: 42, 3: 4016}),
    ],
    ids=["1 motor", "2 motors", "3 motors"],
)
def test_sync_read_by_list(data_name, ids_values, dummy_motors):
    bus = MockMotorsBus("/dev/dummy-port", dummy_motors)
    bus.connect(handshake=False)
    addr, length = DUMMY_CTRL_TABLE_2[data_name]
    ids = list(ids_values)
    expected_values = {f"dummy_{id_}": val for id_, val in ids_values.items()}

    with (
        patch.object(MockMotorsBus, "_sync_read", return_value=(ids_values, 0)) as mock__sync_read,
        patch.object(MockMotorsBus, "_decode_sign", return_value=ids_values) as mock__decode_sign,
        patch.object(MockMotorsBus, "_normalize", return_value=ids_values) as mock__normalize,
    ):
        returned_dict = bus.sync_read(data_name, [f"dummy_{id_}" for id_ in ids])

    assert returned_dict == expected_values
    mock__sync_read.assert_called_once_with(
        addr,
        length,
        ids,
        num_retry=0,
        raise_on_error=True,
        err_msg=f"Failed to sync read '{data_name}' on {ids=} after 1 tries.",
    )
    mock__decode_sign.assert_called_once_with(data_name, ids_values)
    if data_name in bus.normalized_data:
        mock__normalize.assert_called_once_with(data_name, ids_values)


@pytest.mark.parametrize(
    "data_name, ids_values",
    [
        ("Model_Number", {1: 5678, 2: 5799, 3: 5678}),
        ("Present_Position", {1: 1337, 2: 42, 3: 4016}),
        ("Goal_Position", {1: 4008, 2: 199, 3: 3446}),
    ],
    ids=["Model_Number", "Present_Position", "Goal_Position"],
)
def test_sync_read_by_none(data_name, ids_values, dummy_motors):
    bus = MockMotorsBus("/dev/dummy-port", dummy_motors)
    bus.connect(handshake=False)
    addr, length = DUMMY_CTRL_TABLE_2[data_name]
    ids = list(ids_values)
    expected_values = {f"dummy_{id_}": val for id_, val in ids_values.items()}

    with (
        patch.object(MockMotorsBus, "_sync_read", return_value=(ids_values, 0)) as mock__sync_read,
        patch.object(MockMotorsBus, "_decode_sign", return_value=ids_values) as mock__decode_sign,
        patch.object(MockMotorsBus, "_normalize", return_value=ids_values) as mock__normalize,
    ):
        returned_dict = bus.sync_read(data_name)

    assert returned_dict == expected_values
    mock__sync_read.assert_called_once_with(
        addr,
        length,
        ids,
        num_retry=0,
        raise_on_error=True,
        err_msg=f"Failed to sync read '{data_name}' on {ids=} after 1 tries.",
    )
    mock__decode_sign.assert_called_once_with(data_name, ids_values)
    if data_name in bus.normalized_data:
        mock__normalize.assert_called_once_with(data_name, ids_values)


@pytest.mark.parametrize(
    "data_name, value",
    [
        ("Goal_Position", 500),
        ("Goal_Velocity", 4010),
        ("Lock", 0),
    ],
)
def test_sync_write_by_single_value(data_name, value, dummy_motors):
    bus = MockMotorsBus("/dev/dummy-port", dummy_motors)
    bus.connect(handshake=False)
    addr, length = DUMMY_CTRL_TABLE_2[data_name]
    ids_values = {m.id: value for m in dummy_motors.values()}

    with (
        patch.object(MockMotorsBus, "_sync_write", return_value=(ids_values, 0)) as mock__sync_write,
        patch.object(MockMotorsBus, "_encode_sign", return_value=ids_values) as mock__encode_sign,
        patch.object(MockMotorsBus, "_unnormalize", return_value=ids_values) as mock__unnormalize,
    ):
        bus.sync_write(data_name, value)

    mock__sync_write.assert_called_once_with(
        addr,
        length,
        ids_values,
        num_retry=0,
        raise_on_error=True,
        err_msg=f"Failed to sync write '{data_name}' with {ids_values=} after 1 tries.",
    )
    mock__encode_sign.assert_called_once_with(data_name, ids_values)
    if data_name in bus.normalized_data:
        mock__unnormalize.assert_called_once_with(data_name, ids_values)


@pytest.mark.parametrize(
    "data_name, ids_values",
    [
        ("Goal_Position", {1: 1337, 2: 42, 3: 4016}),
        ("Goal_Velocity", {1: 50, 2: 83, 3: 2777}),
        ("Lock", {1: 0, 2: 0, 3: 1}),
    ],
    ids=["Goal_Position", "Goal_Velocity", "Lock"],
)
def test_sync_write_by_value_dict(data_name, ids_values, dummy_motors):
    bus = MockMotorsBus("/dev/dummy-port", dummy_motors)
    bus.connect(handshake=False)
    addr, length = DUMMY_CTRL_TABLE_2[data_name]
    values = {f"dummy_{id_}": val for id_, val in ids_values.items()}

    with (
        patch.object(MockMotorsBus, "_sync_write", return_value=(ids_values, 0)) as mock__sync_write,
        patch.object(MockMotorsBus, "_encode_sign", return_value=ids_values) as mock__encode_sign,
        patch.object(MockMotorsBus, "_unnormalize", return_value=ids_values) as mock__unnormalize,
    ):
        bus.sync_write(data_name, values)

    mock__sync_write.assert_called_once_with(
        addr,
        length,
        ids_values,
        num_retry=0,
        raise_on_error=True,
        err_msg=f"Failed to sync write '{data_name}' with {ids_values=} after 1 tries.",
    )
    mock__encode_sign.assert_called_once_with(data_name, ids_values)
    if data_name in bus.normalized_data:
        mock__unnormalize.assert_called_once_with(data_name, ids_values)
