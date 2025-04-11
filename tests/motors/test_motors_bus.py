import re

import pytest

from lerobot.common.motors.motors_bus import (
    Motor,
    MotorsBus,
    assert_same_address,
    get_address,
    get_ctrl_table,
)

DUMMY_CTRL_TABLE_1 = {
    "Firmware_Version": (0, 1),
    "Model_Number": (1, 2),
    "Present_Position": (3, 4),
    "Goal_Position": (7, 2),
}

DUMMY_CTRL_TABLE_2 = {
    "Model_Number": (0, 2),
    "Firmware_Version": (2, 1),
    "Present_Position": (3, 4),
    "Goal_Position": (7, 4),
    "Lock": (7, 4),
}

DUMMY_MODEL_CTRL_TABLE = {
    "model_1": DUMMY_CTRL_TABLE_1,
    "model_2": DUMMY_CTRL_TABLE_2,
}

DUMMY_BAUDRATE_TABLE = {
    0: 1_000_000,
    1: 500_000,
}

DUMMY_MODEL_BAUDRATE_TABLE = {
    "model_1": DUMMY_BAUDRATE_TABLE,
    "model_2": DUMMY_BAUDRATE_TABLE,
}

DUMMY_ENCODING_TABLE = {
    "Present_Position": 8,
    "Goal_Position": 10,
}

DUMMY_MODEL_ENCODING_TABLE = {
    "model_1": DUMMY_ENCODING_TABLE,
    "model_2": DUMMY_ENCODING_TABLE,
}


class DummyMotorsBus(MotorsBus):
    available_baudrates = [500_000, 1_000_000]
    default_timeout = 1000
    model_baudrate_table = DUMMY_MODEL_BAUDRATE_TABLE
    model_ctrl_table = DUMMY_MODEL_CTRL_TABLE
    model_encoding_table = DUMMY_MODEL_ENCODING_TABLE
    model_number_table = {"model_1": 1234, "model_2": 5678}
    model_resolution_table = {"model_1": 4096, "model_2": 1024}
    normalized_data = ["Present_Position", "Goal_Position"]

    def __init__(self, port: str, motors: dict[str, Motor]):
        super().__init__(port, motors)

    def _assert_protocol_is_compatible(self, instruction_name): ...
    def configure_motors(self): ...
    def disable_torque(self, motors): ...
    def enable_torque(self, motors): ...
    def _get_half_turn_homings(self, positions): ...
    def _encode_sign(self, data_name, ids_values): ...
    def _decode_sign(self, data_name, ids_values): ...
    def _split_into_byte_chunks(self, value, length): ...
    def broadcast_ping(self, num_retry, raise_on_error): ...


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


def test_assert_same_address_different_addresses():
    models = ["model_1", "model_2"]
    with pytest.raises(
        NotImplementedError,
        match=re.escape("At least two motor models use a different address"),
    ):
        assert_same_address(DUMMY_MODEL_CTRL_TABLE, models, "Model_Number")


def test_assert_same_address_different_bytes():
    models = ["model_1", "model_2"]
    with pytest.raises(
        NotImplementedError,
        match=re.escape("At least two motor models use a different bytes representation"),
    ):
        assert_same_address(DUMMY_MODEL_CTRL_TABLE, models, "Goal_Position")


def test__serialize_data_invalid_length():
    bus = DummyMotorsBus("", {})
    with pytest.raises(NotImplementedError):
        bus._serialize_data(100, 3)


def test__serialize_data_negative_numbers():
    bus = DummyMotorsBus("", {})
    with pytest.raises(ValueError):
        bus._serialize_data(-1, 1)


def test__serialize_data_large_number():
    bus = DummyMotorsBus("", {})
    with pytest.raises(ValueError):
        bus._serialize_data(2**32, 4)  # 4-byte max is 0xFFFFFFFF
