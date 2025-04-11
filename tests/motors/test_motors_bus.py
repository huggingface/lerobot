import re

import pytest

from lerobot.common.motors.motors_bus import (
    Motor,
    MotorsBus,
    assert_same_address,
    get_address,
    get_ctrl_table,
)

DUMMY_CTRL_TABLE = {"Present_Position": (13, 4)}

DUMMY_BAUDRATE_TABLE = {
    0: 1_000_000,
    1: 500_000,
}

DUMMY_ENCODING_TABLE = {
    "Present_Position": 8,
}

DUMMY_MODEL_NUMBER_TABLE = {""}


class DummyMotorsBus(MotorsBus):
    available_baudrates = [1_000_000]
    default_timeout = 1000
    model_baudrate_table = {"model": DUMMY_BAUDRATE_TABLE}
    model_ctrl_table = {"model": DUMMY_CTRL_TABLE}
    model_encoding_table = {"model": DUMMY_ENCODING_TABLE}
    model_number_table = {"model": 1234}
    model_resolution_table = {"model": 4096}
    normalized_data = ["Present_Position"]

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


@pytest.fixture
def ctrl_table_1() -> dict:
    return {
        "Firmware_Version": (0, 1),
        "Model_Number": (1, 2),
        "Present_Position": (3, 4),
        "Goal_Position": (7, 2),
    }


@pytest.fixture
def ctrl_table_2() -> dict:
    return {
        "Model_Number": (0, 2),
        "Firmware_Version": (2, 1),
        "Present_Position": (3, 4),
        "Goal_Position": (7, 4),
        "Lock": (7, 4),
    }


@pytest.fixture
def model_ctrl_table(ctrl_table_1, ctrl_table_2) -> dict:
    return {
        "model_1": ctrl_table_1,
        "model_2": ctrl_table_2,
    }


def test_get_ctrl_table(model_ctrl_table, ctrl_table_1):
    model = "model_1"
    ctrl_table = get_ctrl_table(model_ctrl_table, model)
    assert ctrl_table == ctrl_table_1


def test_get_ctrl_table_error(model_ctrl_table):
    model = "model_99"
    with pytest.raises(KeyError, match=f"Control table for {model=} not found."):
        get_ctrl_table(model_ctrl_table, model)


def test_get_address(model_ctrl_table):
    addr, n_bytes = get_address(model_ctrl_table, "model_1", "Firmware_Version")
    assert addr == 0
    assert n_bytes == 1


def test_get_address_error(model_ctrl_table):
    model = "model_1"
    data_name = "Lock"
    with pytest.raises(KeyError, match=f"Address for '{data_name}' not found in {model} control table."):
        get_address(model_ctrl_table, "model_1", data_name)


def test_assert_same_address(model_ctrl_table):
    models = ["model_1", "model_2"]
    assert_same_address(model_ctrl_table, models, "Present_Position")


def test_assert_same_address_different_addresses(model_ctrl_table):
    models = ["model_1", "model_2"]
    with pytest.raises(
        NotImplementedError,
        match=re.escape("At least two motor models use a different address"),
    ):
        assert_same_address(model_ctrl_table, models, "Model_Number")


def test_assert_same_address_different_bytes(model_ctrl_table):
    models = ["model_1", "model_2"]
    with pytest.raises(
        NotImplementedError,
        match=re.escape("At least two motor models use a different bytes representation"),
    ):
        assert_same_address(model_ctrl_table, models, "Goal_Position")


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
