import re

import pytest

from lerobot.common.motors.motors_bus import assert_same_address, get_address, get_ctrl_table

# TODO(aliberts)
# class DummyMotorsBus(MotorsBus):
#     def __init__(self, port: str, motors: dict[str, Motor]):
#         super().__init__(port, motors)


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
