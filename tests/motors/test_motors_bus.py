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
from unittest.mock import patch

import pytest

from lerobot.motors.motors_bus import (
    Motor,
    MotorNormMode,
    assert_same_address,
    get_address,
    get_ctrl_table,
)
from tests.mocks.mock_motors_bus import (
    DUMMY_CTRL_TABLE_1,
    DUMMY_CTRL_TABLE_2,
    DUMMY_MODEL_CTRL_TABLE,
    MockMotorsBus,
)


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
        mock__normalize.assert_called_once_with({id_: value})


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
        mock__unnormalize.assert_called_once_with({id_: value})


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
        mock__normalize.assert_called_once_with({id_: value})


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
        mock__normalize.assert_called_once_with(ids_values)


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
        mock__normalize.assert_called_once_with(ids_values)


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
        mock__unnormalize.assert_called_once_with(ids_values)


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
        mock__unnormalize.assert_called_once_with(ids_values)
