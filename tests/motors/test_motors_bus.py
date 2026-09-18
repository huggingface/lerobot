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

pytest.importorskip("serial", reason="pyserial is required (install lerobot[hardware])")

from lerobot.motors.dynamixel import DynamixelMotorsBus
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import (
    Motor,
    MotorNormMode,
    SerialMotorsBus,
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
        patch.object(MockMotorsBus, "_read", return_value=value) as mock__read,
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
        patch.object(MockMotorsBus, "_write", return_value=None) as mock__write,
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
        patch.object(MockMotorsBus, "_sync_read", return_value={id_: value}) as mock__sync_read,
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
        patch.object(MockMotorsBus, "_sync_read", return_value=ids_values) as mock__sync_read,
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
        patch.object(MockMotorsBus, "_sync_read", return_value=ids_values) as mock__sync_read,
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
        patch.object(MockMotorsBus, "_sync_write", return_value=None) as mock__sync_write,
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
        patch.object(MockMotorsBus, "_sync_write", return_value=None) as mock__sync_write,
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


# The four IO helpers live entirely in SerialMotorsBus now that framing is the
# transport's business, so they are exercised once here instead of once per
# motor family.


@pytest.fixture
def bus(dummy_motors):
    bus = MockMotorsBus("/dev/dummy-port", dummy_motors)
    bus.connect(handshake=False)
    return bus


@pytest.mark.parametrize(
    "addr, length, id_, value",
    [(0, 1, 1, 2), (10, 2, 2, 999), (42, 4, 3, 1337)],
)
def test__read(addr, length, id_, value, bus):
    bus._io.seed(id_, addr, value.to_bytes(length, "little"))

    assert bus._read(addr, length, id_) == value
    assert bus._io.reads == [(id_, addr, length)]


@pytest.mark.parametrize("raise_on_error", (True, False))
def test__read_motor_error(raise_on_error, bus):
    bus._io.status[1] = 0x20

    if raise_on_error:
        with pytest.raises(RuntimeError, match="error status 0x20"):
            bus._read(10, 4, 1, raise_on_error=True)
    else:
        assert bus._read(10, 4, 1, raise_on_error=False) is None


@pytest.mark.parametrize("raise_on_error", (True, False))
def test__read_no_answer(raise_on_error, bus):
    bus._io.absent.add(1)

    if raise_on_error:
        with pytest.raises(ConnectionError, match="Timeout"):
            bus._read(10, 4, 1, raise_on_error=True)
    else:
        assert bus._read(10, 4, 1, raise_on_error=False) is None


@pytest.mark.parametrize(
    "addr, length, id_, value",
    [(0, 1, 1, 2), (10, 2, 2, 999), (42, 4, 3, 1337)],
)
def test__write(addr, length, id_, value, bus):
    bus._write(addr, length, id_, value)

    assert bus._io.writes == [(id_, addr, value.to_bytes(length, "little"))]


def test__write_no_answer(bus):
    bus._io.absent.add(1)

    with pytest.raises(ConnectionError, match="Timeout"):
        bus._write(10, 4, 1, 1337)


@pytest.mark.parametrize(
    "addr, length, ids_values",
    [(0, 1, {1: 4}), (10, 2, {1: 1337, 2: 42}), (42, 4, {1: 1337, 2: 42, 3: 4016})],
)
def test__sync_read(addr, length, ids_values, bus):
    for id_, value in ids_values.items():
        bus._io.seed(id_, addr, value.to_bytes(length, "little"))

    assert bus._sync_read(addr, length, list(ids_values)) == ids_values


def test__sync_read_retries_after_transient_failure(bus):
    bus._io.seed(1, 10, (1337).to_bytes(2, "little"))
    bus._io.fail_times = 1

    assert bus._sync_read(10, 2, [1], num_retry=1) == {1: 1337}


def test__sync_read_no_answer(bus):
    bus._io.absent.add(1)

    with pytest.raises(ConnectionError, match="Timeout"):
        bus._sync_read(10, 2, [1])


@pytest.mark.parametrize(
    "addr, length, ids_values",
    [(0, 1, {1: 4}), (10, 2, {1: 1337, 2: 42}), (42, 4, {1: 1337, 2: 42, 3: 4016})],
)
def test__sync_write(addr, length, ids_values, bus):
    bus._sync_write(addr, length, ids_values)

    expected = [value.to_bytes(length, "little") for value in ids_values.values()]
    assert bus._io.sync_writes == [(list(ids_values), addr, expected)]


def test_ping(bus):
    addr, length = bus.model_number_address
    bus._io.seed(2, addr, (5678).to_bytes(length, "little"))

    assert bus.ping(2) == 5678

    bus._io.absent.add(3)
    assert bus.ping(3) is None


def test_broadcast_ping(bus):
    addr, length = bus.model_number_address
    bus._io.absent = set(range(bus.max_id + 1)) - {2}
    bus._io.seed(2, addr, (5678).to_bytes(length, "little"))

    assert bus.broadcast_ping() == {2: 5678}
    assert bus._io.timeout_ms == bus.default_timeout


@pytest.mark.parametrize(
    "baudrate, expected_timeout_ms",
    [(1_000_000, 5), (115_200, 5), (57_600, 6), (9_600, 33)],
)
def test_scan_timeout_scales_with_baudrate(baudrate, expected_timeout_ms, bus):
    """A sweep pays one timeout per absent ID, so it must fit the baud rate:
    too short and slow links miss motors, too long and a scan takes minutes."""
    bus._io.set_baudrate(baudrate)

    with bus._scan_timeout():
        assert bus._io.timeout_ms == expected_timeout_ms

    assert bus._io.timeout_ms == bus.default_timeout


@pytest.mark.parametrize(
    "model, expected", [("sts3215", FeetechMotorsBus), ("xl330-m077", DynamixelMotorsBus)]
)
def test_serial_motors_bus_resolves_family(model, expected):
    bus = SerialMotorsBus("/dev/dummy-port", {"dummy": Motor(1, model, MotorNormMode.RANGE_M100_100)})

    assert type(bus) is expected


@pytest.mark.parametrize(
    "motors, message",
    [
        ({}, "without any motor"),
        ({"a": Motor(1, "nonexistent", MotorNormMode.RANGE_M100_100)}, "Unknown motor model"),
        (
            {
                "a": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
                "b": Motor(2, "xl330-m077", MotorNormMode.RANGE_M100_100),
            },
            "different families",
        ),
    ],
    ids=["no motors", "unknown model", "mixed families"],
)
def test_serial_motors_bus_cannot_resolve_family(motors, message):
    with pytest.raises(ValueError, match=message):
        SerialMotorsBus("/dev/dummy-port", motors)
