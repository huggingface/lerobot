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

from unittest.mock import MagicMock, patch

import pytest

from lerobot.motors.motors_bus import Motor, MotorNormMode, SerialMotorsBus
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.import_utils import _require_package_cache
from tests.mocks.mock_motors_bus import DUMMY_1, DUMMY_2, MockBus, MockMotorsBus


@pytest.fixture
def dummy_motors() -> dict[str, Motor]:
    return {
        "dummy_1": Motor(1, "model_2", MotorNormMode.RANGE_M100_100),
        "dummy_2": Motor(2, "model_3", MotorNormMode.RANGE_M100_100),
        "dummy_3": Motor(3, "model_2", MotorNormMode.RANGE_0_100),
    }


@pytest.fixture
def mixed_motors() -> dict[str, Motor]:
    """Two servo definitions on one bus."""
    return {
        "dummy_1": Motor(1, "model_1", MotorNormMode.RANGE_M100_100),
        "dummy_2": Motor(2, "model_2", MotorNormMode.RANGE_M100_100),
    }


def test_register_names_resolve_without_case():
    bus = MockMotorsBus("/dev/dummy-port", {"dummy": Motor(1, "model_1", MotorNormMode.RANGE_M100_100)})

    assert bus._has_register("dummy", "Firmware_Version")
    assert bus._has_register("dummy", "firmware_version")
    assert not bus._has_register("dummy", "Lock")


def test_unknown_model_is_refused():
    pytest.importorskip("rustypot")
    with pytest.raises(ValueError, match="Unknown motor model 'sts9999'"):
        SerialMotorsBus("/dev/dummy-port", {"dummy": Motor(1, "sts9999", MotorNormMode.RANGE_M100_100)})


def test_a_missing_rustypot_points_to_the_serial_motors_extra():
    with patch("lerobot.utils.import_utils.is_package_available", return_value=False):
        _require_package_cache.clear()
        try:
            with pytest.raises(ImportError, match=r"lerobot\[serial-motors\]"):
                SerialMotorsBus(
                    "/dev/dummy-port", {"dummy": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100)}
                )
        finally:
            _require_package_cache.clear()


def test_connect_opens_a_bus_with_each_motor_definition(mixed_motors):
    bus = MockMotorsBus("/dev/dummy-port", mixed_motors)
    bus.connect(handshake=False)

    # rustypot counts the timeout in seconds, LeRobot in milliseconds.
    assert (bus._bus.serial_port, bus._bus.baudrate, bus._bus.timeout) == (
        "/dev/dummy-port",
        bus.default_baudrate,
        bus.default_timeout / 1000,
    )
    assert bus._bus.motors == {1: DUMMY_1, 2: DUMMY_2}


@pytest.fixture
def bus(dummy_motors):
    bus = MockMotorsBus("/dev/dummy-port", dummy_motors)
    bus.connect(handshake=False)
    return bus


@pytest.mark.parametrize(
    "data_name, id_, value",
    [
        ("Firmware_Version", 1, 14),
        ("Model_Number", 1, 5678),
        ("Present_Position", 2, 1337),
        ("Present_Velocity", 3, -1337),
    ],
)
def test_read(data_name, id_, value, bus):
    bus._bus.seed(id_, data_name.lower(), value)

    with patch.object(MockMotorsBus, "_normalize", return_value={id_: value}) as mock__normalize:
        assert bus.read(data_name, f"dummy_{id_}") == value

    assert bus._bus.reads == [(id_, data_name.lower())]
    if data_name in bus.normalized_data:
        mock__normalize.assert_called_once_with({id_: value})


def test_a_read_fails_on_an_error_status(bus):
    bus._bus.status[1] = 0x20

    with pytest.raises(RuntimeError, match="error status 0x20"):
        bus.read("Present_Position", "dummy_1", normalize=False)


@pytest.mark.parametrize(
    "data_name, id_, value",
    [
        ("Goal_Position", 1, 1337),
        ("Goal_Velocity", 2, -1337),
        ("Lock", 3, 1),
    ],
)
def test_write(data_name, id_, value, bus):
    with patch.object(MockMotorsBus, "_unnormalize", return_value={id_: value}) as mock__unnormalize:
        bus.write(data_name, f"dummy_{id_}", value)

    assert bus._bus.writes == [(id_, data_name.lower(), value)]
    if data_name in bus.normalized_data:
        mock__unnormalize.assert_called_once_with({id_: value})


@pytest.mark.parametrize(
    "motors, ids_values",
    [
        ("dummy_1", {1: 5678}),
        (["dummy_1", "dummy_2"], {1: 1337, 2: 42}),
        (None, {1: 1337, 2: -42, 3: 4016}),
    ],
    ids=["by name", "by list", "every motor"],
)
@pytest.mark.parametrize("data_name", ["Model_Number", "Present_Position", "Present_Velocity"])
def test_sync_read(data_name, motors, ids_values, bus):
    for id_, value in ids_values.items():
        bus._bus.seed(id_, data_name.lower(), value)

    with patch.object(MockMotorsBus, "_normalize", return_value=ids_values) as mock__normalize:
        returned = bus.sync_read(data_name, motors)

    assert returned == {f"dummy_{id_}": value for id_, value in ids_values.items()}
    assert bus._bus.sync_reads == [(list(ids_values), data_name.lower())]
    if data_name in bus.normalized_data:
        mock__normalize.assert_called_once_with(ids_values)


@pytest.mark.parametrize(
    "values, ids_values",
    [
        (500, {1: 500, 2: 500, 3: 500}),
        ({"dummy_1": 1337, "dummy_2": -42, "dummy_3": 4016}, {1: 1337, 2: -42, 3: 4016}),
    ],
    ids=["one value for every motor", "a value per motor"],
)
@pytest.mark.parametrize("data_name", ["Goal_Position", "Goal_Velocity", "Lock"])
def test_sync_write(data_name, values, ids_values, bus):
    with patch.object(MockMotorsBus, "_unnormalize", return_value=ids_values) as mock__unnormalize:
        bus.sync_write(data_name, values)

    assert bus._bus.sync_writes == [(list(ids_values), data_name.lower(), list(ids_values.values()))]
    if data_name in bus.normalized_data:
        mock__unnormalize.assert_called_once()


@pytest.mark.parametrize(
    "access",
    [
        lambda bus: bus.read("Present_Position", "dummy_1", normalize=False),
        lambda bus: bus.write("Goal_Position", "dummy_1", 1337, normalize=False),
        lambda bus: bus.sync_read("Present_Position", normalize=False),
        lambda bus: bus.sync_write("Goal_Position", 1337, normalize=False),
    ],
    ids=["read", "write", "sync_read", "sync_write"],
)
def test_a_motor_that_does_not_answer_is_a_connection_error(access, bus):
    bus._bus.absent.add(1)

    with pytest.raises(ConnectionError, match="Timeout"):
        access(bus)


def test_retries_are_left_to_rustypot(bus):
    bus.read("Present_Position", "dummy_1", normalize=False, num_retry=1)
    bus.write("Goal_Position", "dummy_1", 1337, normalize=False, num_retry=2)
    bus.sync_read("Present_Position", "dummy_1", normalize=False, num_retry=3)
    bus.sync_write("Goal_Position", {"dummy_1": 1337}, normalize=False, num_retry=4)

    assert bus._bus.retries == [1, 2, 3, 4]


def test_ping(bus):
    bus._bus.seed(2, "model_number", 5678)

    assert bus.ping(2) == 5678

    bus._bus.absent.add(3)
    assert bus.ping(3) is None

    bus._bus.status[2] = 0x20
    assert bus.ping(2) is None


def test_a_motor_is_found_with_a_broadcast_ping_where_its_servo_answers_one(bus):
    bus._bus.seed(9, "model_number", 5678)

    assert bus._find_single_motor("dummy_3") == (1_000_000, 9)
    assert (bus._bus.broadcast_scans, bus._bus.scans) == ([DUMMY_2], [])


def test_a_motor_whose_servo_answers_no_broadcast_ping_is_found_with_a_sweep():
    bus = MockMotorsBus("/dev/dummy-port", {"dummy": Motor(1, "model_1", MotorNormMode.RANGE_M100_100)})
    bus.connect(handshake=False)
    bus._bus.seed(9, "model_number", 1234)

    assert bus._find_single_motor("dummy") == (1_000_000, 9)
    assert (bus._bus.broadcast_scans, bus._bus.scans) == ([], [DUMMY_1])


def test_setup_refuses_a_motor_of_another_model(bus):
    bus._bus.seed(9, "model_number", 5799)

    with pytest.raises(RuntimeError, match="different than the one expected: 5678"):
        bus._find_single_motor("dummy_3")


def test_the_motor_search_tries_the_default_baudrate_then_the_factory_one(bus):
    """A motor already set up answers on the first try, a new one on the second; then the
    rest, in a fixed order."""
    with (
        patch.object(MockBus, "set_baudrate", autospec=True) as set_baudrate,
        pytest.raises(RuntimeError, match="was not found"),
    ):
        bus._find_single_motor("dummy_3")

    assert [call.args[1] for call in set_baudrate.call_args_list] == [1_000_000, 500_000, 250_000]


def test_setup_motor_gives_the_motor_its_id_and_the_default_baudrate(dummy_motors):
    """A new motor answers at an id the bus does not have (9 here); rustypot reaches it
    there through its definition."""
    bus = MockMotorsBus("/dev/dummy-port", dummy_motors)

    bus.setup_motor("dummy_3", initial_baudrate=500_000, initial_id=9)

    assert bus._bus.setups == [
        ("change_id", DUMMY_2, 9, 3),
        ("change_baudrate", DUMMY_2, 3, 1_000_000),
    ]
    assert len(bus.opened) == 1
    assert bus._bus.baudrate == bus.default_baudrate


def test_a_failed_setup_puts_the_port_back_at_the_default_baudrate(dummy_motors):
    bus = MockMotorsBus("/dev/dummy-port", dummy_motors)
    bus.connect(handshake=False)
    bus._bus.absent.add(9)

    with pytest.raises(ConnectionError, match="'dummy_3'"):
        bus.setup_motor("dummy_3", initial_baudrate=500_000, initial_id=9)

    assert bus.is_connected
    assert bus._bus.baudrate == bus.default_baudrate


def test_constructing_does_not_open_the_port(dummy_motors):
    bus = MockMotorsBus("/dev/dummy-port", dummy_motors)

    assert not bus.is_connected


def test_connect_reports_a_missing_port(dummy_motors):
    bus = MockMotorsBus("/dev/nope", dummy_motors)

    with (
        patch.object(
            MockMotorsBus, "_bus_class", return_value=MagicMock(side_effect=OSError("no such device"))
        ),
        pytest.raises(ConnectionError, match="lerobot-find-port"),
    ):
        bus.connect(handshake=False)

    assert not bus.is_connected


def test_a_failed_handshake_closes_the_port(dummy_motors):
    bus = MockMotorsBus("/dev/dummy-port", dummy_motors)

    with (
        patch.object(MockMotorsBus, "_handshake", side_effect=RuntimeError("motor check failed")),
        pytest.raises(RuntimeError, match="motor check failed"),
    ):
        bus.connect()

    assert not bus.is_connected
    assert bus.opened[-1].closed


def test_a_motor_of_another_model_fails_the_handshake(bus):
    """model_2 and model_3 share a control table, yet the motor set as model_3 must be one."""
    for id_ in bus.ids:
        bus._bus.seed(id_, "model_number", 5678)

    with pytest.raises(RuntimeError, match=r"2 \(dummy_2\): expected 5799, found 5678"):
        bus._handshake()


def test_a_lost_reply_costs_a_tick_not_a_second(bus):
    """One reply lost in a 30 Hz loop must not stall it."""
    assert bus._bus.timeout <= 0.05


def test_torque_off_asks_for_every_motor_and_names_the_ones_that_failed(bus):
    """rustypot tries every motor, so one that does not answer leaves no other under torque."""
    bus._bus.absent.update({1, 3})

    with pytest.raises(ConnectionError, match=r"'dummy_1' \(id 1\).*'dummy_3' \(id 3\)"):
        bus.disable_torque(num_retry=2)

    assert bus._bus.torques == [([1, 2, 3], False, 2)]
    assert bus._bus.stored(2, "torque_enable") == 0


def test_torque_on_goes_to_the_motors_asked(bus):
    bus.enable_torque(["dummy_1", "dummy_3"])

    assert bus._bus.torques == [([1, 3], True, 0)]


def test_torque_needs_a_connected_bus(dummy_motors):
    bus = MockMotorsBus("/dev/dummy-port", dummy_motors)

    with pytest.raises(DeviceNotConnectedError):
        bus.disable_torque()


def test_an_operating_mode_is_written_as_the_value_of_its_name(bus):
    bus.set_operating_mode("velocity", ["dummy_1", "dummy_2"])

    assert bus._bus.writes == [(1, "operating_mode", 1), (2, "operating_mode", 1)]


def test_a_mode_the_servo_does_not_have_is_refused(bus):
    with pytest.raises(ValueError, match="no 'step' operating mode"):
        bus.set_operating_mode("step")

    assert bus._bus.writes == []


def test_disconnect_closes_the_port_when_torque_off_fails(bus):
    rustypot_bus = bus._bus
    rustypot_bus.absent.add(1)

    with pytest.raises(ConnectionError):
        bus.disconnect()

    assert rustypot_bus.closed
    assert not bus.is_connected


def test_disconnect_closes_the_port(bus):
    rustypot_bus = bus._bus

    bus.disconnect(disable_torque=False)
    bus.disconnect(disable_torque=False)

    assert rustypot_bus.closed
    assert not bus.is_connected


def test_changing_the_baudrate_keeps_the_port_open(bus):
    bus.set_baudrate(57_600)

    assert bus._bus.baudrate == 57_600
    assert bus.is_connected


def test_the_port_reopens_at_the_default_baudrate(bus):
    bus.set_baudrate(57_600)
    bus.disconnect(disable_torque=False)
    bus.connect(handshake=False)

    assert bus._bus.baudrate == bus.default_baudrate
