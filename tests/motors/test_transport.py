#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""The rustypot adapter itself.

The bus tests swap the transport out for a fake, so what is checked here is the
part they never reach: which controller a protocol maps to, how a port is opened,
closed and rebuilt, and what the raw bindings are handed.
"""

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("rustypot", reason="rustypot is required (install lerobot[feetech])")

import rustypot

from lerobot.motors.transport import PROTOCOL_V1, PROTOCOL_V2, RustypotTransport


@pytest.fixture
def controllers():
    """Stand in for rustypot's two generated controller classes."""
    v1, v2 = MagicMock(name="Sts3215PyController"), MagicMock(name="Xl330PyController")
    with (
        patch.object(rustypot, "Sts3215PyController", v1),
        patch.object(rustypot, "Xl330PyController", v2),
    ):
        yield {PROTOCOL_V1: v1, PROTOCOL_V2: v2}


@pytest.fixture
def transport(controllers):
    transport = RustypotTransport("/dev/dummy-port", PROTOCOL_V2, baudrate=1_000_000, timeout_ms=20)
    transport.open()
    return transport


def test_rejects_unknown_protocol():
    with pytest.raises(ValueError, match="Unknown protocol"):
        RustypotTransport("/dev/dummy-port", "v3", baudrate=1_000_000, timeout_ms=20)


def test_constructing_does_not_touch_the_port(controllers):
    transport = RustypotTransport("/dev/dummy-port", PROTOCOL_V2, baudrate=1_000_000, timeout_ms=20)

    assert not transport.is_open
    controllers[PROTOCOL_V2].assert_not_called()


@pytest.mark.parametrize("protocol", [PROTOCOL_V1, PROTOCOL_V2])
def test_open_builds_the_controller_of_the_protocol(protocol, controllers):
    """One controller per protocol -- the raw API is the same across a family."""
    transport = RustypotTransport("/dev/dummy-port", protocol, baudrate=500_000, timeout_ms=20)
    transport.open()

    other = controllers[PROTOCOL_V1 if protocol == PROTOCOL_V2 else PROTOCOL_V2]
    other.assert_not_called()
    # rustypot counts the timeout in seconds, LeRobot in milliseconds.
    controllers[protocol].assert_called_once_with(
        serial_port="/dev/dummy-port", baudrate=500_000, timeout=0.02
    )
    assert transport.is_open


def test_open_is_idempotent(transport, controllers):
    transport.open()

    controllers[PROTOCOL_V2].assert_called_once()


def test_open_reports_a_missing_port(controllers):
    controllers[PROTOCOL_V2].side_effect = OSError("no such device")
    transport = RustypotTransport("/dev/nope", PROTOCOL_V2, baudrate=1_000_000, timeout_ms=20)

    with pytest.raises(ConnectionError, match="lerobot-find-port"):
        transport.open()

    assert not transport.is_open


def test_close_releases_the_port(transport, controllers):
    controller = controllers[PROTOCOL_V2].return_value

    transport.close()

    controller.close.assert_called_once()
    assert not transport.is_open


def test_close_is_idempotent(transport, controllers):
    transport.close()
    transport.close()

    controllers[PROTOCOL_V2].return_value.close.assert_called_once()


@pytest.mark.parametrize(
    "setter, value, expected_kwargs",
    [
        ("set_baudrate", 57_600, {"baudrate": 57_600, "timeout": 0.02}),
        ("set_timeout", 5, {"baudrate": 1_000_000, "timeout": 0.005}),
    ],
)
def test_changing_a_port_setting_rebuilds_the_controller(
    setter, value, expected_kwargs, transport, controllers
):
    """rustypot fixes baud rate and timeout when the controller is built."""
    controller = controllers[PROTOCOL_V2]
    controller.reset_mock()

    getattr(transport, setter)(value)

    controller.assert_called_once_with(serial_port="/dev/dummy-port", **expected_kwargs)
    assert transport.is_open


def test_setting_a_port_setting_to_its_current_value_is_a_no_op(transport, controllers):
    controllers[PROTOCOL_V2].reset_mock()

    transport.set_baudrate(transport.baudrate)

    controllers[PROTOCOL_V2].assert_not_called()
    controllers[PROTOCOL_V2].return_value.close.assert_not_called()


def test_changing_a_port_setting_while_closed_leaves_it_closed(controllers):
    transport = RustypotTransport("/dev/dummy-port", PROTOCOL_V2, baudrate=1_000_000, timeout_ms=20)

    transport.set_baudrate(57_600)

    assert transport.baudrate == 57_600
    assert not transport.is_open
    controllers[PROTOCOL_V2].assert_not_called()


def test_read_returns_bytes_and_the_status_error(transport, controllers):
    controllers[PROTOCOL_V2].return_value.read_raw_data_with_error.return_value = ([0x34, 0x12], 2)

    assert transport.read(1, 132, 2) == (b"\x34\x12", 2)
    controllers[PROTOCOL_V2].return_value.read_raw_data_with_error.assert_called_once_with(1, 132, 2)


def test_write_hands_the_bindings_a_list_of_ints(transport, controllers):
    controllers[PROTOCOL_V2].return_value.write_raw_data_with_error.return_value = 0

    assert transport.write(1, 116, b"\x34\x12") == 0
    controllers[PROTOCOL_V2].return_value.write_raw_data_with_error.assert_called_once_with(
        1, 116, [0x34, 0x12]
    )


def test_sync_read_returns_one_bytes_object_per_motor(transport, controllers):
    controllers[PROTOCOL_V2].return_value.sync_read_raw_data.return_value = [[0x01, 0x02], [0x03, 0x04]]

    assert transport.sync_read([1, 2], 132, 2) == [b"\x01\x02", b"\x03\x04"]
    controllers[PROTOCOL_V2].return_value.sync_read_raw_data.assert_called_once_with([1, 2], 132, 2)


def test_sync_write_hands_the_bindings_lists_of_ints(transport, controllers):
    transport.sync_write([1, 2], 116, [b"\x01\x02", b"\x03\x04"])

    controllers[PROTOCOL_V2].return_value.sync_write_raw_data.assert_called_once_with(
        [1, 2], 116, [[0x01, 0x02], [0x03, 0x04]]
    )
