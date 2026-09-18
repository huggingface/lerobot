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

"""Dynamixel-specific behaviour.

Register access itself is family-agnostic and covered once in test_motors_bus.py;
what is tested here is what Dynamixel does differently -- little-endian registers,
two's-complement homing offsets, and a drive mode that is a motor register rather
than a calibration-only field.
"""

from unittest.mock import MagicMock, patch

import pytest

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.dynamixel import DynamixelMotorsBus
from lerobot.motors.dynamixel.tables import X_SERIES_CONTROL_TABLE
from lerobot.motors.encoding_utils import encode_twos_complement
from tests.mocks.mock_motors_bus import MockTransport


@pytest.fixture
def dummy_motors() -> dict[str, Motor]:
    return {
        "dummy_1": Motor(1, "xl430-w250", MotorNormMode.RANGE_M100_100),
        "dummy_2": Motor(2, "xm540-w270", MotorNormMode.RANGE_M100_100),
        "dummy_3": Motor(3, "xl330-m077", MotorNormMode.RANGE_M100_100),
    }


@pytest.fixture
def dummy_calibration(dummy_motors) -> dict[str, MotorCalibration]:
    drive_modes = [0, 1, 0]
    homings = [-709, -2006, 1624]
    mins = [43, 27, 145]
    maxes = [1335, 3608, 3999]
    return {
        motor: MotorCalibration(
            id=m.id,
            drive_mode=drive_modes[m.id - 1],
            homing_offset=homings[m.id - 1],
            range_min=mins[m.id - 1],
            range_max=maxes[m.id - 1],
        )
        for motor, m in dummy_motors.items()
    }


def make_bus(motors, calibration=None) -> DynamixelMotorsBus:
    bus = DynamixelMotorsBus(port="/dev/dummy-port", motors=motors, calibration=calibration)
    bus._io = MockTransport()
    bus.connect(handshake=False)
    return bus


def seed(bus, data_name: str, motor_id: int, value: int) -> None:
    addr, length = X_SERIES_CONTROL_TABLE[data_name]
    bus._io.seed(motor_id, addr, bytes(bus._split_into_byte_chunks(value, length)))


def written(bus, data_name: str, motor_id: int) -> int:
    addr, length = X_SERIES_CONTROL_TABLE[data_name]
    return bus._join_byte_chunks(bus._io.stored(motor_id, addr, length), length)


@pytest.mark.parametrize(
    "value, length, expected",
    [
        (0x12,       1, [0x12]),
        (0x1234,     2, [0x34, 0x12]),
        (0x12345678, 4, [0x78, 0x56, 0x34, 0x12]),
    ],
    ids=[
        "1 byte",
        "2 bytes",
        "4 bytes",
    ],
)  # fmt: skip
def test__split_into_byte_chunks(value, length, expected):
    bus = DynamixelMotorsBus("", {})
    assert bus._split_into_byte_chunks(value, length) == expected
    assert bus._join_byte_chunks(bytes(expected), length) == value


def test_abc_implementation(dummy_motors):
    """Instantiation should raise an error if the class doesn't implement abstract methods/properties."""
    DynamixelMotorsBus(port="/dev/dummy-port", motors=dummy_motors)


def test_is_calibrated(dummy_motors, dummy_calibration):
    bus = make_bus(dummy_motors, dummy_calibration)
    for cal in dummy_calibration.values():
        seed(bus, "Drive_Mode", cal.id, cal.drive_mode)
        seed(bus, "Homing_Offset", cal.id, encode_twos_complement(cal.homing_offset, 4))
        seed(bus, "Min_Position_Limit", cal.id, cal.range_min)
        seed(bus, "Max_Position_Limit", cal.id, cal.range_max)

    assert bus.is_calibrated


def test_reset_calibration(dummy_motors):
    bus = make_bus(dummy_motors)

    bus.reset_calibration()

    for motor in dummy_motors.values():
        assert written(bus, "Homing_Offset", motor.id) == 0
        assert written(bus, "Min_Position_Limit", motor.id) == 0
        assert written(bus, "Max_Position_Limit", motor.id) == 4095


def test_set_half_turn_homings(dummy_motors):
    """Homing offsets are assumed to start at 0, so Present_Position == Actual_Position."""
    current_positions = {1: 1337, 2: 42, 3: 3672}
    expected_homings = {1: 710, 2: 2005, 3: -1625}  # 2047 - position

    bus = make_bus(dummy_motors)
    for id_, position in current_positions.items():
        seed(bus, "Present_Position", id_, position)
    bus.reset_calibration = MagicMock()

    bus.set_half_turn_homings()

    bus.reset_calibration.assert_called_once()
    for id_, homing in expected_homings.items():
        assert written(bus, "Homing_Offset", id_) == encode_twos_complement(homing, 4)


def test_record_ranges_of_motion(dummy_motors):
    sweeps = [
        {"dummy_1": 351, "dummy_2": 28, "dummy_3": 4002},
        {"dummy_1": 42, "dummy_2": 3600, "dummy_3": 2999},
        {"dummy_1": 1337, "dummy_2": 2444, "dummy_3": 146},
    ]
    bus = make_bus(dummy_motors)

    with (
        patch("lerobot.motors.motors_bus.enter_pressed", side_effect=[False, True]),
        patch("lerobot.motors.motors_bus.time.sleep") as mock_sleep,
        patch.object(bus, "sync_read", side_effect=sweeps) as mock_sync_read,
    ):
        mins, maxes = bus.record_ranges_of_motion(display_values=False)

    assert mock_sync_read.call_count == 3
    assert all(call.kwargs["num_retry"] == 5 for call in mock_sync_read.call_args_list)
    mock_sleep.assert_called_once_with(0.02)
    assert mins == {"dummy_1": 42, "dummy_2": 28, "dummy_3": 146}
    assert maxes == {"dummy_1": 1337, "dummy_2": 3600, "dummy_3": 4002}
