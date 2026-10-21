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

"""Dynamixel behaviour of `SerialMotorsBus`, from the rustypot definitions.

Register access itself is family-agnostic and covered once in test_motors_bus.py;
what is tested here is what Dynamixel does differently -- two definitions sharing
a bus, homing offsets of the opposite sign, operating mode values, and a drive mode
that is a motor register rather than a calibration-only field.
"""

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("rustypot", reason="rustypot is required (install lerobot[serial-motors])")

import rustypot

from lerobot.motors import Motor, MotorCalibration, MotorNormMode, SerialMotorsBus
from lerobot.motors.dynamixel import DynamixelMotorsBus
from tests.mocks.mock_motors_bus import make_bus, seed, written


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


def test_each_motor_keeps_its_own_definition(dummy_motors):
    """Each motor is mapped to the definition of its model: the XL430's covers the XM540."""
    bus = SerialMotorsBus("", dummy_motors)

    assert bus._definitions == {
        1: rustypot.Xl430PyController.definition(),
        2: rustypot.Xl430PyController.definition(),
        3: rustypot.Xl330PyController.definition(),
    }


def test_dynamixel_motors_bus_is_a_deprecated_alias(dummy_motors):
    with pytest.warns(DeprecationWarning, match="SerialMotorsBus"):
        bus = DynamixelMotorsBus(port="/dev/dummy-port", motors=dummy_motors)

    assert isinstance(bus, SerialMotorsBus)


def test_operating_modes_take_the_dynamixel_values(dummy_motors):
    bus = make_bus(dummy_motors)

    bus.set_operating_mode("position", "dummy_1")
    bus.set_operating_mode("current_based_position", "dummy_3")

    assert (written(bus, "Operating_Mode", 1), written(bus, "Operating_Mode", 3)) == (3, 5)


def test_configure_motors_only_shortens_the_return_delay(dummy_motors):
    """Dynamixel servos have no Feetech acceleration registers."""
    bus = make_bus(dummy_motors)

    bus.configure_motors()

    assert {register for _, register, _ in bus._bus.writes} == {"return_delay_time"}


def test_the_drive_mode_is_left_to_the_motor(dummy_motors, dummy_calibration):
    """dummy_2 is set to drive mode 1, which its Drive_Mode register applies: normalising
    must not invert it a second time."""
    bus = make_bus(dummy_motors, dummy_calibration)

    assert bus._normalize({2: 27}) == {2: -100.0}


@pytest.mark.parametrize(
    "command",
    [
        lambda bus: bus.write("Goal_Position", "dummy_2", -100.0),
        lambda bus: bus.sync_write("Goal_Position", {"dummy_2": -100.0}),
    ],
    ids=["write", "sync_write"],
)
def test_a_command_is_not_inverted_a_second_time(command, dummy_motors, dummy_calibration):
    """dummy_2 has drive mode 1, applied by its Drive_Mode register: the goal is sent as is."""
    bus = make_bus(dummy_motors, dummy_calibration)

    command(bus)

    assert written(bus, "Goal_Position", 2) == 27


def calibrated_bus(motors: dict[str, Motor], calibration: dict[str, MotorCalibration]) -> SerialMotorsBus:
    """A bus whose motors hold `calibration`, motor by motor name."""
    bus = make_bus(motors, calibration)
    for motor, cal in calibration.items():
        motor_id = motors[motor].id
        seed(bus, "Drive_Mode", motor_id, cal.drive_mode)
        seed(bus, "Homing_Offset", motor_id, cal.homing_offset)
        seed(bus, "Min_Position_Limit", motor_id, cal.range_min)
        seed(bus, "Max_Position_Limit", motor_id, cal.range_max)
    return bus


def test_is_calibrated(dummy_motors, dummy_calibration):
    assert calibrated_bus(dummy_motors, dummy_calibration).is_calibrated


@pytest.mark.parametrize(
    "register, value",
    [("Min_Position_Limit", 44), ("Max_Position_Limit", 1336), ("Homing_Offset", -708), ("Drive_Mode", 1)],
)
def test_a_motor_off_its_calibration_is_not_calibrated(register, value, dummy_motors, dummy_calibration):
    bus = calibrated_bus(dummy_motors, dummy_calibration)
    seed(bus, register, 1, value)

    assert not bus.is_calibrated


def test_the_calibration_ids_are_not_compared(dummy_motors, dummy_calibration):
    """Calibrations are matched by motor name: the id a file holds does not count."""
    calibration = {motor: replace(cal, id=cal.id + 10) for motor, cal in dummy_calibration.items()}

    assert calibrated_bus(dummy_motors, calibration).is_calibrated


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
        assert written(bus, "Homing_Offset", id_) == homing
