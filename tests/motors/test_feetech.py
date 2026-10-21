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

"""Feetech behaviour of `SerialMotorsBus`, from the rustypot definitions.

Register access itself is family-agnostic and covered once in test_motors_bus.py;
what is tested here is what Feetech does differently -- STS and SCS definitions
sharing a bus, what the SCS lacks, the acceleration registers, the firmware check,
a drive mode applied in software, and the Phase register quirk of the sts3215.
"""

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("rustypot", reason="rustypot is required (install lerobot[serial-motors])")

import rustypot

from lerobot.motors import Motor, MotorCalibration, MotorNormMode, SerialMotorsBus
from lerobot.motors.feetech import FeetechMotorsBus
from tests.mocks.mock_motors_bus import make_bus, seed, written


@pytest.fixture
def dummy_motors() -> dict[str, Motor]:
    return {
        "dummy_1": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
        "dummy_2": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
        "dummy_3": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
    }


@pytest.fixture
def dummy_calibration(dummy_motors) -> dict[str, MotorCalibration]:
    homings = [-709, -2006, 1624]
    mins = [43, 27, 145]
    maxes = [1335, 3608, 3999]
    return {
        motor: MotorCalibration(
            id=m.id,
            drive_mode=0,
            homing_offset=homings[m.id - 1],
            range_min=mins[m.id - 1],
            range_max=maxes[m.id - 1],
        )
        for motor, m in dummy_motors.items()
    }


def test_sts_and_scs_share_a_bus():
    """Each motor is mapped to the definition of its model, STS and SCS on one bus."""
    bus = SerialMotorsBus(
        "",
        {
            "sts": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
            "scs": Motor(2, "scs0009", MotorNormMode.RANGE_M100_100),
        },
    )

    assert bus._definitions == {
        1: rustypot.Sts3215PyController.definition(),
        2: rustypot.Scs0009PyController.definition(),
    }


def test_feetech_motors_bus_is_a_deprecated_alias(dummy_motors):
    with pytest.warns(DeprecationWarning, match="SerialMotorsBus"):
        bus = FeetechMotorsBus(port="/dev/dummy-port", motors=dummy_motors)

    assert isinstance(bus, SerialMotorsBus)


def test_protocol_version_is_still_accepted(dummy_motors):
    """Code written for the SDK-based bus passes it; each motor's model now says it."""
    with pytest.warns(DeprecationWarning, match="protocol_version"):
        FeetechMotorsBus(port="/dev/dummy-port", motors=dummy_motors, protocol_version=0)


def test_motors_on_different_firmware_fail_the_handshake(dummy_motors):
    bus = make_bus(dummy_motors)
    for motor in dummy_motors.values():
        seed(bus, "Model_Number", motor.id, 777)
        seed(bus, "Firmware_Major_Version", motor.id, 3)
        seed(bus, "Firmware_Minor_Version", motor.id, 10)
    seed(bus, "Firmware_Minor_Version", 2, 9)

    with pytest.raises(RuntimeError, match="different firmware versions"):
        bus._handshake()


def test_the_firmware_is_compared_within_each_servo_family():
    """STS and SCS servos run firmware series of their own: only motors of one definition
    must agree."""
    bus = make_bus(
        {
            "sts": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
            "scs": Motor(2, "scs0009", MotorNormMode.RANGE_M100_100),
        }
    )
    for motor_id, model_number, major in [(1, 777, 3), (2, 1284, 1)]:
        seed(bus, "Model_Number", motor_id, model_number)
        seed(bus, "Firmware_Major_Version", motor_id, major)

    bus._handshake()


@pytest.mark.parametrize(
    "access",
    [
        lambda bus: bus.read("Present_Position", "dummy_1", normalize=False),
        lambda bus: bus.write("Goal_Position", "dummy_1", 1337, normalize=False),
        lambda bus: bus.sync_read("Present_Position", normalize=False),
    ],
    ids=["read", "write", "sync_read"],
)
def test_an_alarm_does_not_fail_an_access(access, dummy_motors):
    """A Feetech status only reports alarms: an overloaded motor still carries the access out."""
    bus = make_bus(dummy_motors)
    bus._bus.status[1] = 0x20

    access(bus)


def test_operating_modes_take_the_feetech_values(dummy_motors):
    bus = make_bus(dummy_motors)

    bus.set_operating_mode("position", "dummy_1")
    bus.set_operating_mode("velocity", "dummy_2")

    assert (written(bus, "Operating_Mode", 1), written(bus, "Operating_Mode", 2)) == (0, 1)


def test_configure_motors_writes_the_accelerations_each_servo_has():
    """The STS has both acceleration registers, the SCS only `Acceleration`."""
    bus = make_bus(
        {
            "sts": Motor(1, "sts3250", MotorNormMode.RANGE_M100_100),
            "scs": Motor(2, "scs0009", MotorNormMode.RANGE_M100_100),
        }
    )

    bus.configure_motors(maximum_acceleration=30, acceleration=40)

    assert bus._bus.writes == [
        (1, "return_delay_time", 0),
        (1, "maximum_acceleration", 30),
        (1, "acceleration", 40),
        (2, "return_delay_time", 0),
        (2, "acceleration", 40),
    ]


def test_the_drive_mode_is_applied_in_software(dummy_motors, dummy_calibration):
    """A Feetech has no Drive_Mode register: the calibration's drive mode inverts the
    normalised value."""
    calibration = {**dummy_calibration}
    calibration["dummy_2"] = MotorCalibration(
        id=2, drive_mode=1, homing_offset=0, range_min=27, range_max=3608
    )
    bus = make_bus(dummy_motors, calibration)

    assert bus._normalize({2: 27}) == {2: 100.0}


@pytest.mark.parametrize(
    "command",
    [
        lambda bus: bus.write("Goal_Position", "dummy_2", -100.0),
        lambda bus: bus.sync_write("Goal_Position", {"dummy_2": -100.0}),
    ],
    ids=["write", "sync_write"],
)
def test_a_command_is_inverted_in_software(command, dummy_motors, dummy_calibration):
    """dummy_2 has drive mode 1: a goal at the bottom of the range lands on its top."""
    calibration = {**dummy_calibration}
    calibration["dummy_2"] = MotorCalibration(
        id=2, drive_mode=1, homing_offset=0, range_min=27, range_max=3608
    )
    bus = make_bus(dummy_motors, calibration)

    command(bus)

    assert written(bus, "Goal_Position", 2) == 3608


def test_a_sm8512bl_set_as_sts3215_fails_the_handshake(dummy_motors):
    """Same control table, but not the motor the robot was built with."""
    bus = make_bus(dummy_motors)
    for motor in dummy_motors.values():
        seed(bus, "Model_Number", motor.id, 777)
    seed(bus, "Model_Number", 2, 11272)

    with pytest.raises(RuntimeError, match="expected 777, found 11272"):
        bus._handshake()


def test_is_calibrated(dummy_motors, dummy_calibration):
    bus = make_bus(dummy_motors, dummy_calibration)
    for cal in dummy_calibration.values():
        seed(bus, "Min_Position_Limit", cal.id, cal.range_min)
        seed(bus, "Max_Position_Limit", cal.id, cal.range_max)
        seed(bus, "Homing_Offset", cal.id, cal.homing_offset)

    assert bus.is_calibrated


def test_a_drive_mode_kept_in_software_does_not_count(dummy_motors, dummy_calibration):
    """A Feetech has no Drive_Mode register: the drive mode lives in the calibration alone."""
    calibration = {**dummy_calibration}
    calibration["dummy_2"] = MotorCalibration(
        id=2, drive_mode=1, homing_offset=-2006, range_min=27, range_max=3608
    )
    bus = make_bus(dummy_motors, calibration)
    for cal in calibration.values():
        seed(bus, "Min_Position_Limit", cal.id, cal.range_min)
        seed(bus, "Max_Position_Limit", cal.id, cal.range_max)
        seed(bus, "Homing_Offset", cal.id, cal.homing_offset)

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
    expected_homings = {1: -710, 2: -2005, 3: 1625}  # position - 2047

    bus = make_bus(dummy_motors)
    for id_, position in current_positions.items():
        seed(bus, "Present_Position", id_, position)
    bus.reset_calibration = MagicMock()

    bus.set_half_turn_homings()

    bus.reset_calibration.assert_called_once()
    for id_, homing in expected_homings.items():
        assert written(bus, "Homing_Offset", id_) == homing


@pytest.mark.parametrize(
    "position, expected_homing",
    [
        (0, -2047),
        (2047, 0),
        (4094, 2047),
        # 4095 - 2047 = 2048 does not fit an 11-bit sign-magnitude register: clamp to 2047.
        (4095, 2047),
    ],
)
def test_half_turn_homings_fit_the_sign_magnitude_register(position, expected_homing, dummy_motors):
    bus = make_bus(dummy_motors)
    seed(bus, "Present_Position", 1, position)

    bus.set_half_turn_homings("dummy_1")

    assert written(bus, "Homing_Offset", 1) == expected_homing


@pytest.mark.parametrize(
    "initial_phase, expected_phase",
    [
        (0b00010000, 0b00000000),  # bit 4 set - cleared
        (0b11111111, 0b11101111),  # all bits set - bit 4 cleared, others preserved
        (0b00000000, 0b00000000),  # bit 4 already 0 - unchanged
    ],
    ids=["bit4_set", "all_bits_set", "bit4_already_cleared"],
)
def test_configure_motors_clears_sts3215_phase_bit4(initial_phase, expected_phase, dummy_motors):
    """Phase register bit 4 (angle feedback mode) must be cleared for sts3215, other bits preserved."""
    bus = make_bus(dummy_motors)
    for motor in dummy_motors.values():
        seed(bus, "Phase", motor.id, initial_phase)

    with patch.object(bus, "write", wraps=bus.write) as mock_write:
        bus.configure_motors()

    write_data_names = [call.args[0] for call in mock_write.call_args_list]
    if initial_phase != expected_phase:
        for motor in dummy_motors.values():
            assert written(bus, "Phase", motor.id) == expected_phase
    else:  # ensure that phase is written only if it needs to be changed
        assert "Phase" not in write_data_names


def test_configure_motors_skips_phase_for_non_sts3215():
    """Phase register must not be touched for motors other than sts3215."""
    motors = {
        "dummy_1": Motor(1, "sts3250", MotorNormMode.RANGE_M100_100),
        "dummy_2": Motor(2, "sts3250", MotorNormMode.RANGE_M100_100),
        "dummy_3": Motor(3, "sts3250", MotorNormMode.RANGE_M100_100),
    }
    bus = make_bus(motors)

    with patch.object(bus, "read", wraps=bus.read) as mock_read:
        bus.configure_motors()
        read_data_names = [call.args[0] for call in mock_read.call_args_list]

    assert "Phase" not in read_data_names


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


def test_scs_has_no_homing_offset():
    """Calibration on the SCS is ranges only: the register does not exist, the bus neither
    reads nor writes it, and refuses a half-turn homing before touching any register."""
    bus = make_bus({"dummy": Motor(1, "scs0009", MotorNormMode.RANGE_M100_100)})
    seed(bus, "Min_Position_Limit", 1, 10)
    seed(bus, "Max_Position_Limit", 1, 1000)

    calibration = bus.read_calibration()

    assert calibration["dummy"].homing_offset == 0
    bus.write_calibration(calibration)
    bus.reset_calibration()
    writes = len(bus._bus.writes)
    with pytest.raises(ValueError, match="no homing offset"):
        bus.set_half_turn_homings()

    assert len(bus._bus.writes) == writes
    assert all(register != "homing_offset" for _, register, _ in bus._bus.writes)
    assert all(register != "homing_offset" for _, register in bus._bus.reads)
    assert all(register != "homing_offset" for _, register in bus._bus.sync_reads)
