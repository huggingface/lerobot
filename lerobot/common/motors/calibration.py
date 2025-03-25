#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import json
import select
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .dynamixel.dynamixel import DynamixelMotorsBus
from .feetech.feetech import FeetechMotorsBus
from .motors_bus import MotorsBus


@dataclass
class MotorCalibration:
    name: str
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


def find_offset(motorbus: MotorsBus):
    input("Move robot to the middle of its range of motion and press ENTER....")
    for name in motorbus.names:
        # Also reset to defaults
        if isinstance(motorbus, FeetechMotorsBus):
            motorbus.write("Offset", name, 0, raw_value=True)
            motorbus.write("Min_Angle_Limit", name, 0, raw_value=True)
            motorbus.write("Max_Angle_Limit", name, 4095, raw_value=True)
        elif isinstance(motorbus, DynamixelMotorsBus):
            motorbus.write("Homing_Offset", name, 0, raw_value=True)
            motorbus.write("Min_Position_Limit", name, 0, raw_value=True)
            motorbus.write("Max_Position_Limit", name, 4095, raw_value=True)
        else:
            raise ValueError("Motorbus instance is unknown")

    middle_values = motorbus.sync_read("Present_Position", raw_values=True)

    offsets = {}
    for name, pos in middle_values.items():
        offset = pos - 2047  # Center the middle reading at 2047.
        set_offset(motorbus, offset, name)
        offsets[name] = offset

    return offsets


def find_min_max(motorbus: MotorsBus):
    print("Move all joints (except wrist_roll; id = 5) sequentially through their entire ranges of motion.")
    print("Recording positions. Press ENTER to stop...")

    recorded_data = {name: [] for name in motorbus.names}

    while True:
        positions = motorbus.sync_read("Present_Position", raw_values=True)
        for name in motorbus.names:
            recorded_data[name].append(positions[name])
        time.sleep(0.01)

        # Check if user pressed Enter
        ready_to_read, _, _ = select.select([sys.stdin], [], [], 0)
        if ready_to_read:
            line = sys.stdin.readline()
            if line.strip() == "":
                break

    min_max = {}
    for name in motorbus.names:
        motor_values = recorded_data[name]
        raw_min = min(motor_values)
        raw_max = max(motor_values)

        if name == "wrist_roll":
            physical_min = 0
            physical_max = 4095
        else:
            physical_min = int(raw_min)
            physical_max = int(raw_max)

        set_min_max(motorbus, physical_min, physical_max, name)
        min_max[name] = {"min": physical_min, "max": physical_max}

    return min_max


def set_calibration(motorbus: MotorsBus, calibration_fpath: Path) -> None:
    with open(calibration_fpath) as f:
        calibration = json.load(f)

    motorbus.calibration = {int(id_): val for id_, val in calibration.items()}

    for _, cal_data in motorbus.calibration.items():
        name = cal_data.get("name")
        if name not in motorbus.names:
            print(f"Motor name '{name}' from calibration not found in arm names.")
            continue

        set_offset(motorbus, cal_data["homing_offset"], name)
        set_min_max(motorbus, cal_data["min"], cal_data["max"], name)


def set_offset(motorbus: MotorsBus, homing_offset: int, name: str):
    homing_offset = int(homing_offset)

    # Clamp to [-2047..+2047]
    if homing_offset > 2047:
        homing_offset = 2047
        print(
            f"Warning: '{homing_offset}' is getting clamped because its larger then 2047; This should not happen!"
        )
    elif homing_offset < -2047:
        homing_offset = -2047
        print(
            f"Warning: '{homing_offset}' is getting clamped because its smaller then -2047; This should not happen!"
        )

    direction_bit = 1 if homing_offset < 0 else 0  # Determine the direction (sign) bit and magnitude
    magnitude = abs(homing_offset)
    servo_offset = (
        direction_bit << 11
    ) | magnitude  # Combine sign bit (bit 11) with the magnitude (bits 0..10)

    if isinstance(motorbus, FeetechMotorsBus):
        motorbus.write("Offset", name, servo_offset, raw_value=True)
        read_offset = motorbus.sync_read("Offset", name, raw_values=True)
    elif isinstance(motorbus, DynamixelMotorsBus):
        motorbus.write("Homing_Offset", name, servo_offset, raw_value=True)
        read_offset = motorbus.sync_read("Homing_Offset", name, raw_values=True)
    else:
        raise ValueError("Motorbus instance is unknown")

    if read_offset[name] != servo_offset:
        raise ValueError(
            f"Offset not set correctly for motor '{name}'. read: {read_offset} does not equal {servo_offset}"
        )


def set_min_max(motorbus: MotorsBus, min: int, max: int, name: str):
    if isinstance(motorbus, FeetechMotorsBus):
        motorbus.write("Min_Angle_Limit", name, min, raw_value=True)
        motorbus.write("Max_Angle_Limit", name, max, raw_value=True)

        read_min = motorbus.sync_read("Min_Angle_Limit", name, raw_values=True)
        read_max = motorbus.sync_read("Max_Angle_Limit", name, raw_values=True)
    elif isinstance(motorbus, DynamixelMotorsBus):
        motorbus.write("Min_Position_Limit", name, min, raw_value=True)
        motorbus.write("Max_Position_Limit", name, max, raw_value=True)

        read_min = motorbus.sync_read("Min_Position_Limit", name, raw_values=True)
        read_max = motorbus.sync_read("Max_Position_Limit", name, raw_values=True)
    else:
        raise ValueError("Motorbus instance is unknown")

    if read_min[name] != min:
        raise ValueError(
            f"Min is not set correctly for motor '{name}'. read: {read_min} does not equal {min}"
        )

    if read_max[name] != max:
        raise ValueError(
            f"Max is not set correctly for motor '{name}'. read: {read_max} does not equal {max}"
        )
