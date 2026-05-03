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

import logging

from lerobot.motors import Motor, MotorCalibration

from .feetech import (
    FeetechMotorsBus,
    OperatingMode,
)

logger = logging.getLogger(__name__)


def calibrate_partial(
    bus: FeetechMotorsBus,
    existing_calibration: dict[str, MotorCalibration],
    motors: list[str],
    device_name: str,
    full_turn_motors: frozenset[str] = frozenset(),
) -> dict[str, MotorCalibration]:
    if not existing_calibration:
        raise ValueError(
            "No existing calibration found. Run full calibration first before calibrating specific motors."
        )

    updated_calibration = existing_calibration.copy()
    motors_to_calibrate: dict[str, Motor] = {}
    logger.info(f"\nRunning partial calibration of {device_name} for motors: {set(motors)}")
    bus.disable_torque()
    for motor in motors:
        if motor not in bus.motors:
            raise ValueError(
                f"Motor '{motor}' not found in the bus. Available motors: {list(bus.motors.keys())}"
            )
        motors_to_calibrate[motor] = bus.motors[motor]
        bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    # calibrate motors one by one
    logger.info(
        "Calibration of selected motors will be done sequentially. During calibration, all the motors will have their torque disabled and can be moved freely, but only the specified motor will be calibrated."
    )
    for motor, m in motors_to_calibrate.items():
        input(f"Move {motor} to the middle of its range of motion and press ENTER....")
        homing_offset = int(bus.set_half_turn_homings([motor])[motor])
        if motor in full_turn_motors:
            range_min_val, range_max_val = 0, 4095
            input(
                f"'{motor}' is set as the full turn motor with a fixed range of [0, 4095]. Homing offset was recorded but no further calibration is needed. Press ENTER to continue..."
            )
        else:
            input(
                f"Move {motor} through its entire range of motion (calibration of other motors won't be affected).\nRecording positions. Press ENTER to stop..."
            )
            range_mins, range_maxs = bus.record_ranges_of_motion([motor])
            range_min_val, range_max_val = int(range_mins[motor]), int(range_maxs[motor])

        updated_calibration[motor] = MotorCalibration(
            id=m.id,
            drive_mode=0,
            homing_offset=homing_offset,
            range_min=range_min_val,
            range_max=range_max_val,
        )

    return updated_calibration
