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
import numpy as np

from ..motors_bus import MotorsBus, TorqueMode
from .feetech import (
    CalibrationMode,
    FeetechMotorsBus,
)

URL_TEMPLATE = (
    "https://raw.githubusercontent.com/huggingface/lerobot/main/media/{robot}/{arm}_{position}.webp"
)


def disable_torque(arm: MotorsBus):
    if (arm.read("Torque_Enable") != TorqueMode.DISABLED.value).any():
        raise ValueError("To run calibration, the torque must be disabled on all motors.")


def get_calibration_modes(arm: MotorsBus):
    """Returns calibration modes for each motor (DEGREE for rotational, LINEAR for gripper)."""
    return [
        CalibrationMode.LINEAR.name if name == "gripper" else CalibrationMode.DEGREE.name
        for name in arm.motor_names
    ]


def reset_offset(motor_id, motor_bus):
    # Open the write lock, changes to EEPROM do NOT persist yet
    motor_bus.write("Lock", 1)

    # Set offset to 0
    motor_name = motor_bus.motor_names[motor_id - 1]
    motor_bus.write("Offset", 0, motor_names=[motor_name])

    # Close the write lock, changes to EEPROM do persist
    motor_bus.write("Lock", 0)

    # Confirm that the offset is zero by reading it back
    confirmed_offset = motor_bus.read("Offset")[motor_id - 1]
    print(f"Offset for motor {motor_id} reset to: {confirmed_offset}")
    return confirmed_offset


def calibrate_homing_motor(motor_id, motor_bus):
    reset_offset(motor_id, motor_bus)

    home_ticks = motor_bus.read("Present_Position")[motor_id - 1]  # Read index starts at 0
    print(f"Encoder offset (present position in homing position): {home_ticks}")

    return home_ticks


def calibrate_linear_motor(motor_id, motor_bus):
    motor_names = motor_bus.motor_names
    motor_name = motor_names[motor_id - 1]

    reset_offset(motor_id, motor_bus)

    input(f"Close the {motor_name}, then press Enter...")
    start_pos = motor_bus.read("Present_Position")[motor_id - 1]  # Read index starts ar 0
    print(f"  [Motor {motor_id}] start position recorded: {start_pos}")

    input(f"Open the {motor_name} fully, then press Enter...")
    end_pos = motor_bus.read("Present_Position")[motor_id - 1]  # Read index starts ar 0
    print(f"  [Motor {motor_id}] end position recorded: {end_pos}")

    return start_pos, end_pos


def single_motor_calibration(arm: MotorsBus, motor_id: int):
    """Calibrates a single motor and returns its calibration data for updating the calibration file."""

    disable_torque(arm)
    print(f"\n--- Calibrating Motor {motor_id} ---")

    start_pos = 0
    end_pos = 0
    encoder_offset = 0

    if motor_id == 6:
        start_pos, end_pos = calibrate_linear_motor(motor_id, arm)
    else:
        input("Move the motor to (zero) position, then press Enter...")
        encoder_offset = calibrate_homing_motor(motor_id, arm)

    print(f"Calibration for motor ID:{motor_id} done.")

    # Create a calibration dictionary for the single motor
    calib_dict = {
        "homing_offset": int(encoder_offset),
        "drive_mode": 0,
        "start_pos": int(start_pos),
        "end_pos": int(end_pos),
        "calib_mode": get_calibration_modes(arm)[motor_id - 1],
        "motor_name": arm.motor_names[motor_id - 1],
    }

    return calib_dict


def run_full_arm_calibration(arm: MotorsBus, robot_type: str, arm_name: str, arm_type: str):
    """
    Runs a full calibration process for all motors in a robotic arm.

    This function calibrates each motor in the arm, determining encoder offsets and
    start/end positions for linear and rotational motors. The calibration data is then
    stored in a dictionary for later use.

    **Calibration Process:**
    - The user is prompted to move the arm to its homing position before starting.
    - Motors with rotational motion are calibrated using a homing method.
    - Linear actuators (e.g., grippers) are calibrated separately.
    - Encoder offsets, start positions, and end positions are recorded.

    **Example Usage:**
    ```python
    run_full_arm_calibration(arm, "so100", "left", "follower")
    ```
    """
    disable_torque(arm)

    print(f"\nRunning calibration of {robot_type} {arm_name} {arm_type}...")

    print("\nMove arm to homing position (middle)")
    print(
        "See: " + URL_TEMPLATE.format(robot=robot_type, arm=arm_type, position="zero")
    )  # TODO(pepijn): replace with new instruction homing pos (all motors in middle) in tutorial
    input("Press Enter to continue...")

    start_positions = np.zeros(len(arm.motor_indices))
    end_positions = np.zeros(len(arm.motor_indices))
    encoder_offsets = np.zeros(len(arm.motor_indices))

    modes = get_calibration_modes(arm)

    for i, motor_id in enumerate(arm.motor_indices):
        if modes[i] == CalibrationMode.DEGREE.name:
            encoder_offsets[i] = calibrate_homing_motor(motor_id, arm)
            start_positions[i] = 0
            end_positions[i] = 0

    for i, motor_id in enumerate(arm.motor_indices):
        if modes[i] == CalibrationMode.LINEAR.name:
            start_positions[i], end_positions[i] = calibrate_linear_motor(motor_id, arm)
            encoder_offsets[i] = 0

    print("\nMove arm to rest position")
    input("Press Enter to continue...")

    print(f"\n calibration of {robot_type} {arm_name} {arm_type} done!")

    # Force drive_mode values (can be static)
    drive_modes = [0, 1, 0, 0, 1, 0]

    calib_dict = {
        "homing_offset": encoder_offsets.astype(int).tolist(),
        "drive_mode": drive_modes,
        "start_pos": start_positions.astype(int).tolist(),
        "end_pos": end_positions.astype(int).tolist(),
        "calib_mode": get_calibration_modes(arm),
        "motor_names": arm.motor_names,
    }
    return calib_dict


def run_full_auto_arm_calibration(arm: MotorsBus, robot_type: str, arm_name: str, arm_type: str):
    """TODO(pepijn): Add this method later as extra
    Example of usage:
    ```python
    run_full_auto_arm_calibration(arm, "so100", "left", "follower")
    ```
    """
    print(f"\nRunning calibration of {robot_type} {arm_name} {arm_type}...")


def apply_feetech_offsets_from_calibration(motorsbus: FeetechMotorsBus, calibration_dict: dict):
    """
    Reads 'calibration_dict' containing 'homing_offset' and 'motor_names',
    then writes each motor's offset to the servo's internal Offset (0x1F) in EPROM.

    This version is modified so each homed position (originally 0) will now read
    2047, i.e. 180° away from 0 in the 4096-count circle. Offsets are permanently
    stored in EEPROM, so the servo's Present_Position is hardware-shifted even
    after power cycling.

    Steps:
      1) Subtract 2047 from the old offset (so 0 -> 2047).
      2) Clamp to [-2047..+2047].
      3) Encode sign bit and magnitude into a 12-bit number.
    """

    homing_offsets = calibration_dict["homing_offset"]
    motor_names = calibration_dict["motor_names"]
    start_pos = calibration_dict["start_pos"]

    # Open the write lock, changes to EEPROM do NOT persist yet
    motorsbus.write("Lock", 1)

    # For each motor, set the 'Offset' parameter
    for m_name, old_offset in zip(motor_names, homing_offsets, strict=False):
        # If bus doesn’t have a motor named m_name, skip
        if m_name not in motorsbus.motors:
            print(f"Warning: '{m_name}' not found in motorsbus.motors; skipping offset.")
            continue

        if m_name == "gripper":
            old_offset = start_pos  # If gripper set the offset to the start position of the gripper
            continue

        # Shift the offset so the homed position reads 2047
        new_offset = old_offset - 2047

        # Clamp to [-2047..+2047]
        if new_offset > 2047:
            new_offset = 2047
            print(
                f"Warning: '{new_offset}' is getting clamped because its larger then 2047; This should not happen!"
            )
        elif new_offset < -2047:
            new_offset = -2047
            print(
                f"Warning: '{new_offset}' is getting clamped because its smaller then -2047; This should not happen!"
            )

        # Determine the direction (sign) bit and magnitude
        direction_bit = 1 if new_offset < 0 else 0
        magnitude = abs(new_offset)

        # Combine sign bit (bit 11) with the magnitude (bits 0..10)
        servo_offset = (direction_bit << 11) | magnitude

        # Write offset to servo
        motorsbus.write("Offset", servo_offset, motor_names=m_name)
        print(
            f"Set offset for {m_name}: "
            f"old_offset={old_offset}, new_offset={new_offset}, servo_encoded={magnitude} + direction={direction_bit}"
        )

    motorsbus.write("Lock", 0)
    print("Offsets have been saved to EEPROM successfully.")
