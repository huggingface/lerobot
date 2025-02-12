"""Logic to calibrate a robot arm built with feetech motors"""
# TODO(rcadene, aliberts): move this logic into the robot code when refactoring

import numpy as np

from lerobot.common.robot_devices.motors.feetech import (
    CalibrationMode,
    TorqueMode,
)
from lerobot.common.robot_devices.motors.utils import MotorsBus

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


def calibrate_homing_motor(motor_id, motor_bus):
    """
    1) Reads servo ticks.
    2) Calculates the offset so that 'home_ticks' becomes 0.
    3) Returns the offset
    """

    home_ticks = motor_bus.read("Present_Position")[motor_id - 1]  # Read index starts at 0

    # Calculate how many ticks to shift so that 'home_ticks' becomes 0
    raw_offset = -home_ticks  # negative of home_ticks
    encoder_offset = raw_offset % 4096  # wrap to [0..4095]

    # Convert to a signed range [-2048..2047]
    if encoder_offset > 2047:
        encoder_offset -= 4096

    print(f"Encoder offset: {encoder_offset}")

    return encoder_offset


def calibrate_linear_motor(motor_id, motor_bus):
    motor_names = motor_bus.motor_names
    motor_name = motor_names[motor_id - 1]

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
        input("Move the motor to middle (home) position, then press Enter...")
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

    print("\nMove arm to homing position")
    print(
        "See: " + URL_TEMPLATE.format(robot=robot_type, arm=arm_type, position="zero")
    )  # TODO(pepijn): replace with new instruction homing pos (all motors in center)
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

    calib_dict = {
        "homing_offset": encoder_offsets.astype(int).tolist(),
        "drive_mode": np.zeros(len(arm.motor_indices), dtype=int).tolist(),
        "start_pos": start_positions.astype(int).tolist(),
        "end_pos": end_positions.astype(int).tolist(),
        "calib_mode": get_calibration_modes(arm),
        "motor_names": arm.motor_names,
    }
    return calib_dict


def run_full_auto_arm_calibration(arm: MotorsBus, robot_type: str, arm_name: str, arm_type: str):
    """TODO: Add this method later as extra
    Example of usage:
    ```python
    run_full_auto_arm_calibration(arm, "so100", "left", "follower")
    ```
    """
    print(f"\nRunning calibration of {robot_type} {arm_name} {arm_type}...")
