"""
LCR Auto Configure: This program is used to automatically configure the Low Cost Robot (LCR) for the user.

The program will:
1. Disable all torque motors of provided LCR.
2. Ask the user to move the LCR to the position 1 (see CONFIGURING.md for more details).
3. Record the position of the LCR.
4. Ask the user to move the LCR to the position 2 (see CONFIGURING.md for more details).
5. Record the position of the LCR.
6. Ask the user to move back the LCR to the position 1.
7. Record the position of the LCR.
8. Calculate the offset of the LCR and save it to the configuration file.

It will also enable all appropriate operating modes for the LCR according if the LCR is a puppet or a master.
"""

import argparse
import time

import numpy as np

from lerobot.common.robot_devices.motors.dynamixel import DynamixelBus, OperatingMode, DriveMode, u32_to_i32, \
    i32_to_u32, retrieve_ids_and_command, TorqueMode


def pause():
    """
    Pause the program until the user presses the enter key.
    """
    input("Press Enter to continue...")


def prepare_configuration(arm: DynamixelBus):
    """
    Prepare the configuration for the LCR.
    :param arm: DynamixelBus
    """

    # To be configured, all servos must be in "torque disable" mode
    arm.sync_write_torque_enable(TorqueMode.DISABLED.value)

    # We need to work with 'extended position mode' (4) for all servos, because in joint mode (1) the servos can't
    # rotate more than 360 degrees (from 0 to 4095) And some mistake can happen while assembling the arm,
    # you could end up with a servo with a position 0 or 4095 at a crucial point See [
    # https://emanual.robotis.com/docs/en/dxl/x/x_series/#operating-mode11]
    arm.sync_write_operating_mode(OperatingMode.EXTENDED_POSITION.value, [1, 2, 3, 4, 5])

    # Gripper is always 'position control current based' (5)
    arm.write_operating_mode(OperatingMode.CURRENT_CONTROLLED_POSITION.value, 6)

    # We need to reset the homing offset for all servos
    arm.sync_write_homing_offset(0)

    # We need to work with 'normal drive mode' (0) for all servos
    arm.sync_write_drive_mode(DriveMode.NON_INVERTED.value)


def invert_appropriate_positions(positions: np.array, inverted: list[bool]) -> np.array:
    """
    Invert the appropriate positions.
    :param positions: numpy array of positions
    :param inverted: list of booleans to determine if the position should be inverted
    :return: numpy array of inverted positions
    """
    for i, invert in enumerate(inverted):
        if not invert and positions[i] is not None:
            positions[i] = -positions[i]

    return positions


def calculate_corrections(positions: np.array, inverted: list[bool]) -> np.array:
    """
    Calculate the corrections for the positions.
    :param positions: numpy array of positions
    :param inverted: list of booleans to determine if the position should be inverted
    :return: numpy array of corrections
    """

    wanted = wanted_position_1()

    correction = invert_appropriate_positions(positions, inverted)

    for i in range(len(positions)):
        if correction[i] is not None:
            if inverted[i]:
                correction[i] -= wanted[i]
            else:
                correction[i] += wanted[i]

    return correction


def calculate_nearest_rounded_positions(positions: np.array) -> np.array:
    """
    Calculate the nearest rounded positions.
    :param positions: numpy array of positions
    :return: numpy array of nearest rounded positions
    """

    return np.array(
        [round(positions[i] / 1024) * 1024 if positions[i] is not None else None for i in range(len(positions))])


def configure_homing(arm: DynamixelBus, inverted: list[bool]) -> np.array:
    """
    Configure the homing for the LCR.
    :param arm: DynamixelBus
    :param inverted: list of booleans to determine if the position should be inverted
    """

    # Reset homing offset for the servos
    arm.sync_write_homing_offset(0)

    # Get the present positions of the servos
    present_positions = arm.sync_read_present_position_i32()

    nearest_positions = calculate_nearest_rounded_positions(present_positions)

    correction = calculate_corrections(nearest_positions, inverted)

    # Write the homing offset for the servos
    arm.sync_write_homing_offset(correction)


def configure_drive_mode(arm: DynamixelBus):
    """
    Configure the drive mode for the LCR.
    :param arm: DynamixelBus
    :param homing: numpy array of homing
    """

    # Get current positions
    present_positions = arm.sync_read_present_position_i32()

    nearest_positions = calculate_nearest_rounded_positions(present_positions)

    # construct 'inverted' list comparing nearest_positions and wanted_position_2
    inverted = []

    for i in range(len(nearest_positions)):
        inverted.append(nearest_positions[i] != wanted_position_2()[i])

    # Write the drive mode for the servos
    arm.sync_write_drive_mode(
        [DriveMode.INVERTED.value if i else DriveMode.NON_INVERTED.value for i in inverted])

    return inverted


def wanted_position_1() -> np.array:
    """
    The present position wanted in position 1 for the arm
    """
    return np.array([0, -1024, 1024, 0, 0, 0])


def wanted_position_2() -> np.array:
    """
    The present position wanted in position 2 for the arm
    """
    return np.array([1024, 0, 0, 1024, 1024, -1024])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LCR Auto Configure: This program is used to automatically configure the Low Cost Robot (LCR) for "
                    "the user.")

    parser.add_argument("--port", type=str, required=True, help="The port of the LCR.")

    args = parser.parse_args()

    arm = DynamixelBus(
        args.port, {
            1: "x_series",
            2: "x_series",
            3: "x_series",
            4: "x_series",
            5: "x_series",
            6: "x_series",
        }
    )

    prepare_configuration(arm)

    # Ask the user to move the LCR to the position 1
    print("Please move the LCR to the position 1")
    pause()

    configure_homing(arm, [False, False, False, False, False, False])

    # Ask the user to move the LCR to the position 2
    print("Please move the LCR to the position 2")
    pause()

    inverted = configure_drive_mode(arm)

    # Ask the user to move back the LCR to the position 1
    print("Please move back the LCR to the position 1")
    pause()

    configure_homing(arm, inverted)

    print("Configuration done!")
    print("Make sure everything is working properly:")

    while True:
        positions = arm.sync_read_present_position_i32()
        print(positions)

        time.sleep(1)
