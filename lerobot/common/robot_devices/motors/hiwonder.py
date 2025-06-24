import traceback

import numpy as np
import serial

from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError

BAUDRATE = 115_200
TIMEOUT_S = 1
COMMAND_HEADER = [0x55, 0x55]  # Header for all commands sent to Hiwonder motors
BYTE_MASK = 0xFF

# https://drive.google.com/file/d/1DQGBBng8UFziv5hg15qmyvakItG3w6Ss/view?usp=sharing

# data_name: (command, length)
HIWONDER_CONTROL_TABLE = {
    "SERVO_MOVE_TIME_WRITE": (0x01, 0x07),
    "SERVO_MOVE_TIME_READ": (0x02, 0x03),
    "SERVO_MOVE_TIME_WAIT_WRITE": (0x07, 0x07),
    "SERVO_MOVE_TIME_WAIT_READ": (0x08, 0x03),
    "SERVO_MOVE_START": (0x0B, 0x03),
    "SERVO_MOVE_STOP": (0x0C, 0x03),
    "SERVO_ID_WRITE": (0x0D, 0x04),
    "SERVO_ID_READ": (0x0E, 0x03),
    "SERVO_ANGLE_OFFSET_ADJUST": (0x11, 0x04),
    "SERVO_ANGLE_OFFSET_WRITE": (0x12, 0x03),
    "SERVO_ANGLE_OFFSET_READ": (0x13, 0x03),
    "SERVO_ANGLE_LIMIT_WRITE": (0x14, 0x07),
    "SERVO_ANGLE_LIMIT_READ": (0x15, 0x03),
    "SERVO_VIN_LIMIT_WRITE": (0x16, 0x07),
    "SERVO_VIN_LIMIT_READ": (0x17, 0x03),
    "SERVO_TEMP_MAX_LIMIT_WRITE": (0x18, 0x04),
    "SERVO_TEMP_MAX_LIMIT_READ": (0x19, 0x03),
    "SERVO_TEMP_READ": (0x1A, 0x03),
    "SERVO_VIN_READ": (0x1B, 0x03),
    "SERVO_POS_READ": (0x1C, 0x03),
    "SERVO_OR_MOTOR_MODE_WRITE": (0x1D, 0x07),
    "SERVO_OR_MOTOR_MODE_READ": (0x1E, 0x03),
    "SERVO_LOAD_OR_UNLOAD_WRITE": (0x1F, 0x04),
    "SERVO_LOAD_OR_UNLOAD_READ": (0x20, 0x03),
    "SERVO_LED_CTRL_WRITE": (0x21, 0x04),
    "SERVO_LED_CTRL_READ": (0x22, 0x03),
    "SERVO_LED_ERROR_WRITE": (0x23, 0x04),
    "SERVO_LED_ERROR_READ": (0x24, 0x03),
}


def calculate_checksum(data):
    """Calculate the checksum for the given data."""
    checksum = ~(sum(data)) & BYTE_MASK
    return checksum


def low_byte(value):
    """Extract the low byte of a 16-bit integer."""
    return value & BYTE_MASK


def high_byte(value):
    """Extract the high byte of a 16-bit integer."""
    return (value >> 8) & BYTE_MASK


class HiwonderMotorsBus:
    """
    The HiwonderMotorBus class allows to efficiently read and write to the attached motors. It relies on
    the [Hiwonder Bus Communication Protoco](https://drive.google.com/file/d/1JKyt_OUg9V6cIBC-SiX6IIAACsvz86aB/view?usp=sharing).

    A HiwonderMotorBus instance requires a port (e.g. `HiwonderMotorBus(port="/dev/tty.usbmodem575E0031751"`)).

    Example of usage for 1 motor connected to the bus:
    ```python
    motor_name = "gripper"
    motor_index = 6
    motor_model = "lx-16a"

    motors_bus = HiwonderMotorBus(
        port="/dev/tty.usbmodem575E0031751",
        motors={motor_name: (motor_index, motor_model)},
    )
    motors_bus.connect()

    position = motors_bus.read()

    # move from a few motor steps as an example
    few_steps = 30
    motors_bus.write(position + few_steps)

    # when done, consider disconnecting
    motors_bus.disconnect()
    ```
    """

    def __init__(self, port: str, motors: dict[str, tuple[int, str]]):
        self.port = port
        self.serial = None
        self.is_connected = False
        self.motors = motors

    def connect(self):
        """Open the serial port and establish a connection."""
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"HiwonderMotorsBus({self.port}) is already connected. Do not call `motors_bus.connect()` twice."
            )
        try:
            self.serial = serial.Serial(self.port, baudrate=BAUDRATE, timeout=TIMEOUT_S)
            self.is_connected = True
            print("Connection established")
        except Exception:
            traceback.print_exc()
            print("\nCould not open port.\n")
            raise

    def disconnect(self):
        """Close the serial port connection."""
        if self.serial:
            self.serial.close()
            self.is_connected = False
            print("Connection closed")

    @property
    def motor_names(self) -> list[str]:
        return list(self.motors.keys())

    def write(self, values: int | float | np.ndarray, motor_names: str | list[str] | None = None):
        if not self.is_connected:
            raise RobotDeviceAlreadyConnectedError(f"HiwonderMotorsBus({self.port}) is not connected.")

        if motor_names is None:
            motor_names = self.motor_names

        if isinstance(motor_names, str):
            motor_names = [motor_names]

        if isinstance(values, (int, float)):
            values = [values] * len(
                motor_names
            )  # Replicate value for each motor if a single value is provided

        motor_commands = zip(motor_names, values, strict=False)

        for motor_name, value in motor_commands:
            motor_id, _ = self.motors[motor_name]  # Using only the ID for the command
            duration = 0  # Get to the next position as soon as possible

            # Command structure: 55 55 ID LEN CMD P1 P2 P3 P4 CHK
            # Retrieve command and length from the control table
            cmd, length = HIWONDER_CONTROL_TABLE["SERVO_MOVE_TIME_WRITE"]
            # Prepare the packet
            data = [
                motor_id,
                length,
                cmd,
                low_byte(value),
                high_byte(value),
                low_byte(duration),
                high_byte(duration),
            ]
            checksum = calculate_checksum(data)
            command = COMMAND_HEADER + data + [checksum]

            self.serial.write(bytearray(command))

    def read(self, motor_names: str | list[str] | None = None):
        """Send a command to read the current position of the motors."""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"HiwonderMotorsBus({self.port}) is not connected. You need to run `motors_bus.connect()`."
            )

        if motor_names is None:
            motor_names = self.motor_names

        if isinstance(motor_names, str):
            motor_names = [motor_names]

        motor_ids = [self.motors[name][0] for name in motor_names]  # Extract motor IDs based on names

        positions = {}
        for motor_name, motor_id in zip(motor_names, motor_ids, strict=False):
            # Command structure for reading position: 55 55 ID LEN CMD CHK
            # Retrieve command and length from the control table
            cmd, length = HIWONDER_CONTROL_TABLE["SERVO_POS_READ"]

            # Prepare the command packet
            data = [motor_id, length, cmd]
            checksum = calculate_checksum(data)
            command = COMMAND_HEADER + data + [checksum]

            # Send the command
            self.serial.write(bytearray(command))

            # Wait for response and handle it
            try:
                response = self.serial.read(8)  # Adjust size based on expected response
                if len(response) == 8:
                    header1, header2, resp_id, resp_len, resp_cmd, param1, param2, resp_chk = response
                    # Calculate expected checksum and compare
                    expected_checksum = calculate_checksum(response[2:-1])
                    if resp_chk == expected_checksum:
                        # Combine param1 and param2 to form the actual position
                        position = param1 + (param2 << 8)
                        positions[motor_name] = position  # Store position with motor ID as the key
                    else:
                        print(f"Checksum mismatch for motor {motor_id}.")
                else:
                    print(f"No response or invalid response for motor {motor_id}.")
            except Exception as e:
                print(f"Failed to read position for motor {motor_id}: {str(e)}")

        return positions


# Example usage
if __name__ == "__main__":
    pass
