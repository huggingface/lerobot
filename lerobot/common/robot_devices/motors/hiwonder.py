import time
import traceback

import serial

from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError


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

    position = motors_bus.read("Present_Position")

    # move from a few motor steps as an example
    few_steps = 30
    motors_bus.write("Goal_Position", position + few_steps)

    # when done, consider disconnecting
    motors_bus.disconnect()
    ```
    """

    def __init__(self, port):
        self.port = port
        self.serial = None
        self.is_connected = False

    def connect(self):
        """Open the serial port and establish a connection."""
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                f"HiwonderMotorsBus({self.port}) is already connected. Do not call `motors_bus.connect()` twice."
            )
        try:
            # TODO made baudrate, timeout constant
            self.serial = serial.Serial(self.port, baudrate=115200, timeout=1)
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

    def calculate_checksum(self, data):
        """Calculate the checksum for the given data."""
        checksum = ~(sum(data)) & 0xFF
        return checksum

    def write_position(self, motor_id, position, duration):
        """Send a command to move the motor to a specific position over a specified duration."""
        if not self.is_connected:
            print("Serial port not connected.")
            return

        # Command structure: 55 55 ID LEN CMD P1 P2 P3 P4 CHK
        cmd = 0x01  # Assuming command for move
        length = 0x07  # Length of the remaining bytes
        pos_low = position & 0xFF
        pos_high = (position >> 8) & 0xFF
        dur_low = duration & 0xFF
        dur_high = (duration >> 8) & 0xFF

        # Prepare the packet
        data = [motor_id, length, cmd, pos_low, pos_high, dur_low, dur_high]
        checksum = self.calculate_checksum(data)
        command = [0x55, 0x55] + data + [checksum]

        # Send the packet
        self.serial.write(bytearray(command))
        print(f"Position write command sent: {' '.join(f'{byte:02X}' for byte in command)}")

    def read_position(self, motor_id):
        """Send a command to read the current position of the motor."""
        if not self.is_connected:
            print("Serial port not connected.")
            return

        # Command structure for reading position: 55 55 ID LEN CMD CHK
        cmd = 0x1C  # Command value for reading position
        length = 0x03  # Length is always 3 for the read position command

        # Prepare the command packet
        data = [motor_id, length, cmd]
        checksum = self.calculate_checksum(data)
        command = [0x55, 0x55] + data + [checksum]

        # Send the command
        self.serial.write(bytearray(command))
        print(f"Position read command sent: {' '.join(f'{byte:02X}' for byte in command)}")

        # Wait for response and handle it
        try:
            response = self.serial.read(8)  # Adjust size based on expected response
            if len(response) == 8:
                header1, header2, resp_id, resp_len, resp_cmd, param1, param2, resp_chk = response
                # Calculate expected checksum and compare
                expected_checksum = self.calculate_checksum(response[2:-1])
                if resp_chk == expected_checksum:
                    # Combine param1 and param2 to form the actual position
                    position = param1 + (param2 << 8)
                    print(f"Current position of motor {motor_id}: {position}")
                    return position
                else:
                    print("Checksum mismatch.")
            else:
                print("No response or invalid response")
        except Exception as e:
            print(f"Failed to read position: {e}")


# Example usage
if __name__ == "__main__":
    hw_bus = HiwonderMotorsBus("/dev/ttyUSB0")  # Adjust as necessary for your setup
    hw_bus.connect()
    hw_bus.send_command(1, 1000, 2000)  # Move motor ID 1 to position 1000 in 2000 ms
    time.sleep(2.5)  # Wait for the movement to complete
    hw_bus.send_command(1, 500, 2000)  # Move motor ID 1 to position 500 in 2000 ms
    time.sleep(2.5)  # Wait for the movement to complete
    hw_bus.disconnect()
