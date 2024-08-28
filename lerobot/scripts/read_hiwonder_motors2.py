import time

from lerobot.common.robot_devices.motors.hiwonder import (
    HiwonderMotorsBus,  # assuming the class is saved in hiwonder_motors_bus.py
)


def main():
    # Replace '/dev/ttyUSB0' with your actual port
    # Try connecting with 9600 baud rate
    hw_bus = HiwonderMotorsBus("/dev/ttyUSB0")
    hw_bus.connect()
    # hw_bus.read_motor_id()
    hw_bus.write_position(1, 10, 2000)  # Move motor ID 1 to position 1000 in 2000 ms
    time.sleep(3.0)  # Wait for the movement to complete
    hw_bus.read_position(1)
    hw_bus.write_position(1, 500, 2000)  # Move motor ID 1 to position 500 in 2000 ms
    time.sleep(3.0)  # Wait for the movement to complete
    hw_bus.read_position(1)
    hw_bus.write_position(1, 1000, 2000)  # Move motor ID 1 to position 500 in 2000 ms
    time.sleep(3.0)  # Wait for the movement to complete
    hw_bus.read_position(1)

    hw_bus.write_position(2, 10, 2000)  # Move motor ID 1 to position 1000 in 2000 ms
    time.sleep(3.0)  # Wait for the movement to complete
    hw_bus.read_position(2)
    hw_bus.write_position(2, 500, 2000)  # Move motor ID 1 to position 500 in 2000 ms
    time.sleep(3.0)  # Wait for the movement to complete
    hw_bus.read_position(2)
    hw_bus.write_position(2, 1000, 2000)  # Move motor ID 1 to position 500 in 2000 ms
    time.sleep(3.0)  # Wait for the movement to complete
    hw_bus.read_position(2)
    hw_bus.disconnect()


if __name__ == "__main__":
    main()
