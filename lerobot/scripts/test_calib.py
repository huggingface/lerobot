#!/usr/bin/env python

import time

from lerobot.common.robots.so100.configuration_so100 import SO100RobotConfig
from lerobot.common.robots.so100.robot_so100 import SO100Robot


def main():
    config = SO100RobotConfig(
        port="/dev/tty.usbmodem58760430031",
    )

    # Create the SO100 robot object
    robot = SO100Robot(config)

    # Connect the robot (this will internally call `calibrate()` once the bus is connected)
    print("Connecting the robot (this automatically calibrates motors).")
    robot.connect()

    # If you prefer to call calibration separately or again at any point:
    print("Running calibration explicitly...")
    robot.calibrate()

    # Wait briefly, then disconnect
    time.sleep(1)
    print("Disconnecting robot.")
    robot.disconnect()


if __name__ == "__main__":
    main()
