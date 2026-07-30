#!/usr/bin/env python
"""Set up only the LeKiwi Orbbec camera pan/tilt motors: camera_pan (10) and
camera_tilt (11), daisy-chained after the base wheels (7, 8, 9).

Connect ONE new servo at a time when prompted. Each servo is (re)assigned the
target id and the bus' default baud-rate, so start from factory-fresh servos
(or any id) -- the tool scans for whatever is currently connected.

Usage:
    uv run python setup_camera_motors.py --port /dev/tty.usbmodem58760431551
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Set up only the LeKiwi Orbbec camera pan/tilt motors.")
    parser.add_argument(
        "--port",
        required=True,
        help="USB port of the LeKiwi MotorsBus (e.g. /dev/tty.usbmodem58760431551).",
    )
    args = parser.parse_args()

    from lerobot.robots.lekiwi import LeKiwi, LeKiwiConfig

    # use_camera_head=True registers camera_pan (10) and camera_tilt (11) on the bus.
    robot = LeKiwi(LeKiwiConfig(port=args.port, use_camera_head=True))

    # handshake=False so we don't require all motors (arm 1-6, wheels 7-9) to be present/responding.
    robot.bus.connect(handshake=False)
    try:
        for motor in reversed(robot.camera_motors):  # tilt (11), then pan (10)
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            robot.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {robot.bus.motors[motor].id}")
    finally:
        # disable_torque=False: only the camera motors are connected, so attempting to
        # disable torque on the absent arm/wheel motors would raise a ConnectionError.
        robot.bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
