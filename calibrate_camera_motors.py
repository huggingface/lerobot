#!/usr/bin/env python
"""Calibrate ONLY the LeKiwi Orbbec camera pan/tilt servos (camera_pan=10, camera_tilt=11).

The existing arm/base calibration is preserved -- this only records homing offset +
range of motion for the two camera servos and merges them into the calibration file.
Run a full calibration first (arm + wheels) before using this.

Usage:
    uv run python calibrate_camera_motors.py --port /dev/tty.usbmodem58760431551 --id my_lekiwi
"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate only the LeKiwi Orbbec camera pan/tilt servos."
    )
    parser.add_argument(
        "--port",
        required=True,
        help="USB port of the LeKiwi MotorsBus (e.g. /dev/tty.usbmodem58760431551).",
    )
    parser.add_argument(
        "--id",
        default=None,
        help="Robot id used to locate the calibration file (must match the id used elsewhere).",
    )
    args = parser.parse_args()

    from lerobot.robots.lekiwi import LeKiwi, LeKiwiConfig

    # use_camera_head=True registers camera_pan (10) / camera_tilt (11); the existing calibration
    # file (if any) for this id is loaded automatically in __init__.
    config = LeKiwiConfig(port=args.port, use_camera_head=True)
    if args.id is not None:
        config.id = args.id
    robot = LeKiwi(config)

    # handshake=False so only the two camera servos need to be connected (chained to each other:
    # board -> camera_pan (10) -> camera_tilt (11)); the arm/wheel motors don't have to be present.
    robot.bus.connect(handshake=False)
    try:
        robot.calibrate_camera_head()
    finally:
        # disable_torque=False: only the camera motors are connected, so disabling torque on the
        # absent arm/wheel motors would raise a ConnectionError.
        robot.bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
