#!/usr/bin/env python

import time

import numpy as np

from lerobot.common.constants import OBS_STATE
from lerobot.common.robots.so100.configuration_so100 import SO100RobotConfig
from lerobot.common.robots.so100.robot_so100 import SO100Robot


def main():
    config_follower = SO100RobotConfig(
        port="/dev/tty.usbmodem58760431101",
    )

    robot_follower = SO100Robot(config_follower)

    print("Connecting the robot (this automatically calibrates motors).")
    robot_follower.connect()

    config_leader = SO100RobotConfig(
        port="/dev/tty.usbmodem58760429511",
    )

    robot_leader = SO100Robot(config_leader)

    print("Connecting the robot (this automatically calibrates motors).")
    robot_leader.connect()

    print("Starting Teleop!")

    try:
        while True:
            obs = robot_leader.get_observation()
            positions = obs[OBS_STATE]
            robot_follower.send_action(np.array(positions))

            time.sleep(0.02)
    except KeyboardInterrupt:
        print("Exiting loop.")
    finally:
        robot_leader.disconnect()
        robot_follower.disconnect()


if __name__ == "__main__":
    main()
