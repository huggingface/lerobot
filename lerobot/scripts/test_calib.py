#!/usr/bin/env python

import time

from lerobot.common.motors.motors_bus import TorqueMode
from lerobot.common.robots.so100.configuration_so100 import SO100RobotConfig
from lerobot.common.robots.so100.robot_so100 import SO100Robot


def main():
    config_follower = SO100RobotConfig(
        port="/dev/tty.usbmodem58760431201",
    )

    robot_follower = SO100Robot(config_follower)

    print("Connecting the robot (this automatically calibrates motors).")
    robot_follower.connect()

    config_leader = SO100RobotConfig(
        port="/dev/tty.usbmodem58760430821",
    )

    robot_leader = SO100Robot(config_leader)

    print("Connecting the robot (this automatically calibrates motors).")
    robot_leader.connect()

    print("Starting Teleop!")
    for name in robot_leader.arm.names:
        robot_leader.arm.write("Torque_Enable", name, TorqueMode.DISABLED.value)

    try:
        while True:
            motor_names = [
                "shoulder_pan",
            ]  # "shoulder_lift", "elbow_flex", "wrist_flex","wrist_roll","gripper",
            leader_pos = robot_leader.arm.sync_read("Present_Position", motor_names)
            robot_follower.arm.sync_write("Goal_Position", leader_pos)

            time.sleep(0.02)
    except KeyboardInterrupt:
        print("Exiting loop.")
    finally:
        robot_leader.disconnect()
        robot_follower.disconnect()


if __name__ == "__main__":
    main()
