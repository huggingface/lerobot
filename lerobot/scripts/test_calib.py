#!/usr/bin/env python

import time

from lerobot.common.motors.motors_bus import TorqueMode
from lerobot.common.robots.so100.configuration_so100 import SO100RobotConfig
from lerobot.common.robots.so100.robot_so100 import SO100Robot
from lerobot.common.teleoperators.so100.configuration_so100 import SO100TeleopConfig
from lerobot.common.teleoperators.so100.teleop_so100 import SO100Teleop


def main():
    config_follower = SO100RobotConfig()

    robot_follower = SO100Robot(config_follower)

    print("Connecting the robot (this automatically calibrates motors).")
    robot_follower.connect()

    config_leader = SO100TeleopConfig()

    robot_leader = SO100Teleop(config_leader)

    print("Connecting the robot (this automatically calibrates motors).")
    robot_leader.connect()

    print("Starting Teleop!")
    for name in robot_leader.arm.names:
        robot_leader.arm.write("Torque_Enable", name, TorqueMode.DISABLED.value)

    try:
        while True:
            leader_pos = robot_leader.arm.sync_read("Present_Position")
            robot_follower.arm.sync_write("Goal_Position", leader_pos)

            time.sleep(0.02)
    except KeyboardInterrupt:
        print("Exiting loop.")
    finally:
        robot_leader.disconnect()
        robot_follower.disconnect()


if __name__ == "__main__":
    main()
