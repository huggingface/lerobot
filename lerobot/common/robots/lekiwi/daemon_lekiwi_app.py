import logging
import time

import numpy as np

from lerobot.common.robots.lekiwi.configuration_daemon_lekiwi import DaemonLeKiwiRobotConfig
from lerobot.common.robots.lekiwi.daemon_lekiwi import DaemonLeKiwiRobot, RobotMode
from lerobot.common.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.common.teleoperators.so100 import SO100Teleop, SO100TeleopConfig


def main():
    logging.info("Configuring Teleop Devices")
    leader_arm_config = SO100TeleopConfig(port="/dev/tty.usbmodem58760429271")
    leader_arm = SO100Teleop(leader_arm_config)

    keyboard_config = KeyboardTeleopConfig()
    keyboard = KeyboardTeleop(keyboard_config)

    logging.info("Connecting Teleop Devices")
    leader_arm.connect()
    keyboard.connect()

    logging.info("Configuring LeKiwiRobot Daemon")
    robot_config = DaemonLeKiwiRobotConfig()
    robot = DaemonLeKiwiRobot(robot_config)

    logging.info("Connecting remote LeKiwiRobot")
    robot.connect()
    robot.robot_mode = RobotMode.TELEOP

    logging.info("Starting LeKiwiRobot teleoperation")
    start = time.perf_counter()
    duration = 0
    while duration < 20:
        arm_action = leader_arm.get_action()
        base_action = keyboard.get_action()
        action = np.append(arm_action, base_action) if base_action.size > 0 else arm_action
        _action_sent = robot.send_action(action)
        _observation = robot.get_observation()

        # dataset.save(action_sent, obs)

        # TODO(Steven): Deal with policy action space
        # robot.set_mode(RobotMode.AUTO)
        # policy_action = policy.get_action() # This might be in body frame, key space or smt else
        # robot.send_action(policy_action)
        duration = time.perf_counter() - start

    logging.info("Disconnecting Teleop Devices and LeKiwiRobot Daemon")
    robot.disconnect()
    leader_arm.disconnect()
    keyboard.disconnect()

    logging.info("Finished LeKiwiRobot cleanly")


if __name__ == "__main__":
    main()
