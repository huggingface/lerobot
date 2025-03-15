import logging
import time

import numpy as np

from ...teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from ...teleoperators.so100 import SO100Teleop, SO100TeleopConfig
from .configuration_daemon_lekiwi import DaemonLeKiwiRobotConfig
from .daemon_lekiwi import DaemonLeKiwiRobot, RobotMode


def main():
    logging.info("Configuring Teleop Devices")
    leader_arm_config = SO100TeleopConfig(port="/dev/tty.usbmodem585A0085511")
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
    robot.connect()  # Establishes ZMQ sockets with the remote mobile robot
    robot.robot_mode = RobotMode.TELEOP

    logging.info("Starting LeKiwiRobot teleoperation")
    start = time.perf_counter()
    duration = 0
    while duration < 20:
        arm_action = leader_arm.get_action()
        base_action = keyboard.get_action()
        action = np.concatenate((arm_action, base_action))
        _action_sent = robot.send_action(action)  # Translates to motor space + sends over ZMQ
        _observation = robot.get_observation()  # Receives over ZMQ, translate to body-frame vel

        # dataset.save(action_sent, obs)

        # TODO(Steven): Deal with policy action space
        # robot.set_mode(RobotMode.AUTO)
        # policy_action = policy.get_action() # This might be in body frame, key space or smt else
        # robot.send_action(policy_action)

        duration = time.perf_counter() - start

    logging.info("Disconnecting Teleop Devices and LeKiwiRobot Daemon")
    robot.disconnect()  # Cleans ZMQ comms
    leader_arm.disconnect()
    keyboard.disconnect()

    logging.info("Finished LeKiwiRobot cleanly")


if __name__ == "__main__":
    main()
