import logging

from .configuration_lekiwi import LeKiwiRobotConfig
from .lekiwi_robot import LeKiwiRobot


def main():
    logging.info("Configuring LeKiwiRobot")
    robot_config = LeKiwiRobotConfig()
    robot = LeKiwiRobot(robot_config)

    logging.info("Connecting LeKiwiRobot")
    robot.connect()

    # Remotely teleoperated
    logging.info("Starting LeKiwiRobot teleoperation")
    robot.run()

    logging.info("Finished LeKiwiRobot cleanly")


if __name__ == "__main__":
    main()
