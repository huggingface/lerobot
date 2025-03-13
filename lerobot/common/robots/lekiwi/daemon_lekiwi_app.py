from ...teleoperators.so100 import SO100Teleop, SO100TeleopConfig
from ...teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from .configuration_daemon_lekiwi import DaemonLeKiwiRobotConfig
from .daemon_lekiwi import DaemonLeKiwiRobot
import time
import logging

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
    robot.connect() # Establishes ZMQ sockets with the remote mobile robot

    logging.info("Starting LeKiwiRobot teleoperation")
    start = time.perf_counter()
    duration = 0
    while duration < 20:
        
        arm_action = leader_arm.get_action() # 6 motors 
        base_action = keyboard.get_action() # n keys pressed
        action = {
            **arm_action,
            **base_action
        }
        robot.set_mode(TELEOP)
        action_sent = robot.send_action(action) # Translates to motor space + sends over ZMQ
        obs = robot.get_observation() # Receives over ZMQ, translate to body-frame vel

        dataset.save(action_sent, obs)

        # TODO(Steven)
        robot.set_mode(AUTO)
        policy_action = policy.get_action() # This might be in body frame or in key space
        robot.send_action(policy_action) # This has no way to know

        teleop_step() # teleop
        send_action() #policy

        duration = time.perf_counter() - start

    logging.info("Disconnecting Teleop Devices and LeKiwiRobot Daemon")
    robot.disconnect() # Cleans ZMQ comms
    leader_arm.disconnect()
    keyboard.disconnect()

    logging.info("Finished LeKiwiRobot cleanly")

if __name__ == "__main__":
    main()