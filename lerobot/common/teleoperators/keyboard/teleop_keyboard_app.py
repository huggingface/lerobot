import logging
import time

from lerobot.common.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig


def main():
    logging.info("Configuring Keyboard Teleop")
    keyboard_config = KeyboardTeleopConfig()
    keyboard = KeyboardTeleop(keyboard_config)

    logging.info("Connecting Keyboard Teleop")
    keyboard.connect()

    logging.info("Starting Keyboard capture")
    i = 0
    while i < 20:
        action = keyboard.get_action()
        print("Captured keys: %s", action)
        time.sleep(1)
        i += 1

    keyboard.disconnect()
    logging.info("Finished LeKiwiRobot cleanly")


if __name__ == "__main__":
    main()
