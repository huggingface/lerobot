# Base-only teleoperation for LeKiwi without arm

import time

from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.utils.robot_utils import precise_sleep

FPS = 30


def main():
    robot_config = LeKiwiClientConfig(
        remote_ip="10.42.0.87",
        id="my_kiwi",
        include_arm=False,
        cameras={},
    )
    keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")

    robot = LeKiwiClient(robot_config)
    keyboard = KeyboardTeleop(keyboard_config)

    robot.connect()
    keyboard.connect()

    if not robot.is_connected or not keyboard.is_connected:
        raise ValueError("Robot or keyboard is not connected!")

    print("Starting base-only teleop loop...")
    print("Controls: W/S forward/back, A/D left/right, Z/X rotate, R/F speed up/down, Q quit")

    while True:
        t0 = time.perf_counter()

        observation = robot.get_observation()

        keyboard_keys = keyboard.get_action()
        base_action = robot._from_keyboard_to_base_action(keyboard_keys)

        if base_action:
            robot.send_action(base_action)

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))


if __name__ == "__main__":
    main()
