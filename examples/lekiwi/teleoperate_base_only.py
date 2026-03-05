# Base-only teleoperation for LeKiwi with OAK-D Lite on the Pi
#
# On the Pi:
#   cd ~/lerobot && mamba activate lerobot
#   python -m lerobot.robots.lekiwi.lekiwi_host \
#     --robot.id=my_kiwi \
#     --robot.include_arm=false \
#     '--robot.cameras={"front": {"type": "oak", "fps": 30, "width": 640, "height": 400}}'
#
# On your laptop:
#   cd ~/Desktop/code/lerobot && conda activate lerobot
#   python examples/lekiwi/teleoperate_base_only.py

import time

import cv2

from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.cameras.oak.configuration_oak import OAKCameraConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.utils.robot_utils import precise_sleep

FPS = 30


def main():
    robot_config = LeKiwiClientConfig(
        remote_ip="10.42.0.87",
        id="my_kiwi",
        include_arm=False,
        cameras={
            "front": OAKCameraConfig(fps=FPS, width=640, height=400),
        },
    )
    keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")

    robot = LeKiwiClient(robot_config)
    keyboard = KeyboardTeleop(keyboard_config)

    robot.connect()
    keyboard.connect()

    if not robot.is_connected or not keyboard.is_connected:
        raise ValueError("Robot or keyboard is not connected!")

    print("Starting base-only teleop with OAK-D Lite...")
    print("Controls: W/S forward/back, A/D left/right, Z/X rotate, R/F speed up/down, Q quit")

    cv2.namedWindow("LeKiwi", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("LeKiwi", 640, 400)

    try:
        while True:
            t0 = time.perf_counter()

            observation = robot.get_observation()

            keyboard_keys = keyboard.get_action()
            base_action = robot._from_keyboard_to_base_action(keyboard_keys)

            if base_action:
                robot.send_action(base_action)

            if "front" in observation and hasattr(observation["front"], "shape"):
                cv2.imshow("LeKiwi", observation["front"])

            if cv2.waitKey(1) == ord("q"):
                break

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
