

import time
from lerobot.common.robot_devices.robots.aloha import AlohaRobot
import torch


def record_dataset():
    robot = AlohaRobot(use_cameras=True)
    robot.init_teleop()

    while True:
        now = time.time()
        observation, action = robot.teleop_step(record_data=True)

        dt_s = (time.time() - now)
        print(f"Latency (ms): {dt_s * 1000:.2f}\tFrequency: {1 / dt_s:.2f}")

if __name__ == "__main__":
    record_dataset()