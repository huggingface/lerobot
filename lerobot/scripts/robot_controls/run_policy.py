

import time
from lerobot.common.robot_devices.robots.aloha import AlohaRobot
import torch



def teleoperate():
    robot = AlohaRobot(use_cameras=False)
    robot.init_teleop()

    while True:
        now = time.time()
        robot.teleop_step(record_data=False)

        dt_s = (time.time() - now)
        print(f"Latency (ms): {dt_s * 1000:.2f}\tFrequency: {1 / dt_s:.2f}")


def record_teleop_data():
    robot = AlohaRobot(use_cameras=True)
    robot.init_teleop()

    while True:
        now = time.time()
        observation, action = robot.teleop_step(record_data=True)

        dt_s = (time.time() - now)
        print(f"Latency (ms): {dt_s * 1000:.2f}\tFrequency: {1 / dt_s:.2f}")



def evaluate_policy(policy):
    robot = AlohaRobot(use_cameras=True)
    observation = robot.init_evaluate()

    while True:
        now = time.time()
        with torch.inference_mode():
            action = policy.select_action(observation)

        observation, action = robot.step(action, record_data=False)

        dt_s = (time.time() - now)
        print(f"Latency (ms): {dt_s * 1000:.2f}\tFrequency: {1 / dt_s:.2f}")

