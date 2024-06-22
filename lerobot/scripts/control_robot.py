

import argparse
import time
from lerobot.common.robot_devices.robots.aloha import AlohaRobot
from lerobot.scripts.robot_controls.record_dataset import record_dataset

CONTROL_MODES = [
    "teleoperate",
    "record_dataset",
    "replay_episode",
    "run_policy",
    "disable_torque",
]


def teleoperate():
    robot = AlohaRobot(use_cameras=False)
    robot.init_teleop()

    while True:
        now = time.time()
        robot.teleop_step()

        dt_s = (time.time() - now)
        print(f"Latency (ms): {dt_s * 1000:.2f}\tFrequency: {1 / dt_s:.2f}")


def record_dataset():
    robot = AlohaRobot(use_cameras=True)
    robot.init_teleop()

    timestamps = []
    observations = []
    actions = []

    start_time = time.time()
    while True:
        now = time.time()
        observation, action = robot.teleop_step(record_data=True)
        dt_s = (time.time() - now)

        timestamp = time.time() - start_time
        timestamps.append(timestamp)
        observations.append(observation)
        actions.append(action)

        print(f"Latency (ms): {dt_s * 1000:.2f}\tFrequency: {1 / dt_s:.2f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=CONTROL_MODES, default="teleoperate")
    args = parser.parse_args()

    if args.mode == "teleoperate":
        teleoperate()
    elif args.mode == "record_dataset":
        record_dataset()
    elif args.mode == "replay_episode":
        replay_episode()
    # elif args.mode == "find_camera_ids":
    #     find_camera_ids()
    elif args.mode == "disable_torque":
        disable_torque()
