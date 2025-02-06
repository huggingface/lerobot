import argparse
import time

import cv2
import numpy as np

from lerobot.common.robot_devices.control_utils import is_headless
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.utils.utils import init_hydra_config


def find_joint_bounds(
    robot,
    control_time_s=20,
    display_cameras=False,
):
    # TODO(rcadene): Add option to record logs
    if not robot.is_connected:
        robot.connect()

        control_time_s = float("inf")

    timestamp = 0
    start_episode_t = time.perf_counter()
    pos_list = []
    while timestamp < control_time_s:
        observation, action = robot.teleop_step(record_data=True)

        pos_list.append(robot.follower_arms["main"].read("Present_Position"))

        if display_cameras and not is_headless():
            image_keys = [key for key in observation if "image" in key]
            for key in image_keys:
                cv2.imshow(key, cv2.cvtColor(observation[key].numpy(), cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        timestamp = time.perf_counter() - start_episode_t
        if timestamp > 60:
            max = np.max(np.stack(pos_list), 0)
            min = np.min(np.stack(pos_list), 0)
            print(f"Max angle position per joint {max}")
            print(f"Min angle position per joint {min}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot-path",
        type=str,
        default="lerobot/configs/robot/koch.yaml",
        help="Path to robot yaml file used to instantiate the robot using `make_robot` factory function.",
    )
    parser.add_argument(
        "--robot-overrides",
        type=str,
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )
    parser.add_argument("--control-time-s", type=float, default=20, help="Maximum episode length in seconds")
    args = parser.parse_args()
    robot_cfg = init_hydra_config(args.robot_path, args.robot_overrides)

    robot = make_robot(robot_cfg)
    find_joint_bounds(robot, control_time_s=args.control_time_s)
