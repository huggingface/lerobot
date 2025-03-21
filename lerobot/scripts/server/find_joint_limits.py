import argparse
import time

import cv2
import numpy as np

from lerobot.common.robot_devices.control_utils import is_headless
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.utils.utils import init_hydra_config
from lerobot.scripts.server.kinematics import RobotKinematics


def find_joint_bounds(
    robot,
    control_time_s=30,
    display_cameras=False,
):
    if not robot.is_connected:
        robot.connect()

    start_episode_t = time.perf_counter()
    pos_list = []
    while True:
        observation, action = robot.teleop_step(record_data=True)

        # Wait for 5 seconds to stabilize the robot initial position
        if time.perf_counter() - start_episode_t < 5:
            continue

        pos_list.append(robot.follower_arms["main"].read("Present_Position"))

        if display_cameras and not is_headless():
            image_keys = [key for key in observation if "image" in key]
            for key in image_keys:
                cv2.imshow(
                    key, cv2.cvtColor(observation[key].numpy(), cv2.COLOR_RGB2BGR)
                )
            cv2.waitKey(1)

        if time.perf_counter() - start_episode_t > control_time_s:
            max = np.max(np.stack(pos_list), 0)
            min = np.min(np.stack(pos_list), 0)
            print(f"Max angle position per joint {max}")
            print(f"Min angle position per joint {min}")
            break


def find_ee_bounds(
    robot,
    control_time_s=30,
    display_cameras=False,
):
    if not robot.is_connected:
        robot.connect()

    start_episode_t = time.perf_counter()
    ee_list = []
    while True:
        observation, action = robot.teleop_step(record_data=True)

        # Wait for 5 seconds to stabilize the robot initial position
        if time.perf_counter() - start_episode_t < 5:
            continue

        joint_positions = robot.follower_arms["main"].read("Present_Position")
        print(f"Joint positions: {joint_positions}")
        ee_list.append(RobotKinematics.fk_gripper_tip(joint_positions)[:3, 3])

        if display_cameras and not is_headless():
            image_keys = [key for key in observation if "image" in key]
            for key in image_keys:
                cv2.imshow(
                    key, cv2.cvtColor(observation[key].numpy(), cv2.COLOR_RGB2BGR)
                )
            cv2.waitKey(1)

        if time.perf_counter() - start_episode_t > control_time_s:
            max = np.max(np.stack(ee_list), 0)
            min = np.min(np.stack(ee_list), 0)
            print(f"Max ee position {max}")
            print(f"Min ee position {min}")
            break


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
    parser.add_argument(
        "--mode",
        type=str,
        default="joint",
        choices=["joint", "ee"],
        help="Mode to run the script in. Can be 'joint' or 'ee'.",
    )
    parser.add_argument(
        "--control-time-s",
        type=int,
        default=30,
        help="Time step to use for control.",
    )
    args = parser.parse_args()
    robot_cfg = init_hydra_config(args.robot_path, args.robot_overrides)

    robot = make_robot(robot_cfg)
    if args.mode == "joint":
        find_joint_bounds(robot, args.control_time_s)
    elif args.mode == "ee":
        find_ee_bounds(robot, args.control_time_s)
    if robot.is_connected:
        robot.disconnect()
