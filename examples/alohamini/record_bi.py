#!/usr/bin/env python3

"""
Usage:
  python examples/alohamini/record_bi.py \
    --dataset liyitenga/alohamini_test1 \
    --num_episodes 1 \
    --fps 30 \
    --episode_time 45 \
    --reset_time 8 \
    --task_description "alohamini_test1" \
    --remote_ip 192.168.50.84 \
    --leader_id so101_leader_bi_t6
"""

import argparse
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.processor import make_default_processors
from lerobot.robots.alohamini.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.alohamini.lekiwi_client import LeKiwiClient
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.bi_so_leader import BiSOLeader, BiSOLeaderConfig
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so_leader import SOLeaderConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun


def main():
    parser = argparse.ArgumentParser(description="Record episodes with bi-arm teleoperation")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset repo_id, e.g. liyitenga/record_20250914225057"
    )
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to record")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--episode_time", type=int, default=60, help="Duration of each episode (seconds)")
    parser.add_argument(
        "--reset_time", type=int, default=10, help="Reset duration between episodes (seconds)"
    )
    parser.add_argument(
        "--task_description", type=str, default="My task description4", help="Task description"
    )
    parser.add_argument("--remote_ip", type=str, default="127.0.0.1", help="Robot host IP")
    parser.add_argument("--robot_id", type=str, default="alohamini_host", help="Robot ID")
    parser.add_argument("--leader_id", type=str, default="so101_leader_bi", help="Leader arm device ID")
    parser.add_argument("--resume", action="store_true", help="Resume recording on existing dataset")

    args = parser.parse_args()

    # === Robot and teleop config ===
    robot_config = LeKiwiClientConfig(remote_ip=args.remote_ip, id=args.robot_id)
    leader_arm_config = BiSOLeaderConfig(
        left_arm_config=SOLeaderConfig(
            port="/dev/am_arm_leader_left",
        ),
        right_arm_config=SOLeaderConfig(
            port="/dev/am_arm_leader_right",
        ),
        id=args.leader_id,
    )
    keyboard_config = KeyboardTeleopConfig()

    robot = LeKiwiClient(robot_config)
    leader_arm = BiSOLeader(leader_arm_config)
    keyboard = KeyboardTeleop(keyboard_config)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # === Dataset setup ===
    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    dataset_root = Path(args.dataset.split("/")[-1])

    if args.resume:
        print("Resuming existing dataset:", args.dataset)
        dataset = LeRobotDataset(
            args.dataset,
        )
        dataset.start_image_writer(num_threads=4)
    else:
        dataset = LeRobotDataset.create(
            repo_id=args.dataset,
            fps=args.fps,
            features=dataset_features,
            robot_type=robot.name,
            use_videos=True,
            image_writer_threads=4,
        )
        print(f"Dataset created with id: {dataset.repo_id}")

    # === Connect devices ===
    robot.connect()
    leader_arm.connect()
    keyboard.connect()

    listener, events = init_keyboard_listener()
    init_rerun(session_name="lekiwi_record")

    if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected:
        raise ValueError("Robot or teleop is not connected!")

    print("Starting record loop...")
    recorded_episodes = 0

    while recorded_episodes < args.num_episodes and not events["stop_recording"]:
        log_say(f"Recording episode {recorded_episodes + 1} of {args.num_episodes}")

        # === Main record loop ===
        record_loop(
            robot=robot,
            events=events,
            fps=args.fps,
            dataset=dataset,
            teleop=[leader_arm, keyboard],
            control_time_s=args.episode_time,
            single_task=args.task_description,
            display_data=True,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )

        # === Reset environment ===
        if not events["stop_recording"] and (
            (recorded_episodes < args.num_episodes - 1) or events["rerecord_episode"]
        ):
            log_say("Reset the environment")
            record_loop(
                robot=robot,
                events=events,
                fps=args.fps,
                teleop=[leader_arm, keyboard],
                control_time_s=args.reset_time,
                single_task=args.task_description,
                display_data=True,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
            )

        if events["rerecord_episode"]:
            log_say("Re-record episode")
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        dataset.save_episode()
        recorded_episodes += 1

    # === Clean up ===
    log_say("Stop recording")
    robot.disconnect()
    leader_arm.disconnect()
    keyboard.disconnect()
    listener.stop()
    dataset.finalize()
    dataset.push_to_hub()


if __name__ == "__main__":
    main()
