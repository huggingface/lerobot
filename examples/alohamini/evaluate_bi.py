#!/usr/bin/env python3

"""
Usage:
  python examples/alohamini/evaluate_bi.py \
    --num_episodes 3 \
    --fps 20 \
    --episode_time 45 \
    --task_description "Pick and place task" \
    --hf_model_id liyitenga/act_policy \
    --hf_dataset_id liyitenga/eval_dataset \
    --remote_ip 127.0.0.1 \
    --robot_id alohamini \
    --hf_model_id ./outputs/train/act_your_dataset1/checkpoints/020000/pretrained_model
"""

import argparse

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.robots.alohamini import LeKiwiClient, LeKiwiClientConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun


def main():
    parser = argparse.ArgumentParser(description="Evaluate Alohamini Robot with a pretrained policy")
    parser.add_argument("--num_episodes", type=int, default=2, help="Number of episodes to record")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--episode_time", type=int, default=60, help="Duration of each episode in seconds")
    parser.add_argument(
        "--task_description", type=str, default="My task description", help="Description of the task"
    )
    parser.add_argument("--hf_model_id", type=str, required=True, help="HuggingFace model repo id")
    parser.add_argument("--hf_dataset_id", type=str, required=True, help="HuggingFace dataset repo id")
    parser.add_argument("--remote_ip", type=str, default="127.0.0.1", help="Alohamini host IP address")
    parser.add_argument("--robot_id", type=str, default="Alohamini", help="Robot ID")

    args = parser.parse_args()

    # === Robot config ===
    robot_config = LeKiwiClientConfig(remote_ip=args.remote_ip, id=args.robot_id)
    robot = LeKiwiClient(robot_config)
    robot.connect()

    # === Policy ===
    policy = ACTPolicy.from_pretrained(args.hf_model_id)

    # === Dataset features ===
    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    dataset = LeRobotDataset.create(
        repo_id=args.hf_dataset_id,
        fps=args.fps,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    # === Policy Processors ===
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy,
        pretrained_path=args.hf_model_id,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    listener, events = init_keyboard_listener()
    init_rerun(session_name="alohamini_evaluate")

    if not robot.is_connected:
        raise ValueError("Robot is not connected!")

    print("Starting evaluate loop...")
    recorded_episodes = 0

    while recorded_episodes < args.num_episodes and not events["stop_recording"]:
        log_say(f"Running inference, recording eval episode {recorded_episodes + 1} of {args.num_episodes}")

        record_loop(
            robot=robot,
            events=events,
            fps=args.fps,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset=dataset,
            control_time_s=args.episode_time,
            single_task=args.task_description,
            display_data=True,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )

        if not events["stop_recording"]:
            dataset.save_episode()
            recorded_episodes += 1

    log_say("Stop recording")
    robot.disconnect()
    listener.stop()
    dataset.finalize()
    dataset.push_to_hub()


if __name__ == "__main__":
    main()
