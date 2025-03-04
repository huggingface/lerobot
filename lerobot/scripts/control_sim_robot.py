"""
Utilities to control a robot in simulation.

Useful to record a dataset, replay a recorded episode and record an evaluation dataset.

Examples of usage:


- Unlimited teleoperation at a limited frequency of 30 Hz, to simulate data recording frequency.
  You can modify this value depending on how fast your simulation can run:
```bash
python lerobot/scripts/control_robot.py teleoperate \
    --fps 30 \
    --robot-path lerobot/configs/robot/your_robot_config.yaml \
    --sim-config lerobot/configs/env/your_sim_config.yaml
```

- Record one episode in order to test replay:
```bash
python lerobot/scripts/control_sim_robot.py record \
    --robot-path lerobot/configs/robot/your_robot_config.yaml \
    --sim-config lerobot/configs/env/your_sim_config.yaml \
    --fps 30 \
    --repo-id $USER/robot_sim_test \
    --num-episodes 1 \
    --run-compute-stats 0
```

Enable the --push-to-hub 1 to push the recorded dataset to the huggingface hub.

- Visualize dataset:
```bash
python lerobot/scripts/visualize_dataset.py \
    --repo-id $USER/robot_sim_test \
    --episode-index 0
```

- Replay a sequence of test episodes:
```bash
python lerobot/scripts/control_sim_robot.py replay \
    --robot-path lerobot/configs/robot/your_robot_config.yaml \
    --sim-config lerobot/configs/env/your_sim_config.yaml \
    --fps 30 \
    --repo-id $USER/robot_sim_test \
    --episode 0
```
Note: The seed is saved, therefore, during replay we can load the same environment state as the one during collection.

- Record a full dataset in order to train a policy,
30 seconds of recording for each episode, and 10 seconds to reset the environment in between episodes:
```bash
python lerobot/scripts/control_sim_robot.py record \
    --robot-path lerobot/configs/robot/your_robot_config.yaml \
    --sim-config lerobot/configs/env/your_sim_config.yaml \
    --fps 30 \
    --repo-id $USER/robot_sim_test \
    --num-episodes 50 \
    --episode-time-s 30 \
```

**NOTE**: You can use your keyboard to control data recording flow.
- Tap right arrow key '->' to early exit while recording an episode and go to resetting the environment.
- Tap right arrow key '->' to early exit while resetting the environment and got to recording the next episode.
- Tap left arrow key '<-' to early exit and re-record the current episode.
- Tap escape key 'esc' to stop the data recording.
This might require a sudo permission to allow your terminal to monitor keyboard events.

**NOTE**: You can resume/continue data recording by running the same data recording command twice.
"""

import argparse
import importlib
import logging
import time
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.control_utils import (
    init_keyboard_listener,
    init_policy,
    is_headless,
    log_control_info,
    predict_action,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
    stop_recording,
)
from lerobot.common.robot_devices.robots.utils import Robot, make_robot
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.utils import init_hydra_config, init_logging, log_say

raise NotImplementedError("This script is currently deactivated")

DEFAULT_FEATURES = {
    "next.reward": {
        "dtype": "float32",
        "shape": (1,),
        "names": None,
    },
    "next.success": {
        "dtype": "bool",
        "shape": (1,),
        "names": None,
    },
    "seed": {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    },
    "timestamp": {
        "dtype": "float32",
        "shape": (1,),
        "names": None,
    },
}


########################################################################################
# Utilities
########################################################################################
def none_or_int(value):
    if value == "None":
        return None
    return int(value)


def init_sim_calibration(robot, cfg):
    # Constants necessary for transforming the joint pos of the real robot to the sim
    # depending on the robot description used in that sim.
    start_pos = np.array(robot.leader_arms.main.calibration["start_pos"])
    axis_directions = np.array(cfg.get("axis_directions", [1]))
    offsets = np.array(cfg.get("offsets", [0])) * np.pi

    return {"start_pos": start_pos, "axis_directions": axis_directions, "offsets": offsets}


def real_positions_to_sim(real_positions, axis_directions, start_pos, offsets):
    """Counts - starting position -> radians -> align axes -> offset"""
    return axis_directions * (real_positions - start_pos) * 2.0 * np.pi / 4096 + offsets


########################################################################################
# Control modes
########################################################################################


def teleoperate(env, robot: Robot, process_action_fn, teleop_time_s=None):
    env = env()
    env.reset()
    start_teleop_t = time.perf_counter()
    while True:
        leader_pos = robot.leader_arms.main.read("Present_Position")
        action = process_action_fn(leader_pos)
        env.step(np.expand_dims(action, 0))
        if teleop_time_s is not None and time.perf_counter() - start_teleop_t > teleop_time_s:
            print("Teleoperation processes finished.")
            break


def record(
    env,
    robot: Robot,
    process_action_from_leader,
    root: Path,
    repo_id: str,
    task: str,
    fps: int | None = None,
    tags: list[str] | None = None,
    pretrained_policy_name_or_path: str = None,
    policy_overrides: bool | None = None,
    episode_time_s: int = 30,
    num_episodes: int = 50,
    video: bool = True,
    push_to_hub: bool = True,
    num_image_writer_processes: int = 0,
    num_image_writer_threads_per_camera: int = 4,
    display_cameras: bool = False,
    play_sounds: bool = True,
    resume: bool = False,
    local_files_only: bool = False,
    run_compute_stats: bool = True,
) -> LeRobotDataset:
    # Load pretrained policy
    policy = None
    if pretrained_policy_name_or_path is not None:
        policy, policy_fps, device, use_amp = init_policy(pretrained_policy_name_or_path, policy_overrides)

        if fps is None:
            fps = policy_fps
            logging.warning(f"No fps provided, so using the fps from policy config ({policy_fps}).")

    if policy is None and process_action_from_leader is None:
        raise ValueError("Either policy or process_action_fn has to be set to enable control in sim.")

    # initialize listener before sim env
    listener, events = init_keyboard_listener()

    # create sim env
    env = env()

    # Create empty dataset or load existing saved episodes
    num_cameras = sum([1 if "image" in key else 0 for key in env.observation_space])

    # get image keys
    image_keys = [key for key in env.observation_space if "image" in key]
    state_keys_dict = env_cfg.state_keys

    if resume:
        dataset = LeRobotDataset(
            repo_id,
            root=root,
            local_files_only=local_files_only,
        )
        dataset.start_image_writer(
            num_processes=num_image_writer_processes,
            num_threads=num_image_writer_threads_per_camera * num_cameras,
        )
        sanity_check_dataset_robot_compatibility(dataset, robot, fps, video)
    else:
        features = DEFAULT_FEATURES
        # add image keys to features
        for key in image_keys:
            shape = env.observation_space[key].shape
            if not key.startswith("observation.image."):
                key = "observation.image." + key
            features[key] = {"dtype": "video", "names": ["channels", "height", "width"], "shape": shape}

        for key, obs_key in state_keys_dict.items():
            features[key] = {
                "dtype": "float32",
                "names": None,
                "shape": env.observation_space[obs_key].shape,
            }

        features["action"] = {"dtype": "float32", "shape": env.action_space.shape, "names": None}

        # Create empty dataset or load existing saved episodes
        sanity_check_dataset_name(repo_id, policy)
        dataset = LeRobotDataset.create(
            repo_id,
            fps,
            root=root,
            features=features,
            use_videos=video,
            image_writer_processes=num_image_writer_processes,
            image_writer_threads=num_image_writer_threads_per_camera * num_cameras,
        )

    recorded_episodes = 0
    while True:
        log_say(f"Recording episode {dataset.num_episodes}", play_sounds)

        if events is None:
            events = {"exit_early": False}

        if episode_time_s is None:
            episode_time_s = float("inf")

        timestamp = 0
        start_episode_t = time.perf_counter()

        seed = np.random.randint(0, 1e5)
        observation, info = env.reset(seed=seed)

        while timestamp < episode_time_s:
            start_loop_t = time.perf_counter()

            if policy is not None:
                action = predict_action(observation, policy, device, use_amp)
            else:
                leader_pos = robot.leader_arms.main.read("Present_Position")
                action = process_action_from_leader(leader_pos)

            observation, reward, terminated, _, info = env.step(action)

            success = info.get("is_success", False)
            env_timestamp = info.get("timestamp", dataset.episode_buffer["size"] / fps)

            frame = {
                "action": torch.from_numpy(action),
                "next.reward": reward,
                "next.success": success,
                "seed": seed,
                "timestamp": env_timestamp,
            }

            for key in image_keys:
                if not key.startswith("observation.image"):
                    frame["observation.image." + key] = observation[key]
                else:
                    frame[key] = observation[key]

            for key, obs_key in state_keys_dict.items():
                frame[key] = torch.from_numpy(observation[obs_key])

            dataset.add_frame(frame)

            if display_cameras and not is_headless():
                for key in image_keys:
                    cv2.imshow(key, cv2.cvtColor(observation[key], cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

            if fps is not None:
                dt_s = time.perf_counter() - start_loop_t
                busy_wait(1 / fps - dt_s)

            dt_s = time.perf_counter() - start_loop_t
            log_control_info(robot, dt_s, fps=fps)

            timestamp = time.perf_counter() - start_episode_t
            if events["exit_early"] or terminated:
                events["exit_early"] = False
                break

        if events["rerecord_episode"]:
            log_say("Re-record episode", play_sounds)
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        dataset.save_episode(task=task)
        recorded_episodes += 1

        if events["stop_recording"] or recorded_episodes >= num_episodes:
            break
        else:
            logging.info("Waiting for a few seconds before starting next episode recording...")
            busy_wait(3)

    log_say("Stop recording", play_sounds, blocking=True)
    stop_recording(robot, listener, display_cameras)

    if run_compute_stats:
        logging.info("Computing dataset statistics")
    dataset.consolidate(run_compute_stats)

    if push_to_hub:
        dataset.push_to_hub(tags=tags)

    log_say("Exiting", play_sounds)
    return dataset


def replay(
    env, root: Path, repo_id: str, episode: int, fps: int | None = None, local_files_only: bool = True
):
    env = env()

    local_dir = Path(root) / repo_id
    if not local_dir.exists():
        raise ValueError(local_dir)

    dataset = LeRobotDataset(repo_id, root=root, local_files_only=local_files_only)
    items = dataset.hf_dataset.select_columns("action")
    seeds = dataset.hf_dataset.select_columns("seed")["seed"]

    from_idx = dataset.episode_data_index["from"][episode].item()
    to_idx = dataset.episode_data_index["to"][episode].item()
    env.reset(seed=seeds[from_idx].item())
    logging.info("Replaying episode")
    log_say("Replaying episode", play_sounds=True)
    for idx in range(from_idx, to_idx):
        start_episode_t = time.perf_counter()
        action = items[idx]["action"]
        env.step(action.unsqueeze(0).numpy())
        dt_s = time.perf_counter() - start_episode_t
        busy_wait(1 / fps - dt_s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Set common options for all the subparsers
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--robot-path",
        type=str,
        default="lerobot/configs/robot/koch.yaml",
        help="Path to robot yaml file used to instantiate the robot using `make_robot` factory function.",
    )

    base_parser.add_argument(
        "--sim-config",
        help="Path to a yaml config you want to use for initializing a sim environment based on gym ",
    )

    parser_record = subparsers.add_parser("teleoperate", parents=[base_parser])

    parser_record = subparsers.add_parser("record", parents=[base_parser])
    parser_record.add_argument(
        "--fps", type=none_or_int, default=None, help="Frames per second (set to None to disable)"
    )
    parser_record.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory where the dataset will be stored locally at '{root}/{repo_id}' (e.g. 'data/hf_username/dataset_name').",
    )
    parser_record.add_argument(
        "--repo-id",
        type=str,
        default="lerobot/test",
        help="Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).",
    )
    parser_record.add_argument(
        "--episode-time-s",
        type=int,
        default=60,
        help="Number of seconds for data recording for each episode.",
    )
    parser_record.add_argument(
        "--task",
        type=str,
        required=True,
        help="A description of the task preformed during recording that can be used as a language instruction.",
    )
    parser_record.add_argument("--num-episodes", type=int, default=50, help="Number of episodes to record.")
    parser_record.add_argument(
        "--run-compute-stats",
        type=int,
        default=1,
        help="By default, run the computation of the data statistics at the end of data collection. Compute intensive and not required to just replay an episode.",
    )
    parser_record.add_argument(
        "--push-to-hub",
        type=int,
        default=1,
        help="Upload dataset to Hugging Face hub.",
    )
    parser_record.add_argument(
        "--tags",
        type=str,
        nargs="*",
        help="Add tags to your dataset on the hub.",
    )
    parser_record.add_argument(
        "--num-image-writer-processes",
        type=int,
        default=0,
        help=(
            "Number of subprocesses handling the saving of frames as PNG. Set to 0 to use threads only; "
            "set to ≥1 to use subprocesses, each using threads to write images. The best number of processes "
            "and threads depends on your system. We recommend 4 threads per camera with 0 processes. "
            "If fps is unstable, adjust the thread count. If still unstable, try using 1 or more subprocesses."
        ),
    )
    parser_record.add_argument(
        "--num-image-writer-threads-per-camera",
        type=int,
        default=4,
        help=(
            "Number of threads writing the frames as png images on disk, per camera. "
            "Too much threads might cause unstable teleoperation fps due to main thread being blocked. "
            "Not enough threads might cause low camera fps."
        ),
    )
    parser_record.add_argument(
        "--display-cameras",
        type=int,
        default=0,
        help="Visualize image observations with opencv.",
    )
    parser_record.add_argument(
        "--resume",
        type=int,
        default=0,
        help="Resume recording on an existing dataset.",
    )
    parser_replay = subparsers.add_parser("replay", parents=[base_parser])
    parser_replay.add_argument(
        "--fps", type=none_or_int, default=None, help="Frames per second (set to None to disable)"
    )
    parser_replay.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory where the dataset will be stored locally (e.g. 'data/hf_username/dataset_name'). By default, stored in cache folder.",
    )
    parser_replay.add_argument(
        "--repo-id",
        type=str,
        default="lerobot/test",
        help="Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).",
    )
    parser_replay.add_argument("--episode", type=int, default=0, help="Index of the episodes to replay.")

    args = parser.parse_args()

    init_logging()

    control_mode = args.mode
    robot_path = args.robot_path
    env_config_path = args.sim_config
    kwargs = vars(args)
    del kwargs["mode"]
    del kwargs["robot_path"]
    del kwargs["sim_config"]

    # make gym env
    env_cfg = init_hydra_config(env_config_path)
    importlib.import_module(f"gym_{env_cfg.env.type}")

    def env_constructor():
        return gym.make(env_cfg.env.handle, disable_env_checker=True, **env_cfg.env.gym)

    robot = None
    process_leader_actions_fn = None

    if control_mode in ["teleoperate", "record"]:
        # make robot
        robot_overrides = ["~cameras", "~follower_arms"]
        # TODO(rcadene): remove
        robot_cfg = init_hydra_config(robot_path, robot_overrides)
        robot = make_robot(robot_cfg)
        robot.connect()

        calib_kwgs = init_sim_calibration(robot, env_cfg.calibration)

        def process_leader_actions_fn(action):
            return real_positions_to_sim(action, **calib_kwgs)

        robot.leader_arms.main.calibration = None

    if control_mode == "teleoperate":
        teleoperate(env_constructor, robot, process_leader_actions_fn)

    elif control_mode == "record":
        record(env_constructor, robot, process_leader_actions_fn, **kwargs)

    elif control_mode == "replay":
        replay(env_constructor, **kwargs)

    else:
        raise ValueError(
            f"Invalid control mode: '{control_mode}', only valid modes are teleoperate, record and replay."
        )

    if robot and robot.is_connected:
        # Disconnect manually to avoid a "Core dump" during process
        # termination due to camera threads not properly exiting.
        robot.disconnect()
