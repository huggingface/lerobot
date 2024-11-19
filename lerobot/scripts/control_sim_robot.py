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
    --root tmp/data \
    --repo-id $USER/robot_sim_test \
    --num-episodes 1 \
    --run-compute-stats 0
```

Enable the --push-to-hub 1 to push the recorded dataset to the huggingface hub.

- Visualize dataset:
```bash
python lerobot/scripts/visualize_dataset.py \
    --root tmp/data \
    --repo-id $USER/robot_sim_test \
    --episode-index 0
```

- Replay a sequence of test episodes: 
```bash
python lerobot/scripts/control_sim_robot.py replay \
    --robot-path lerobot/configs/robot/your_robot_config.yaml \
    --sim-config lerobot/configs/env/your_sim_config.yaml \
    --fps 30 \
    --root tmp/data \
    --repo-id $USER/robot_sim_test \
    --episodes 0 1 2 3
```
Note: The seed is saved, therefore, during replay we can load the same environment state as the one during collection.

- Record a full dataset in order to train a policy,
30 seconds of recording for each episode, and 10 seconds to reset the environment in between episodes:
```bash
python lerobot/scripts/control_sim_robot.py record \
    --robot-path lerobot/configs/robot/your_robot_config.yaml \
    --sim-config lerobot/configs/env/your_sim_config.yaml \
    --fps 30 \
    --root data \
    --repo-id $USER/robot_sim_test \
    --num-episodes 50 \
    --episode-time-s 30 \
```

**NOTE**: You can use your keyboard to control data recording flow.
- Tap right arrow key '->' to early exit while recording an episode and go to reseting the environment.
- Tap right arrow key '->' to early exit while reseting the environment and got to recording the next episode.
- Tap left arrow key '<-' to early exit and re-record the current episode.
- Tap escape key 'esc' to stop the data recording.
This might require a sudo permission to allow your terminal to monitor keyboard events.

**NOTE**: You can resume/continue data recording by running the same data recording command twice.
To avoid resuming by deleting the dataset, use `--force-override 1`.

"""

import argparse
import importlib
import json
import logging
import time
import traceback
from functools import cache
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import torch
import tqdm
from datasets import Dataset, Features, Sequence, Value
from PIL import Image

from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes, get_default_encoding
from lerobot.common.datasets.utils import calculate_episode_data_index, hf_transform_to_torch
from lerobot.common.datasets.video_utils import VideoFrame
from lerobot.common.robot_devices.control_utils import (
    init_keyboard_listener,
    init_policy,
    log_control_info,
    predict_action,
    stop_recording,
)
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.utils import init_hydra_config, init_logging, log_say


########################################################################################
# Utilities
########################################################################################
def none_or_int(value):
    if value == "None":
        return None
    return int(value)


@cache
def is_headless():
    """Detects if python is running without a monitor."""
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
        return True


def init_sim_calibration(robot, cfg):
    # Constants necessary for transforming the joint pos of the real robot to the sim
    # depending on the robot discription used in that sim.
    start_pos = np.array(robot.leader_arms.main.calibration["start_pos"])
    axis_directions = np.array(cfg.get("axis_directions", [1]))
    offsets = np.array(cfg.get("offsets", [0])) * np.pi

    return {"start_pos": start_pos, "axis_directions": axis_directions, "offsets": offsets}


def real_positions_to_sim(real_positions, axis_directions, start_pos, offsets):
    """Counts - starting position -> radians -> align axes -> offset"""
    return axis_directions * (real_positions - start_pos) * 2.0 * np.pi / 4096 + offsets


def standardize_observation_key_names(observation, image_keys=None, state_keys_dict=None):
    """Change key names for images and states to the standard keys in LeRobot dataset"""
    if image_keys is None:
        image_keys = []

    for key in image_keys:
        if not key.startswith("observation.image"):
            observation["observation.images." + key] = observation.pop(key)

    if state_keys_dict is None:
        state_keys_dict = {}
    for key, obs_key in state_keys_dict.items():
        observation[key] = torch.from_numpy(observation.pop(obs_key))


def add_frame_with_reward(dataset, observation, action, reward, success, seed):
    add_frame(dataset, observation, action)
    ep_dict = dataset["current_episode"]

    if "next.reward" not in ep_dict:
        ep_dict["next.reward"] = []
    if "next.success" not in ep_dict:
        ep_dict["next.success"] = []
    if "seed" not in ep_dict:
        ep_dict["seed"] = []

    ep_dict["next.reward"].append(reward)
    ep_dict["next.success"].append(success)
    ep_dict["seed"].append(seed)


def save_current_episode(dataset):
    episode_index = dataset["num_episodes"]
    ep_dict = dataset["current_episode"]
    episodes_dir = dataset["episodes_dir"]
    rec_info_path = dataset["rec_info_path"]

    ep_dict["next.done"][-1] = True

    for key in ep_dict:
        if "observation" in key and "image" not in key:
            ep_dict[key] = torch.stack(ep_dict[key])

    ep_dict["action"] = torch.stack(ep_dict["action"])
    ep_dict["next.reward"] = torch.tensor(ep_dict["next.reward"])
    ep_dict["next.success"] = torch.tensor(ep_dict["next.success"])
    ep_dict["seed"] = torch.tensor(ep_dict["seed"])
    ep_dict["episode_index"] = torch.tensor(ep_dict["episode_index"])
    ep_dict["frame_index"] = torch.tensor(ep_dict["frame_index"])
    ep_dict["timestamp"] = torch.tensor(ep_dict["timestamp"])
    ep_dict["next.done"] = torch.tensor(ep_dict["next.done"])

    ep_path = episodes_dir / f"episode_{episode_index}.pth"
    torch.save(ep_dict, ep_path)

    rec_info = {
        "last_episode_index": episode_index,
    }
    with open(rec_info_path, "w") as f:
        json.dump(rec_info, f)

    # force re-initialization of episode dictionnary during add_frame
    del dataset["current_episode"]

    dataset["num_episodes"] += 1


def to_hf_dataset(data_dict, video):
    features = {}

    keys = [key for key in data_dict if "observation.images." in key]
    for key in keys:
        if video:
            features[key] = VideoFrame()
        else:
            features[key] = Image()

    features["observation.state"] = Sequence(
        length=data_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None)
    )
    if "observation.velocity" in data_dict:
        features["observation.velocity"] = Sequence(
            length=data_dict["observation.velocity"].shape[1], feature=Value(dtype="float32", id=None)
        )
    if "observation.effort" in data_dict:
        features["observation.effort"] = Sequence(
            length=data_dict["observation.effort"].shape[1], feature=Value(dtype="float32", id=None)
        )
    features["action"] = Sequence(
        length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["next.reward"] = Value(dtype="float32", id=None)
    features["next.success"] = Value(dtype="bool", id=None)

    features["seed"] = Value(dtype="int64", id=None)
    features["episode_index"] = Value(dtype="int64", id=None)
    features["frame_index"] = Value(dtype="int64", id=None)
    features["timestamp"] = Value(dtype="float32", id=None)
    features["next.done"] = Value(dtype="bool", id=None)
    features["index"] = Value(dtype="int64", id=None)
    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset


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
    process_action_fn=None,
    fps: int | None = None,
    root="data",
    repo_id="lerobot/debug",
    pretrained_policy_name_or_path=None,
    policy_overrides=None,
    episode_time_s=30,
    num_episodes=50,
    video=True,
    run_compute_stats=True,
    push_to_hub=True,
    tags=None,
    num_image_writer_processes=0,
    num_image_writers_per_camera=4,
    force_override=False,
    display_cameras=False,
    play_sounds=True,
):
    # Load pretrained policy
    policy = None
    if pretrained_policy_name_or_path is not None:
        policy, policy_fps, device, use_amp = init_policy(pretrained_policy_name_or_path, policy_overrides)

        if fps is None:
            fps = policy_fps
            logging.warning(f"No fps provided, so using the fps from policy config ({policy_fps}).")
        elif fps != policy_fps:
            logging.warning(
                f"There is a mismatch between the provided fps ({fps}) and the one from policy config ({policy_fps})."
            )

    if policy is None and process_action_fn is None:
        raise ValueError("Either policy or process_action_fn has to be set to enable control in sim.")

    # initialize listener before sim env
    listener, events = init_keyboard_listener()

    # create sim env
    env = env()

    # Create empty dataset or load existing saved episodes
    num_cameras = sum([1 if "image" in key else 0 for key in env.observation_space])
    num_image_writers = num_image_writers_per_camera * num_cameras

    image_keys = [key for key in env.observation_space if "image" in key]
    state_keys_dict = env_cfg.state_keys
    write_images = len(image_keys) > 0

    dataset = init_dataset(
        repo_id,
        root,
        force_override,
        fps,
        video,
        write_images=write_images,
        num_image_writer_processes=num_image_writer_processes,
        num_image_writer_threads=num_image_writers * num_cameras,
    )

    while True:
        if dataset["num_episodes"] >= num_episodes:
            break

        episode_index = dataset["num_episodes"]
        log_say(f"Recording episode {episode_index}", play_sounds)

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
                action = process_action_fn(leader_pos)

            observation, reward, terminated, _, info = env.step(action)

            action = {"action": torch.from_numpy(action)}
            success = info.get("is_success", False)

            if dataset is not None:
                standardize_observation_key_names(observation, image_keys, state_keys_dict)
                add_frame_with_reward(dataset, observation, action, reward, success, seed)

            if display_cameras and not is_headless():
                for key in image_keys:
                    cv2.imshow(key, cv2.cvtColor(observation[key].numpy(), cv2.COLOR_RGB2BGR))
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
            delete_current_episode(dataset)
            continue

        # Increment by one dataset["current_episode_index"]
        save_current_episode(dataset)

        if events["stop_recording"]:
            break
        else:
            logging.info("Waiting for a few seconds before starting next episode recording...")
            busy_wait(3)

    log_say("Stop recording", play_sounds, blocking=True)
    stop_recording(robot, listener, display_cameras)

    if "image_writer" in dataset:
        logging.info("Waiting for image writer to terminate...")
        image_writer = dataset["image_writer"]
        stop_image_writer(image_writer, timeout=20)

    log_say("Consolidate episodes", play_sounds)

    num_episodes = dataset["num_episodes"]
    episodes_dir = dataset["episodes_dir"]
    videos_dir = dataset["videos_dir"]

    ep_dicts = []
    for episode_index in tqdm.tqdm(range(num_episodes)):
        ep_path = episodes_dir / f"episode_{episode_index}.pth"
        ep_dict = torch.load(ep_path)
        ep_dicts.append(ep_dict)
    data_dict = concatenate_episodes(ep_dicts)

    if video:
        image_keys = [key for key in data_dict if "image" in key]
        encode_videos(dataset, image_keys, play_sounds)

    hf_dataset = to_hf_dataset(data_dict, video)
    episode_data_index = calculate_episode_data_index(hf_dataset)

    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps,
        "video": video,
    }
    if video:
        info["encoding"] = get_default_encoding()

    lerobot_dataset = LeRobotDataset.from_preloaded(
        repo_id=repo_id,
        hf_dataset=hf_dataset,
        episode_data_index=episode_data_index,
        info=info,
        videos_dir=videos_dir,
    )

    if run_compute_stats:
        log_say("Computing dataset statistics", play_sounds)
        lerobot_dataset.stats = compute_stats(lerobot_dataset)
    else:
        logging.info("Skipping computation of the dataset statistics")
        lerobot_dataset.stats = {}

    save_lerobot_dataset_on_disk(lerobot_dataset)

    if push_to_hub:
        push_lerobot_dataset_to_hub(lerobot_dataset, tags)

    log_say("Exiting", play_sounds)
    return lerobot_dataset


def replay(env, episodes: list, fps: int | None = None, root="data", repo_id="lerobot/debug"):
    env = env()

    local_dir = Path(root) / repo_id
    if not local_dir.exists():
        raise ValueError(local_dir)

    dataset = LeRobotDataset(repo_id, root=root)
    items = dataset.hf_dataset.select_columns("action")
    seeds = dataset.hf_dataset.select_columns("seed")["seed"]

    for episode in episodes:
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

        # wait before playing next episode
        busy_wait(5)


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
        default="data",
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
            "Number of subprocesses handling the saving of frames as PNGs. Set to 0 to use threads only; "
            "set to ≥1 to use subprocesses, each using threads to write images. The best number of processes "
            "and threads depends on your system. We recommend 4 threads per camera with 0 processes. "
            "If fps is unstable, adjust the thread count. If still unstable, try using 1 or more subprocesses."
        ),
    )
    parser_record.add_argument(
        "--num-image-writers-per-camera",
        type=int,
        default=4,
        help=(
            "Number of threads writing the frames as png images on disk, per camera. "
            "Too much threads might cause unstable teleoperation fps due to main thread being blocked. "
            "Not enough threads might cause low camera fps."
        ),
    )
    parser_record.add_argument(
        "--force-override",
        type=int,
        default=0,
        help="By default, data recording is resumed. When set to 1, delete the local directory and start data recording from scratch.",
    )
    parser_record.add_argument(
        "--display-cameras",
        type=int,
        default=0,
        help="Visualize image observations with opencv.",
    )

    parser_replay = subparsers.add_parser("replay", parents=[base_parser])
    parser_replay.add_argument(
        "--fps", type=none_or_int, default=None, help="Frames per second (set to None to disable)"
    )
    parser_replay.add_argument(
        "--root",
        type=Path,
        default="data",
        help="Root directory where the dataset will be stored locally at '{root}/{repo_id}' (e.g. 'data/hf_username/dataset_name').",
    )
    parser_replay.add_argument(
        "--repo-id",
        type=str,
        default="lerobot/test",
        help="Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).",
    )
    parser_replay.add_argument(
        "--episodes", nargs="+", type=int, default=[0], help="Indices of the episodes to replay."
    )

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
    importlib.import_module(f"gym_{env_cfg.env.name}")

    def env_constructor():
        return gym.make(env_cfg.env.handle, disable_env_checker=True, **env_cfg.env.gym)

    robot = None

    if control_mode in ["teleoperate", "record"]:
        # make robot
        robot_overrides = ["~cameras", "~follower_arms"]
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
