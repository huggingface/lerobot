
import argparse
from pathlib import Path
from typing import List
import torch
import os 

# from safetensors.torch import load_file, save_file
from lerobot.common.datasets.populate_dataset import (
    create_lerobot_dataset,
    init_dataset,
    save_current_episode,
    add_frame
)
from lerobot.common.robot_devices.control_utils import (
    record_episode
)
from lerobot.common.envs.factory import make_env
from lerobot.common.utils.utils import init_hydra_config, init_logging, log_say
import matplotlib.pyplot as plt
from PIL import Image


def record_episode(
    env,
    max_frames: int = 500,
    display_cameras=False,
    dataset=None,
    events=None
):

    if events is None:
        events = {"exit_early": False}
    frame_id = 0
    observation, info = env.reset()
    image_keys = [key for key in observation.keys() if "image" in key]
    non_image_keys = [key for key in observation.keys() if "image" not in key]
    if display_cameras:
        plt.ion()
        fig, ax = plt.subplots()
        img = ax.imshow(observation[image_keys[0]][0])
    while frame_id < max_frames:
        action = env.action_space.sample() * 10 # TODO : we had a weight to have bigger actions
        action_dict = {"action": torch.from_numpy(action)[0]}

        for key in non_image_keys:
            observation[key] = torch.from_numpy(observation[key])[0]

        if dataset is not None:
            video_dir =  dataset["videos_dir"]
            current_ep = dataset["num_episodes"]
            for key in image_keys:
                img_dir = os.path.join(video_dir, f"{key}_episode_{current_ep:06d}")
                img = Image.fromarray(observation[f"{key}"][0])
                path = Path(img_dir) / f"frame_{frame_id:06d}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                img.save(str(path), quality=100)
            add_frame(dataset, {"observation.image": observation["observation.images.front"]}, action_dict)

        if display_cameras:
            img.set_data(observation[image_keys[0]][0])
            plt.draw()
            plt.pause(0.0001)
        
        observation, reward, done, truncated, info = env.step(action)
        if done or  truncated:
            observation, info = env.reset()
            break
        frame_id += 1
    if display_cameras:
        plt.close(fig)


def record(
    env,
    root: str,
    repo_id: str,
    fps: int | None = None,
    num_episodes=50,
    video=True,
    run_compute_stats=True,
    tags=None,
    num_image_writer_processes=0,
    num_image_writer_threads_per_camera=4,
    force_override=False,
    display_cameras=True,
    play_sounds=True,
):
    events = None

    # Create empty dataset or load existing saved episodes
    dataset = init_dataset(
        repo_id,
        root,
        force_override,
        fps,
        video,
        write_images=True,
        num_image_writer_processes=num_image_writer_processes,
        num_image_writer_threads=num_image_writer_threads_per_camera*2,
    )

    # Record data
    while True:
        if dataset["num_episodes"] >= num_episodes:
            break

        episode_index = dataset["num_episodes"]
        log_say(f"Recording episode {episode_index}", play_sounds)
        record_episode(
            dataset=dataset,
            env=env,
            events=events,
            display_cameras=display_cameras,
        )

        # Increment by one dataset["current_episode_index"]
        save_current_episode(dataset)

    log_say("Stop recording", play_sounds, blocking=True)

    lerobot_dataset = create_lerobot_dataset(dataset, run_compute_stats, push_to_hub=False, tags=tags, play_sounds=play_sounds)

    log_say("Exiting", play_sounds)
    return lerobot_dataset



if __name__ == "__main__":
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--env",
        type=str,
        default="lerobot/configs/env/lowcostrobot.yaml",
        help="Path to env yaml file to collect random trajectories.",
    )
    base_parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of episodes to record.",
    )
    base_parser.add_argument(
        "--display_cameras",
        type=bool,
        default=False,
        help="Display the cameras while recording.",
    )
    base_parser.add_argument(
        "--root",
        type=Path,
        default="data",
        help="Root directory where the dataset will be stored locally at '{root}/{repo_id}' (e.g. 'data/hf_username/dataset_name').",
    )
    base_parser.add_argument(
        "--repo-id",
        type=str,
        default="alexcbb/lowcostrobot",
        help="Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).",
    )
    base_parser.add_argument("--num-episodes", type=int, default=50, help="Number of episodes to record.")
    base_parser.add_argument(
        "--run-compute-stats",
        type=int,
        default=1,
        help="By default, run the computation of the data statistics at the end of data collection. Compute intensive and not required to just replay an episode.",
    )
    base_parser.add_argument(
        "--tags",
        type=str,
        nargs="*",
        help="Add tags to your dataset on the hub.",
    )
    base_parser.add_argument(
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
    base_parser.add_argument(
        "--num-image-writer-threads-per-camera",
        type=int,
        default=4,
        help=(
            "Number of threads writing the frames as png images on disk, per camera. "
            "Too many threads might cause unstable teleoperation fps due to main thread being blocked. "
            "Not enough threads might cause low camera fps."
        ),
    )
    base_parser.add_argument(
        "--force-override",
        type=int,
        default=0,
        help="By default, data recording is resumed. When set to 1, delete the local directory and start data recording from scratch.",
    )
    args = base_parser.parse_args()

    init_logging()

    env_path = args.env
    del args.env
    kwargs = vars(args)

    # Init environement to collect data
    env_cfg = init_hydra_config(env_path)
    kwargs["fps"] = env_cfg["fps"]
    print(f"FPS: {kwargs['fps']}")
    env = make_env(env_cfg)

    # Record random data
    record(env, **kwargs)

    env.close()

