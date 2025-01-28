import shutil
from pathlib import Path

import h5py
import numpy as np
import torch
import tqdm

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME, LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw


def create_empty_dataset(dataset_name, robot_type, mode="video", has_velocity=False, has_effort=False):
    motors = [
        # TODO(rcadene): verify
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
    ]
    cameras = [
        "cam_high",
        "cam_low",
        "cam_left_wrist",
        "cam_right_wrist",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_velocity:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    dataset = LeRobotDataset.create(
        repo_id=f"cadene/{dataset_name}_v2",
        fps=50,
        robot_type=robot_type,
        features=features,
    )
    return dataset


def get_cameras(hdf5_files):
    with h5py.File(hdf5_files[0], "r") as ep:
        # ignore depth channel, not currently handled
        # TODO(rcadene): add depth
        rgb_cameras = [key for key in ep["/observations/images"].keys() if "depth" not in key]  # noqa: SIM118
    return rgb_cameras


def has_velocity(hdf5_files):
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep


def has_effort(hdf5_files):
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep


def load_raw_images_per_camera(ep, cameras):
    imgs_per_cam = {}
    for camera in cameras:
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4

        if uncompressed:
            # load all images in RAM
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:
            import cv2

            # load one compressed image after the other in RAM and uncompress
            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                imgs_array.append(cv2.imdecode(data, 1))
            imgs_array = np.array(imgs_array)

        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def load_raw_episode_data(ep_path):
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(ep["/observations/qpos"][:])
        action = torch.from_numpy(ep["/action"][:])

        velocity = None
        if "/observations/qvel" in ep:
            velocity = torch.from_numpy(ep["/observations/qvel"][:])

        effort = None
        if "/observations/effort" in ep:
            effort = torch.from_numpy(ep["/observations/effort"][:])

        imgs_per_cam = load_raw_images_per_camera(ep)

    return imgs_per_cam, state, action, velocity, effort


def populate_dataset(dataset, hdf5_files, task, episodes=None):
    if episodes is None:
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]

        imgs_per_cam, state, action, velocity, effort = load_raw_episode_data(ep_path)
        num_frames = state.shape[0]

        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
            }

            for camera, img_array in imgs_per_cam.items():
                frame[f"observation.images.{camera}"] = img_array[i]

            if velocity is not None:
                frame["observation.velocity"] = velocity[i]
            if effort is not None:
                frame["observation.effort"] = effort[i]

            dataset.add_frame(frame)

        dataset.save_episode(task=task)

    return dataset


def port_aloha(raw_dir, raw_repo_id, repo_id, episodes: list[int] | None = None, push_to_hub=True):
    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        download_raw(raw_dir, repo_id=raw_repo_id)

    hdf5_files = sorted(raw_dir.glob("episode_*.hdf5"))

    dataset_name = repo_id.split("/")[1]
    dataset = create_empty_dataset(
        repo_id,
        robot_type="mobile_aloha" if "mobile" in dataset_name else "aloha",
        has_effort=has_effort(hdf5_files),
        has_velocity=has_velocity(hdf5_files),
    )
    dataset = populate_dataset(
        dataset,
        hdf5_files,
        task="DEBUG",
        episodes=episodes,
    )
    dataset.consolidate()

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    raw_repo_id = "lerobot-raw/aloha_sim_insertion_human_raw"
    repo_id = "cadene/aloha_sim_insertion_human_v2"
    port_aloha(f"data/{raw_repo_id}", raw_repo_id, repo_id, episodes=[0, 1], push_to_hub=False)
