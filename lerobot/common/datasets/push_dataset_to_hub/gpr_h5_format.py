"""
This script loads a GPR dataset from raw HDF5 files and converts it to lerobot dataset format.

Example Usage:
    python lerobot/common/datasets/push_dataset_to_hub/gpr_h5_format.py --raw_dir /path/to/h5/files
"""

import argparse
import h5py
import numpy as np
import torch
from pathlib import Path
from datasets import Dataset, Features, Sequence, Value
from tqdm import tqdm
from pprint import pprint

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.common.datasets.push_dataset_to_hub.utils import (
    calculate_episode_data_index,
    concatenate_episodes,
)
from lerobot.common.datasets.utils import hf_transform_to_torch


def check_format(raw_dir) -> bool:
    """Verify HDF5 files have expected structure"""
    print(f"[DEBUG] Checking format for directory: {raw_dir}")
    hdf5_paths = list(raw_dir.glob("*.h5"))
    assert len(hdf5_paths) > 0, "No HDF5 files found"
    print(f"[DEBUG] Found {len(hdf5_paths)} HDF5 files")

    for hdf5_path in hdf5_paths:
        print(f"[DEBUG] Checking file: {hdf5_path}")
        with h5py.File(hdf5_path, "r") as data:
            print(f"[DEBUG] File contents: {list(data.keys())}")
            print(f"[DEBUG] Observations contents: {list(data['observations'].keys())}")
            # Verify required datasets exist
            assert "observations" in data
            assert "q" in data["observations"]
            assert "dq" in data["observations"]
            assert "ang_vel" in data["observations"]
            assert "euler" in data["observations"]
            assert "prev_actions" in data
            assert "curr_actions" in data

            # Verify shapes match
            num_frames = data["observations"]["q"].shape[0]
            assert data["observations"]["dq"].shape[0] == num_frames
            assert data["observations"]["ang_vel"].shape[0] == num_frames
            assert data["observations"]["euler"].shape[0] == num_frames
            assert data["prev_actions"].shape[0] == num_frames
            assert data["curr_actions"].shape[0] == num_frames


def load_from_raw(
    raw_dir: Path,
    videos_dir: Path,
    fps: int,
    video: bool,
    episodes: list[int] | None = None,
    encoding: dict | None = None,
):
    """Load data from HDF5 files into standardized format"""
    print(f"[DEBUG] Loading raw data from: {raw_dir}")
    print(f"[DEBUG] Videos dir: {videos_dir}, FPS: {fps}, Video: {video}")
    print(f"[DEBUG] Episodes to load: {episodes}")
    hdf5_files = sorted(raw_dir.glob("*.h5"))
    num_episodes = len(hdf5_files)
    print(f"[DEBUG] Found {len(hdf5_files)} total HDF5 files")

    ep_dicts = []
    ep_ids = episodes if episodes else range(num_episodes)
    print(f"[DEBUG] Processing episodes: {list(ep_ids)}")

    for ep_idx in tqdm(ep_ids):
        ep_path = hdf5_files[ep_idx]
        print(f"[DEBUG] Processing episode {ep_idx} from file: {ep_path}")
        with h5py.File(ep_path, "r") as ep:
            # Load data
            print(f"[DEBUG] Episode data shapes:")
            print(f"[DEBUG] - joint_pos: {ep['observations']['q'][:].shape}")
            print(f"[DEBUG] - joint_vel: {ep['observations']['dq'][:].shape}")
            print(f"[DEBUG] - ang_vel: {ep['observations']['ang_vel'][:].shape}")
            print(f"[DEBUG] - euler_rotation: {ep['observations']['euler'][:].shape}")
            print(f"[DEBUG] - prev_actions: {ep['prev_actions'][:].shape}")
            print(f"[DEBUG] - curr_actions: {ep['curr_actions'][:].shape}")

            joint_pos = torch.from_numpy(ep["observations"]["q"][:])
            joint_vel = torch.from_numpy(ep["observations"]["dq"][:])
            ang_vel = torch.from_numpy(ep["observations"]["ang_vel"][:])
            euler_rotation = torch.from_numpy(ep["observations"]["euler"][:])
            prev_actions = torch.from_numpy(ep["prev_actions"][:])
            curr_actions = torch.from_numpy(ep["curr_actions"][:])

            num_frames = joint_pos.shape[0]

            # Create done signal (True for last frame)
            done = torch.zeros(num_frames, dtype=torch.bool)
            done[-1] = True

            ep_dict = {
                "observation.joint_pos": joint_pos,
                "observation.joint_vel": joint_vel,
                "observation.ang_vel": ang_vel,
                "observation.euler_rotation": euler_rotation,
                "prev_actions": prev_actions,
                "action": curr_actions,
                "episode_index": torch.tensor([ep_idx] * num_frames),
                "frame_index": torch.arange(0, num_frames, 1),
                "timestamp": torch.arange(0, num_frames, 1) / fps,
                "next.done": done,
            }
            ep_dicts.append(ep_dict)

            print(f"[DEBUG] Created episode dict with {num_frames} frames")

    print(f"[DEBUG] Concatenating {len(ep_dicts)} episodes")
    data_dict = concatenate_episodes(ep_dicts)
    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)
    print(f"[DEBUG] Final data_dict shapes:")
    for key, value in data_dict.items():
        print(f"[DEBUG] - {key}: {value.shape}")
    return data_dict


def to_hf_dataset(data_dict, video) -> Dataset:
    """Convert to HuggingFace dataset format"""
    print("[DEBUG] Converting to HuggingFace dataset format")
    print(f"[DEBUG] Input data_dict keys: {list(data_dict.keys())}")
    features = {
        "observation.joint_pos": Sequence(
            length=data_dict["observation.joint_pos"].shape[1],
            feature=Value(dtype="float32", id=None),
        ),
        "observation.joint_vel": Sequence(
            length=data_dict["observation.joint_vel"].shape[1],
            feature=Value(dtype="float32", id=None),
        ),
        "observation.ang_vel": Sequence(
            length=data_dict["observation.ang_vel"].shape[1],
            feature=Value(dtype="float32", id=None),
        ),
        "observation.euler_rotation": Sequence(
            length=data_dict["observation.euler_rotation"].shape[1],
            feature=Value(dtype="float32", id=None),
        ),
        "prev_actions": Sequence(
            length=data_dict["prev_actions"].shape[1],
            feature=Value(dtype="float32", id=None),
        ),
        "action": Sequence(
            length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)
        ),
        "episode_index": Value(dtype="int64", id=None),
        "frame_index": Value(dtype="int64", id=None),
        "timestamp": Value(dtype="float32", id=None),
        "next.done": Value(dtype="bool", id=None),
        "index": Value(dtype="int64", id=None),
    }

    print("[DEBUG] Creating HuggingFace dataset")
    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    print(f"[DEBUG] Dataset size: {len(hf_dataset)}")
    print("[DEBUG] Setting transform function")
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset


def from_raw_to_lerobot_format(
    raw_dir: Path,
    videos_dir: Path,
    fps: int | None = None,
    video: bool = True,
    episodes: list[int] | None = None,
    encoding: dict | None = None,
):
    """Main function to convert raw data to LeRobot format"""
    print(f"[DEBUG] Starting conversion from raw to LeRobot format")
    print(f"[DEBUG] Parameters:")
    print(f"[DEBUG] - raw_dir: {raw_dir}")
    print(f"[DEBUG] - videos_dir: {videos_dir}")
    print(f"[DEBUG] - fps: {fps}")
    print(f"[DEBUG] - video: {video}")
    print(f"[DEBUG] - episodes: {episodes}")
    print(f"[DEBUG] - encoding: {encoding}")
    check_format(raw_dir)

    if fps is None:
        fps = 50  # Default FPS for your dataset
        print(f"[DEBUG] Using default FPS: {fps}")

    data_dict = load_from_raw(raw_dir, videos_dir, fps, video, episodes, encoding)
    hf_dataset = to_hf_dataset(data_dict, video)
    print("[DEBUG] Calculating episode data index")
    episode_data_index = calculate_episode_data_index(hf_dataset)

    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps,
        "video": video,
    }
    print(f"[DEBUG] Final info: {info}")

    return hf_dataset, episode_data_index, info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert GPR HDF5 dataset to LeRobot format"
    )
    parser.add_argument(
        "--raw_dir", type=str, required=True, help="Directory containing raw HDF5 files"
    )
    parser.add_argument(
        "--videos_dir",
        type=str,
        default="data/temp",
        help="Directory for video output (default: data/temp)",
    )
    parser.add_argument(
        "--fps", type=int, default=50, help="Frames per second (default: 50)"
    )
    parser.add_argument(
        "--video", action="store_true", help="Enable video processing (default: False)"
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    videos_dir = Path(args.videos_dir)
    videos_dir.mkdir(parents=True, exist_ok=True)

    print("Converting raw data to LeRobot format...")
    hf_dataset, episode_data_index, info = from_raw_to_lerobot_format(
        raw_dir=raw_dir, videos_dir=videos_dir, fps=args.fps, video=args.video
    )
    print("Conversion completed!")
    print("\nDataset info:")
    pprint(hf_dataset)
