"""
This script demonstrates loading and testing the GPR (General Purpose Robot) as a Lerobot dataset locally. (not working yet)

Example Usage:
    python examples/12_load_gpr_dataset.py --raw_dir /path/to/h5/files
"""

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from pprint import pprint
import shutil
import argparse

from lerobot.common.datasets.push_dataset_to_hub.gpr_h5_format import (
    from_raw_to_lerobot_format,
)
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

GPR_FEATURES = {
    "observation.joint_pos": {
        "dtype": "float32",
        "shape": (12,),  # Adjust based on your robot's DOF
        "names": ["joint_positions"],
    },
    "observation.joint_vel": {
        "dtype": "float32",
        "shape": (12,),
        "names": ["joint_velocities"],
    },
    "observation.ang_vel": {
        "dtype": "float32",
        "shape": (3,),
        "names": ["angular_velocity"],
    },
    "observation.euler_rotation": {
        "dtype": "float32",
        "shape": (3,),
        "names": ["euler_angles"],
    },
    "action": {
        "dtype": "float32",
        "shape": (12,),
        "names": ["joint_commands"],
    },
}


def test_gpr_dataset(raw_dir: Path, videos_dir: Path, fps: int):
    # Setup paths
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Create temporary repo_id for local testing
    repo_id = "gpr_test"

    # Convert raw data to LeRobot format
    print("Converting raw data to LeRobot format...")
    hf_dataset, episode_data_index, info = from_raw_to_lerobot_format(
        raw_dir=raw_dir,
        videos_dir=videos_dir,
        fps=fps,  # Your simulation fps
        video=False,  # No video data
    )

    # Delete the existing dataset folder if it exists
    dataset_path = Path.home() / ".cache/huggingface/lerobot/gpr_test"
    if dataset_path.exists():
        print(f"Deleting existing dataset folder: {dataset_path}")
        shutil.rmtree(dataset_path)

    # Create dataset instance
    print("\nCreating dataset...")
    dataset = LeRobotDataset.from_preloaded(
        repo_id=repo_id,
        hf_dataset=hf_dataset,
        episode_data_index=episode_data_index,
        info=info,
        videos_dir=videos_dir,
    )

    #########################################################
    # From this point on its copy paste from lerobot/examples/1_load_lerobot_dataset.py

    # And see how many frames you have:
    print(f"Selected episodes: {dataset.episodes}")
    print(f"Number of episodes selected: {dataset.num_episodes}")
    print(f"Number of frames selected: {dataset.num_frames}")

    # Or simply load the entire dataset:
    dataset = LeRobotDataset(repo_id)
    print(f"Number of episodes selected: {dataset.num_episodes}")
    print(f"Number of frames selected: {dataset.num_frames}")

    # The previous metadata class is contained in the 'meta' attribute of the dataset:
    print(dataset.meta)

    # LeRobotDataset actually wraps an underlying Hugging Face dataset
    # (see https://huggingface.co/docs/datasets for more information).
    print(dataset.hf_dataset)

    # LeRobot datasets also subclasses PyTorch datasets so you can do everything you know and love from working
    # with the latter, like iterating through the dataset.
    # The __getitem__ iterates over the frames of the dataset. Since our datasets are also structured by
    # episodes, you can access the frame indices of any episode using the episode_data_index. Here, we access
    # frame indices associated to the first episode:
    episode_index = 0
    from_idx = dataset.episode_data_index["from"][episode_index].item()
    to_idx = dataset.episode_data_index["to"][episode_index].item()

    # Then we grab all the image frames from the first camera:
    camera_key = dataset.meta.camera_keys[0]
    frames = [dataset[idx][camera_key] for idx in range(from_idx, to_idx)]

    # The objects returned by the dataset are all torch.Tensors
    print(type(frames[0]))
    print(frames[0].shape)

    # Since we're using pytorch, the shape is in pytorch, channel-first convention (c, h, w).
    # We can compare this shape with the information available for that feature
    pprint(dataset.features[camera_key])
    # In particular:
    print(dataset.features[camera_key]["shape"])
    # The shape is in (h, w, c) which is a more universal format.

    # For many machine learning applications we need to load the history of past observations or trajectories of
    # future actions. Our datasets can load previous and future frames for each key/modality, using timestamps
    # differences with the current loaded frame. For instance:
    delta_timestamps = {
        # loads 4 images: 1 second before current frame, 500 ms before, 200 ms before, and current frame
        camera_key: [-1, -0.5, -0.20, 0],
        # loads 8 state vectors: 1.5 seconds before, 1 second before, ... 200 ms, 100 ms, and current frame
        "observation.state": [-1.5, -1, -0.5, -0.20, -0.10, 0],
        # loads 64 action vectors: current frame, 1 frame in the future, 2 frames, ... 63 frames in the future
        "action": [t / dataset.fps for t in range(64)],
    }
    # Note that in any case, these delta_timestamps values need to be multiples of (1/fps) so that added to any
    # timestamp, you still get a valid timestamp.

    dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)
    print(f"\n{dataset[0][camera_key].shape=}")  # (4, c, h, w)
    print(f"{dataset[0]['observation.state'].shape=}")  # (6, c)
    print(f"{dataset[0]['action'].shape=}\n")  # (64, c)

    # Finally, our datasets are fully compatible with PyTorch dataloaders and samplers because they are just
    # PyTorch datasets.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=32,
        shuffle=True,
    )

    for batch in dataloader:
        print(f"{batch[camera_key].shape=}")  # (32, 4, c, h, w)
        print(f"{batch['observation.state'].shape=}")  # (32, 5, c)
        print(f"{batch['action'].shape=}")  # (32, 64, c)
        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and test GPR dataset")
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

    args = parser.parse_args()

    test_gpr_dataset(
        raw_dir=Path(args.raw_dir), videos_dir=Path(args.videos_dir), fps=args.fps
    )
