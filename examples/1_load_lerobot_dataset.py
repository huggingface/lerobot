"""
This script demonstrates the use of `LeRobotDataset` class for handling and processing robotic datasets from Hugging Face.
It illustrates how to load datasets, manipulate them, and apply transformations suitable for machine learning tasks in PyTorch.

Features included in this script:
- Loading a dataset and accessing its properties.
- Filtering data by episode number.
- Converting tensor data for visualization.
- Saving video files from dataset frames.
- Using advanced dataset features like timestamp-based frame selection.
- Demonstrating compatibility with PyTorch DataLoader for batch processing.

The script ends with examples of how to batch process data using PyTorch's DataLoader.
"""

from pathlib import Path

import imageio
import torch

import lerobot
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

print("List of available datasets", lerobot.available_datasets)
# # >>> ['lerobot/aloha_sim_insertion_human', 'lerobot/aloha_sim_insertion_scripted',
# #     'lerobot/aloha_sim_transfer_cube_human', 'lerobot/aloha_sim_transfer_cube_scripted',
# #     'lerobot/pusht', 'lerobot/xarm_lift_medium']

repo_id = "lerobot/pusht"

# You can easily load a dataset from a Hugging Face repositery
dataset = LeRobotDataset(repo_id)

# LeRobotDataset is actually a thin wrapper around an underlying Hugging Face dataset  (see https://huggingface.co/docs/datasets/index for more information).
# TODO(rcadene): update to make the print pretty
print(f"{dataset=}")
print(f"{dataset.hf_dataset=}")

# and provides additional utilities for robotics and compatibility with pytorch
print(f"number of samples/frames: {dataset.num_samples=}")
print(f"number of episodes: {dataset.num_episodes=}")
print(f"average number of frames per episode: {dataset.num_samples / dataset.num_episodes:.3f}")
print(f"frames per second used during data collection: {dataset.fps=}")
print(f"keys to access images from cameras: {dataset.image_keys=}")

# While the LeRobotDataset adds helpers for working within our library, we still expose the underling Hugging Face dataset.
# It may be freely replaced or modified in place. Here we use the filtering to keep only frames from episode 5.
# TODO(rcadene): remove this example of accessing hf_dataset
dataset.hf_dataset = dataset.hf_dataset.filter(lambda frame: frame["episode_index"] == 5)

# LeRobot datsets actually subclass PyTorch datasets. So you can do everything you know and love from working with the latter, for example: iterating through the dataset. Here we grab all the image frames.
frames = [sample["observation.image"] for sample in dataset]

# but frames are now float32 range [0,1] channel first (c,h,w) to follow pytorch convention,
# to view them, we convert to uint8 range [0,255]
frames = [(frame * 255).type(torch.uint8) for frame in frames]
# and to channel last (h,w,c)
frames = [frame.permute((1, 2, 0)).numpy() for frame in frames]

# and finally save them to a mp4 video
Path("outputs/examples/1_load_lerobot_dataset").mkdir(parents=True, exist_ok=True)
imageio.mimsave("outputs/examples/1_load_lerobot_dataset/episode_5.mp4", frames, fps=dataset.fps)

# For many machine learning applications we need to load histories of past observations, or trajectorys of future actions. Our datasets can load previous and future frames for each key/modality,
# using timestamps differences with the current loaded frame. For instance:
delta_timestamps = {
    # loads 4 images: 1 second before current frame, 500 ms before, 200 ms before, and current frame
    "observation.image": [-1, -0.5, -0.20, 0],
    # loads 8 state vectors: 1.5 seconds before, 1 second before, ... 20 ms, 10 ms, and current frame
    "observation.state": [-1.5, -1, -0.5, -0.20, -0.10, -0.02, -0.01, 0],
    # loads 64 action vectors: current frame, 1 frame in the future, 2 frames, ... 63 frames in the future
    "action": [t / dataset.fps for t in range(64)],
}
dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)
print(f"{dataset[0]['observation.image'].shape=}")  # (4,c,h,w)
print(f"{dataset[0]['observation.state'].shape=}")  # (8,c)
print(f"{dataset[0]['action'].shape=}")  # (64,c)

# Finally, our datasets are fully compatible with PyTorch dataloaders and samplers
# because they are just PyTorch datasets.
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=1,
    batch_size=32,
    shuffle=True,
)
for batch in dataloader:
    print(f"{batch['observation.image'].shape=}")  # (32,4,c,h,w)
    print(f"{batch['observation.state'].shape=}")  # (32,8,c)
    print(f"{batch['action'].shape=}")  # (32,64,c)
    break
