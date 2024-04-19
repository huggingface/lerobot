"""
This script demonstrates the visualization of various robotic datasets from Hugging Face hub.
It covers the steps from loading the datasets, filtering specific episodes, and converting the frame data to MP4 videos.
Importantly, the dataset format is agnostic to any deep learning library and doesn't require using `lerobot` functions.
It is compatible with pytorch, jax, numpy, etc.

As an example, this script saves frames of episode number 5 of the PushT dataset to a mp4 video and saves the result here:
`outputs/examples/1_visualize_hugging_face_datasets/episode_5.mp4`

This script supports several Hugging Face datasets, among which:
1. [Pusht](https://huggingface.co/datasets/lerobot/pusht)
2. [Xarm Lift Medium](https://huggingface.co/datasets/lerobot/xarm_lift_medium)
3. [Aloha Sim Insertion Human](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human)
4. [Aloha Sim Insertion Scripted](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_scripted)
5. [Aloha Sim Transfer Cube Human](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human)
6. [Aloha Sim Transfer Cube Scripted](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_scripted)

To try a different Hugging Face dataset, you can replace this line:
```python
hf_dataset, fps = load_dataset("lerobot/pusht", split="train"), 10
```
by one of these:
```python
hf_dataset, fps = load_dataset("lerobot/xarm_lift_medium", split="train"), 15
hf_dataset, fps = load_dataset("lerobot/aloha_sim_insertion_human", split="train"), 50
hf_dataset, fps = load_dataset("lerobot/aloha_sim_insertion_scripted", split="train"), 50
hf_dataset, fps = load_dataset("lerobot/aloha_sim_transfer_cube_human", split="train"), 50
hf_dataset, fps = load_dataset("lerobot/aloha_sim_transfer_cube_scripted", split="train"), 50
```
"""

from pathlib import Path

import imageio
from datasets import load_dataset

# TODO(rcadene): list available datasets on lerobot page using `datasets`

# download/load hugging face dataset in pyarrow format
hf_dataset, fps = load_dataset("lerobot/pusht", revision="v1.0", split="train"), 10

# display name of dataset and its features
print(f"{hf_dataset=}")
print(f"{hf_dataset.features=}")

# display useful statistics about frames and episodes, which are sequences of frames from the same video
print(f"number of frames: {len(hf_dataset)=}")
print(f"number of episodes: {len(hf_dataset.unique('episode_id'))=}")
print(f"average number of frames per episode: {len(hf_dataset) / len(hf_dataset.unique('episode_id')):.3f}")

# select the frames belonging to episode number 5
hf_dataset = hf_dataset.filter(lambda frame: frame["episode_id"] == 5)

# load all frames of episode 5 in RAM in PIL format
frames = hf_dataset["observation.image"]

# save episode frames to a mp4 video
Path("outputs/examples/1_load_hugging_face_dataset").mkdir(parents=True, exist_ok=True)
imageio.mimsave("outputs/examples/1_load_hugging_face_dataset/episode_5.mp4", frames, fps=fps)
