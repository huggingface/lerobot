# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script demonstrates the use of `LeRobotDataset` class for handling and processing robotic datasets from Hugging Face.
It illustrates how to load datasets, manipulate them, and apply transformations suitable for machine learning tasks in PyTorch.

Features included in this script:
- Viewing a dataset's metadata and exploring its properties.
- Loading an existing dataset from the hub or a subset of it.
- Accessing frames by episode number.
- Using advanced dataset features like timestamp-based frame selection.
- Demonstrating compatibility with PyTorch DataLoader for batch processing.

The script ends with examples of how to batch process data using PyTorch's DataLoader.
"""

from pprint import pprint

import torch
from huggingface_hub import HfApi

import lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


def main():
    # We ported a number of existing datasets ourselves, use this to see the list:
    print("List of available datasets:")
    pprint(lerobot.available_datasets)

    # You can also browse through the datasets created/ported by the community on the hub using the hub api:
    hub_api = HfApi()
    repo_ids = [info.id for info in hub_api.list_datasets(task_categories="robotics", tags=["LeRobot"])]
    pprint(repo_ids)

    # Or simply explore them in your web browser directly at:
    # https://huggingface.co/datasets?other=LeRobot

    # Let's take this one for this example
    repo_id = "lerobot/aloha_mobile_cabinet"
    # We can have a look and fetch its metadata to know more about it:
    ds_meta = LeRobotDatasetMetadata(repo_id)

    # By instantiating just this class, you can quickly access useful information about the content and the
    # structure of the dataset without downloading the actual data yet (only metadata files — which are
    # lightweight).
    print(f"Total number of episodes: {ds_meta.total_episodes}")
    print(f"Average number of frames per episode: {ds_meta.total_frames / ds_meta.total_episodes:.3f}")
    print(f"Frames per second used during data collection: {ds_meta.fps}")
    print(f"Robot type: {ds_meta.robot_type}")
    print(f"keys to access images from cameras: {ds_meta.camera_keys=}\n")

    print("Tasks:")
    print(ds_meta.tasks)
    print("Features:")
    pprint(ds_meta.features)

    # You can also get a short summary by simply printing the object:
    print(ds_meta)

    # You can then load the actual dataset from the hub.
    # Either load any subset of episodes:
    dataset = LeRobotDataset(repo_id, episodes=[0, 10, 11, 23])

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
    # episodes, you can access the frame indices of any episode using dataset.meta.episodes. Here, we access
    # frame indices associated to the first episode:
    episode_index = 0
    from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
    to_idx = dataset.meta.episodes["dataset_to_index"][episode_index]

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
        # loads 6 state vectors: 1.5 seconds before, 1 second before, ... 200 ms, 100 ms, and current frame
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

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=32,
        shuffle=True,
    )
    for batch in dataloader:
        print(f"{batch[camera_key].shape=}")  # (32, 4, c, h, w)
        print(f"{batch['observation.state'].shape=}")  # (32, 6, c)
        print(f"{batch['action'].shape=}")  # (32, 64, c)
        break


if __name__ == "__main__":
    main()
