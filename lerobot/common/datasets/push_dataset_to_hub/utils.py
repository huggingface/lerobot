#!/usr/bin/env python

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
import inspect
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict

import datasets
import numpy
import PIL
import torch

from lerobot.common.datasets.video_utils import encode_video_frames


def concatenate_episodes(ep_dicts):
    data_dict = {}

    keys = ep_dicts[0].keys()
    for key in keys:
        if torch.is_tensor(ep_dicts[0][key][0]):
            data_dict[key] = torch.cat([ep_dict[key] for ep_dict in ep_dicts])
        else:
            if key not in data_dict:
                data_dict[key] = []
            for ep_dict in ep_dicts:
                for x in ep_dict[key]:
                    data_dict[key].append(x)

    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)
    return data_dict


def save_images_concurrently(imgs_array: numpy.array, out_dir: Path, max_workers: int = 4):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_image(img_array, i, out_dir):
        img = PIL.Image.fromarray(img_array)
        img.save(str(out_dir / f"frame_{i:06d}.png"), quality=100)

    num_images = len(imgs_array)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        [executor.submit(save_image, imgs_array[i], i, out_dir) for i in range(num_images)]


def get_default_encoding() -> dict:
    """Returns the default ffmpeg encoding parameters used by `encode_video_frames`."""
    signature = inspect.signature(encode_video_frames)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty and k in ["vcodec", "pix_fmt", "g", "crf"]
    }


def check_repo_id(repo_id: str) -> None:
    if len(repo_id.split("/")) != 2:
        raise ValueError(
            f"""`repo_id` is expected to contain a community or user id `/` the name of the dataset
            (e.g. 'lerobot/pusht'), but contains '{repo_id}'."""
        )


# TODO(aliberts): remove
def calculate_episode_data_index(hf_dataset: datasets.Dataset) -> Dict[str, torch.Tensor]:
    """
    Calculate episode data index for the provided HuggingFace Dataset. Relies on episode_index column of hf_dataset.

    Parameters:
    - hf_dataset (datasets.Dataset): A HuggingFace dataset containing the episode index.

    Returns:
    - episode_data_index: A dictionary containing the data index for each episode. The dictionary has two keys:
        - "from": A tensor containing the starting index of each episode.
        - "to": A tensor containing the ending index of each episode.
    """
    episode_data_index = {"from": [], "to": []}

    current_episode = None
    """
    The episode_index is a list of integers, each representing the episode index of the corresponding example.
    For instance, the following is a valid episode_index:
      [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2]

    Below, we iterate through the episode_index and populate the episode_data_index dictionary with the starting and
    ending index of each episode. For the episode_index above, the episode_data_index dictionary will look like this:
        {
            "from": [0, 3, 7],
            "to": [3, 7, 12]
        }
    """
    if len(hf_dataset) == 0:
        episode_data_index = {
            "from": torch.tensor([]),
            "to": torch.tensor([]),
        }
        return episode_data_index
    for idx, episode_idx in enumerate(hf_dataset["episode_index"]):
        if episode_idx != current_episode:
            # We encountered a new episode, so we append its starting location to the "from" list
            episode_data_index["from"].append(idx)
            # If this is not the first episode, we append the ending location of the previous episode to the "to" list
            if current_episode is not None:
                episode_data_index["to"].append(idx)
            # Let's keep track of the current episode index
            current_episode = episode_idx
        else:
            # We are still in the same episode, so there is nothing for us to do here
            pass
    # We have reached the end of the dataset, so we append the ending location of the last episode to the "to" list
    episode_data_index["to"].append(idx + 1)

    for k in ["from", "to"]:
        episode_data_index[k] = torch.tensor(episode_data_index[k])

    return episode_data_index
