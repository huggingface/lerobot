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
import json
import re
import warnings
from functools import cache
from pathlib import Path
from typing import Dict

import datasets
import torch
from datasets import load_dataset, load_from_disk
from huggingface_hub import DatasetCard, HfApi, hf_hub_download, snapshot_download
from PIL import Image as PILImage
from safetensors.torch import load_file
from torchvision import transforms

DATASET_CARD_TEMPLATE = """
---
# Metadata will go there
---
This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).

"""


def flatten_dict(d, parent_key="", sep="/"):
    """Flatten a nested dictionary structure by collapsing nested keys into one key with a separator.

    For example:
    ```
    >>> dct = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}`
    >>> print(flatten_dict(dct))
    {"a/b": 1, "a/c/d": 2, "e": 3}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d, sep="/"):
    outdict = {}
    for key, value in d.items():
        parts = key.split(sep)
        d = outdict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return outdict


def hf_transform_to_torch(items_dict: dict[torch.Tensor | None]):
    """Get a transform function that convert items from Hugging Face dataset (pyarrow)
    to torch tensors. Importantly, images are converted from PIL, which corresponds to
    a channel last representation (h w c) of uint8 type, to a torch image representation
    with channel first (c h w) of float32 type in range [0,1].
    """
    for key in items_dict:
        first_item = items_dict[key][0]
        if isinstance(first_item, PILImage.Image):
            to_tensor = transforms.ToTensor()
            items_dict[key] = [to_tensor(img) for img in items_dict[key]]
        elif isinstance(first_item, str):
            # TODO (michel-aractingi): add str2embedding via language tokenizer
            # For now we leave this part up to the user to choose how to address
            # language conditioned tasks
            pass
        elif isinstance(first_item, dict) and "path" in first_item and "timestamp" in first_item:
            # video frame will be processed downstream
            pass
        elif first_item is None:
            pass
        else:
            items_dict[key] = [torch.tensor(x) for x in items_dict[key]]
    return items_dict


@cache
def get_hf_dataset_safe_version(repo_id: str, version: str) -> str:
    api = HfApi()
    dataset_info = api.list_repo_refs(repo_id, repo_type="dataset")
    branches = [b.name for b in dataset_info.branches]
    if version not in branches:
        warnings.warn(
            f"""You are trying to load a dataset from {repo_id} created with a previous version of the
            codebase. The following versions are available: {branches}.
            The requested version ('{version}') is not found. You should be fine since
            backward compatibility is maintained. If you encounter a problem, contact LeRobot maintainers on
            Discord ('https://discord.com/invite/s3KuuzsPFb') or open an issue on github.""",
            stacklevel=1,
        )
        if "main" not in branches:
            raise ValueError(f"Version 'main' not found on {repo_id}")
        return "main"
    else:
        return version


def load_hf_dataset(repo_id: str, version: str, root: Path, split: str) -> datasets.Dataset:
    """hf_dataset contains all the observations, states, actions, rewards, etc."""
    if root is not None:
        hf_dataset = load_from_disk(str(Path(root) / repo_id / "train"))
        # TODO(rcadene): clean this which enables getting a subset of dataset
        if split != "train":
            if "%" in split:
                raise NotImplementedError(f"We dont support splitting based on percentage for now ({split}).")
            match_from = re.search(r"train\[(\d+):\]", split)
            match_to = re.search(r"train\[:(\d+)\]", split)
            if match_from:
                from_frame_index = int(match_from.group(1))
                hf_dataset = hf_dataset.select(range(from_frame_index, len(hf_dataset)))
            elif match_to:
                to_frame_index = int(match_to.group(1))
                hf_dataset = hf_dataset.select(range(to_frame_index))
            else:
                raise ValueError(
                    f'`split` ({split}) should either be "train", "train[INT:]", or "train[:INT]"'
                )
    else:
        safe_version = get_hf_dataset_safe_version(repo_id, version)
        hf_dataset = load_dataset(repo_id, revision=safe_version, split=split)

    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset


def load_episode_data_index(repo_id, version, root) -> dict[str, torch.Tensor]:
    """episode_data_index contains the range of indices for each episode

    Example:
    ```python
    from_id = episode_data_index["from"][episode_id].item()
    to_id = episode_data_index["to"][episode_id].item()
    episode_frames = [dataset[i] for i in range(from_id, to_id)]
    ```
    """
    if root is not None:
        path = Path(root) / repo_id / "meta_data" / "episode_data_index.safetensors"
    else:
        safe_version = get_hf_dataset_safe_version(repo_id, version)
        path = hf_hub_download(
            repo_id, "meta_data/episode_data_index.safetensors", repo_type="dataset", revision=safe_version
        )

    return load_file(path)


def load_stats(repo_id, version, root) -> dict[str, dict[str, torch.Tensor]]:
    """stats contains the statistics per modality computed over the full dataset, such as max, min, mean, std

    Example:
    ```python
    normalized_action = (action - stats["action"]["mean"]) / stats["action"]["std"]
    ```
    """
    if root is not None:
        path = Path(root) / repo_id / "meta_data" / "stats.safetensors"
    else:
        safe_version = get_hf_dataset_safe_version(repo_id, version)
        path = hf_hub_download(
            repo_id, "meta_data/stats.safetensors", repo_type="dataset", revision=safe_version
        )

    stats = load_file(path)
    return unflatten_dict(stats)


def load_info(repo_id, version, root) -> dict:
    """info contains useful information regarding the dataset that are not stored elsewhere

    Example:
    ```python
    print("frame per second used to collect the video", info["fps"])
    ```
    """
    if root is not None:
        path = Path(root) / repo_id / "meta_data" / "info.json"
    else:
        safe_version = get_hf_dataset_safe_version(repo_id, version)
        path = hf_hub_download(repo_id, "meta_data/info.json", repo_type="dataset", revision=safe_version)

    with open(path) as f:
        info = json.load(f)
    return info


def load_videos(repo_id, version, root) -> Path:
    if root is not None:
        path = Path(root) / repo_id / "videos"
    else:
        # TODO(rcadene): we download the whole repo here. see if we can avoid this
        safe_version = get_hf_dataset_safe_version(repo_id, version)
        repo_dir = snapshot_download(repo_id, repo_type="dataset", revision=safe_version)
        path = Path(repo_dir) / "videos"

    return path


def load_previous_and_future_frames(
    item: dict[str, torch.Tensor],
    hf_dataset: datasets.Dataset,
    episode_data_index: dict[str, torch.Tensor],
    delta_timestamps: dict[str, list[float]],
    tolerance_s: float,
) -> dict[torch.Tensor]:
    """
    Given a current item in the dataset containing a timestamp (e.g. 0.6 seconds), and a list of time differences of
    some modalities (e.g. delta_timestamps={"observation.image": [-0.8, -0.2, 0, 0.2]}), this function computes for each
    given modality (e.g. "observation.image") a list of query timestamps (e.g. [-0.2, 0.4, 0.6, 0.8]) and loads the closest
    frames in the dataset.

    Importantly, when no frame can be found around a query timestamp within a specified tolerance window, this function
    raises an AssertionError. When a timestamp is queried before the first available timestamp of the episode or after
    the last available timestamp, the violation of the tolerance doesnt raise an AssertionError, and the function
    populates a boolean array indicating which frames are outside of the episode range. For instance, this boolean array
    is useful during batched training to not supervise actions associated to timestamps coming after the end of the
    episode, or to pad the observations in a specific way. Note that by default the observation frames before the start
    of the episode are the same as the first frame of the episode.

    Parameters:
    - item (dict): A dictionary containing all the data related to a frame. It is the result of `dataset[idx]`. Each key
      corresponds to a different modality (e.g., "timestamp", "observation.image", "action").
    - hf_dataset (datasets.Dataset): A dictionary containing the full dataset. Each key corresponds to a different
      modality (e.g., "timestamp", "observation.image", "action").
    - episode_data_index (dict): A dictionary containing two keys ("from" and "to") associated to dataset indices.
      They indicate the start index and end index of each episode in the dataset.
    - delta_timestamps (dict): A dictionary containing lists of delta timestamps for each possible modality to be
      retrieved. These deltas are added to the item timestamp to form the query timestamps.
    - tolerance_s (float, optional): The tolerance level (in seconds) used to determine if a data point is close enough to the query
      timestamp by asserting `tol > difference`. It is suggested to set `tol` to a smaller value than the
      smallest expected inter-frame period, but large enough to account for jitter.

    Returns:
    - The same item with the queried frames for each modality specified in delta_timestamps, with an additional key for
      each modality (e.g. "observation.image_is_pad").

    Raises:
    - AssertionError: If any of the frames unexpectedly violate the tolerance level. This could indicate synchronization
      issues with timestamps during data collection.
    """
    # get indices of the frames associated to the episode, and their timestamps
    ep_id = item["episode_index"].item()
    ep_data_id_from = episode_data_index["from"][ep_id].item()
    ep_data_id_to = episode_data_index["to"][ep_id].item()
    ep_data_ids = torch.arange(ep_data_id_from, ep_data_id_to, 1)

    # load timestamps
    ep_timestamps = hf_dataset.select_columns("timestamp")[ep_data_id_from:ep_data_id_to]["timestamp"]
    ep_timestamps = torch.stack(ep_timestamps)

    # we make the assumption that the timestamps are sorted
    ep_first_ts = ep_timestamps[0]
    ep_last_ts = ep_timestamps[-1]
    current_ts = item["timestamp"].item()

    for key in delta_timestamps:
        # get timestamps used as query to retrieve data of previous/future frames
        delta_ts = delta_timestamps[key]
        query_ts = current_ts + torch.tensor(delta_ts)

        # compute distances between each query timestamp and all timestamps of all the frames belonging to the episode
        dist = torch.cdist(query_ts[:, None], ep_timestamps[:, None], p=1)
        min_, argmin_ = dist.min(1)

        # TODO(rcadene): synchronize timestamps + interpolation if needed

        is_pad = min_ > tolerance_s

        # check violated query timestamps are all outside the episode range
        assert ((query_ts[is_pad] < ep_first_ts) | (ep_last_ts < query_ts[is_pad])).all(), (
            f"One or several timestamps unexpectedly violate the tolerance ({min_} > {tolerance_s=}) inside episode range."
            "This might be due to synchronization issues with timestamps during data collection."
        )

        # get dataset indices corresponding to frames to be loaded
        data_ids = ep_data_ids[argmin_]

        # load frames modality
        item[key] = hf_dataset.select_columns(key)[data_ids][key]

        if isinstance(item[key][0], dict) and "path" in item[key][0]:
            # video mode where frame are expressed as dict of path and timestamp
            item[key] = item[key]
        else:
            item[key] = torch.stack(item[key])

        item[f"{key}_is_pad"] = is_pad

    return item


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


def reset_episode_index(hf_dataset: datasets.Dataset) -> datasets.Dataset:
    """Reset the `episode_index` of the provided HuggingFace Dataset.

    `episode_data_index` (and related functionality such as `load_previous_and_future_frames`) requires the
    `episode_index` to be sorted, continuous (1,1,1 and not 1,2,1) and start at 0.

    This brings the `episode_index` to the required format.
    """
    if len(hf_dataset) == 0:
        return hf_dataset
    unique_episode_idxs = torch.stack(hf_dataset["episode_index"]).unique().tolist()
    episode_idx_to_reset_idx_mapping = {
        ep_id: reset_ep_id for reset_ep_id, ep_id in enumerate(unique_episode_idxs)
    }

    def modify_ep_idx_func(example):
        example["episode_index"] = episode_idx_to_reset_idx_mapping[example["episode_index"].item()]
        return example

    hf_dataset = hf_dataset.map(modify_ep_idx_func)

    return hf_dataset


def cycle(iterable):
    """The equivalent of itertools.cycle, but safe for Pytorch dataloaders.

    See https://github.com/pytorch/pytorch/issues/23900 for information on why itertools.cycle is not safe.
    """
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def create_branch(repo_id, *, branch: str, repo_type: str | None = None):
    """Create a branch on a existing Hugging Face repo. Delete the branch if it already
    exists before creating it.
    """
    api = HfApi()

    branches = api.list_repo_refs(repo_id, repo_type=repo_type).branches
    refs = [branch.ref for branch in branches]
    ref = f"refs/heads/{branch}"
    if ref in refs:
        api.delete_branch(repo_id, repo_type=repo_type, branch=branch)

    api.create_branch(repo_id, repo_type=repo_type, branch=branch)


def create_lerobot_dataset_card(tags: list | None = None, text: str | None = None) -> DatasetCard:
    card = DatasetCard(DATASET_CARD_TEMPLATE)
    card.data.task_categories = ["robotics"]
    card.data.tags = ["LeRobot"]
    if tags is not None:
        card.data.tags += tags
    if text is not None:
        card.text += text
    return card
