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
import importlib.resources
import json
import logging
import textwrap
from itertools import accumulate
from pathlib import Path
from pprint import pformat
from typing import Any

import datasets
import jsonlines
import numpy as np
import pyarrow.compute as pc
import torch
from datasets.table import embed_table_storage
from huggingface_hub import DatasetCard, DatasetCardData, HfApi
from PIL import Image as PILImage
from torchvision import transforms

from lerobot.common.robot_devices.robots.utils import Robot

DEFAULT_CHUNK_SIZE = 1000  # Max number of episodes per chunk

INFO_PATH = "meta/info.json"
EPISODES_PATH = "meta/episodes.jsonl"
STATS_PATH = "meta/stats.json"
TASKS_PATH = "meta/tasks.jsonl"

DEFAULT_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
DEFAULT_PARQUET_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
DEFAULT_IMAGE_PATH = "images/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.png"

DATASET_CARD_TEMPLATE = """
---
# Metadata will go there
---
This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).

## {}

"""

DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
    "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
    "index": {"dtype": "int64", "shape": (1,), "names": None},
    "task_index": {"dtype": "int64", "shape": (1,), "names": None},
}


def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
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


def unflatten_dict(d: dict, sep: str = "/") -> dict:
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


def serialize_dict(stats: dict[str, torch.Tensor | np.ndarray | dict]) -> dict:
    serialized_dict = {key: value.tolist() for key, value in flatten_dict(stats).items()}
    return unflatten_dict(serialized_dict)


def write_parquet(dataset: datasets.Dataset, fpath: Path) -> None:
    # Embed image bytes into the table before saving to parquet
    format = dataset.format
    dataset = dataset.with_format("arrow")
    dataset = dataset.map(embed_table_storage, batched=False)
    dataset = dataset.with_format(**format)
    dataset.to_parquet(fpath)


def load_json(fpath: Path) -> Any:
    with open(fpath) as f:
        return json.load(f)


def write_json(data: dict, fpath: Path) -> None:
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_jsonlines(fpath: Path) -> list[Any]:
    with jsonlines.open(fpath, "r") as reader:
        return list(reader)


def write_jsonlines(data: dict, fpath: Path) -> None:
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with jsonlines.open(fpath, "w") as writer:
        writer.write_all(data)


def append_jsonlines(data: dict, fpath: Path) -> None:
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with jsonlines.open(fpath, "a") as writer:
        writer.write(data)


def load_info(local_dir: Path) -> dict:
    info = load_json(local_dir / INFO_PATH)
    for ft in info["features"].values():
        ft["shape"] = tuple(ft["shape"])
    return info


def load_stats(local_dir: Path) -> dict:
    if not (local_dir / STATS_PATH).exists():
        return None
    stats = load_json(local_dir / STATS_PATH)
    stats = {key: torch.tensor(value) for key, value in flatten_dict(stats).items()}
    return unflatten_dict(stats)


def load_tasks(local_dir: Path) -> dict:
    tasks = load_jsonlines(local_dir / TASKS_PATH)
    return {item["task_index"]: item["task"] for item in sorted(tasks, key=lambda x: x["task_index"])}


def load_episodes(local_dir: Path) -> dict:
    return load_jsonlines(local_dir / EPISODES_PATH)


def load_image_as_numpy(fpath: str | Path, dtype="float32", channel_first: bool = True) -> np.ndarray:
    img = PILImage.open(fpath).convert("RGB")
    img_array = np.array(img, dtype=dtype)
    if channel_first:  # (H, W, C) -> (C, H, W)
        img_array = np.transpose(img_array, (2, 0, 1))
    if "float" in dtype:
        img_array /= 255.0
    return img_array


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
        elif first_item is None:
            pass
        else:
            items_dict[key] = [torch.tensor(x) for x in items_dict[key]]
    return items_dict


def _get_major_minor(version: str) -> tuple[int]:
    split = version.strip("v").split(".")
    return int(split[0]), int(split[1])


class BackwardCompatibilityError(Exception):
    def __init__(self, repo_id, version):
        message = textwrap.dedent(f"""
            BackwardCompatibilityError: The dataset you requested ({repo_id}) is in {version} format.

            We introduced a new format since v2.0 which is not backward compatible with v1.x.
            Please, use our conversion script. Modify the following command with your own task description:
            ```
            python lerobot/common/datasets/v2/convert_dataset_v1_to_v2.py \\
                --repo-id {repo_id} \\
                --single-task "TASK DESCRIPTION."  # <---- /!\\ Replace TASK DESCRIPTION /!\\
            ```

            A few examples to replace TASK DESCRIPTION: "Pick up the blue cube and place it into the bin.",
            "Insert the peg into the socket.", "Slide open the ziploc bag.", "Take the elevator to the 1st floor.",
            "Open the top cabinet, store the pot inside it then close the cabinet.", "Push the T-shaped block onto the T-shaped target.",
            "Grab the spray paint on the shelf and place it in the bin on top of the robot dog.", "Fold the sweatshirt.", ...

            If you encounter a problem, contact LeRobot maintainers on [Discord](https://discord.com/invite/s3KuuzsPFb)
            or open an [issue on GitHub](https://github.com/huggingface/lerobot/issues/new/choose).
        """)
        super().__init__(message)


def check_version_compatibility(
    repo_id: str, version_to_check: str, current_version: str, enforce_breaking_major: bool = True
) -> None:
    current_major, _ = _get_major_minor(current_version)
    major_to_check, _ = _get_major_minor(version_to_check)
    if major_to_check < current_major and enforce_breaking_major:
        raise BackwardCompatibilityError(repo_id, version_to_check)
    elif float(version_to_check.strip("v")) < float(current_version.strip("v")):
        logging.warning(
            f"""The dataset you requested ({repo_id}) was created with a previous version ({version_to_check}) of the
            codebase. The current codebase version is {current_version}. You should be fine since
            backward compatibility is maintained. If you encounter a problem, contact LeRobot maintainers on
            Discord ('https://discord.com/invite/s3KuuzsPFb') or open an issue on github.""",
        )


def get_hub_safe_version(repo_id: str, version: str) -> str:
    api = HfApi()
    dataset_info = api.list_repo_refs(repo_id, repo_type="dataset")
    branches = [b.name for b in dataset_info.branches]
    if version not in branches:
        num_version = float(version.strip("v"))
        hub_num_versions = [float(v.strip("v")) for v in branches if v.startswith("v")]
        if num_version >= 2.0 and all(v < 2.0 for v in hub_num_versions):
            raise BackwardCompatibilityError(repo_id, version)

        logging.warning(
            f"""You are trying to load a dataset from {repo_id} created with a previous version of the
            codebase. The following versions are available: {branches}.
            The requested version ('{version}') is not found. You should be fine since
            backward compatibility is maintained. If you encounter a problem, contact LeRobot maintainers on
            Discord ('https://discord.com/invite/s3KuuzsPFb') or open an issue on github.""",
        )
        if "main" not in branches:
            raise ValueError(f"Version 'main' not found on {repo_id}")
        return "main"
    else:
        return version


def get_hf_features_from_features(features: dict) -> datasets.Features:
    hf_features = {}
    for key, ft in features.items():
        if ft["dtype"] == "video":
            continue
        elif ft["dtype"] == "image":
            hf_features[key] = datasets.Image()
        elif ft["shape"] == (1,):
            hf_features[key] = datasets.Value(dtype=ft["dtype"])
        else:
            assert len(ft["shape"]) == 1
            hf_features[key] = datasets.Sequence(
                length=ft["shape"][0], feature=datasets.Value(dtype=ft["dtype"])
            )

    return datasets.Features(hf_features)


def get_features_from_robot(robot: Robot, use_videos: bool = True) -> dict:
    camera_ft = {}
    if robot.cameras:
        camera_ft = {
            key: {"dtype": "video" if use_videos else "image", **ft}
            for key, ft in robot.camera_features.items()
        }
    return {**robot.motor_features, **camera_ft, **DEFAULT_FEATURES}


def create_empty_dataset_info(
    codebase_version: str,
    fps: int,
    robot_type: str,
    features: dict,
    use_videos: bool,
) -> dict:
    return {
        "codebase_version": codebase_version,
        "robot_type": robot_type,
        "total_episodes": 0,
        "total_frames": 0,
        "total_tasks": 0,
        "total_videos": 0,
        "total_chunks": 0,
        "chunks_size": DEFAULT_CHUNK_SIZE,
        "fps": fps,
        "splits": {},
        "data_path": DEFAULT_PARQUET_PATH,
        "video_path": DEFAULT_VIDEO_PATH if use_videos else None,
        "features": features,
    }


def get_episode_data_index(
    episode_dicts: list[dict], episodes: list[int] | None = None
) -> dict[str, torch.Tensor]:
    episode_lengths = {ep_idx: ep_dict["length"] for ep_idx, ep_dict in enumerate(episode_dicts)}
    if episodes is not None:
        episode_lengths = {ep_idx: episode_lengths[ep_idx] for ep_idx in episodes}

    cumulative_lenghts = list(accumulate(episode_lengths.values()))
    return {
        "from": torch.LongTensor([0] + cumulative_lenghts[:-1]),
        "to": torch.LongTensor(cumulative_lenghts),
    }


def calculate_total_episode(
    hf_dataset: datasets.Dataset, raise_if_not_contiguous: bool = True
) -> dict[str, torch.Tensor]:
    episode_indices = sorted(hf_dataset.unique("episode_index"))
    total_episodes = len(episode_indices)
    if raise_if_not_contiguous and episode_indices != list(range(total_episodes)):
        raise ValueError("episode_index values are not sorted and contiguous.")
    return total_episodes


def calculate_episode_data_index(hf_dataset: datasets.Dataset) -> dict[str, torch.Tensor]:
    episode_lengths = []
    table = hf_dataset.data.table
    total_episodes = calculate_total_episode(hf_dataset)
    for ep_idx in range(total_episodes):
        ep_table = table.filter(pc.equal(table["episode_index"], ep_idx))
        episode_lengths.insert(ep_idx, len(ep_table))

    cumulative_lenghts = list(accumulate(episode_lengths))
    return {
        "from": torch.LongTensor([0] + cumulative_lenghts[:-1]),
        "to": torch.LongTensor(cumulative_lenghts),
    }


def check_timestamps_sync(
    hf_dataset: datasets.Dataset,
    episode_data_index: dict[str, torch.Tensor],
    fps: int,
    tolerance_s: float,
    raise_value_error: bool = True,
) -> bool:
    """
    This check is to make sure that each timestamps is separated to the next by 1/fps +/- tolerance to
    account for possible numerical error.
    """
    timestamps = torch.stack(hf_dataset["timestamp"])
    diffs = torch.diff(timestamps)
    within_tolerance = torch.abs(diffs - 1 / fps) <= tolerance_s

    # We mask differences between the timestamp at the end of an episode
    # and the one at the start of the next episode since these are expected
    # to be outside tolerance.
    mask = torch.ones(len(diffs), dtype=torch.bool)
    ignored_diffs = episode_data_index["to"][:-1] - 1
    mask[ignored_diffs] = False
    filtered_within_tolerance = within_tolerance[mask]

    if not torch.all(filtered_within_tolerance):
        # Track original indices before masking
        original_indices = torch.arange(len(diffs))
        filtered_indices = original_indices[mask]
        outside_tolerance_filtered_indices = torch.nonzero(~filtered_within_tolerance)  # .squeeze()
        outside_tolerance_indices = filtered_indices[outside_tolerance_filtered_indices]
        episode_indices = torch.stack(hf_dataset["episode_index"])

        outside_tolerances = []
        for idx in outside_tolerance_indices:
            entry = {
                "timestamps": [timestamps[idx], timestamps[idx + 1]],
                "diff": diffs[idx],
                "episode_index": episode_indices[idx].item(),
            }
            outside_tolerances.append(entry)

        if raise_value_error:
            raise ValueError(
                f"""One or several timestamps unexpectedly violate the tolerance inside episode range.
                This might be due to synchronization issues with timestamps during data collection.
                \n{pformat(outside_tolerances)}"""
            )
        return False

    return True


def check_delta_timestamps(
    delta_timestamps: dict[str, list[float]], fps: int, tolerance_s: float, raise_value_error: bool = True
) -> bool:
    """This will check if all the values in delta_timestamps are multiples of 1/fps +/- tolerance.
    This is to ensure that these delta_timestamps added to any timestamp from a dataset will themselves be
    actual timestamps from the dataset.
    """
    outside_tolerance = {}
    for key, delta_ts in delta_timestamps.items():
        within_tolerance = [abs(ts * fps - round(ts * fps)) / fps <= tolerance_s for ts in delta_ts]
        if not all(within_tolerance):
            outside_tolerance[key] = [
                ts for ts, is_within in zip(delta_ts, within_tolerance, strict=True) if not is_within
            ]

    if len(outside_tolerance) > 0:
        if raise_value_error:
            raise ValueError(
                f"""
                The following delta_timestamps are found outside of tolerance range.
                Please make sure they are multiples of 1/{fps} +/- tolerance and adjust
                their values accordingly.
                \n{pformat(outside_tolerance)}
                """
            )
        return False

    return True


def get_delta_indices(delta_timestamps: dict[str, list[float]], fps: int) -> dict[str, list[int]]:
    delta_indices = {}
    for key, delta_ts in delta_timestamps.items():
        delta_indices[key] = (torch.tensor(delta_ts) * fps).long().tolist()

    return delta_indices


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


def create_branch(repo_id, *, branch: str, repo_type: str | None = None) -> None:
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


def create_lerobot_dataset_card(
    tags: list | None = None,
    dataset_info: dict | None = None,
    **kwargs,
) -> DatasetCard:
    """
    Keyword arguments will be used to replace values in ./lerobot/common/datasets/card_template.md.
    Note: If specified, license must be one of https://huggingface.co/docs/hub/repositories-licenses.
    """
    card_tags = ["LeRobot"]
    card_template_path = importlib.resources.path("lerobot.common.datasets", "card_template.md")

    if tags:
        card_tags += tags
    if dataset_info:
        dataset_structure = "[meta/info.json](meta/info.json):\n"
        dataset_structure += f"```json\n{json.dumps(dataset_info, indent=4)}\n```\n"
        kwargs = {**kwargs, "dataset_structure": dataset_structure}
    card_data = DatasetCardData(
        license=kwargs.get("license"),
        tags=card_tags,
        task_categories=["robotics"],
        configs=[
            {
                "config_name": "default",
                "data_files": "data/*/*.parquet",
            }
        ],
    )

    return DatasetCard.from_template(
        card_data=card_data,
        template_path=str(card_template_path),
        **kwargs,
    )
