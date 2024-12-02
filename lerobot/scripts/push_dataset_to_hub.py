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
"""
Use this script to convert your dataset into LeRobot dataset format and upload it to the Hugging Face hub,
or store it locally. LeRobot dataset format is lightweight, fast to load from, and does not require any
installation of neural net specific packages like pytorch, tensorflow, jax.

Example of how to download raw datasets, convert them into LeRobotDataset format, and push them to the hub:
```
python lerobot/scripts/push_dataset_to_hub.py \
--raw-dir data/pusht_raw \
--raw-format pusht_zarr \
--repo-id lerobot/pusht

python lerobot/scripts/push_dataset_to_hub.py \
--raw-dir data/xarm_lift_medium_raw \
--raw-format xarm_pkl \
--repo-id lerobot/xarm_lift_medium

python lerobot/scripts/push_dataset_to_hub.py \
--raw-dir data/aloha_sim_insertion_scripted_raw \
--raw-format aloha_hdf5 \
--repo-id lerobot/aloha_sim_insertion_scripted

python lerobot/scripts/push_dataset_to_hub.py \
--raw-dir data/umi_cup_in_the_wild_raw \
--raw-format umi_zarr \
--repo-id lerobot/umi_cup_in_the_wild
```
"""

import argparse
import json
import shutil
import warnings
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi
from safetensors.torch import save_file

from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub.utils import check_repo_id
from lerobot.common.datasets.utils import create_branch, create_lerobot_dataset_card, flatten_dict


def get_from_raw_to_lerobot_format_fn(raw_format: str):
    if raw_format == "pusht_zarr":
        from lerobot.common.datasets.push_dataset_to_hub.pusht_zarr_format import from_raw_to_lerobot_format
    elif raw_format == "umi_zarr":
        from lerobot.common.datasets.push_dataset_to_hub.umi_zarr_format import from_raw_to_lerobot_format
    elif raw_format == "aloha_hdf5":
        from lerobot.common.datasets.push_dataset_to_hub.aloha_hdf5_format import from_raw_to_lerobot_format
    elif raw_format in ["rlds", "openx"]:
        from lerobot.common.datasets.push_dataset_to_hub.openx_rlds_format import from_raw_to_lerobot_format
    elif raw_format == "dora_parquet":
        from lerobot.common.datasets.push_dataset_to_hub.dora_parquet_format import from_raw_to_lerobot_format
    elif raw_format == "xarm_pkl":
        from lerobot.common.datasets.push_dataset_to_hub.xarm_pkl_format import from_raw_to_lerobot_format
    elif raw_format == "cam_png":
        from lerobot.common.datasets.push_dataset_to_hub.cam_png_format import from_raw_to_lerobot_format
    else:
        raise ValueError(
            f"The selected {raw_format} can't be found. Did you add it to `lerobot/scripts/push_dataset_to_hub.py::get_from_raw_to_lerobot_format_fn`?"
        )

    return from_raw_to_lerobot_format


def save_meta_data(
    info: dict[str, Any], stats: dict, episode_data_index: dict[str, list], meta_data_dir: Path
):
    meta_data_dir.mkdir(parents=True, exist_ok=True)

    # save info
    info_path = meta_data_dir / "info.json"
    with open(str(info_path), "w") as f:
        json.dump(info, f, indent=4)

    # save stats
    stats_path = meta_data_dir / "stats.safetensors"
    save_file(flatten_dict(stats), stats_path)

    # save episode_data_index
    episode_data_index = {key: torch.tensor(episode_data_index[key]) for key in episode_data_index}
    ep_data_idx_path = meta_data_dir / "episode_data_index.safetensors"
    save_file(episode_data_index, ep_data_idx_path)


def push_meta_data_to_hub(repo_id: str, meta_data_dir: str | Path, revision: str | None):
    """Expect all meta data files to be all stored in a single "meta_data" directory.
    On the hugging face repositery, they will be uploaded in a "meta_data" directory at the root.
    """
    api = HfApi()
    api.upload_folder(
        folder_path=meta_data_dir,
        path_in_repo="meta_data",
        repo_id=repo_id,
        revision=revision,
        repo_type="dataset",
    )


def push_dataset_card_to_hub(
    repo_id: str,
    revision: str | None,
    tags: list | None = None,
    license: str = "apache-2.0",
    **card_kwargs,
):
    """Creates and pushes a LeRobotDataset Card with appropriate tags to easily find it on the hub."""
    card = create_lerobot_dataset_card(tags=tags, license=license, **card_kwargs)
    card.push_to_hub(repo_id=repo_id, repo_type="dataset", revision=revision)


def push_videos_to_hub(repo_id: str, videos_dir: str | Path, revision: str | None):
    """Expect mp4 files to be all stored in a single "videos" directory.
    On the hugging face repositery, they will be uploaded in a "videos" directory at the root.
    """
    api = HfApi()
    api.upload_folder(
        folder_path=videos_dir,
        path_in_repo="videos",
        repo_id=repo_id,
        revision=revision,
        repo_type="dataset",
        allow_patterns="*.mp4",
    )


def push_dataset_to_hub(
    raw_dir: Path,
    raw_format: str,
    repo_id: str,
    push_to_hub: bool = True,
    local_dir: Path | None = None,
    fps: int | None = None,
    video: bool = True,
    batch_size: int = 32,
    num_workers: int = 8,
    episodes: list[int] | None = None,
    force_override: bool = False,
    resume: bool = False,
    cache_dir: Path = Path("/tmp"),
    tests_data_dir: Path | None = None,
    encoding: dict | None = None,
):
    check_repo_id(repo_id)
    user_id, dataset_id = repo_id.split("/")

    # Robustify when `raw_dir` is str instead of Path
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise NotADirectoryError(
            f"{raw_dir} does not exists. Check your paths or run this command to download an existing raw dataset on the hub: "
            f"`python lerobot/common/datasets/push_dataset_to_hub/_download_raw.py --raw-dir your/raw/dir --repo-id your/repo/id_raw`"
        )

    if local_dir:
        # Robustify when `local_dir` is str instead of Path
        local_dir = Path(local_dir)

        # Send warning if local_dir isn't well formated
        if local_dir.parts[-2] != user_id or local_dir.parts[-1] != dataset_id:
            warnings.warn(
                f"`local_dir` ({local_dir}) doesn't contain a community or user id `/` the name of the dataset that match the `repo_id` (e.g. 'data/lerobot/pusht'). Following this naming convention is advised, but not mandatory.",
                stacklevel=1,
            )

        # Check we don't override an existing `local_dir` by mistake
        if local_dir.exists():
            if force_override:
                shutil.rmtree(local_dir)
            elif not resume:
                raise ValueError(f"`local_dir` already exists ({local_dir}). Use `--force-override 1`.")

        meta_data_dir = local_dir / "meta_data"
        videos_dir = local_dir / "videos"
    else:
        # Temporary directory used to store images, videos, meta_data
        meta_data_dir = Path(cache_dir) / "meta_data"
        videos_dir = Path(cache_dir) / "videos"

    if raw_format is None:
        # TODO(rcadene, adilzouitine): implement auto_find_raw_format
        raise NotImplementedError()
        # raw_format = auto_find_raw_format(raw_dir)

    # convert dataset from original raw format to LeRobot format
    from_raw_to_lerobot_format = get_from_raw_to_lerobot_format_fn(raw_format)

    hf_dataset, episode_data_index, info = from_raw_to_lerobot_format(
        raw_dir,
        videos_dir,
        fps,
        video,
        episodes,
        encoding,
    )

    lerobot_dataset = LeRobotDataset.from_preloaded(
        repo_id=repo_id,
        hf_dataset=hf_dataset,
        episode_data_index=episode_data_index,
        info=info,
        videos_dir=videos_dir,
    )
    stats = compute_stats(lerobot_dataset, batch_size, num_workers)

    if local_dir:
        hf_dataset = hf_dataset.with_format(None)  # to remove transforms that cant be saved
        hf_dataset.save_to_disk(str(local_dir / "train"))

    if push_to_hub or local_dir:
        # mandatory for upload
        save_meta_data(info, stats, episode_data_index, meta_data_dir)

    if push_to_hub:
        hf_dataset.push_to_hub(repo_id, revision="main")
        push_meta_data_to_hub(repo_id, meta_data_dir, revision="main")
        push_dataset_card_to_hub(repo_id, revision="main")
        if video:
            push_videos_to_hub(repo_id, videos_dir, revision="main")
        create_branch(repo_id, repo_type="dataset", branch=CODEBASE_VERSION)

    if tests_data_dir:
        # get the first episode
        num_items_first_ep = episode_data_index["to"][0] - episode_data_index["from"][0]
        test_hf_dataset = hf_dataset.select(range(num_items_first_ep))
        episode_data_index = {k: v[:1] for k, v in episode_data_index.items()}

        test_hf_dataset = test_hf_dataset.with_format(None)
        test_hf_dataset.save_to_disk(str(tests_data_dir / repo_id / "train"))

        tests_meta_data = tests_data_dir / repo_id / "meta_data"
        save_meta_data(info, stats, episode_data_index, tests_meta_data)

        # copy videos of first episode to tests directory
        episode_index = 0
        tests_videos_dir = tests_data_dir / repo_id / "videos"
        tests_videos_dir.mkdir(parents=True, exist_ok=True)
        for key in lerobot_dataset.camera_keys:
            fname = f"{key}_episode_{episode_index:06d}.mp4"
            shutil.copy(videos_dir / fname, tests_videos_dir / fname)

    if local_dir is None:
        # clear cache
        shutil.rmtree(meta_data_dir)
        shutil.rmtree(videos_dir)

    return lerobot_dataset


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Directory containing input raw datasets (e.g. `data/aloha_mobile_chair_raw` or `data/pusht_raw).",
    )
    # TODO(rcadene): add automatic detection of the format
    parser.add_argument(
        "--raw-format",
        type=str,
        required=True,
        help="Dataset type (e.g. `pusht_zarr`, `umi_zarr`, `aloha_hdf5`, `xarm_pkl`, `dora_parquet`, `rlds`, `openx`).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repositery identifier on Hugging Face: a community or a user name `/` the name of the dataset (e.g. `lerobot/pusht`, `cadene/aloha_sim_insertion_human`).",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        help="When provided, writes the dataset converted to LeRobotDataset format in this directory  (e.g. `data/lerobot/aloha_mobile_chair`).",
    )
    parser.add_argument(
        "--push-to-hub",
        type=int,
        default=1,
        help="Upload to hub.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        help="Frame rate used to collect videos. If not provided, use the default one specified in the code.",
    )
    parser.add_argument(
        "--video",
        type=int,
        default=1,
        help="Convert each episode of the raw dataset to an mp4 video. This option allows 60 times lower disk space consumption and 25 faster loading time during training.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size loaded by DataLoader for computing the dataset statistics.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of processes of Dataloader for computing the dataset statistics.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        help="When provided, only converts the provided episodes (e.g `--episodes 2 3 4`). Useful to test the code on 1 episode.",
    )
    parser.add_argument(
        "--force-override",
        type=int,
        default=0,
        help="When set to 1, removes provided output directory if it already exists. By default, raises a ValueError exception.",
    )
    parser.add_argument(
        "--resume",
        type=int,
        default=0,
        help="When set to 1, resumes a previous run.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=False,
        default="/tmp",
        help="Directory to store the temporary videos and images generated while creating the dataset.",
    )
    parser.add_argument(
        "--tests-data-dir",
        type=Path,
        help=(
            "When provided, save tests artifacts into the given directory "
            "(e.g. `--tests-data-dir tests/data` will save to tests/data/{--repo-id})."
        ),
    )

    args = parser.parse_args()
    push_dataset_to_hub(**vars(args))


if __name__ == "__main__":
    main()
