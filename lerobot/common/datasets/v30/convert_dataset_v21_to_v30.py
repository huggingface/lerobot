"""
This script will help you convert any LeRobot dataset already pushed to the hub from codebase version 2.1 to
3.0. It will:

- Generate per-episodes stats and writes them in `episodes_stats.jsonl`
- Check consistency between these new stats and the old ones.
- Remove the deprecated `stats.json`.
- Update codebase_version in `info.json`.
- Push this new version to the hub on the 'main' branch and tags it with "v2.1".

Usage:

```bash
python lerobot/common/datasets/v30/convert_dataset_v21_to_v30.py \
    --repo-id=lerobot/pusht
```

"""

import argparse
import logging
from pathlib import Path
import sys

from datasets import Dataset
from huggingface_hub import snapshot_download
import tqdm

from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_PATH,
    DEFAULT_FILE_SIZE_IN_MB,
    DEFAULT_VIDEO_PATH,
    concat_video_files,
    flatten_dict,
    get_parquet_num_frames,
    get_video_duration_in_s,
    get_video_size_in_mb,
    legacy_load_episodes,
    legacy_load_episodes_stats,
    load_info,
    legacy_load_tasks,
    update_chunk_file_indices,
    write_episodes,
    write_episodes_stats,
    write_info,
    write_tasks,
)
import subprocess
import tempfile
import pandas as pd
import pyarrow.parquet as pq

V21 = "v2.1"


"""
-------------------------
OLD
data/chunk-000/episode_000000.parquet

NEW
data/chunk-000/file_000.parquet
-------------------------
OLD
videos/chunk-000/CAMERA/episode_000000.mp4

NEW
videos/chunk-000/file_000.mp4
-------------------------
OLD
episodes.jsonl
{"episode_index": 1, "tasks": ["Put the blue block in the green bowl"], "length": 266}

NEW
meta/episodes/chunk-000/episodes_000.parquet
episode_index | video_chunk_index | video_file_index | data_chunk_index | data_file_index | tasks | length
-------------------------
OLD
tasks.jsonl
{"task_index": 1, "task": "Put the blue block in the green bowl"}

NEW
meta/tasks/chunk-000/file_000.parquet
task_index | task
-------------------------
OLD
episodes_stats.jsonl

NEW
meta/episodes_stats/chunk-000/file_000.parquet
episode_index | mean | std | min | max
-------------------------
UPDATE
meta/info.json
-------------------------
"""

def get_parquet_file_size_in_mb(parquet_path):
    metadata = pq.read_metadata(parquet_path)
    uncompressed_size = metadata.num_rows * metadata.row_group(0).total_byte_size
    return uncompressed_size / (1024 ** 2)



def generate_flat_ep_stats(episodes_stats):
    for ep_idx, ep_stats in episodes_stats.items():
        flat_ep_stats = flatten_dict(ep_stats)
        flat_ep_stats["episode_index"] = ep_idx
        yield flat_ep_stats

def convert_episodes_stats(root, new_root):
    episodes_stats = legacy_load_episodes_stats(root)
    ds_episodes_stats = Dataset.from_generator(lambda: generate_flat_ep_stats(episodes_stats))
    write_episodes_stats(ds_episodes_stats, new_root)

def generate_task_dict(tasks):
    for task_index, task in tasks.items():
        yield {"task_index": task_index, "task": task}

def convert_tasks(root, new_root):
    tasks, _ = legacy_load_tasks(root)
    ds_tasks = Dataset.from_generator(lambda: generate_task_dict(tasks))
    write_tasks(ds_tasks, new_root)


def concat_data_files(paths_to_cat, new_root, chunk_idx, file_idx):
    # TODO(rcadene): to save RAM use Dataset.from_parquet(file) and concatenate_datasets
    dataframes = [pd.read_parquet(file) for file in paths_to_cat]
    # Concatenate all DataFrames along rows
    concatenated_df = pd.concat(dataframes, ignore_index=True)

    path = new_root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    concatenated_df.to_parquet(path, index=False)


def convert_data(root, new_root):
    data_dir = root / "data"

    ep_paths = [path for path in data_dir.glob("*/*.parquet")]
    ep_paths = sorted(ep_paths)

    episodes_metadata = []
    ep_idx = 0
    chunk_idx = 0
    file_idx = 0
    size_in_mb = 0
    num_frames = 0
    paths_to_cat = []
    for ep_path in ep_paths:
        ep_size_in_mb = get_parquet_file_size_in_mb(ep_path)
        ep_num_frames = get_parquet_num_frames(ep_path)
        ep_metadata = {
            "episode_index": ep_idx,
            "data/chunk_index": chunk_idx,
            "data/file_index": file_idx,
            "data/from_index": num_frames,
            "data/to_index": num_frames + ep_num_frames,
        }
        size_in_mb += ep_size_in_mb
        num_frames += ep_num_frames
        episodes_metadata.append(ep_metadata)
        ep_idx += 1

        if size_in_mb < DEFAULT_FILE_SIZE_IN_MB:
            paths_to_cat.append(ep_path)
            continue

        concat_data_files(paths_to_cat, new_root, chunk_idx, file_idx)

        # Reset for the next file
        size_in_mb = ep_size_in_mb
        num_frames = ep_num_frames
        paths_to_cat = [ep_path]

        chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, DEFAULT_CHUNK_SIZE)

    # Write remaining data if any
    if paths_to_cat:
        concat_data_files(paths_to_cat, new_root, chunk_idx, file_idx)

    return episodes_metadata



def get_video_keys(root):
    info = load_info(root)
    features = info["features"]
    image_keys = [key for key, ft in features.items() if ft["dtype"] == "image"]
    if len(image_keys) != 0:
        raise NotImplementedError()

    video_keys = [key for key, ft in features.items() if ft["dtype"] == "video"]
    return video_keys


def convert_videos(root: Path, new_root: Path):
    video_keys = get_video_keys(root)
    video_keys = sorted(video_keys)

    eps_metadata_per_cam = []
    for camera in video_keys:
        eps_metadata = convert_videos_of_camera(root, new_root, camera)
        eps_metadata_per_cam.append(eps_metadata)
    
    num_eps_per_cam = [len(eps_cam_map) for eps_cam_map in eps_metadata_per_cam]
    if len(set(num_eps_per_cam)) != 1:
        raise ValueError(f"All cams dont have same number of episodes ({num_eps_per_cam}).")
    
    episods_metadata = []
    num_cameras = len(video_keys)
    num_episodes = num_eps_per_cam[0]
    for ep_idx in range(num_episodes):
        # Sanity check
        ep_ids = [eps_metadata_per_cam[cam_idx][ep_idx]["episode_index"] for cam_idx in range(num_cameras)]
        ep_ids += [ep_idx]
        if len(set(ep_ids)) != 1:
            raise ValueError(f"All episode indices need to match ({ep_ids}).")

        ep_dict = {}
        for cam_idx in range(num_cameras):
            ep_dict.update(eps_metadata_per_cam[cam_idx][ep_idx])
        episods_metadata.append(ep_dict)
            
    return episods_metadata


def convert_videos_of_camera(root: Path, new_root: Path, video_key):
    # Access old paths to mp4
    videos_dir = root / "videos"
    ep_paths = [path for path in videos_dir.glob(f"*/{video_key}/*.mp4")]
    ep_paths = sorted(ep_paths)

    episodes_metadata = []
    ep_idx = 0
    chunk_idx = 0
    file_idx = 0
    size_in_mb = 0
    duration_in_s = 0.0
    paths_to_cat = []
    for ep_path in tqdm.tqdm(ep_paths, desc=f"convert videos of {video_key}"):
        ep_size_in_mb = get_video_size_in_mb(ep_path)
        ep_duration_in_s = get_video_duration_in_s(ep_path)
        ep_metadata = {
            "episode_index": ep_idx,
            f"{video_key}/chunk_index": chunk_idx,
            f"{video_key}/file_index": file_idx,
            f"{video_key}/from_timestamp": duration_in_s,
            f"{video_key}/to_timestamp": duration_in_s + ep_duration_in_s,
        }
        size_in_mb += ep_size_in_mb
        duration_in_s += ep_duration_in_s
        episodes_metadata.append(ep_metadata)
        ep_idx += 1

        if size_in_mb < DEFAULT_FILE_SIZE_IN_MB:
            paths_to_cat.append(ep_path)
            continue

        concat_video_files(paths_to_cat, new_root, video_key, chunk_idx, file_idx)

        # Reset for the next file
        size_in_mb = ep_size_in_mb
        duration_in_s = ep_duration_in_s
        paths_to_cat = [ep_path]

        chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, DEFAULT_CHUNK_SIZE)

    # Write remaining videos if any
    if paths_to_cat:
        concat_video_files(paths_to_cat, new_root, video_key, chunk_idx, file_idx)

    return episodes_metadata

def generate_episode_dict(episodes, episodes_data, episodes_videos):
    for ep, ep_data, ep_video in zip(episodes.values(), episodes_data, episodes_videos):
        ep_idx = ep["episode_index"]
        ep_idx_data = ep_data["episode_index"]
        ep_idx_video = ep_video["episode_index"]

        if len(set([ep_idx, ep_idx_data, ep_idx_video])) != 1:
            raise ValueError(f"Number of episodes is not the same ({ep_idx=},{ep_idx_data=},{ep_idx_video=}).")

        ep_dict = {**ep_data, **ep_video, **ep}
        yield ep_dict

def convert_episodes(root, new_root, episodes_data, episodes_videos):
    episodes = legacy_load_episodes(root)

    num_eps = len(episodes)
    num_eps_data = len(episodes_data)
    num_eps_video = len(episodes_videos)
    if len(set([num_eps, num_eps_data, num_eps_video])) != 1:
        raise ValueError(f"Number of episodes is not the same ({num_eps=},{num_eps_data=},{num_eps_video=}).")

    ds_episodes = Dataset.from_generator(lambda: generate_episode_dict(episodes, episodes_data, episodes_videos))
    write_episodes(ds_episodes, new_root)

def convert_info(root, new_root):
    info = load_info(root)
    info["codebase_version"] = "v3.0"
    del info["total_chunks"]
    del info["total_videos"]
    info["files_size_in_mb"] = DEFAULT_FILE_SIZE_IN_MB
    # TODO(rcadene): chunk- or chunk_ or file- or file_
    info["data_path"] = DEFAULT_DATA_PATH
    info["video_path"] = DEFAULT_VIDEO_PATH
    info["fps"] = float(info["fps"])
    for key in info["features"]:
        if info["features"][key]["dtype"] == "video":
            # already has fps in video_info
            continue
        info["features"][key]["fps"] = info["fps"]
    write_info(info, new_root)

def convert_dataset(
    repo_id: str,
    branch: str | None = None,
    num_workers: int = 4,
):
    root = HF_LEROBOT_HOME / repo_id
    new_root = HF_LEROBOT_HOME / f"{repo_id}_v30"

    snapshot_download(
        repo_id,
        repo_type="dataset",
        revision=V21,
        local_dir=root,
    )

    convert_info(root, new_root)
    convert_episodes_stats(root, new_root)
    convert_tasks(root, new_root)    
    episodes_data_mapping = convert_data(root, new_root)
    episodes_videos_mapping = convert_videos(root, new_root)
    convert_episodes(root, new_root, episodes_data_mapping, episodes_videos_mapping)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository identifier on Hugging Face: a community or a user name `/` the name of the dataset "
        "(e.g. `lerobot/pusht`, `cadene/aloha_sim_insertion_human`).",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Repo branch to push your dataset. Defaults to the main branch.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for parallelizing stats compute. Defaults to 4.",
    )

    args = parser.parse_args()
    convert_dataset(**vars(args))
