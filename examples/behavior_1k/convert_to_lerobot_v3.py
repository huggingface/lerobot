#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Convert Behavior Dataset to LeRobotDataset v3.0 format"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import jsonlines
import pandas as pd
import pyarrow as pa
import tqdm
from datasets import Dataset, Features, Image

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_DATA_PATH,
    DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    DEFAULT_VIDEO_PATH,
    LEGACY_EPISODES_PATH,
    LEGACY_EPISODES_STATS_PATH,
    LEGACY_TASKS_PATH,
    cast_stats_to_numpy,
    flatten_dict,
    get_file_size_in_mb,
    get_parquet_file_size_in_mb,
    get_parquet_num_frames,
    load_info,
    update_chunk_file_indices,
    write_episodes,
    write_info,
    write_stats,
    write_tasks,
)
from lerobot.datasets.video_utils import concatenate_video_files, get_video_duration_in_s
from lerobot.utils.utils import init_logging

# script to convert one single task to v3.1
# TASK = 1
NEW_ROOT = Path("/fsx/jade_choghari/tmp/bb")


def get_total_episodes_task(local_dir: Path, task_id: int, task_ranges: dict, step) -> int:
    """
    Calculates the total number of episodes for a single, specified task.
    """
    # Simply load the episodes for the task and count them.
    episodes = legacy_load_episodes_task(
        local_dir=local_dir, task_id=task_id, task_ranges=task_ranges, step=step
    )
    return len(episodes)


NUM_CAMERAS = 9


def get_total_frames_task(local_dir, meta_path, task_id: int, task_ranges: dict, step: int) -> int:
    episodes_metadata = legacy_load_episodes_task(
        local_dir=local_dir, task_id=task_id, task_ranges=task_ranges, step=step
    )
    total_frames = 0
    # like 'duration'
    for ep in episodes_metadata.values():
        duration_s = ep["length"]
        total_frames += int(duration_s)
    return total_frames


def convert_info(
    root, new_root, data_file_size_in_mb, video_file_size_in_mb, meta_path, task_id: int, task_ranges, step
):
    info = load_info(root)
    info["codebase_version"] = "v3.0"
    del info["total_videos"]
    info["data_files_size_in_mb"] = data_file_size_in_mb
    info["video_files_size_in_mb"] = video_file_size_in_mb
    info["data_path"] = DEFAULT_DATA_PATH
    info["video_path"] = DEFAULT_VIDEO_PATH if info["video_path"] is not None else None
    info["fps"] = int(info["fps"])
    for key in info["features"]:
        if info["features"][key]["dtype"] == "video":
            # already has fps in video_info
            continue
        info["features"][key]["fps"] = info["fps"]

    info["total_episodes"] = get_total_episodes_task(root, task_id, task_ranges, step)
    info["total_videos"] = info["total_episodes"] * NUM_CAMERAS
    info["total_frames"] = get_total_frames_task(root, meta_path, task_id, task_ranges, step)
    info["total_tasks"] = 1
    write_info(info, new_root)


def load_jsonlines(fpath: Path) -> list[any]:
    with jsonlines.open(fpath, "r") as reader:
        return list(reader)


def legacy_load_tasks(local_dir: Path) -> tuple[dict, dict]:
    tasks = load_jsonlines(local_dir / LEGACY_TASKS_PATH)
    # return tasks dict such that
    tasks = {item["task_index"]: item["task"] for item in sorted(tasks, key=lambda x: x["task_index"])}
    task_to_task_index = {task: task_index for task_index, task in tasks.items()}
    return tasks, task_to_task_index


def convert_tasks(root, new_root, task_id: int):
    tasks, _ = legacy_load_tasks(root)
    if task_id not in tasks:
        raise ValueError(f"Task ID {task_id} not found in tasks (available: {list(tasks.keys())})")
    tasks = {task_id: tasks[task_id]}
    task_indices = tasks.keys()
    task_strings = tasks.values()
    df_tasks = pd.DataFrame({"task_index": task_indices}, index=task_strings)
    write_tasks(df_tasks, new_root)


def concat_data_files(paths_to_cat, new_root, chunk_idx, file_idx, image_keys):
    # TODO(rcadene): to save RAM use Dataset.from_parquet(file) and concatenate_datasets
    dataframes = [pd.read_parquet(file) for file in paths_to_cat]
    # Concatenate all DataFrames along rows
    concatenated_df = pd.concat(dataframes, ignore_index=True)

    path = new_root / DEFAULT_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(image_keys) > 0:
        schema = pa.Schema.from_pandas(concatenated_df)
        features = Features.from_arrow_schema(schema)
        for key in image_keys:
            features[key] = Image()
        schema = features.arrow_schema
    else:
        schema = None

    concatenated_df.to_parquet(path, index=False, schema=schema)


def get_image_keys(root):
    info = load_info(root)
    features = info["features"]
    image_keys = [key for key, ft in features.items() if ft["dtype"] == "image"]
    return image_keys


def convert_data(root: Path, new_root: Path, data_file_size_in_mb: int, task_index: int):
    task_dir_name = f"task-00{task_index}"
    data_dir = root / "data" / task_dir_name
    ep_paths = sorted(data_dir.glob("*.parquet"))
    image_keys = get_image_keys(root)

    ep_idx = 0
    chunk_idx = 0
    file_idx = 0
    size_in_mb = 0
    num_frames = 0
    paths_to_cat = []
    episodes_metadata = []

    logging.info(f"Converting data files from {len(ep_paths)} episodes")

    for ep_path in tqdm.tqdm(ep_paths, desc="convert data files"):
        ep_size_in_mb = get_parquet_file_size_in_mb(ep_path)
        ep_num_frames = get_parquet_num_frames(ep_path)
        ep_metadata = {
            "episode_index": ep_idx,
            "data/chunk_index": chunk_idx,
            "data/file_index": file_idx,
            "dataset_from_index": num_frames,
            "dataset_to_index": num_frames + ep_num_frames,
        }
        size_in_mb += ep_size_in_mb
        num_frames += ep_num_frames
        episodes_metadata.append(ep_metadata)
        ep_idx += 1

        if size_in_mb < data_file_size_in_mb:
            paths_to_cat.append(ep_path)
            continue

        if paths_to_cat:
            concat_data_files(paths_to_cat, new_root, chunk_idx, file_idx, image_keys)

        # Reset for the next file
        size_in_mb = ep_size_in_mb
        paths_to_cat = [ep_path]

        chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, DEFAULT_CHUNK_SIZE)

    # Write remaining data if any
    if paths_to_cat:
        concat_data_files(paths_to_cat, new_root, chunk_idx, file_idx, image_keys)

    return episodes_metadata


def convert_videos_of_camera(
    root: Path, new_root: Path, video_key: str, video_file_size_in_mb: int, task_index: int
):
    # Access old paths to mp4
    # videos_dir = root / "videos"
    # ep_paths = sorted(videos_dir.glob(f"*/{video_key}/*.mp4"))
    task_dir_name = f"task-00{task_index}"
    videos_dir = root / "videos" / task_dir_name / video_key
    ep_paths = sorted(videos_dir.glob("*.mp4"))
    print("ep_paths", ep_paths)
    ep_idx = 0
    chunk_idx = 0
    file_idx = 0
    size_in_mb = 0
    duration_in_s = 0.0
    paths_to_cat = []
    episodes_metadata = []

    for ep_path in tqdm.tqdm(ep_paths, desc=f"convert videos of {video_key}"):
        ep_size_in_mb = get_file_size_in_mb(ep_path)
        ep_duration_in_s = get_video_duration_in_s(ep_path)

        # Check if adding this episode would exceed the limit
        if size_in_mb + ep_size_in_mb >= video_file_size_in_mb and len(paths_to_cat) > 0:
            # Size limit would be exceeded, save current accumulation WITHOUT this episode
            concatenate_video_files(
                paths_to_cat,
                new_root
                / DEFAULT_VIDEO_PATH.format(video_key=video_key, chunk_index=chunk_idx, file_index=file_idx),
            )

            # Update episodes metadata for the file we just saved
            for i, _ in enumerate(paths_to_cat):
                past_ep_idx = ep_idx - len(paths_to_cat) + i
                episodes_metadata[past_ep_idx][f"videos/{video_key}/chunk_index"] = chunk_idx
                episodes_metadata[past_ep_idx][f"videos/{video_key}/file_index"] = file_idx

            # Move to next file and start fresh with current episode
            chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, DEFAULT_CHUNK_SIZE)
            size_in_mb = 0
            duration_in_s = 0.0
            paths_to_cat = []

        # Add current episode metadata
        ep_metadata = {
            "episode_index": ep_idx,
            f"videos/{video_key}/chunk_index": chunk_idx,  # Will be updated when file is saved
            f"videos/{video_key}/file_index": file_idx,  # Will be updated when file is saved
            f"videos/{video_key}/from_timestamp": duration_in_s,
            f"videos/{video_key}/to_timestamp": duration_in_s + ep_duration_in_s,
        }
        episodes_metadata.append(ep_metadata)

        # Add current episode to accumulation
        paths_to_cat.append(ep_path)
        size_in_mb += ep_size_in_mb
        duration_in_s += ep_duration_in_s
        ep_idx += 1

    # Write remaining videos if any
    if paths_to_cat:
        concatenate_video_files(
            paths_to_cat,
            new_root
            / DEFAULT_VIDEO_PATH.format(video_key=video_key, chunk_index=chunk_idx, file_index=file_idx),
        )

        # Update episodes metadata for the final file
        for i, _ in enumerate(paths_to_cat):
            past_ep_idx = ep_idx - len(paths_to_cat) + i
            episodes_metadata[past_ep_idx][f"videos/{video_key}/chunk_index"] = chunk_idx
            episodes_metadata[past_ep_idx][f"videos/{video_key}/file_index"] = file_idx

    return episodes_metadata


def get_video_keys(root):
    info = load_info(root)
    features = info["features"]
    video_keys = [key for key, ft in features.items() if ft["dtype"] == "video"]
    return video_keys


def convert_videos(root: Path, new_root: Path, video_file_size_in_mb: int, task_id: int):
    logging.info(f"Converting videos from {root} to {new_root}")

    video_keys = get_video_keys(root)
    if len(video_keys) == 0:
        return None

    video_keys = sorted(video_keys)

    eps_metadata_per_cam = []
    for camera in video_keys:
        eps_metadata = convert_videos_of_camera(root, new_root, camera, video_file_size_in_mb, task_id)
        eps_metadata_per_cam.append(eps_metadata)

    num_eps_per_cam = [len(eps_cam_map) for eps_cam_map in eps_metadata_per_cam]
    if len(set(num_eps_per_cam)) != 1:
        raise ValueError(f"All cams dont have same number of episodes ({num_eps_per_cam}).")

    episods_metadata = []
    num_cameras = len(video_keys)
    num_episodes = num_eps_per_cam[0]
    for ep_idx in tqdm.tqdm(range(num_episodes), desc="convert videos"):
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


def infer_task_episode_ranges(episodes_jsonl_path: Path) -> dict:
    """
    Parse the Behavior-1K episodes.jsonl metadata and infer contiguous episode ranges per unique task.
    Returns a dict:
      { task_id: { "task_string": ..., "ep_start": ..., "ep_end": ... } }
    """
    task_ranges = {}
    task_id = 0
    current_task_str = None
    ep_start = None
    ep_end = None

    with open(episodes_jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            ep = json.loads(line)
            ep_idx = ep["episode_index"]
            task_str = ep["tasks"][0] if ep["tasks"] else "UNKNOWN"

            if current_task_str is None:
                current_task_str = task_str
                ep_start = ep_idx
                ep_end = ep_idx
            elif task_str == current_task_str:
                ep_end = ep_idx
            else:
                # close previous task group
                task_ranges[task_id] = {
                    "task_string": current_task_str,
                    "ep_start": ep_start,
                    "ep_end": ep_end,
                }
                task_id += 1
                # start new one
                current_task_str = task_str
                ep_start = ep_idx
                ep_end = ep_idx

    # store last task
    if current_task_str is not None:
        task_ranges[task_id] = {
            "task_string": current_task_str,
            "ep_start": ep_start,
            "ep_end": ep_end,
        }

    return task_ranges


def legacy_load_episodes_task(local_dir: Path, task_id: int, task_ranges: dict, step: int = 10) -> dict:
    """
    Load only the episodes belonging to a specific task, inferred automatically from episode ranges.

    Args:
        local_dir (Path): Root path containing legacy meta/episodes.jsonl
        task_id (int): Which task to load (key from the inferred task_ranges dict)
        task_ranges (dict): Mapping from infer_task_episode_ranges()
        step (int): Episode index step (Behavior-1K = 10)
    """
    all_episodes = legacy_load_episodes(local_dir)

    # get the range for this task
    if task_id not in task_ranges:
        raise ValueError(f"Task id {task_id} not found in task_ranges")

    ep_start = task_ranges[task_id]["ep_start"]
    ep_end = task_ranges[task_id]["ep_end"]

    task_episode_indices = range(ep_start, ep_end + step, step)
    return {i: all_episodes[i] for i in task_episode_indices if i in all_episodes}


def legacy_load_episodes(local_dir: Path) -> dict:
    episodes = load_jsonlines(local_dir / LEGACY_EPISODES_PATH)
    return {item["episode_index"]: item for item in sorted(episodes, key=lambda x: x["episode_index"])}


def legacy_load_episodes_stats(local_dir: Path) -> dict:
    episodes_stats = load_jsonlines(local_dir / LEGACY_EPISODES_STATS_PATH)
    return {
        item["episode_index"]: cast_stats_to_numpy(item["stats"])
        for item in sorted(episodes_stats, key=lambda x: x["episode_index"])
    }


def legacy_load_episodes_stats_task(local_dir: Path, task_id: int, task_ranges: dict, step: int = 10) -> dict:
    all_stats = legacy_load_episodes_stats(local_dir)

    if task_id not in task_ranges:
        raise ValueError(f"Task id {task_id} not found in task_ranges")

    ep_start = task_ranges[task_id]["ep_start"]
    ep_end = task_ranges[task_id]["ep_end"]

    task_episode_indices = range(ep_start, ep_end + step, step)
    return {i: all_stats[i] for i in task_episode_indices if i in all_stats}


def generate_episode_metadata_dict(
    episodes_legacy_metadata, episodes_metadata, episodes_stats, episodes_videos=None
):
    num_episodes = len(episodes_metadata)
    episodes_legacy_metadata_vals = list(episodes_legacy_metadata.values())
    episodes_stats_vals = list(episodes_stats.values())
    episodes_stats_keys = list(episodes_stats.keys())

    for i in range(num_episodes):
        ep_legacy_metadata = episodes_legacy_metadata_vals[i]
        ep_metadata = episodes_metadata[i]
        ep_stats = episodes_stats_vals[i]

        ep_ids_set = {
            ep_legacy_metadata["episode_index"],
            ep_metadata["episode_index"],
            episodes_stats_keys[i],
        }

        if episodes_videos is None:
            ep_video = {}
        else:
            ep_video = episodes_videos[i]
            ep_ids_set.add(ep_video["episode_index"])
        # we skip this check because ep_ids have a step of 10, whereas we convert with a step of 1
        # if len(ep_ids_set) != 1:
        #     raise ValueError(f"Number of episodes is not the same ({ep_ids_set}).")

        ep_dict = {**ep_metadata, **ep_video, **ep_legacy_metadata, **flatten_dict({"stats": ep_stats})}
        ep_dict["meta/episodes/chunk_index"] = 0
        ep_dict["meta/episodes/file_index"] = 0
        yield ep_dict


def convert_episodes_metadata(
    root, new_root, episodes_metadata, task_id: int, task_ranges, episodes_video_metadata=None
):
    logging.info(f"Converting episodes metadata from {root} to {new_root}")

    # filter by task
    episodes_legacy_metadata = legacy_load_episodes_task(root, task_id=task_id, task_ranges=task_ranges)
    episodes_stats = legacy_load_episodes_stats_task(root, task_id=task_id, task_ranges=task_ranges)

    num_eps_set = {len(episodes_legacy_metadata), len(episodes_metadata)}
    if episodes_video_metadata is not None:
        num_eps_set.add(len(episodes_video_metadata))

    if len(num_eps_set) != 1:
        raise ValueError(f"Number of episodes is not the same ({num_eps_set}).")

    ds_episodes = Dataset.from_generator(
        lambda: generate_episode_metadata_dict(
            episodes_legacy_metadata, episodes_metadata, episodes_stats, episodes_video_metadata
        )
    )
    write_episodes(ds_episodes, new_root)

    stats = aggregate_stats(list(episodes_stats.values()))
    write_stats(stats, new_root)


def convert_dataset_local(
    data_path: Path,
    new_repo: Path,
    task_id: int,
    data_file_size_in_mb: int = DEFAULT_DATA_FILE_SIZE_IN_MB,
    video_file_size_in_mb: int = DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    force_conversion: bool = False,
):
    """
    Convert a local dataset to v3.x format, task-by-task, without using the Hugging Face Hub.

    Args:
        data_path (Path): path to local dataset root (e.g. /fsx/.../2025-challenge-demos)
        new_repo (Path): path where converted dataset will be written (e.g. /fsx/.../behavior1k_v3)
        task_id (int): which task to convert (index)
        data_file_size_in_mb (int): max size per data chunk
        video_file_size_in_mb (int): max size per video chunk
        force_conversion (bool): overwrite existing conversion if True
    """

    root = Path(data_path)
    new_root = Path(new_repo)

    # Clean up if needed
    if new_root.exists() and force_conversion:
        shutil.rmtree(new_root)
    new_root.mkdir(parents=True, exist_ok=True)

    print(f"🔹 Starting conversion for task {task_id}")
    print(f"Input root: {root}")
    print(f"Output root: {new_root}")
    # Infer task episode ranges
    episodes_meta_path = root / "meta" / "episodes.jsonl"
    task_ranges = infer_task_episode_ranges(episodes_meta_path)
    convert_info(
        root,
        new_root,
        data_file_size_in_mb,
        video_file_size_in_mb,
        episodes_meta_path,
        task_id,
        task_ranges,
        step=10,
    )
    convert_tasks(root, new_root, task_id)
    episodes_metadata = convert_data(root, new_root, data_file_size_in_mb, task_index=task_id)
    episodes_videos_metadata = convert_videos(root, new_root, video_file_size_in_mb, task_id=task_id)
    convert_episodes_metadata(
        root,
        new_root,
        episodes_metadata,
        task_id=task_id,
        task_ranges=task_ranges,
        episodes_video_metadata=episodes_videos_metadata,
    )

    print(f"✅ Conversion complete for task {task_id}")
    print(f"Converted dataset written to: {new_root}")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    init_logging()

    parser = argparse.ArgumentParser(
        description="Convert Behavior-1K tasks to LeRobot v3 format (local only)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the local Behavior-1K dataset (e.g. /fsx/francesco_capuano/.cache/behavior-1k/2025-challenge-demos)",
    )
    parser.add_argument(
        "--new-repo",
        type=str,
        required=True,
        help="Path to the output directory for the converted dataset",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        required=True,
        help="Task index to convert (e.g. 0, 1, 2, ...)",
    )
    parser.add_argument(
        "--data-file-size-in-mb",
        type=int,
        default=DEFAULT_DATA_FILE_SIZE_IN_MB,
        help=f"Maximum size per data chunk (default: {DEFAULT_DATA_FILE_SIZE_IN_MB})",
    )
    parser.add_argument(
        "--video-file-size-in-mb",
        type=int,
        default=DEFAULT_VIDEO_FILE_SIZE_IN_MB,
        help=f"Maximum size per video chunk (default: {DEFAULT_VIDEO_FILE_SIZE_IN_MB})",
    )
    parser.add_argument(
        "--force-conversion",
        action="store_true",
        help="Force overwrite of existing conversion output if present.",
    )

    args = parser.parse_args()

    convert_dataset_local(
        data_path=Path(args.data_path),
        new_repo=Path(args.new_repo),
        task_id=args.task_id,
        data_file_size_in_mb=args.data_file_size_in_mb,
        video_file_size_in_mb=args.video_file_size_in_mb,
        force_conversion=args.force_conversion,
    )
