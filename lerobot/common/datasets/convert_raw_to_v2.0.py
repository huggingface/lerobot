#!/usr/bin/env python
# convert_lerobot_dataset.py
import argparse
import contextlib
import filecmp
import logging
import math
import subprocess
import tempfile
import jsonlines
import numpy as np
from typing import Union, Dict, List
import decord
from decord import VideoReader, cpu
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import json
import shutil
from pathlib import Path
import datasets
from datasets import Dataset
from datasets import Dataset, Features, Value, Image
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import pyarrow.parquet as pq
import pyarrow.compute as pc
import re
import pyarrow as pa
import pyarrow.ipc as ipc
from tqdm import tqdm
import torch
from safetensors.torch import load_file
import os
import shutil

from lerobot.common.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_PARQUET_PATH,
    DEFAULT_VIDEO_PATH,
    EPISODES_PATH,
    INFO_PATH,
    STATS_PATH,
    TASKS_PATH,
    create_branch,
    create_lerobot_dataset_card,
    flatten_dict,
    get_safe_version,
    load_json,
    unflatten_dict,
    write_json,
    write_jsonlines,
)

from lerobot.common.datasets.video_utils import (
    VideoFrame,  # noqa: F401
    get_image_pixel_channels,
    get_video_info,
)
from lerobot.common.robot_devices.robots.configs import RobotConfig
from lerobot.common.robot_devices.robots.utils import make_robot_config


episode_lengths = []
episode_num_list = []
robot_type = "unknown"

V16 = "v1.6"
V20 = "v2.0"
task_map_path = Path(__file__).resolve().parent / "task_map.json"
EPISODES_STATS_PATH = "meta/episodes_stats.jsonl"
GITATTRIBUTES_REF = "aliberts/gitattributes_reference"
V1_VIDEO_FILE = "{video_key}_episode_{episode_index:06d}.mp4"
V1_INFO_PATH = "meta_data/info.json"
V1_STATS_PATH = "meta_data/stats.safetensors"
DEFAULT_CHUNK_SIZE = 1000  # 每个chunk包含的episode数量
DEFAULT_PARQUET_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
DEFAULT_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

def merge_parquets():
    x = 1


def load_task_map() -> list[dict]:
    with open(task_map_path, "r") as f:
        return json.load(f)
    
def get_task_index(ep_idx: int, task_map: list[dict]) -> int:
    for entry in task_map:
        start, end = entry["episode_range"]
        if start <= ep_idx <= end:
            return entry["task_index"]
    raise ValueError(f"Episode {ep_idx} not found in task map.")

def split_parquet_by_episodes(
    dataset: Dataset,
    total_episodes: int,
    total_chunks: int,
    output_dir: Path,
) -> list:
    table = dataset.data.table
    episode_lengths_temp = []

    task_map = load_task_map()

    for ep_chunk in range(total_chunks):
        ep_chunk_start = DEFAULT_CHUNK_SIZE * ep_chunk
        ep_chunk_end = min(DEFAULT_CHUNK_SIZE * (ep_chunk + 1), total_episodes)
        chunk_dir = "/".join(DEFAULT_PARQUET_PATH.split("/")[:-1]).format(episode_chunk=ep_chunk)
        (output_dir / chunk_dir).mkdir(parents=True, exist_ok=True)

        for ep_idx in range(ep_chunk_start, ep_chunk_end):
            ep_table = table.filter(pc.equal(table["episode_index"], ep_idx))

            columns_to_drop = [
                "observation.images.cam_high",
                "observation.images.cam_low",
                "observation.images.cam_right_wrist",
                "observation.images.cam_left_wrist"
            ]
            existing_cols = [col for col in columns_to_drop if col in ep_table.column_names]
            ep_table = ep_table.drop(existing_cols)

            task_index_val = get_task_index(ep_idx, task_map)
            task_index = pa.array([task_index_val] * len(ep_table))
            ep_table = ep_table.append_column("task_index", task_index)

            episode_lengths_temp.insert(ep_idx, len(ep_table))

            output_file = output_dir / DEFAULT_PARQUET_PATH.format(
                episode_chunk=ep_chunk, episode_index=ep_idx
            )
            pq.write_table(ep_table, output_file)

    return episode_lengths_temp

def merge_and_convert_arrow_to_v2(input_dirs:list[Path], input_arrow: list[Path], output_dir: Path):
    total_episode_offset = 0
    total_index_offset = 0
    merged_tables = []

    for input_dir in input_dirs:
        arrow_file = next(input_dir.glob("train/data-*.arrow"))
        print(f"Loading Arrow file: {arrow_file}")
        if not arrow_file.exists():
            print(f"Warning: {arrow_file} not found.")
            continue

        with open(arrow_file, "rb") as f:
            reader = ipc.open_stream(f)
            table = reader.read_all()

        if "episode_index" in table.schema.names:
            idx_array = table["index"]
            new_index_idx = pa.array([i + total_index_offset for i in idx_array.to_pylist()])
            table = table.set_column(table.schema.get_field_index("index"),
                                     "index",
                                     new_index_idx)
            total_index_offset += len(idx_array)

            episode_num_list.append(len(idx_array))
            episode_idx_array = table["episode_index"]
            new_episode_idx = pa.array([i + total_episode_offset for i in episode_idx_array.to_pylist()])
            table = table.set_column(table.schema.get_field_index("episode_index"),
                                     "episode_index",
                                     new_episode_idx)
            episode_idx_array_np = episode_idx_array.to_numpy()
            max_episode_index = episode_idx_array_np.max() + 1                       
            total_episode_offset += max_episode_index
        else:
            print(f"Warning: 'episode_index' not in table schema from {arrow_file}, skipping.")
            continue

        merged_tables.append(table)

    if merged_tables:
        final_table = pa.concat_tables(merged_tables)
        output_parquet_dir = output_dir
        output_parquet_dir.mkdir(parents=True, exist_ok=True)
        output_parquet = output_parquet_dir / "train-00000-of-00001.parquet"
        pq.write_table(final_table, output_parquet)
        print(f"Saved merged parquet to: {output_parquet}")
    else:
        print("No valid tables loaded. Nothing to write.")

    dataset = datasets.load_dataset("parquet", data_dir=output_parquet_dir, split="train")

    episode_indices = sorted(dataset.unique("episode_index"))
    total_episodes = len(episode_indices)

    assert episode_indices == list(range(total_episodes))
    total_chunks = total_episodes // DEFAULT_CHUNK_SIZE
    if total_episodes % DEFAULT_CHUNK_SIZE != 0:
        total_chunks += 1
    print(f"Found {total_episodes} episodes")

    (output_dir / "meta").mkdir(parents=True, exist_ok=True)
    episode_lengths = split_parquet_by_episodes(dataset, total_episodes, total_chunks, output_dir)
    return episode_lengths, dataset

def convert_arrow_to_v2(input_dir:Path, input_arrow: Path, output_dir: Path):
    print(f"Loading Arrow file: {input_arrow}")
    output_parquet_dir = input_dir.with_name(f"{input_dir.name}_parquet")
    output_parquet = output_parquet_dir / "train-00000-of-00001.parquet"
    output_parquet_dir.mkdir(parents=True, exist_ok=True)
    with open(input_arrow, "rb") as f:
        reader = ipc.open_stream(f)
        table = reader.read_all()
    pq.write_table(table, output_parquet)
    
    dataset = datasets.load_dataset("parquet", data_dir=output_parquet_dir, split="train")


    episode_indices = sorted(dataset.unique("episode_index"))
    total_episodes = len(episode_indices)

    assert episode_indices == list(range(total_episodes))
    total_chunks = total_episodes // DEFAULT_CHUNK_SIZE
    if total_episodes % DEFAULT_CHUNK_SIZE != 0:
        total_chunks += 1
    print(f"Found {total_episodes} episodes")

    (output_dir / "meta").mkdir(parents=True, exist_ok=True)
    episode_lengths = split_parquet_by_episodes(dataset, total_episodes, total_chunks, output_dir)
    return episode_lengths, dataset

def merge_reorganize_videos(input_dir: list[Path], output_dir: Path):
    print("Reorganizing video files...")
    ep_idx_offset = 0
    for temp_dir in input_dir:
        videos_dir = temp_dir / "videos"
        video_files = list(videos_dir.glob("*.mp4"))
        test = 0
        max_ep_idx = 0
        
        for video_path in video_files:
            # if (test > 1):
            #     break
            test += 1
            match = re.match(r"(.+?)_episode_(\d+)\.mp4", video_path.name)
            if not match:
                print(f"Skipping invalid video filename: {video_path.name}")
                continue
                
            video_key = match.group(1)
            ep_idx = int(match.group(2))
            if ep_idx > max_ep_idx:
                max_ep_idx = ep_idx
            chunk_idx = (ep_idx + ep_idx_offset) // DEFAULT_CHUNK_SIZE
            
            target_dir = output_dir / f"videos/chunk-{chunk_idx:03d}/{video_key}"
            target_dir.mkdir(parents=True, exist_ok=True)
            
            target_path = target_dir / f"episode_{(ep_idx + ep_idx_offset):06d}.mp4"
            shutil.copy2(video_path, target_path)
            print(f"Reorganized: {video_path.name} -> {target_path.relative_to(output_dir)}")
        ep_idx_offset += max_ep_idx + 1

def reorganize_videos(videos_dir: Path, output_dir: Path):
    print("Reorganizing video files...")
    video_files = list(videos_dir.glob("*.mp4"))
    test = 0
    for video_path in video_files:
        # if (test > 1):
        #     break
        test += 1
        match = re.match(r"(.+?)_episode_(\d+)\.mp4", video_path.name)
        if not match:
            print(f"Skipping invalid video filename: {video_path.name}")
            continue
            
        video_key = match.group(1)
        ep_idx = int(match.group(2))
        chunk_idx = ep_idx // DEFAULT_CHUNK_SIZE
        
        target_dir = output_dir / f"videos/chunk-{chunk_idx:03d}/{video_key}"
        target_dir.mkdir(parents=True, exist_ok=True)
        
        target_path = target_dir / f"episode_{ep_idx:06d}.mp4"
        shutil.copy2(video_path, target_path)
        print(f"Reorganized: {video_path.name} -> {target_path.relative_to(output_dir)}")


def merge_convert_stats_to_json(v1_dir_list: list[Path], v2_dir: Path) -> None:
    v2_dir_temp = v2_dir / "meta_test"
    v2_dir_final = v2_dir / "meta" / "stats.json"
    v2_dir_final.parent.mkdir(exist_ok=True, parents=True)
    for idx, v1_dir in enumerate(v1_dir_list):
        safetensor_path = v1_dir / V1_STATS_PATH
        print(f"Loading: {safetensor_path}")
        
        stats = load_file(safetensor_path)
        serialized_stats = {key: value.tolist() for key, value in stats.items()}
        serialized_stats = unflatten_dict(serialized_stats)

        suffix = v1_dir.name 
        json_path = v2_dir_temp / f"stats_{suffix}_{idx}.json"
        json_path.parent.mkdir(exist_ok=True, parents=True)

        with open(json_path, "w") as f:
            json.dump(serialized_stats, f, indent=4)
        
        with open(json_path) as f:
            stats_json = json.load(f)
        
        stats_json = flatten_dict(stats_json)
        stats_json = {key: torch.tensor(value) for key, value in stats_json.items()}
        for key in stats:
            torch.testing.assert_close(stats_json[key], stats[key])
    data_list = []
    for json_file in v2_dir_temp.glob("*.json"):
        with open(json_file, "r") as f:
            data = json.load(f)
        data_list.append(data)
    if (len(data_list) != len(episode_num_list)):
        print("wrong size")
        return
    out_data = {}
    for key, stat_dict in data_list[0].items():
        inner_data = {}
        for key_in, stat_dict_in in stat_dict.items():
            inner_data[key_in] = []
        out_data[key] = inner_data

    total_cnt = 0
    for i in range(len(data_list)):
        total_cnt += episode_num_list[i]
        for key, stat_dict in data_list[i].items():
            if "cam" in key:

                for j in range(len(data_list[i][key]["max"])):
                    if len(out_data[key]["max"]) == 0:
                        out_data[key]["max"] = data_list[i][key]["max"]
                        break
                    if data_list[i][key]["max"][j][0][0] > out_data[key]["max"][j][0][0]:
                        out_data[key]["max"][j][0][0] = data_list[i][key]["max"][j][0][0]

                for j in range(len(data_list[i][key]["min"])):
                    if len(out_data[key]["min"]) == 0:
                        out_data[key]["min"] = data_list[i][key]["min"]
                        break
                    if data_list[i][key]["min"][j][0][0] < out_data[key]["min"][j][0][0]:
                        out_data[key]["min"][j][0][0] = data_list[i][key]["min"][j][0][0]

                delta_list = []
                for j in range(len(data_list[i][key]["mean"])):
                    if len(out_data[key]["mean"]) == 0:
                        out_data[key]["mean"] = data_list[i][key]["mean"]
                        delta_list = [0.0] * len(data_list[i][key]["mean"])
                        break

                    delta = data_list[i][key]["mean"][j][0][0] - out_data[key]["mean"][j][0][0]
                    
                    mean = (
                        episode_num_list[i - 1] * out_data[key]["mean"][j][0][0] +
                        episode_num_list[i] * data_list[i][key]["mean"][j][0][0]
                    ) / total_cnt
                    out_data[key]["mean"][j][0][0] = mean
                    delta_list.append(delta)

                for j in range(len(data_list[i][key]["std"])):
                    if len(out_data[key]["std"]) == 0:
                        out_data[key]["std"] = data_list[i][key]["std"]
                        break
                    M1 = episode_num_list[i - 1] * out_data[key]["std"][j][0][0]**2
                    M2 = episode_num_list[i] * data_list[i][key]["std"][j][0][0]**2
                    M = M1 + M2 + (episode_num_list[i - 1] * episode_num_list[i] / total_cnt) * (delta_list[j]**2)
                    std = (M / total_cnt)**0.5
                    out_data[key]["std"][j][0][0] = std
            else:
                for j in range(len(data_list[i][key]["max"])):
                    if len(out_data[key]["max"]) == 0:
                        out_data[key]["max"] = data_list[i][key]["max"]
                        break
                    if data_list[i][key]["max"][j] > out_data[key]["max"][j]:
                        out_data[key]["max"][j] = data_list[i][key]["max"][j]

                for j in range(len(data_list[i][key]["min"])):
                    if len(out_data[key]["min"]) == 0:
                        out_data[key]["min"] = data_list[i][key]["min"]
                        break
                    if data_list[i][key]["min"][j] < out_data[key]["min"][j]:
                        out_data[key]["min"][j] = data_list[i][key]["min"][j]

                delta_list = []
                for j in range(len(data_list[i][key]["mean"])):
                    if len(out_data[key]["mean"]) == 0:
                        out_data[key]["mean"] = data_list[i][key]["mean"]
                        delta_list = [0.0] * len(data_list[i][key]["mean"])
                        break
                    delta = data_list[i][key]["mean"][j] - out_data[key]["mean"][j]
                    
                    mean = (
                        episode_num_list[i - 1] * out_data[key]["mean"][j] +
                        episode_num_list[i] * data_list[i][key]["mean"][j]
                    ) / total_cnt
                    out_data[key]["mean"][j] = mean
                    delta_list.append(delta)

                for j in range(len(data_list[i][key]["std"])):
                    if len(out_data[key]["std"]) == 0:
                        out_data[key]["std"] = data_list[i][key]["std"]
                        break

                    M1 = episode_num_list[i - 1] * out_data[key]["std"][j]**2
                    M2 = episode_num_list[i] * data_list[i][key]["std"][j]**2
                    M = M1 + M2 + (episode_num_list[i - 1] * episode_num_list[i] / total_cnt) * (delta_list[j]**2)
                    std = (M / total_cnt)**0.5
                    out_data[key]["std"][j] = std
    with open(v2_dir_final, "w") as f:
        json.dump(out_data, f, indent=4)
                    

def convert_stats_to_json(v1_dir: Path, v2_dir: Path) -> None:
    safetensor_path = v1_dir / V1_STATS_PATH
    print(safetensor_path)
    stats = load_file(safetensor_path)
    serialized_stats = {key: value.tolist() for key, value in stats.items()}
    serialized_stats = unflatten_dict(serialized_stats)

    json_path = v2_dir / STATS_PATH
    json_path.parent.mkdir(exist_ok=True, parents=True)
    with open(json_path, "w") as f:
        json.dump(serialized_stats, f, indent=4)

    with open(json_path) as f:
        stats_json = json.load(f)

    stats_json = flatten_dict(stats_json)
    stats_json = {key: torch.tensor(value) for key, value in stats_json.items()}
    for key in stats:
        torch.testing.assert_close(stats_json[key], stats[key])

def get_features_from_hf_dataset(
    dataset: Dataset, robot_config: RobotConfig | None = None
) -> dict[str, list]:
    robot_config = None
    features = {}
    for key, ft in dataset.features.items():
        if isinstance(ft, datasets.Value):
            dtype = ft.dtype
            shape = (1,)
            names = None
        if isinstance(ft, datasets.Sequence):
            assert isinstance(ft.feature, datasets.Value)
            dtype = ft.feature.dtype
            shape = (ft.length,)
            motor_names = (
                robot_config["names"][key] if robot_config else [f"motor_{i}" for i in range(ft.length)]
            )
            assert len(motor_names) == shape[0]
            names = {"motors": motor_names}
        elif isinstance(ft, datasets.Image):
            dtype = "image"
            image = dataset[0][key]  
            channels = get_image_pixel_channels(image)
            shape = (image.height, image.width, channels)
            names = ["height", "width", "channels"]
        elif ft._type == "VideoFrame":
            dtype = "video"
            shape = None 
            names = ["height", "width", "channels"]

        features[key] = {
            "dtype": dtype,
            "shape": shape,
            "names": names,
        }
    return features

def add_task_index_by_episodes(dataset: Dataset, tasks_by_episodes: dict) -> tuple[Dataset, list[str]]:
    df = dataset.to_pandas()
    tasks = list(set(tasks_by_episodes.values()))
    tasks_to_task_index = {task: task_idx for task_idx, task in enumerate(tasks)}
    episodes_to_task_index = {ep_idx: tasks_to_task_index[task] for ep_idx, task in tasks_by_episodes.items()}
    df["task_index"] = df["episode_index"].map(episodes_to_task_index).astype(int)

    features = dataset.features
    features["task_index"] = datasets.Value(dtype="int64")
    dataset = Dataset.from_pandas(df, features=features, split="train")
    return dataset, tasks

def get_videos_info(local_dir: Path, video_keys: list[str]) -> dict:
    video_files = [
        DEFAULT_VIDEO_PATH.format(episode_chunk=0, video_key=vid_key, episode_index=0)
        for vid_key in video_keys
    ]

    videos_info_dict = {}
    for vid_key, vid_path in zip(video_keys, video_files, strict=True):
        videos_info_dict[vid_key] = get_video_info(local_dir / vid_path)

    return videos_info_dict



def merge_convert_metadata(input_dir_list:list[Path], dataset: Dataset, output_dir: Path, episode_lengths_in):
    print("Converting metadata...")

    output_meta_dir = output_dir / "meta"
    output_meta_dir.mkdir(exist_ok=True)
    features = get_features_from_hf_dataset(dataset, None)
    video_keys = [key for key, ft in features.items() if ft["dtype"] == "video"]
    # Episodes & chunks
    episode_indices = sorted(dataset.unique("episode_index"))
    total_episodes = len(episode_indices)
    assert episode_indices == list(range(total_episodes))
    total_videos = total_episodes * len(video_keys)
    total_chunks = total_episodes // DEFAULT_CHUNK_SIZE
    if total_episodes % DEFAULT_CHUNK_SIZE != 0:
        total_chunks += 1

    # tasks_by_episodes = {ep_idx: single_task for ep_idx in episode_indices}
    # new_dataset, tasks = add_task_index_by_episodes(dataset, tasks_by_episodes)
    # tasks_by_episodes = {ep_idx: [task] for ep_idx, task in tasks_by_episodes.items()}

    # assert set(tasks) == {task for ep_tasks in tasks_by_episodes.values() for task in ep_tasks}

    tasks_by_episodes = {}

    
    task_map = load_task_map()
    for task in task_map:
        start, end = task["episode_range"]
        for ep_idx in range(start, end + 1):
            tasks_by_episodes[ep_idx] = [task["task_detail"]]

    tasks = [{"task_index": t["task_index"], "task": t["task_detail"]} for t in task_map]
    write_jsonlines(tasks, output_dir / TASKS_PATH)
    episodes = [
        {"episode_index": ep_idx, "tasks": tasks_by_episodes[ep_idx], "length": episode_lengths_in[ep_idx]}
        for ep_idx in episode_indices
    ]
    write_jsonlines(episodes, output_dir / EPISODES_PATH)


    features["task_index"] = {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    }
    metadata_v1 = load_json(input_dir_list[0] / V1_INFO_PATH)

    if video_keys:
        assert metadata_v1.get("video", False)
        dataset = dataset.remove_columns(video_keys)
        videos_info = get_videos_info(output_dir, video_keys=video_keys)
        for key in video_keys:
            features[key]["shape"] = (
                videos_info[key].pop("video.height"),
                videos_info[key].pop("video.width"),
                videos_info[key].pop("video.channels"),
            )
            features[key]["video_info"] = videos_info[key]
            assert math.isclose(videos_info[key]["video.fps"], metadata_v1["fps"], rel_tol=1e-3)
            if "encoding" in metadata_v1:
                assert videos_info[key]["video.pix_fmt"] == metadata_v1["encoding"]["pix_fmt"]
    else:
        assert metadata_v1.get("video", 0) == 0
        videos_info = None

    

    # Assemble metadata v2.0
    metadata_v2_0 = {
        "codebase_version": V20,
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": len(dataset),
        "total_tasks": len(tasks),
        "total_videos": total_videos,
        "total_chunks": total_chunks,
        "chunks_size": DEFAULT_CHUNK_SIZE,
        "fps": metadata_v1["fps"],
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": DEFAULT_PARQUET_PATH,
        "video_path": DEFAULT_VIDEO_PATH if video_keys else None,
        "features": features,
    }
    write_json(metadata_v2_0, output_dir / INFO_PATH)
    merge_convert_stats_to_json(input_dir_list, output_dir)

def convert_metadata(input_dir:Path, dataset: Dataset, meta_dir: Path, output_dir: Path, episode_lengths_in):
    print("Converting metadata...")
    task_map = load_task_map()
    if (len(task_map) > 1):
        print("wrong task num, please check")
        return
    single_task = task_map[0]["task_detail"]
    output_meta_dir = output_dir / "meta"
    output_meta_dir.mkdir(exist_ok=True)
    features = get_features_from_hf_dataset(dataset, None)
    video_keys = [key for key, ft in features.items() if ft["dtype"] == "video"]
    # Episodes & chunks
    episode_indices = sorted(dataset.unique("episode_index"))
    total_episodes = len(episode_indices)
    assert episode_indices == list(range(total_episodes))
    total_videos = total_episodes * len(video_keys)
    total_chunks = total_episodes // DEFAULT_CHUNK_SIZE
    if total_episodes % DEFAULT_CHUNK_SIZE != 0:
        total_chunks += 1

    tasks_by_episodes = {ep_idx: single_task for ep_idx in episode_indices}
    new_dataset, tasks = add_task_index_by_episodes(dataset, tasks_by_episodes)
    tasks_by_episodes = {ep_idx: [task] for ep_idx, task in tasks_by_episodes.items()}

    assert set(tasks) == {task for ep_tasks in tasks_by_episodes.values() for task in ep_tasks}
    tasks = [{"task_index": task_idx, "task": task} for task_idx, task in enumerate(tasks)]
    write_jsonlines(tasks, output_dir / TASKS_PATH)
    features["task_index"] = {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    }
    metadata_v1 = load_json(input_dir / V1_INFO_PATH)

    if video_keys:
        assert metadata_v1.get("video", False)
        new_dataset = new_dataset.remove_columns(video_keys)
        videos_info = get_videos_info(output_dir, video_keys=video_keys)
        for key in video_keys:
            features[key]["shape"] = (
                videos_info[key].pop("video.height"),
                videos_info[key].pop("video.width"),
                videos_info[key].pop("video.channels"),
            )
            features[key]["video_info"] = videos_info[key]
            assert math.isclose(videos_info[key]["video.fps"], metadata_v1["fps"], rel_tol=1e-3)
            if "encoding" in metadata_v1:
                assert videos_info[key]["video.pix_fmt"] == metadata_v1["encoding"]["pix_fmt"]
    else:
        assert metadata_v1.get("video", 0) == 0
        videos_info = None

    episodes = [
        {"episode_index": ep_idx, "tasks": tasks_by_episodes[ep_idx], "length": episode_lengths_in[ep_idx]}
        for ep_idx in episode_indices
    ]
    write_jsonlines(episodes, output_dir / EPISODES_PATH)

    # Assemble metadata v2.0
    metadata_v2_0 = {
        "codebase_version": V20,
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": len(new_dataset),
        "total_tasks": len(tasks),
        "total_videos": total_videos,
        "total_chunks": total_chunks,
        "chunks_size": DEFAULT_CHUNK_SIZE,
        "fps": metadata_v1["fps"],
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": DEFAULT_PARQUET_PATH,
        "video_path": DEFAULT_VIDEO_PATH if video_keys else None,
        "features": features,
    }
    write_json(metadata_v2_0, output_dir / INFO_PATH)
    convert_stats_to_json(input_dir, output_dir)
    
def main(input_dir_list: list[Path], output_dir: Path):
    if len(input_dir_list) >= 2:
        print(f"Merging and converting dataset from {input_dir_list} to {output_dir}")
        arrow_file_list = []
        for temp_dir in input_dir_list:
            arrow_file = next(temp_dir.glob("train/data-*.arrow"))
            arrow_file_list.append(arrow_file)
        episode_lengths, dataset_out = merge_and_convert_arrow_to_v2(input_dir_list, arrow_file_list, output_dir)
        merge_reorganize_videos(input_dir_list, output_dir)
        merge_convert_metadata(input_dir_list, dataset_out, output_dir, episode_lengths)
        target_dir = output_dir / "meta_test"
        if target_dir.exists() and target_dir.is_dir():
            shutil.rmtree(target_dir)
            print(f"Deleted folder: {target_dir}")
        train_file_path = output_dir / "train-00000-of-00001.parquet"
        if train_file_path.exists() and train_file_path.is_file():
            train_file_path.unlink()
            print(f"Deleted file: {train_file_path}")
        return
    else:
        input_dir = input_dir_list[0]
        print(f"Converting dataset from {input_dir} to {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_parquet_dir = input_dir.with_name(f"{input_dir.name}_parquet")
        # # 1. 处理arrow文件
        arrow_file = next(input_dir.glob("train/data-*.arrow"))
        # print(str(arrow_file))
        episode_lengths, dataset_out = convert_arrow_to_v2(input_dir, arrow_file, output_dir)
        
        # # 2. 处理视频文件
        videos_dir = input_dir / "videos"
        reorganize_videos(videos_dir, output_dir)
        
        # #3. 处理元数据
        meta_dir = input_dir / "meta_data"
        convert_metadata(input_dir, dataset_out, meta_dir, output_dir, episode_lengths)
        shutil.rmtree(output_parquet_dir)

    # input_path = Path("/home/h666/code/dataset/hf_dataset/zcai/aloha2/collect_dish_0126_merged_resized6d_new_2")
    # output_path = Path("/home/h666/code/dataset/hf_dataset/zcai/aloha2/collect_dish_0126_merged_resized6d_new_2")
    # convert_stats(input_path, output_path, 0)
    # input_path_new = "/home/h666/code/dataset/hf_dataset/zcai/aloha2/collect_dish_0126_merged_resized6d_new_2"
    # convert_stats_to_jsonl(input_path_new)

    # with open("/home/h666/code/dataset/hf_dataset/zcai/aloha2/collect_dish_0126_merged_resized6d/meta_data/stats.safetensors", "rb") as f:
    #     print(f.read(64))
    # with open("/home/h666/code/dataset/hf_dataset/zcai/aloha2/pick_and_place_0124_rf10/train/data-00000-of-00001.arrow", "rb") as f:
    #     reader = ipc.open_stream(f)
    #     table_new = reader.read_all()
    # table_new = pq.read_table("/home/h666/code/dataset/hf_dataset/zcai/aloha2/pick_and_place_0124_rf10/train/data-00000-of-00001.arrow")
    # table_new = pq.read_table("/home/h666/code/dataset/hf_dataset/zcai/aloha2/pick_and_place_0124_rf10_parquet_2/train-00000-of-00001.parquet")
    # table_new = pq.read_table("/home/h666/episode_000091.parquet")
    # table_new = pq.read_table("/home/h666/code/dataset/hf_dataset/zcai/aloha2/collect_dish_0126_merged_resized6d_new_4/data/chunk-000/episode_000000.parquet",  columns=["observation.images.cam_high"])
    
    # table_new = pq.read_table("/data/TR2/hugging_face/pick_and_place_0124_rf10_0423/train-00000-of-00001.parquet")    # print(table_new)   
    # table_new = pq.read_table("/data/TR2/hugging_face/pick_and_place_0124_rf10_0423/data/chunk-000/episode_000014.parquet") 
    # table_new = pq.read_table("/data/TR2/hugging_face/pick_and_place_0124_rf10_0423/data/chunk-000/episode_000714.parquet")            
    
    
    
    # flat_table = table_new.flatten()

    # 显示特定列
    # col = flat_table.column("observation.images.cam_high")
    # print(table_new.column_names)
    # print(flat_table)
    # print(table_new.to_pandas())


    # df = table_new.to_pandas()
    # df.to_csv("train-00000-of-00001_new_2.csv", index=False)



    # print(f"Conversion complete! Output saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, nargs='+', required=True, 
                       help="Path to input dataset directory (containing train/, videos/, meta_data/)")
    parser.add_argument("--output-dir", type=Path, required=True,
                       help="Path to output converted dataset")
    args = parser.parse_args()
    
    main(args.input_dir, args.output_dir)