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
This script will help you convert any LeRobot dataset already pushed to the hub from codebase version 1.6 to
2.0. You will be required to provide the 'tasks', which is a short but accurate description in plain English
for each of the task performed in the dataset. This will allow to easily train models with task-conditioning.

We support 3 different scenarios for these tasks (see instructions below):
    1. Single task dataset: all episodes of your dataset have the same single task.
    2. Single task episodes: the episodes of your dataset each contain a single task but they can differ from
      one episode to the next.
    3. Multi task episodes: episodes of your dataset may each contain several different tasks.


Can you can also provide a robot config .yaml file (not mandatory) to this script via the option
'--robot-config' so that it writes information about the robot (robot type, motors names) this dataset was
recorded with. For now, only Aloha/Koch type robots are supported with this option.


# 1. Single task dataset
If your dataset contains a single task, you can simply provide it directly via the CLI with the
'--single-task' option.

Examples:

```bash
python lerobot/common/datasets/v2/convert_dataset_v1_to_v2.py \
    --repo-id lerobot/aloha_sim_insertion_human_image \
    --single-task "Insert the peg into the socket." \
    --robot-config lerobot/configs/robot/aloha.yaml \
    --local-dir data
```

```bash
python lerobot/common/datasets/v2/convert_dataset_v1_to_v2.py \
    --repo-id aliberts/koch_tutorial \
    --single-task "Pick the Lego block and drop it in the box on the right." \
    --robot-config lerobot/configs/robot/koch.yaml \
    --local-dir data
```


# 2. Single task episodes
If your dataset is a multi-task dataset, you have two options to provide the tasks to this script:

- If your dataset already contains a language instruction column in its parquet file, you can simply provide
  this column's name with the '--tasks-col' arg.

    Example:

    ```bash
    python lerobot/common/datasets/v2/convert_dataset_v1_to_v2.py \
        --repo-id lerobot/stanford_kuka_multimodal_dataset \
        --tasks-col "language_instruction" \
        --local-dir data
    ```

- If your dataset doesn't contain a language instruction, you should provide the path to a .json file with the
  '--tasks-path' arg. This file should have the following structure where keys correspond to each
  episode_index in the dataset, and values are the language instruction for that episode.

    Example:

    ```json
    {
        "0": "Do something",
        "1": "Do something else",
        "2": "Do something",
        "3": "Go there",
        ...
    }
    ```

# 3. Multi task episodes
If you have multiple tasks per episodes, your dataset should contain a language instruction column in its
parquet file, and you must provide this column's name with the '--tasks-col' arg.

Example:

```bash
python lerobot/common/datasets/v2/convert_dataset_v1_to_v2.py \
    --repo-id lerobot/stanford_kuka_multimodal_dataset \
    --tasks-col "language_instruction" \
    --local-dir data
```
"""

import argparse
import contextlib
import filecmp
import json
import logging
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import datasets
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from datasets import Dataset
from huggingface_hub import HfApi
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError
from safetensors.torch import load_file

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

V16 = "v1.6"
V20 = "v2.0"

GITATTRIBUTES_REF = "aliberts/gitattributes_reference"
V1_VIDEO_FILE = "{video_key}_episode_{episode_index:06d}.mp4"
V1_INFO_PATH = "meta_data/info.json"
V1_STATS_PATH = "meta_data/stats.safetensors"


def parse_robot_config(robot_cfg: RobotConfig) -> tuple[str, dict]:
    if robot_cfg.type in ["aloha", "koch"]:
        state_names = [
            f"{arm}_{motor}" if len(robot_cfg.follower_arms) > 1 else motor
            for arm in robot_cfg.follower_arms
            for motor in robot_cfg.follower_arms[arm].motors
        ]
        action_names = [
            # f"{arm}_{motor}" for arm in ["left", "right"] for motor in robot_cfg["leader_arms"][arm]["motors"]
            f"{arm}_{motor}" if len(robot_cfg.leader_arms) > 1 else motor
            for arm in robot_cfg.leader_arms
            for motor in robot_cfg.leader_arms[arm].motors
        ]
    # elif robot_cfg["robot_type"] == "stretch3": TODO
    else:
        raise NotImplementedError(
            "Please provide robot_config={'robot_type': ..., 'names': ...} directly to convert_dataset()."
        )

    return {
        "robot_type": robot_cfg.type,
        "names": {
            "observation.state": state_names,
            "observation.effort": state_names,
            "action": action_names,
        },
    }


def convert_stats_to_json(v1_dir: Path, v2_dir: Path) -> None:
    safetensor_path = v1_dir / V1_STATS_PATH
    stats = load_file(safetensor_path)
    serialized_stats = {key: value.tolist() for key, value in stats.items()}
    serialized_stats = unflatten_dict(serialized_stats)

    json_path = v2_dir / STATS_PATH
    json_path.parent.mkdir(exist_ok=True, parents=True)
    with open(json_path, "w") as f:
        json.dump(serialized_stats, f, indent=4)

    # Sanity check
    with open(json_path) as f:
        stats_json = json.load(f)

    stats_json = flatten_dict(stats_json)
    stats_json = {key: torch.tensor(value) for key, value in stats_json.items()}
    for key in stats:
        torch.testing.assert_close(stats_json[key], stats[key])


def get_features_from_hf_dataset(
    dataset: Dataset, robot_config: RobotConfig | None = None
) -> dict[str, list]:
    robot_config = parse_robot_config(robot_config)
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
            image = dataset[0][key]  # Assuming first row
            channels = get_image_pixel_channels(image)
            shape = (image.height, image.width, channels)
            names = ["height", "width", "channels"]
        elif ft._type == "VideoFrame":
            dtype = "video"
            shape = None  # Add shape later
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


def add_task_index_from_tasks_col(
    dataset: Dataset, tasks_col: str
) -> tuple[Dataset, dict[str, list[str]], list[str]]:
    df = dataset.to_pandas()

    # HACK: This is to clean some of the instructions in our version of Open X datasets
    prefix_to_clean = "tf.Tensor(b'"
    suffix_to_clean = "', shape=(), dtype=string)"
    df[tasks_col] = df[tasks_col].str.removeprefix(prefix_to_clean).str.removesuffix(suffix_to_clean)

    # Create task_index col
    tasks_by_episode = df.groupby("episode_index")[tasks_col].unique().apply(lambda x: x.tolist()).to_dict()
    tasks = df[tasks_col].unique().tolist()
    tasks_to_task_index = {task: idx for idx, task in enumerate(tasks)}
    df["task_index"] = df[tasks_col].map(tasks_to_task_index).astype(int)

    # Build the dataset back from df
    features = dataset.features
    features["task_index"] = datasets.Value(dtype="int64")
    dataset = Dataset.from_pandas(df, features=features, split="train")
    dataset = dataset.remove_columns(tasks_col)

    return dataset, tasks, tasks_by_episode


def split_parquet_by_episodes(
    dataset: Dataset,
    total_episodes: int,
    total_chunks: int,
    output_dir: Path,
) -> list:
    table = dataset.data.table
    episode_lengths = []
    for ep_chunk in range(total_chunks):
        ep_chunk_start = DEFAULT_CHUNK_SIZE * ep_chunk
        ep_chunk_end = min(DEFAULT_CHUNK_SIZE * (ep_chunk + 1), total_episodes)
        chunk_dir = "/".join(DEFAULT_PARQUET_PATH.split("/")[:-1]).format(episode_chunk=ep_chunk)
        (output_dir / chunk_dir).mkdir(parents=True, exist_ok=True)
        for ep_idx in range(ep_chunk_start, ep_chunk_end):
            ep_table = table.filter(pc.equal(table["episode_index"], ep_idx))
            episode_lengths.insert(ep_idx, len(ep_table))
            output_file = output_dir / DEFAULT_PARQUET_PATH.format(
                episode_chunk=ep_chunk, episode_index=ep_idx
            )
            pq.write_table(ep_table, output_file)

    return episode_lengths


def move_videos(
    repo_id: str,
    video_keys: list[str],
    total_episodes: int,
    total_chunks: int,
    work_dir: Path,
    clean_gittatributes: Path,
    branch: str = "main",
) -> None:
    """
    HACK: Since HfApi() doesn't provide a way to move files directly in a repo, this function will run git
    commands to fetch git lfs video files references to move them into subdirectories without having to
    actually download them.
    """
    _lfs_clone(repo_id, work_dir, branch)

    videos_moved = False
    video_files = [str(f.relative_to(work_dir)) for f in work_dir.glob("videos*/*.mp4")]
    if len(video_files) == 0:
        video_files = [str(f.relative_to(work_dir)) for f in work_dir.glob("videos*/*/*/*.mp4")]
        videos_moved = True  # Videos have already been moved

    assert len(video_files) == total_episodes * len(video_keys)

    lfs_untracked_videos = _get_lfs_untracked_videos(work_dir, video_files)

    current_gittatributes = work_dir / ".gitattributes"
    if not filecmp.cmp(current_gittatributes, clean_gittatributes, shallow=False):
        fix_gitattributes(work_dir, current_gittatributes, clean_gittatributes)

    if lfs_untracked_videos:
        fix_lfs_video_files_tracking(work_dir, video_files)

    if videos_moved:
        return

    video_dirs = sorted(work_dir.glob("videos*/"))
    for ep_chunk in range(total_chunks):
        ep_chunk_start = DEFAULT_CHUNK_SIZE * ep_chunk
        ep_chunk_end = min(DEFAULT_CHUNK_SIZE * (ep_chunk + 1), total_episodes)
        for vid_key in video_keys:
            chunk_dir = "/".join(DEFAULT_VIDEO_PATH.split("/")[:-1]).format(
                episode_chunk=ep_chunk, video_key=vid_key
            )
            (work_dir / chunk_dir).mkdir(parents=True, exist_ok=True)

            for ep_idx in range(ep_chunk_start, ep_chunk_end):
                target_path = DEFAULT_VIDEO_PATH.format(
                    episode_chunk=ep_chunk, video_key=vid_key, episode_index=ep_idx
                )
                video_file = V1_VIDEO_FILE.format(video_key=vid_key, episode_index=ep_idx)
                if len(video_dirs) == 1:
                    video_path = video_dirs[0] / video_file
                else:
                    for dir in video_dirs:
                        if (dir / video_file).is_file():
                            video_path = dir / video_file
                            break

                video_path.rename(work_dir / target_path)

    commit_message = "Move video files into chunk subdirectories"
    subprocess.run(["git", "add", "."], cwd=work_dir, check=True)
    subprocess.run(["git", "commit", "-m", commit_message], cwd=work_dir, check=True)
    subprocess.run(["git", "push"], cwd=work_dir, check=True)


def fix_lfs_video_files_tracking(work_dir: Path, lfs_untracked_videos: list[str]) -> None:
    """
    HACK: This function fixes the tracking by git lfs which was not properly set on some repos. In that case,
    there's no other option than to download the actual files and reupload them with lfs tracking.
    """
    for i in range(0, len(lfs_untracked_videos), 100):
        files = lfs_untracked_videos[i : i + 100]
        try:
            subprocess.run(["git", "rm", "--cached", *files], cwd=work_dir, capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            print("git rm --cached ERROR:")
            print(e.stderr)
        subprocess.run(["git", "add", *files], cwd=work_dir, check=True)

    commit_message = "Track video files with git lfs"
    subprocess.run(["git", "commit", "-m", commit_message], cwd=work_dir, check=True)
    subprocess.run(["git", "push"], cwd=work_dir, check=True)


def fix_gitattributes(work_dir: Path, current_gittatributes: Path, clean_gittatributes: Path) -> None:
    shutil.copyfile(clean_gittatributes, current_gittatributes)
    subprocess.run(["git", "add", ".gitattributes"], cwd=work_dir, check=True)
    subprocess.run(["git", "commit", "-m", "Fix .gitattributes"], cwd=work_dir, check=True)
    subprocess.run(["git", "push"], cwd=work_dir, check=True)


def _lfs_clone(repo_id: str, work_dir: Path, branch: str) -> None:
    subprocess.run(["git", "lfs", "install"], cwd=work_dir, check=True)
    repo_url = f"https://huggingface.co/datasets/{repo_id}"
    env = {"GIT_LFS_SKIP_SMUDGE": "1"}  # Prevent downloading LFS files
    subprocess.run(
        ["git", "clone", "--branch", branch, "--single-branch", "--depth", "1", repo_url, str(work_dir)],
        check=True,
        env=env,
    )


def _get_lfs_untracked_videos(work_dir: Path, video_files: list[str]) -> list[str]:
    lfs_tracked_files = subprocess.run(
        ["git", "lfs", "ls-files", "-n"], cwd=work_dir, capture_output=True, text=True, check=True
    )
    lfs_tracked_files = set(lfs_tracked_files.stdout.splitlines())
    return [f for f in video_files if f not in lfs_tracked_files]


def get_videos_info(repo_id: str, local_dir: Path, video_keys: list[str], branch: str) -> dict:
    # Assumes first episode
    video_files = [
        DEFAULT_VIDEO_PATH.format(episode_chunk=0, video_key=vid_key, episode_index=0)
        for vid_key in video_keys
    ]
    hub_api = HfApi()
    hub_api.snapshot_download(
        repo_id=repo_id, repo_type="dataset", local_dir=local_dir, revision=branch, allow_patterns=video_files
    )
    videos_info_dict = {}
    for vid_key, vid_path in zip(video_keys, video_files, strict=True):
        videos_info_dict[vid_key] = get_video_info(local_dir / vid_path)

    return videos_info_dict


def convert_dataset(
    repo_id: str,
    local_dir: Path,
    single_task: str | None = None,
    tasks_path: Path | None = None,
    tasks_col: Path | None = None,
    robot_config: RobotConfig | None = None,
    test_branch: str | None = None,
    **card_kwargs,
):
    v1 = get_safe_version(repo_id, V16)
    v1x_dir = local_dir / V16 / repo_id
    v20_dir = local_dir / V20 / repo_id
    v1x_dir.mkdir(parents=True, exist_ok=True)
    v20_dir.mkdir(parents=True, exist_ok=True)

    hub_api = HfApi()
    hub_api.snapshot_download(
        repo_id=repo_id, repo_type="dataset", revision=v1, local_dir=v1x_dir, ignore_patterns="videos*/"
    )
    branch = "main"
    if test_branch:
        branch = test_branch
        create_branch(repo_id=repo_id, branch=test_branch, repo_type="dataset")

    metadata_v1 = load_json(v1x_dir / V1_INFO_PATH)
    dataset = datasets.load_dataset("parquet", data_dir=v1x_dir / "data", split="train")
    features = get_features_from_hf_dataset(dataset, robot_config)
    video_keys = [key for key, ft in features.items() if ft["dtype"] == "video"]

    if single_task and "language_instruction" in dataset.column_names:
        logging.warning(
            "'single_task' provided but 'language_instruction' tasks_col found. Using 'language_instruction'.",
        )
        single_task = None
        tasks_col = "language_instruction"

    # Episodes & chunks
    episode_indices = sorted(dataset.unique("episode_index"))
    total_episodes = len(episode_indices)
    assert episode_indices == list(range(total_episodes))
    total_videos = total_episodes * len(video_keys)
    total_chunks = total_episodes // DEFAULT_CHUNK_SIZE
    if total_episodes % DEFAULT_CHUNK_SIZE != 0:
        total_chunks += 1

    # Tasks
    if single_task:
        tasks_by_episodes = {ep_idx: single_task for ep_idx in episode_indices}
        dataset, tasks = add_task_index_by_episodes(dataset, tasks_by_episodes)
        tasks_by_episodes = {ep_idx: [task] for ep_idx, task in tasks_by_episodes.items()}
    elif tasks_path:
        tasks_by_episodes = load_json(tasks_path)
        tasks_by_episodes = {int(ep_idx): task for ep_idx, task in tasks_by_episodes.items()}
        dataset, tasks = add_task_index_by_episodes(dataset, tasks_by_episodes)
        tasks_by_episodes = {ep_idx: [task] for ep_idx, task in tasks_by_episodes.items()}
    elif tasks_col:
        dataset, tasks, tasks_by_episodes = add_task_index_from_tasks_col(dataset, tasks_col)
    else:
        raise ValueError

    assert set(tasks) == {task for ep_tasks in tasks_by_episodes.values() for task in ep_tasks}
    tasks = [{"task_index": task_idx, "task": task} for task_idx, task in enumerate(tasks)]
    write_jsonlines(tasks, v20_dir / TASKS_PATH)
    features["task_index"] = {
        "dtype": "int64",
        "shape": (1,),
        "names": None,
    }

    # Videos
    if video_keys:
        assert metadata_v1.get("video", False)
        dataset = dataset.remove_columns(video_keys)
        clean_gitattr = Path(
            hub_api.hf_hub_download(
                repo_id=GITATTRIBUTES_REF, repo_type="dataset", local_dir=local_dir, filename=".gitattributes"
            )
        ).absolute()
        with tempfile.TemporaryDirectory() as tmp_video_dir:
            move_videos(
                repo_id, video_keys, total_episodes, total_chunks, Path(tmp_video_dir), clean_gitattr, branch
            )
        videos_info = get_videos_info(repo_id, v1x_dir, video_keys=video_keys, branch=branch)
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

    # Split data into 1 parquet file by episode
    episode_lengths = split_parquet_by_episodes(dataset, total_episodes, total_chunks, v20_dir)

    if robot_config is not None:
        robot_type = robot_config.type
        repo_tags = [robot_type]
    else:
        robot_type = "unknown"
        repo_tags = None

    # Episodes
    episodes = [
        {"episode_index": ep_idx, "tasks": tasks_by_episodes[ep_idx], "length": episode_lengths[ep_idx]}
        for ep_idx in episode_indices
    ]
    write_jsonlines(episodes, v20_dir / EPISODES_PATH)

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
    write_json(metadata_v2_0, v20_dir / INFO_PATH)
    convert_stats_to_json(v1x_dir, v20_dir)
    card = create_lerobot_dataset_card(tags=repo_tags, dataset_info=metadata_v2_0, **card_kwargs)

    with contextlib.suppress(EntryNotFoundError, HfHubHTTPError):
        hub_api.delete_folder(repo_id=repo_id, path_in_repo="data", repo_type="dataset", revision=branch)

    with contextlib.suppress(EntryNotFoundError, HfHubHTTPError):
        hub_api.delete_folder(repo_id=repo_id, path_in_repo="meta_data", repo_type="dataset", revision=branch)

    with contextlib.suppress(EntryNotFoundError, HfHubHTTPError):
        hub_api.delete_folder(repo_id=repo_id, path_in_repo="meta", repo_type="dataset", revision=branch)

    hub_api.upload_folder(
        repo_id=repo_id,
        path_in_repo="data",
        folder_path=v20_dir / "data",
        repo_type="dataset",
        revision=branch,
    )
    hub_api.upload_folder(
        repo_id=repo_id,
        path_in_repo="meta",
        folder_path=v20_dir / "meta",
        repo_type="dataset",
        revision=branch,
    )

    card.push_to_hub(repo_id=repo_id, repo_type="dataset", revision=branch)

    if not test_branch:
        create_branch(repo_id=repo_id, branch=V20, repo_type="dataset")


def main():
    parser = argparse.ArgumentParser()
    task_args = parser.add_mutually_exclusive_group(required=True)

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository identifier on Hugging Face: a community or a user name `/` the name of the dataset (e.g. `lerobot/pusht`, `cadene/aloha_sim_insertion_human`).",
    )
    task_args.add_argument(
        "--single-task",
        type=str,
        help="A short but accurate description of the single task performed in the dataset.",
    )
    task_args.add_argument(
        "--tasks-col",
        type=str,
        help="The name of the column containing language instructions",
    )
    task_args.add_argument(
        "--tasks-path",
        type=Path,
        help="The path to a .json file containing one language instruction for each episode_index",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default=None,
        help="Robot config used for the dataset during conversion (e.g. 'koch', 'aloha', 'so100', etc.)",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="Local directory to store the dataset during conversion. Defaults to /tmp/lerobot_dataset_v2",
    )
    parser.add_argument(
        "--license",
        type=str,
        default="apache-2.0",
        help="Repo license. Must be one of https://huggingface.co/docs/hub/repositories-licenses. Defaults to mit.",
    )
    parser.add_argument(
        "--test-branch",
        type=str,
        default=None,
        help="Repo branch to test your conversion first (e.g. 'v2.0.test')",
    )

    args = parser.parse_args()
    if not args.local_dir:
        args.local_dir = Path("/tmp/lerobot_dataset_v2")

    if args.robot is not None:
        robot_config = make_robot_config(args.robot)

    del args.robot

    convert_dataset(**vars(args), robot_config=robot_config)


if __name__ == "__main__":
    main()
