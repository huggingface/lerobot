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
import contextlib
import gc
import logging
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

import datasets
import numpy as np
import packaging.version
import pandas as pd
import PIL.Image
import torch
import torch.utils
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import RevisionNotFoundError

from collections import defaultdict
from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.datasets.image_writer import AsyncImageWriter, write_image
from lerobot.datasets.utils import (
    DEFAULT_EPISODES_PATH,
    DEFAULT_FEATURES,
    DEFAULT_IMAGE_PATH,
    INFO_PATH,
    _validate_feature_names,
    check_delta_timestamps,
    check_version_compatibility,
    create_empty_dataset_info,
    create_lerobot_dataset_card,
    embed_images,
    flatten_dict,
    get_delta_indices,
    get_hf_dataset_cache_dir,
    get_hf_dataset_size_in_mb,
    get_hf_features_from_features,
    get_parquet_file_size_in_mb,
    get_parquet_num_frames,
    get_safe_version,
    get_video_size_in_mb,
    hf_transform_to_torch,
    is_valid_version,
    load_episodes,
    load_info,
    load_nested_dataset,
    load_stats,
    load_tasks,
    to_parquet_with_hf_images,
    update_chunk_file_indices,
    validate_episode_buffer,
    validate_frame,
    write_info,
    write_json,
    write_stats,
    write_tasks,
)
from lerobot.datasets.video_utils import (
    VideoFrame,
    concatenate_video_files,
    decode_video_frames,
    encode_video_frames,
    get_safe_default_codec,
    get_video_duration_in_s,
    get_video_info,
)

CODEBASE_VERSION = "v3.0"
OBS_IMAGE = "observation.image"
OBS_IMAGE_2 = "observation.image_2"
OBS_IMAGE_3 = "observation.image_3"
OBS_STATE = "observation.state"
OBS_ENV_STATE = "observation.env_state"
ACTION = "action"

class LeRobotDatasetMetadata:
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        revision: str | None = None,
        force_cache_sync: bool = False,
    ):
        self.repo_id = repo_id
        self.revision = revision if revision else CODEBASE_VERSION
        self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id

        try:
            if force_cache_sync:
                raise FileNotFoundError
            self.load_metadata()
        except (FileNotFoundError, NotADirectoryError):
            if is_valid_version(self.revision):
                self.revision = get_safe_version(self.repo_id, self.revision)

            (self.root / "meta").mkdir(exist_ok=True, parents=True)
            self.pull_from_repo(allow_patterns="meta/")
            self.load_metadata()

    def load_metadata(self):
        self.info = load_info(self.root)
        check_version_compatibility(self.repo_id, self._version, CODEBASE_VERSION)
        self.tasks = load_tasks(self.root)
        self.episodes = load_episodes(self.root)
        self.stats = load_stats(self.root)

    def pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
    ) -> None:
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

    @property
    def url_root(self) -> str:
        return f"hf://datasets/{self.repo_id}"

    @property
    def _version(self) -> packaging.version.Version:
        """Codebase version used to create this dataset."""
        return packaging.version.parse(self.info["codebase_version"])

    def get_data_file_path(self, ep_index: int) -> Path:
        ep = self.episodes[ep_index]
        chunk_idx = ep["data/chunk_index"]
        file_idx = ep["data/file_index"]
        fpath = self.data_path.format(chunk_index=chunk_idx, file_index=file_idx)
        return Path(fpath)

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        ep = self.episodes[ep_index]
        chunk_idx = ep[f"videos/{vid_key}/chunk_index"]
        file_idx = ep[f"videos/{vid_key}/file_index"]
        fpath = self.video_path.format(video_key=vid_key, chunk_index=chunk_idx, file_index=file_idx)
        return Path(fpath)

    @property
    def data_path(self) -> str:
        """Formattable string for the parquet files."""
        return self.info["data_path"]

    @property
    def video_path(self) -> str | None:
        """Formattable string for the video files."""
        return self.info["video_path"]

    @property
    def robot_type(self) -> str | None:
        """Robot type used in recording this dataset."""
        return self.info["robot_type"]

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.info["fps"]

    @property
    def features(self) -> dict[str, dict]:
        """All features contained in the dataset."""
        return self.info["features"]

    @property
    def image_keys(self) -> list[str]:
        """Keys to access visual modalities stored as images."""
        return [key for key, ft in self.features.items() if ft["dtype"] == "image"]

    @property
    def video_keys(self) -> list[str]:
        """Keys to access visual modalities stored as videos."""
        return [key for key, ft in self.features.items() if ft["dtype"] == "video"]

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access visual modalities (regardless of their storage method)."""
        return [key for key, ft in self.features.items() if ft["dtype"] in ["video", "image"]]

    @property
    def names(self) -> dict[str, list | dict]:
        """Names of the various dimensions of vector modalities."""
        return {key: ft["names"] for key, ft in self.features.items()}

    @property
    def shapes(self) -> dict:
        """Shapes for the different features."""
        return {key: tuple(ft["shape"]) for key, ft in self.features.items()}

    @property
    def total_episodes(self) -> int:
        """Total number of episodes available."""
        return self.info["total_episodes"]

    @property
    def total_frames(self) -> int:
        """Total number of frames saved in this dataset."""
        return self.info["total_frames"]

    @property
    def total_tasks(self) -> int:
        """Total number of different tasks performed in this dataset."""
        return self.info["total_tasks"]

    @property
    def chunks_size(self) -> int:
        """Max number of files per chunk."""
        return self.info["chunks_size"]

    @property
    def data_files_size_in_mb(self) -> int:
        """Max size of data file in mega bytes."""
        return self.info["data_files_size_in_mb"]

    @property
    def video_files_size_in_mb(self) -> int:
        """Max size of video file in mega bytes."""
        return self.info["video_files_size_in_mb"]

    def get_task_index(self, task: str) -> int | None:
        """
        Given a task in natural language, returns its task_index if the task already exists in the dataset,
        otherwise return None.
        """
        if task in self.tasks.index:
            return int(self.tasks.loc[task].task_index)
        else:
            return None

    def save_episode_tasks(self, tasks: list[str]):
        if len(set(tasks)) != len(tasks):
            raise ValueError(f"Tasks are not unique: {tasks}")

        if self.tasks is None:
            new_tasks = tasks
            task_indices = range(len(tasks))
            self.tasks = pd.DataFrame({"task_index": task_indices}, index=tasks)
        else:
            new_tasks = [task for task in tasks if task not in self.tasks.index]
            new_task_indices = range(len(self.tasks), len(self.tasks) + len(new_tasks))
            for task_idx, task in zip(new_task_indices, new_tasks, strict=False):
                self.tasks.loc[task] = task_idx

        if len(new_tasks) > 0:
            # Update on disk
            write_tasks(self.tasks, self.root)

    def _save_episode_metadata(self, episode_dict: dict) -> None:
        """Save episode metadata to a parquet file and update the Hugging Face dataset of episodes metadata.

        This function processes episodes metadata from a dictionary, converts it into a Hugging Face dataset,
        and saves it as a parquet file. It handles both the creation of new parquet files and the
        updating of existing ones based on size constraints. After saving the metadata, it reloads
        the Hugging Face dataset to ensure it is up-to-date.

        Notes: We both need to update parquet files and HF dataset:
        - `pandas` loads parquet file in RAM
        - `datasets` relies on a memory mapping from pyarrow (no RAM). It either converts parquet files to a pyarrow cache on disk,
          or loads directly from pyarrow cache.
        """
        # Convert buffer into HF Dataset
        episode_dict = {key: [value] for key, value in episode_dict.items()}
        ep_dataset = datasets.Dataset.from_dict(episode_dict)
        ep_size_in_mb = get_hf_dataset_size_in_mb(ep_dataset)
        df = pd.DataFrame(ep_dataset)
        num_frames = episode_dict["length"][0]

        if self.episodes is None:
            # Initialize indices and frame count for a new dataset made of the first episode data
            chunk_idx, file_idx = 0, 0
            df["meta/episodes/chunk_index"] = [chunk_idx]
            df["meta/episodes/file_index"] = [file_idx]
            df["dataset_from_index"] = [0]
            df["dataset_to_index"] = [num_frames]
        else:
            # Retrieve information from the latest parquet file
            latest_ep = self.episodes[-1]
            chunk_idx = latest_ep["meta/episodes/chunk_index"]
            file_idx = latest_ep["meta/episodes/file_index"]

            latest_path = self.root / DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
            latest_size_in_mb = get_parquet_file_size_in_mb(latest_path)

            if latest_size_in_mb + ep_size_in_mb >= self.data_files_size_in_mb:
                # Size limit is reached, prepare new parquet file
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, self.chunks_size)

            # Update the existing pandas dataframe with new row
            df["meta/episodes/chunk_index"] = [chunk_idx]
            df["meta/episodes/file_index"] = [file_idx]
            df["dataset_from_index"] = [latest_ep["dataset_to_index"]]
            df["dataset_to_index"] = [latest_ep["dataset_to_index"] + num_frames]

            if latest_size_in_mb + ep_size_in_mb < self.data_files_size_in_mb:
                # Size limit wasnt reached, concatenate latest dataframe with new one
                latest_df = pd.read_parquet(latest_path)
                df = pd.concat([latest_df, df], ignore_index=True)

                # Memort optimization
                del latest_df
                gc.collect()

        # Write the resulting dataframe from RAM to disk
        path = self.root / DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

        if self.episodes is not None:
            # Remove the episodes cache directory, necessary to avoid cache bloat
            cached_dir = get_hf_dataset_cache_dir(self.episodes)
            if cached_dir is not None:
                shutil.rmtree(cached_dir)

        self.episodes = load_episodes(self.root)

    def save_episode(
        self,
        episode_index: int,
        episode_length: int,
        episode_tasks: list[str],
        episode_stats: dict[str, dict],
        episode_metadata: dict,
    ) -> None:
        episode_dict = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": episode_length,
        }
        episode_dict.update(episode_metadata)
        episode_dict.update(flatten_dict({"stats": episode_stats}))
        self._save_episode_metadata(episode_dict)

        # Update info
        self.info["total_episodes"] += 1
        self.info["total_frames"] += episode_length
        self.info["total_tasks"] = len(self.tasks)
        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}

        write_info(self.info, self.root)

        self.stats = aggregate_stats([self.stats, episode_stats]) if self.stats is not None else episode_stats
        write_stats(self.stats, self.root)

    def update_video_info(self, video_key: str | None = None) -> None:
        """
        Warning: this function writes info from first episode videos, implicitly assuming that all videos have
        been encoded the same way. Also, this means it assumes the first episode exists.
        """
        if video_key is not None and video_key not in self.video_keys:
            raise ValueError(f"Video key {video_key} not found in dataset")

        video_keys = [video_key] if video_key is not None else self.video_keys
        for key in video_keys:
            if not self.features[key].get("info", None):
                video_path = self.root / self.video_path.format(
                    video_key=video_key, chunk_index=0, file_index=0
                )
                self.info["features"][key]["info"] = get_video_info(video_path)

    def update_chunk_settings(
        self,
        chunks_size: int | None = None,
        data_files_size_in_mb: int | None = None,
        video_files_size_in_mb: int | None = None,
    ) -> None:
        """Update chunk and file size settings after dataset creation.

        This allows users to customize storage organization without modifying the constructor.
        These settings control how episodes are chunked and how large files can grow before
        creating new ones.

        Args:
            chunks_size: Maximum number of files per chunk directory. If None, keeps current value.
            data_files_size_in_mb: Maximum size for data parquet files in MB. If None, keeps current value.
            video_files_size_in_mb: Maximum size for video files in MB. If None, keeps current value.
        """
        if chunks_size is not None:
            if chunks_size <= 0:
                raise ValueError(f"chunks_size must be positive, got {chunks_size}")
            self.info["chunks_size"] = chunks_size

        if data_files_size_in_mb is not None:
            if data_files_size_in_mb <= 0:
                raise ValueError(f"data_files_size_in_mb must be positive, got {data_files_size_in_mb}")
            self.info["data_files_size_in_mb"] = data_files_size_in_mb

        if video_files_size_in_mb is not None:
            if video_files_size_in_mb <= 0:
                raise ValueError(f"video_files_size_in_mb must be positive, got {video_files_size_in_mb}")
            self.info["video_files_size_in_mb"] = video_files_size_in_mb

        # Update the info file on disk
        write_info(self.info, self.root)

    def get_chunk_settings(self) -> dict[str, int]:
        """Get current chunk and file size settings.

        Returns:
            Dict containing chunks_size, data_files_size_in_mb, and video_files_size_in_mb.
        """
        return {
            "chunks_size": self.chunks_size,
            "data_files_size_in_mb": self.data_files_size_in_mb,
            "video_files_size_in_mb": self.video_files_size_in_mb,
        }

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Total episodes: '{self.total_episodes}',\n"
            f"    Total frames: '{self.total_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "})',\n"
        )

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        robot_type: str | None = None,
        root: str | Path | None = None,
        use_videos: bool = True,
    ) -> "LeRobotDatasetMetadata":
        """Creates metadata for a LeRobotDataset."""
        obj = cls.__new__(cls)
        obj.repo_id = repo_id
        obj.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id

        obj.root.mkdir(parents=True, exist_ok=False)

        features = {**features, **DEFAULT_FEATURES}
        _validate_feature_names(features)

        obj.tasks = None
        obj.episodes = None
        obj.stats = None
        obj.info = create_empty_dataset_info(CODEBASE_VERSION, fps, features, use_videos, robot_type)
        if len(obj.video_keys) > 0 and not use_videos:
            raise ValueError()
        write_json(obj.info, obj.root / INFO_PATH)
        obj.revision = None
        return obj


class LeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
    ):
        """
        2 modes are available for instantiating this class, depending on 2 different use cases:

        1. Your dataset already exists:
            - On your local disk in the 'root' folder. This is typically the case when you recorded your
              dataset locally and you may or may not have pushed it to the hub yet. Instantiating this class
              with 'root' will load your dataset directly from disk. This can happen while you're offline (no
              internet connection).

            - On the Hugging Face Hub at the address https://huggingface.co/datasets/{repo_id} and not on
              your local disk in the 'root' folder. Instantiating this class with this 'repo_id' will download
              the dataset from that address and load it, pending your dataset is compliant with
              codebase_version v3.0. If your dataset has been created before this new format, you will be
              prompted to convert it using our conversion script from v2.1 to v3.0, which you can find at
              lerobot/datasets/v30/convert_dataset_v21_to_v30.py.


        2. Your dataset doesn't already exists (either on local disk or on the Hub): you can create an empty
           LeRobotDataset with the 'create' classmethod. This can be used for recording a dataset or port an
           existing dataset to the LeRobotDataset format.


        In terms of files, LeRobotDataset encapsulates 3 main things:
            - metadata:
                - info contains various information about the dataset like shapes, keys, fps etc.
                - stats stores the dataset statistics of the different modalities for normalization
                - tasks contains the prompts for each task of the dataset, which can be used for
                  task-conditioned training.
            - hf_dataset (from datasets.Dataset), which will read any values from parquet files.
            - videos (optional) from which frames are loaded to be synchronous with data from parquet files.

        A typical LeRobotDataset looks like this from its root path:
        .
        ├── data
        │   ├── chunk-000
        │   │   ├── file-000.parquet
        │   │   ├── file-001.parquet
        │   │   └── ...
        │   ├── chunk-001
        │   │   ├── file-000.parquet
        │   │   ├── file-001.parquet
        │   │   └── ...
        │   └── ...
        ├── meta
        │   ├── episodes
        │   │   ├── chunk-000
        │   │   │   ├── file-000.parquet
        │   │   │   ├── file-001.parquet
        │   │   │   └── ...
        │   │   ├── chunk-001
        │   │   │   └── ...
        │   │   └── ...
        │   ├── info.json
        │   ├── stats.json
        │   └── tasks.parquet
        └── videos
            ├── observation.images.laptop
            │   ├── chunk-000
            │   │   ├── file-000.mp4
            │   │   ├── file-001.mp4
            │   │   └── ...
            │   ├── chunk-001
            │   │   └── ...
            │   └── ...
            ├── observation.images.phone
            │   ├── chunk-000
            │   │   ├── file-000.mp4
            │   │   ├── file-001.mp4
            │   │   └── ...
            │   ├── chunk-001
            │   │   └── ...
            │   └── ...
            └── ...

        Note that this file-based structure is designed to be as versatile as possible. Multiple episodes are
        consolidated into chunked files which improves storage efficiency and loading performance. The
        structure of the dataset is entirely described in the info.json file, which can be easily downloaded
        or viewed directly on the hub before downloading any actual data. The type of files used are very
        simple and do not need complex tools to be read, it only uses .parquet, .json and .mp4 files (and .md
        for the README).

        Args:
            repo_id (str): This is the repo id that will be used to fetch the dataset. Locally, the dataset
                will be stored under root/repo_id.
            root (Path | None, optional): Local directory to use for downloading/writing files. You can also
                set the LEROBOT_HOME environment variable to point to a different location. Defaults to
                '~/.cache/huggingface/lerobot'.
            episodes (list[int] | None, optional): If specified, this will only load episodes specified by
                their episode_index in this list. Defaults to None.
            image_transforms (Callable | None, optional): You can pass standard v2 image transforms from
                torchvision.transforms.v2 here which will be applied to visual modalities (whether they come
                from videos or images). Defaults to None.
            delta_timestamps (dict[list[float]] | None, optional): _description_. Defaults to None.
            tolerance_s (float, optional): Tolerance in seconds used to ensure data timestamps are actually in
                sync with the fps value. It is used at the init of the dataset to make sure that each
                timestamps is separated to the next by 1/fps +/- tolerance_s. This also applies to frames
                decoded from video files. It is also used to check that `delta_timestamps` (when provided) are
                multiples of 1/fps. Defaults to 1e-4.
            revision (str, optional): An optional Git revision id which can be a branch name, a tag, or a
                commit hash. Defaults to current codebase version tag.
            force_cache_sync (bool, optional): Flag to sync and refresh local files first. If True and files
                are already present in the local cache, this will be faster. However, files loaded might not
                be in sync with the version on the hub, especially if you specified 'revision'. Defaults to
                False.
            download_videos (bool, optional): Flag to download the videos. Note that when set to True but the
                video files are already present on local disk, they won't be downloaded again. Defaults to
                True.
            video_backend (str | None, optional): Video backend to use for decoding videos. Defaults to torchcodec when available int the platform; otherwise, defaults to 'pyav'.
                You can also use the 'pyav' decoder used by Torchvision, which used to be the default option, or 'video_reader' which is another decoder of Torchvision.
            batch_encoding_size (int, optional): Number of episodes to accumulate before batch encoding videos.
                Set to 1 for immediate encoding (default), or higher for batched encoding. Defaults to 1.
        """
        super().__init__()
        self.repo_id = repo_id
        self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        self.delta_indices = None
        self.batch_encoding_size = batch_encoding_size
        self.episodes_since_last_encoding = 0

        # Unused attributes
        self.image_writer = None
        self.episode_buffer = None

        self.root.mkdir(exist_ok=True, parents=True)

        # Load metadata
        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=force_cache_sync
        )

        # Load actual data
        try:
            if force_cache_sync:
                raise FileNotFoundError
            self.hf_dataset = self.load_hf_dataset()
            # Check if cached dataset contains all requested episodes
            if not self._check_cached_episodes_sufficient():
                raise FileNotFoundError("Cached dataset doesn't contain all requested episodes")
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            self.revision = get_safe_version(self.repo_id, self.revision)
            self.download(download_videos)
            self.hf_dataset = self.load_hf_dataset()

        # Setup delta_indices
        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

    def push_to_hub(
        self,
        branch: str | None = None,
        tags: list | None = None,
        license: str | None = "apache-2.0",
        tag_version: bool = True,
        push_videos: bool = True,
        private: bool = False,
        allow_patterns: list[str] | str | None = None,
        upload_large_folder: bool = False,
        **card_kwargs,
    ) -> None:
        ignore_patterns = ["images/"]
        if not push_videos:
            ignore_patterns.append("videos/")

        hub_api = HfApi()
        hub_api.create_repo(
            repo_id=self.repo_id,
            private=private,
            repo_type="dataset",
            exist_ok=True,
        )
        if branch:
            hub_api.create_branch(
                repo_id=self.repo_id,
                branch=branch,
                revision=self.revision,
                repo_type="dataset",
                exist_ok=True,
            )

        upload_kwargs = {
            "repo_id": self.repo_id,
            "folder_path": self.root,
            "repo_type": "dataset",
            "revision": branch,
            "allow_patterns": allow_patterns,
            "ignore_patterns": ignore_patterns,
        }
        if upload_large_folder:
            hub_api.upload_large_folder(**upload_kwargs)
        else:
            hub_api.upload_folder(**upload_kwargs)

        card = create_lerobot_dataset_card(
            tags=tags, dataset_info=self.meta.info, license=license, **card_kwargs
        )
        card.push_to_hub(repo_id=self.repo_id, repo_type="dataset", revision=branch)

        if tag_version:
            with contextlib.suppress(RevisionNotFoundError):
                hub_api.delete_tag(self.repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
            hub_api.create_tag(self.repo_id, tag=CODEBASE_VERSION, revision=branch, repo_type="dataset")

    def pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
    ) -> None:
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

    def download(self, download_videos: bool = True) -> None:
        """Downloads the dataset from the given 'repo_id' at the provided version. If 'episodes' is given, this
        will only download those episodes (selected by their episode_index). If 'episodes' is None, the whole
        dataset will be downloaded. Thanks to the behavior of snapshot_download, if the files are already present
        in 'local_dir', they won't be downloaded again.
        """
        # TODO(rcadene, aliberts): implement faster transfer
        # https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads
        ignore_patterns = None if download_videos else "videos/"
        files = None
        if self.episodes is not None:
            files = self.get_episodes_file_paths()
        self.pull_from_repo(allow_patterns=files, ignore_patterns=ignore_patterns)

    def get_episodes_file_paths(self) -> list[Path]:
        episodes = self.episodes if self.episodes is not None else list(range(self.meta.total_episodes))
        fpaths = [str(self.meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        if len(self.meta.video_keys) > 0:
            video_files = [
                str(self.meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self.meta.video_keys
                for ep_idx in episodes
            ]
            fpaths += video_files
        # episodes are stored in the same files, so we return unique paths only
        fpaths = list(set(fpaths))
        return fpaths

    def load_hf_dataset(self) -> datasets.Dataset:
        """hf_dataset contains all the observations, states, actions, rewards, etc."""
        features = get_hf_features_from_features(self.features)
        hf_dataset = load_nested_dataset(self.root / "data", features=features)
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def _check_cached_episodes_sufficient(self) -> bool:
        """Check if the cached dataset contains all requested episodes."""
        if self.hf_dataset is None or len(self.hf_dataset) == 0:
            return False

        # Get available episode indices from cached dataset
        available_episodes = {
            ep_idx.item() if isinstance(ep_idx, torch.Tensor) else ep_idx
            for ep_idx in self.hf_dataset["episode_index"]
        }

        # Determine requested episodes
        if self.episodes is None:
            # Requesting all episodes - check if we have all episodes from metadata
            requested_episodes = set(range(self.meta.total_episodes))
        else:
            # Requesting specific episodes
            requested_episodes = set(self.episodes)

        # Check if all requested episodes are available in cached data
        return requested_episodes.issubset(available_episodes)

    def create_hf_dataset(self) -> datasets.Dataset:
        features = get_hf_features_from_features(self.features)
        ft_dict = {col: [] for col in features}
        hf_dataset = datasets.Dataset.from_dict(ft_dict, features=features, split="train")
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.meta.fps

    @property
    def num_frames(self) -> int:
        """Number of frames in selected episodes."""
        return len(self.hf_dataset) if self.hf_dataset is not None else self.meta.total_frames

    @property
    def num_episodes(self) -> int:
        """Number of episodes selected."""
        return len(self.episodes) if self.episodes is not None else self.meta.total_episodes

    @property
    def features(self) -> dict[str, dict]:
        return self.meta.features

    @property
    def hf_features(self) -> datasets.Features:
        """Features of the hf_dataset."""
        if self.hf_dataset is not None:
            return self.hf_dataset.features
        else:
            return get_hf_features_from_features(self.features)

    def _get_query_indices(self, idx: int, ep_idx: int) -> tuple[dict[str, list[int | bool]]]:
        ep = self.meta.episodes[ep_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]
        query_indices = {
            key: [max(ep_start, min(ep_end - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {  # Pad values outside of current episode range
            f"{key}_is_pad": torch.BoolTensor(
                [(idx + delta < ep_start) | (idx + delta >= ep_end) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                timestamps = self.hf_dataset[query_indices[key]]["timestamp"]
                query_timestamps[key] = torch.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]

        return query_timestamps

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        return {
            key: torch.stack(self.hf_dataset[q_idx][key])
            for key, q_idx in query_indices.items()
            if key not in self.meta.video_keys
        }

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault. This probably happens because a memory reference to the video loader is created in
        the main process and a subprocess fails to access it.
        """
        ep = self.meta.episodes[ep_idx]
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            # Episodes are stored sequentially on a single mp4 to reduce the number of files.
            # Thus we load the start timestamp of the episode on this mp4 and,
            # shift the query timestamp accordingly.
            from_timestamp = ep[f"videos/{vid_key}/from_timestamp"]
            shifted_query_ts = [from_timestamp + ts for ts in query_ts]

            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            frames = decode_video_frames(video_path, shifted_query_ts, self.tolerance_s, self.video_backend)
            item[vid_key] = frames.squeeze(0)

        return item

    def _add_padding_keys(self, item: dict, padding: dict[str, list[bool]]) -> dict:
        for key, val in padding.items():
            item[key] = torch.BoolTensor(val)
        return item

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx) -> dict:
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if self.image_transforms is not None:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])

        # Add task as a string
        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks.iloc[task_idx].name
        return item

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Number of selected episodes: '{self.num_episodes}',\n"
            f"    Number of selected samples: '{self.num_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "})',\n"
        )

    def create_episode_buffer(self, episode_index: int | None = None) -> dict:
        current_ep_idx = self.meta.total_episodes if episode_index is None else episode_index
        ep_buffer = {}
        # size and task are special cases that are not in self.features
        ep_buffer["size"] = 0
        ep_buffer["task"] = []
        for key in self.features:
            ep_buffer[key] = current_ep_idx if key == "episode_index" else []
        return ep_buffer

    def _get_image_file_path(self, episode_index: int, image_key: str, frame_index: int) -> Path:
        fpath = DEFAULT_IMAGE_PATH.format(
            image_key=image_key, episode_index=episode_index, frame_index=frame_index
        )
        return self.root / fpath

    def _get_image_file_dir(self, episode_index: int, image_key: str) -> Path:
        return self._get_image_file_path(episode_index, image_key, frame_index=0).parent

    def _save_image(self, image: torch.Tensor | np.ndarray | PIL.Image.Image, fpath: Path) -> None:
        if self.image_writer is None:
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            write_image(image, fpath)
        else:
            self.image_writer.save_image(image=image, fpath=fpath)

    def add_frame(self, frame: dict) -> None:
        """
        This function only adds the frame to the episode_buffer. Apart from images — which are written in a
        temporary directory — nothing is written to disk. To save those frames, the 'save_episode()' method
        then needs to be called.
        """
        # Convert torch to numpy if needed
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()

        validate_frame(frame, self.features)

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        # Automatically add frame_index and timestamp to episode buffer
        frame_index = self.episode_buffer["size"]
        timestamp = frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)
        self.episode_buffer["task"].append(frame.pop("task"))  # Remove task from frame after processing

        # Add frame features to episode_buffer
        for key in frame:
            if key not in self.features:
                raise ValueError(
                    f"An element of the frame is not in the features. '{key}' not in '{self.features.keys()}'."
                )

            if self.features[key]["dtype"] in ["image", "video"]:
                img_path = self._get_image_file_path(
                    episode_index=self.episode_buffer["episode_index"], image_key=key, frame_index=frame_index
                )
                if frame_index == 0:
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                self._save_image(frame[key], img_path)
                self.episode_buffer[key].append(str(img_path))
            else:
                self.episode_buffer[key].append(frame[key])

        self.episode_buffer["size"] += 1

    def save_episode(self, episode_data: dict | None = None) -> None:
        """
        This will save to disk the current episode in self.episode_buffer.

        Video encoding is handled automatically based on batch_encoding_size:
        - If batch_encoding_size == 1: Videos are encoded immediately after each episode
        - If batch_encoding_size > 1: Videos are encoded in batches.

        Args:
            episode_data (dict | None, optional): Dict containing the episode data to save. If None, this will
                save the current episode in self.episode_buffer, which is filled with 'add_frame'. Defaults to
                None.
        """
        episode_buffer = episode_data if episode_data is not None else self.episode_buffer

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        # size and task are special cases that won't be added to hf_dataset
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        # Update tasks and task indices with new tasks if any
        self.meta.save_episode_tasks(episode_tasks)

        # Given tasks in natural language, find their corresponding task indices
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.features.items():
            # index, episode_index, task_index are already processed above, and image and video
            # are processed separately by storing image path and frame info as meta data
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        # Wait for image writer to end, so that episode stats over images can be computed
        self._wait_image_writer()
        ep_stats = compute_episode_stats(episode_buffer, self.features)

        ep_metadata = self._save_episode_data(episode_buffer)
        has_video_keys = len(self.meta.video_keys) > 0
        use_batched_encoding = self.batch_encoding_size > 1

        if has_video_keys and not use_batched_encoding:
            for video_key in self.meta.video_keys:
                ep_metadata.update(self._save_episode_video(video_key, episode_index))

        # `meta.save_episode` need to be executed after encoding the videos
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats, ep_metadata)

        if has_video_keys and use_batched_encoding:
            # Check if we should trigger batch encoding
            self.episodes_since_last_encoding += 1
            if self.episodes_since_last_encoding == self.batch_encoding_size:
                start_ep = self.num_episodes - self.batch_encoding_size
                end_ep = self.num_episodes
                self._batch_save_episode_video(start_ep, end_ep)
                self.episodes_since_last_encoding = 0

        if not episode_data:
            # Reset episode buffer and clean up temporary images (if not already deleted during video encoding)
            self.clear_episode_buffer(delete_images=len(self.meta.image_keys) > 0)

    def _batch_save_episode_video(self, start_episode: int, end_episode: int | None = None):
        """
        Batch save videos for multiple episodes.

        Args:
            start_episode: Starting episode index (inclusive)
            end_episode: Ending episode index (exclusive). If None, encodes all episodes from start_episode to the current episode.
        """
        if end_episode is None:
            end_episode = self.num_episodes

        logging.info(
            f"Batch encoding {self.batch_encoding_size} videos for episodes {start_episode} to {end_episode - 1}"
        )

        chunk_idx = self.meta.episodes[start_episode]["data/chunk_index"]
        file_idx = self.meta.episodes[start_episode]["data/file_index"]
        episode_df_path = self.root / DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        episode_df = pd.read_parquet(episode_df_path)

        for ep_idx in range(start_episode, end_episode):
            logging.info(f"Encoding videos for episode {ep_idx}")

            if (
                self.meta.episodes[ep_idx]["data/chunk_index"] != chunk_idx
                or self.meta.episodes[ep_idx]["data/file_index"] != file_idx
            ):
                # The current episode is in a new chunk or file.
                # Save previous episode dataframe and update the Hugging Face dataset by reloading it.
                episode_df.to_parquet(episode_df_path)
                self.meta.episodes = load_episodes(self.root)

                # Load new episode dataframe
                chunk_idx = self.meta.episodes[ep_idx]["data/chunk_index"]
                file_idx = self.meta.episodes[ep_idx]["data/file_index"]
                episode_df_path = self.root / DEFAULT_EPISODES_PATH.format(
                    chunk_index=chunk_idx, file_index=file_idx
                )
                episode_df = pd.read_parquet(episode_df_path)

            # Save the current episode's video metadata to the dataframe
            video_ep_metadata = {}
            for video_key in self.meta.video_keys:
                video_ep_metadata.update(self._save_episode_video(video_key, ep_idx))
            video_ep_metadata.pop("episode_index")
            video_ep_df = pd.DataFrame(video_ep_metadata, index=[ep_idx]).convert_dtypes(
                dtype_backend="pyarrow"
            )  # allows NaN values along with integers

            episode_df = episode_df.combine_first(video_ep_df)
            episode_df.to_parquet(episode_df_path)
            self.meta.episodes = load_episodes(self.root)

    def _save_episode_data(self, episode_buffer: dict) -> dict:
        """Save episode data to a parquet file and update the Hugging Face dataset of frames data.

        This function processes episodes data from a buffer, converts it into a Hugging Face dataset,
        and saves it as a parquet file. It handles both the creation of new parquet files and the
        updating of existing ones based on size constraints. After saving the data, it reloads
        the Hugging Face dataset to ensure it is up-to-date.

        Notes: We both need to update parquet files and HF dataset:
        - `pandas` loads parquet file in RAM
        - `datasets` relies on a memory mapping from pyarrow (no RAM). It either converts parquet files to a pyarrow cache on disk,
          or loads directly from pyarrow cache.
        """
        # Convert buffer into HF Dataset
        ep_dict = {key: episode_buffer[key] for key in self.hf_features}
        ep_dataset = datasets.Dataset.from_dict(ep_dict, features=self.hf_features, split="train")
        ep_dataset = embed_images(ep_dataset)
        ep_size_in_mb = get_hf_dataset_size_in_mb(ep_dataset)
        ep_num_frames = len(ep_dataset)
        df = pd.DataFrame(ep_dataset)

        if self.meta.episodes is None:
            # Initialize indices and frame count for a new dataset made of the first episode data
            chunk_idx, file_idx = 0, 0
            latest_num_frames = 0
        else:
            # Retrieve information from the latest parquet file
            latest_ep = self.meta.episodes[-1]
            chunk_idx = latest_ep["data/chunk_index"]
            file_idx = latest_ep["data/file_index"]

            latest_path = self.root / self.meta.data_path.format(chunk_index=chunk_idx, file_index=file_idx)
            latest_size_in_mb = get_parquet_file_size_in_mb(latest_path)
            latest_num_frames = get_parquet_num_frames(latest_path)

            # Determine if a new parquet file is needed
            if latest_size_in_mb + ep_size_in_mb >= self.meta.data_files_size_in_mb:
                # Size limit is reached, prepare new parquet file
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, self.meta.chunks_size)
                latest_num_frames = 0
            else:
                # Update the existing parquet file with new rows
                latest_df = pd.read_parquet(latest_path)
                df = pd.concat([latest_df, df], ignore_index=True)

                # Memort optimization
                del latest_df
                gc.collect()

        # Write the resulting dataframe from RAM to disk
        path = self.root / self.meta.data_path.format(chunk_index=chunk_idx, file_index=file_idx)
        path.parent.mkdir(parents=True, exist_ok=True)
        if len(self.meta.image_keys) > 0:
            to_parquet_with_hf_images(df, path)
        else:
            df.to_parquet(path)

        if self.hf_dataset is not None:
            # Remove hf dataset cache directory, necessary to avoid cache bloat
            cached_dir = get_hf_dataset_cache_dir(self.hf_dataset)
            if cached_dir is not None:
                shutil.rmtree(cached_dir)

        self.hf_dataset = self.load_hf_dataset()

        metadata = {
            "data/chunk_index": chunk_idx,
            "data/file_index": file_idx,
            "dataset_from_index": latest_num_frames,
            "dataset_to_index": latest_num_frames + ep_num_frames,
        }
        return metadata

    def _save_episode_video(self, video_key: str, episode_index: int):
        # Encode episode frames into a temporary video
        ep_path = self._encode_temporary_episode_video(video_key, episode_index)
        ep_size_in_mb = get_video_size_in_mb(ep_path)
        ep_duration_in_s = get_video_duration_in_s(ep_path)

        if self.meta.episodes is None or (
            f"videos/{video_key}/chunk_index" not in self.meta.episodes.column_names
            or f"videos/{video_key}/file_index" not in self.meta.episodes.column_names
        ):
            # Initialize indices for a new dataset made of the first episode data
            chunk_idx, file_idx = 0, 0
            latest_duration_in_s = 0.0
            new_path = self.root / self.meta.video_path.format(
                video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
            )
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(ep_path), str(new_path))
        else:
            # Retrieve information from the latest updated video file (possibly several episodes ago)
            latest_ep = self.meta.episodes[episode_index - 1]
            chunk_idx = latest_ep[f"videos/{video_key}/chunk_index"]
            file_idx = latest_ep[f"videos/{video_key}/file_index"]

            latest_path = self.root / self.meta.video_path.format(
                video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
            )
            latest_size_in_mb = get_video_size_in_mb(latest_path)
            latest_duration_in_s = get_video_duration_in_s(latest_path)

            if latest_size_in_mb + ep_size_in_mb >= self.meta.video_files_size_in_mb:
                # Move temporary episode video to a new video file in the dataset
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, self.meta.chunks_size)
                new_path = self.root / self.meta.video_path.format(
                    video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
                )
                new_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(ep_path), str(new_path))
                latest_duration_in_s = 0.0
            else:
                # Update latest video file
                concatenate_video_files(
                    [latest_path, ep_path],
                    latest_path,
                )

        # Remove temporary directory
        shutil.rmtree(str(ep_path.parent))

        # Update video info (only needed when first episode is encoded since it reads from episode 0)
        if episode_index == 0:
            self.meta.update_video_info(video_key)
            write_info(self.meta.info, self.meta.root)  # ensure video info always written properly

        metadata = {
            "episode_index": episode_index,
            f"videos/{video_key}/chunk_index": chunk_idx,
            f"videos/{video_key}/file_index": file_idx,
            f"videos/{video_key}/from_timestamp": latest_duration_in_s,
            f"videos/{video_key}/to_timestamp": latest_duration_in_s + ep_duration_in_s,
        }
        return metadata

    def clear_episode_buffer(self, delete_images: bool = True) -> None:
        # Clean up image files for the current episode buffer
        if delete_images:
            # Wait for the async image writer to finish
            if self.image_writer is not None:
                self._wait_image_writer()
            episode_index = self.episode_buffer["episode_index"]
            if isinstance(episode_index, np.ndarray):
                episode_index = episode_index.item() if episode_index.size == 1 else episode_index[0]
            for cam_key in self.meta.camera_keys:
                img_dir = self._get_image_file_dir(episode_index, cam_key)
                if img_dir.is_dir():
                    shutil.rmtree(img_dir)

        # Reset the buffer
        self.episode_buffer = self.create_episode_buffer()

    def start_image_writer(self, num_processes: int = 0, num_threads: int = 4) -> None:
        if isinstance(self.image_writer, AsyncImageWriter):
            logging.warning(
                "You are starting a new AsyncImageWriter that is replacing an already existing one in the dataset."
            )

        self.image_writer = AsyncImageWriter(
            num_processes=num_processes,
            num_threads=num_threads,
        )

    def stop_image_writer(self) -> None:
        """
        Whenever wrapping this dataset inside a parallelized DataLoader, this needs to be called first to
        remove the image_writer in order for the LeRobotDataset object to be pickleable and parallelized.
        """
        if self.image_writer is not None:
            self.image_writer.stop()
            self.image_writer = None

    def _wait_image_writer(self) -> None:
        """Wait for asynchronous image writer to finish."""
        if self.image_writer is not None:
            self.image_writer.wait_until_done()

    def _encode_temporary_episode_video(self, video_key: str, episode_index: int) -> dict:
        """
        Use ffmpeg to convert frames stored as png into mp4 videos.
        Note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
        since video encoding with ffmpeg is already using multithreading.
        """
        temp_path = Path(tempfile.mkdtemp(dir=self.root)) / f"{video_key}_{episode_index:03d}.mp4"
        img_dir = self._get_image_file_dir(episode_index, video_key)
        encode_video_frames(img_dir, temp_path, self.fps, overwrite=True)
        shutil.rmtree(img_dir)
        return temp_path

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        root: str | Path | None = None,
        robot_type: str | None = None,
        use_videos: bool = True,
        tolerance_s: float = 1e-4,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
    ) -> "LeRobotDataset":
        """Create a LeRobot Dataset from scratch in order to record data."""
        obj = cls.__new__(cls)
        obj.meta = LeRobotDatasetMetadata.create(
            repo_id=repo_id,
            fps=fps,
            robot_type=robot_type,
            features=features,
            root=root,
            use_videos=use_videos,
        )
        obj.repo_id = obj.meta.repo_id
        obj.root = obj.meta.root
        obj.revision = None
        obj.tolerance_s = tolerance_s
        obj.image_writer = None
        obj.batch_encoding_size = batch_encoding_size
        obj.episodes_since_last_encoding = 0

        if image_writer_processes or image_writer_threads:
            obj.start_image_writer(image_writer_processes, image_writer_threads)

        # TODO(aliberts, rcadene, alexander-soare): Merge this with OnlineBuffer/DataBuffer
        obj.episode_buffer = obj.create_episode_buffer()

        obj.episodes = None
        obj.hf_dataset = obj.create_hf_dataset()
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.delta_indices = None
        obj.video_backend = video_backend if video_backend is not None else get_safe_default_codec()
        return obj

ROBOT_TYPE_KEYS_MAPPING = {
    "lerobot/stanford_hydra_dataset": "static_single_arm",
    "lerobot/iamlab_cmu_pickup_insert": "static_single_arm",
    "lerobot/berkeley_fanuc_manipulation": "static_single_arm",
    "lerobot/toto": "static_single_arm",
    "lerobot/roboturk": "static_single_arm",
    "lerobot/jaco_play": "static_single_arm",
    "lerobot/taco_play": "static_single_arm_7statedim",
}
class MultiLeRobotDatasetMeta:
    def __init__(
        self,
        datasets: list[LeRobotDataset],
        repo_ids: list[str],
        keys_to_max_dim: dict[str, int],
        train_on_all_features: bool = False,
    ):
        self.repo_ids = repo_ids
        self.keys_to_max_dim = keys_to_max_dim
        self.train_on_all_features = train_on_all_features
        self.robot_types = [ds.meta.info["robot_type"] for ds in datasets]

        # assign robot_type if missing
        for ds in datasets:
            ds.meta.info["robot_type"] = ROBOT_TYPE_KEYS_MAPPING.get(ds.repo_id, ds.meta.info["robot_type"])
            ds.robot_type = ds.meta.info["robot_type"]

        # step 1: compute disabled features
        self.disabled_features = set()
        if not self.train_on_all_features:
            intersection = set(datasets[0].features)
            for ds in datasets:
                intersection.intersection_update(ds.features)
            if not intersection:
                raise RuntimeError("No common features across datasets.")
            for repo_id, ds in zip(repo_ids, datasets, strict=False):
                extra = set(ds.features) - intersection
                logging.warning(f"Disabling {extra} for repo {repo_id}")
                self.disabled_features.update(extra)

        # step 2: build union_features excluding disabled
        self.union_features = {}
        for ds in datasets:
            for k, v in ds.features.items():
                if k not in self.disabled_features:
                    self.union_features[k] = v

        # step 3: reshape feature schema
        self.features = reshape_features_to_max_dim(
            self.union_features, reshape_dim=-1, keys_to_max_dim=self.keys_to_max_dim
        )

        # step 4: aggregate stats
        self.stats = aggregate_stats_per_robot_type(datasets)
        for robot_type_, stats_ in self.stats.items():
            for feat_key, feat_stats in stats_.items():
                if feat_key in [ACTION, OBS_ENV_STATE, OBS_STATE]:
                    for k, v in feat_stats.items():
                        pad_value = 0 if k in ["min", "mean"] else 1
                        self.stats[robot_type_][feat_key][k] = pad_tensor(
                            v,
                            max_size=self.keys_to_max_dim.get(feat_key, -1),
                            pad_dim=-1,
                            pad_value=pad_value,
                        )

        # step 5: episodes & tasks
        self.episodes = {repo_id: ds.meta.episodes for repo_id, ds in zip(repo_ids, datasets, strict=False)}
        self.tasks = {repo_id: ds.meta.tasks for repo_id, ds in zip(repo_ids, datasets, strict=False)}
        self.info = {repo_id: ds.meta.info for repo_id, ds in zip(repo_ids, datasets, strict=False)}


class MultiLeRobotDatasetCleaner:
    def __init__(
        self,
        datasets: list[LeRobotDataset],
        repo_ids: list[str],
        sampling_weights: list[float],
        datasets_repo_ids: list[str],
        min_fps: int = 1,
        max_fps: int = 100,
    ):
        self.original_datasets = datasets
        self.original_repo_ids = repo_ids
        self.original_weights = sampling_weights
        self.original_datasets_repo_ids = datasets_repo_ids

        # step 1: remove datasets with invalid fps

        # step 2: keep datasets with same features per robot type
        consistent_datasets, keep_mask = keep_datasets_with_the_same_features_per_robot_type(
            datasets
        )

        self.cleaned_datasets = consistent_datasets
        self.keep_mask = keep_mask
        self.cleaned_weights = [sampling_weights[i] for i in range(len(datasets)) if keep_mask[i]]
        self.cleaned_repo_ids = [repo_ids[i] for i in range(len(datasets)) if keep_mask[i]]
        self.cleaned_datasets_repo_ids = [
            datasets_repo_ids[i] for i in range(len(datasets)) if keep_mask[i]
        ]

        self.cumulative_sizes = np.array(
            [0] + list(torch.cumsum(torch.tensor([len(d) for d in consistent_datasets]), dim=0))
        )
        self.cleaned_weights = np.array(self.cleaned_weights, dtype=np.float32)

# --- at the top of the file (same imports as before) ---
from collections import defaultdict
from typing import Callable
import copy
import numpy as np
import torch
import datasets
from pathlib import Path

# If you already have these in your codebase, reuse them
try:
    from lerobot.common.constants import (
        ACTION, OBS_ENV_STATE, OBS_STATE, OBS_IMAGE, OBS_IMAGE_2, OBS_IMAGE_3
    )
except Exception:
    # Fallbacks if constants are already strings elsewhere
    ACTION = "action"
    OBS_ENV_STATE = "observation.env_state"
    OBS_STATE = "observation.state"
    OBS_IMAGE = "observation.image"
    OBS_IMAGE_2 = "observation.image_2"
    OBS_IMAGE_3 = "observation.image_3"

IGNORED_KEYS = ["observation.effort"]
class MultiLeRobotDataset(torch.utils.data.Dataset):
    # ... keep your existing docstring ...

    def __init__(
        self,
        repo_ids: list[str],
        root: str | Path | None = None,
        episodes: dict | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerances_s: dict | None = None,
        download_videos: bool = True,
        video_backend: str | None = None,
        # --- NEW: simple add-ons ---
        sampling_weights: list[float] | None = None,
        feature_keys_mapping: dict[str, dict[str, str]] | None = None,
        max_action_dim: int | None = None,
        max_state_dim: int | None = None,
        max_num_images: int | None = None,
        max_image_dim: int | None = None,
        train_on_all_features: bool = False,
        min_fps: int = 1,
        max_fps: int = 100,
        ignore_keys: list[str] | None = None,  # exact or glob patterns
    ):
        super().__init__()
        self.repo_ids = repo_ids
        self.root = Path(root) if root else HF_LEROBOT_HOME
        self.tolerances_s = tolerances_s if tolerances_s else dict.fromkeys(repo_ids, 0.0001)

        # --- NEW: store mapping and simple knobs ---
        self.feature_keys_mapping: dict[str, dict[str, str]] = feature_keys_mapping or {}
        self.train_on_all_features = train_on_all_features
        self.max_action_dim = max_action_dim
        self.max_state_dim = max_state_dim
        self.max_image_dim = max_image_dim
        self.max_num_images = max_num_images  # (optional, we don’t enforce count, we enforce names)
        self._ignore_patterns = list(ignore_keys or [])
        # Build underlying single datasets
        _datasets = []
        datasets_repo_ids = []
        self.sampling_weights = []

        sampling_weights = sampling_weights if sampling_weights is not None else [1] * len(repo_ids)
        assert len(sampling_weights) == len(repo_ids), (
            "The number of sampling weights must match the number of datasets. "
            f"Got {len(sampling_weights)} weights for {len(repo_ids)} datasets."
        )
        for i, repo_id in enumerate(repo_ids):
            try:
                _datasets.append(
                    LeRobotDataset(
                        repo_id,
                        root=self.root / repo_id,
                        episodes=episodes.get(repo_id, None) if episodes else None,
                        image_transforms=image_transforms,  # transforms applied inside single ds
                        delta_timestamps=delta_timestamps.get(repo_id, None) if delta_timestamps else None,
                        tolerance_s=self.tolerances_s[repo_id],
                        download_videos=download_videos,
                        video_backend=video_backend,
                    )
                )
                datasets_repo_ids.append(repo_id)
                self.sampling_weights.append(float(sampling_weights[i]))
            except Exception as e:
                print(f"Failed to load dataset: {repo_id} due to Exception: {e}")

        print(
            f"Finish loading {len(_datasets)} datasets, with sampling weights: "
            f"{self.sampling_weights} corresponding to: {datasets_repo_ids}"
        )

        # Bookkeeping for mapping & canonical image inventory
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps.get(repo_id, None) if delta_timestamps else None
        self._datasets = _datasets
        self.datasets_repo_ids = datasets_repo_ids

        # --- NEW: compute “canonical image keys” (targets across all mappings) ---
        self._canonical_image_keys: set[str] = set()
        self._source_keys_per_repo: dict[str, set[str]] = {}
        self._target_keys_per_repo: dict[str, set[str]] = {}
        for rid, mapping in self.feature_keys_mapping.items():
            src_keys = set(mapping.keys())
            tgt_keys = set(mapping.values())
            self._source_keys_per_repo[rid] = src_keys
            self._target_keys_per_repo[rid] = tgt_keys
            # union of target names (we will ensure these exist at __getitem__)
            self._canonical_image_keys |= {
                k for k in tgt_keys if self._is_image_key_like(k)
            }

        # If user didn’t give any mapping, fall back to native keys (no-ops)
        if not self._canonical_image_keys and self.train_on_all_features:
            # discover all image-like keys from raw features
            for ds in self._datasets:
                for k, v in ds.hf_features.items():
                    if isinstance(v, (datasets.Image, VideoFrame)):
                        self._canonical_image_keys.add(k)

        # Cleaner: keep fps & consistent feature sets per robot type (unchanged)
        cleaner = MultiLeRobotDatasetCleaner(
            datasets=self._datasets,
            repo_ids=repo_ids,
            sampling_weights=self.sampling_weights,
            datasets_repo_ids=self.datasets_repo_ids,
            min_fps=min_fps,
            max_fps=max_fps,
        )
        self._datasets = cleaner.cleaned_datasets
        self.sampling_weights = cleaner.cleaned_weights
        self.repo_ids = cleaner.cleaned_repo_ids
        self.datasets_repo_ids = cleaner.cleaned_datasets_repo_ids
        self.cumulative_sizes = cleaner.cumulative_sizes

        # Meta (unchanged): we give it dim maxima; it will reshape/pad vectors
        self.meta = MultiLeRobotDatasetMeta(
            datasets=self._datasets,
            repo_ids=self.repo_ids,
            keys_to_max_dim={
                ACTION: self.max_action_dim if self.max_action_dim is not None else -1,
                OBS_ENV_STATE: self.max_state_dim if self.max_state_dim is not None else -1,
                OBS_STATE: self.max_state_dim if self.max_state_dim is not None else -1,
                OBS_IMAGE: self.max_image_dim if self.max_image_dim is not None else -1,
                OBS_IMAGE_2: self.max_image_dim if self.max_image_dim is not None else -1,
                OBS_IMAGE_3: self.max_image_dim if self.max_image_dim is not None else -1,
            },
            train_on_all_features=train_on_all_features,
        )

        # --- NEW: track dropped (source) keys so collate won’t expect them
        # Anything that we *rename away* should be considered disabled,
        # otherwise downstream may expect them to exist.
        self._dropped_keys = set()
        for rid, mapping in self.feature_keys_mapping.items():
            self._dropped_keys |= set(mapping.keys())

        # Merge with meta’s disabled features
        self.disabled_features = set(self.meta.disabled_features) | self._dropped_keys

        self.stats = self.meta.stats

        # --- NEW: cache an example image shape per canonical key (lazy, filled on first use)
        self._cached_img_shape: dict[str, torch.Size] = {}

    # ---------------------- NEW small helpers ----------------------

    def _is_image_key_like(self, key: str) -> bool:
        # A loose heuristic: rely on name OR on features later
        return ("image" in key) or ("cam_" in key) or ("images." in key)
    
    def _should_ignore(self, key: str) -> bool:
        # exact or glob-style match
        for pat in self._ignore_patterns:
            if key == pat or fnmatch.fnmatch(key, pat):
                return True
        return False
    def _apply_feature_mapping(self, item: dict, repo_id: str) -> dict:
        """
        Rename features according to feature_keys_mapping[repo_id].
        - Moves tensor/image under target key.
        - Drops source key if moved.
        - Adds *_is_pad=False for image targets we fill/keep.
        """
        mapping = self.feature_keys_mapping.get(repo_id, {}) or {}
        if not mapping:
            return item

        for src, tgt in mapping.items():
            if src in item:
                # Move value
                item[tgt] = item[src]
                # Drop the source to avoid duplication
                del item[src]
        return item

    def _ensure_union_image_keys(self, item: dict) -> dict:
        """
        Ensure that every canonical image key exists.
        When missing, create a zero tensor matching (B,C,H,W) or (C,H,W) of an available image.
        Also add boolean mask at f"{key}_is_pad".
        """
        if not self.train_on_all_features or not self._canonical_image_keys:
            return item

        # find any existing image tensor in item to copy shape/dtype
        exemplar = None
        for k in list(item.keys()):
            v = item[k]
            if torch.is_tensor(v) and v.ndim in (3, 4, 5):  # (C,H,W) or (B,C,H,W) or (B,T,C,H,W)
                exemplar = v
                break

        # fallback to a safe 3x224x224 if nothing found
        def _fallback_image():
            return torch.zeros(3, 224, 224, dtype=torch.uint8)

        for key in self._canonical_image_keys:
            if key not in item:
                img = torch.zeros_like(exemplar) if exemplar is not None else _fallback_image()
                item[key] = img
                item[f"{key}_is_pad"] = torch.tensor(True, dtype=torch.bool)
            else:
                # Add a mask saying it’s *not* padded
                if f"{key}_is_pad" not in item:
                    item[f"{key}_is_pad"] = torch.tensor(False, dtype=torch.bool)
        return item

    # ---------------------- existing API below (mostly unchanged) ----------------------

    @property
    def repo_id_to_index(self):
        return {repo_id: i for i, repo_id in enumerate(self.repo_ids)}

    @property
    def repo_index_to_id(self):
        return {v: k for k, v in self.repo_id_to_index}

    @property
    def fps(self) -> int:
        return self._datasets[0].meta.info["fps"]

    @property
    def video(self) -> bool:
        return self._datasets[0].meta.info.get("video", False)

    @property
    def features(self) -> datasets.Features:
        """
        Extend native HF features with any *target* keys introduced by mapping.
        We copy the source spec for targets that didn’t exist in any raw dataset.
        """
        features: dict[str, datasets.features.Feature] = {}
        for dataset in self._datasets:
            for k, v in dataset.hf_features.items():
                if k not in self.disabled_features:
                    features[k] = v

        # Add mapped target image specs if not present yet
        for rid, mapping in self.feature_keys_mapping.items():
            ds = None
            # find the dataset object to read feature spec for source
            for _ds, _rid in zip(self._datasets, self.repo_ids, strict=False):
                if _rid == rid:
                    ds = _ds
                    break
            if ds is None:
                continue
            for src, tgt in mapping.items():
                if tgt not in features and src in ds.hf_features:
                    features[tgt] = ds.hf_features[src]

        return features

    @property
    def camera_keys(self) -> list[str]:
        keys = []
        for key, feats in self.features.items():
            if isinstance(feats, (datasets.Image, VideoFrame)):
                keys.append(key)
        return keys

    @property
    def video_frame_keys(self) -> list[str]:
        video_frame_keys = []
        for key, feats in self.features.items():
            if isinstance(feats, VideoFrame):
                video_frame_keys.append(key)
        return video_frame_keys

    @property
    def num_frames(self) -> int:
        return sum(d.num_frames for d in self._datasets)

    @property
    def num_episodes(self) -> int:
        return sum(d.num_episodes for d in self._datasets)

    @property
    def tolerance_s(self) -> float:
        return 1 / self.fps - 1e-4

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        dataset_idx = np.searchsorted(self.cumulative_sizes, idx, side="right").item() - 1
        local_idx = (idx - self.cumulative_sizes[dataset_idx]).item()
        item = self._datasets[dataset_idx][local_idx]

        # Identify which repo this sample came from
        repo_id = self.datasets_repo_ids[dataset_idx]

        # --- NEW: apply mapping and ensure union of image keys ---
        item = self._apply_feature_mapping(item, repo_id)
        item = self._ensure_union_image_keys(item)

        # annotate dataset index for downstream
        item["dataset_index"] = torch.tensor(dataset_idx)

        # Pad vector features to max dims using meta (unchanged)
        item = create_padded_features(item, self.meta.features)

        # Drop any disabled (including original source keys we remapped away)
        for data_key in self.disabled_features:
            if data_key in item:
                del item[data_key]
        for k in IGNORED_KEYS:
            if k in item:
                item.pop(k)
        # Convert any datasets.Image still present to tensor
        if self.image_transforms is not None:
            for cam in [k for k in item.keys() if self._is_image_key_like(k)]:
                val = item[cam]
                if not torch.is_tensor(val):
                    item[cam] = self.image_transforms(val)
        # 🔑 Pad actions if too short
        if "actions" in item and self.max_action_dim is not None:
            act = item["actions"]
            if act.shape[-1] < self.max_action_dim:
                pad_len = self.max_action_dim - act.shape[-1]
                item["actions"] = torch.cat([act, torch.zeros(pad_len, dtype=act.dtype)], dim=-1)
                item["actions_padding_mask"] = torch.cat(
                    [torch.zeros_like(act, dtype=torch.bool), torch.ones(pad_len, dtype=torch.bool)],
                    dim=-1,
                )

        # pad obs_state if too short
        if "obs_state" in item and self.max_state_dim is not None:
            st = item["obs_state"]
            if st.shape[-1] < self.max_state_dim:
                pad_len = self.max_state_dim - st.shape[-1]
                item["obs_state"] = torch.cat([st, torch.zeros(pad_len, dtype=st.dtype)], dim=-1)
                item["obs_state_padding_mask"] = torch.cat(
                    [torch.zeros_like(st, dtype=torch.bool), torch.ones(pad_len, dtype=torch.bool)],
                    dim=-1,
                )
        # actions
        if "actions" in item and self.max_action_dim is not None:
            act = item["actions"]
            if act.shape[-1] < self.max_action_dim:
                pad_len = self.max_action_dim - act.shape[-1]
                item["actions"] = torch.cat([act, torch.zeros(pad_len, dtype=act.dtype)], dim=-1)
                mask = torch.cat(
                    [torch.zeros_like(act, dtype=torch.bool), torch.ones(pad_len, dtype=torch.bool)],
                    dim=-1,
                )
            else:
                mask = torch.zeros(self.max_action_dim, dtype=torch.bool)  # 👈 all False if no padding
            item["actions_padding_mask"] = mask
        # obs state
        if "obs_state" in item and self.max_state_dim is not None:
            st = item["obs_state"]
            if st.shape[-1] < self.max_state_dim:
                pad_len = self.max_state_dim - st.shape[-1]
                item["obs_state"] = torch.cat([st, torch.zeros(pad_len, dtype=st.dtype)], dim=-1)
                mask = torch.cat(
                    [torch.zeros_like(st, dtype=torch.bool), torch.ones(pad_len, dtype=torch.bool)],
                    dim=-1,
                )
            else:
                mask = torch.zeros(self.max_state_dim, dtype=torch.bool)  # 👈 always add mask
            item["obs_state_padding_mask"] = mask

        return item

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  Repository IDs: '{self.repo_ids}',\n"
            f"  Number of Samples: {self.num_frames},\n"
            f"  Number of Episodes: {self.num_episodes},\n"
            f"  Type: {'video (.mp4)' if self.video else 'image (.png)'},\n"
            f"  Recorded Frames per Second: {self.fps},\n"
            f"  Camera Keys: {self.camera_keys},\n"
            f"  Video Frame Keys: {self.video_frame_keys if self.video else 'N/A'},\n"
            f"  Transformations: {self.image_transforms},\n"
            f")"
        )

def keep_datasets_with_the_same_features_per_robot_type(ls_datasets: list) -> list:
    """
    Filters datasets to only keep those with consistent feature shapes per robot type.

    Args:
        ls_datasets (List): List of datasets, each with a `meta.info['robot_type']`
            and `meta.episodes_stats` dictionary.

    Returns:
        List: Filtered list of datasets with consistent feature shapes.
    """
    robot_types = {ds.meta.info["robot_type"] for ds in ls_datasets}
    datasets_to_remove = set()

    for robot_type in robot_types:
        # Collect all stats dicts for this robot type
        stats_list = [
            ep_stats
            for ds in ls_datasets
            if ds.meta.info["robot_type"] == robot_type
            for ep_stats in episode_stats_values(ds.meta)
        ]
        if not stats_list:
            continue

        # Determine the most common shape for each key
        all_keys = {key for stats in stats_list for key in stats}
        for ds in ls_datasets:
            if ds.meta.info["robot_type"] != robot_type:
                continue
            for key in all_keys:
                shape_counter = defaultdict(int)

                for stats in stats_list:
                    value = stats.get(key)
                    if (
                        value and "mean" in value and isinstance(value["mean"], (torch.Tensor, np.ndarray))
                    ):  # FIXME(mshukor): check all stats; min, mean, max
                        shape_counter[value["mean"].shape] += 1
                if not shape_counter:
                    continue

                # Identify the most frequent shape
                main_shape = max(shape_counter, key=shape_counter.get)
                # Flag datasets that don't match the main shape
                # for ds in ls_datasets:
                first_ep_stats = next(iter(episode_stats_values(ds.meta)), None)
                if not first_ep_stats:
                    continue
                value = first_ep_stats.get(key)
                if (
                    value
                    and "mean" in value
                    and isinstance(value["mean"], (torch.Tensor, np.ndarray))
                    and value["mean"].shape != main_shape
                ):
                    datasets_to_remove.add(ds)
                    break

    # Filter out inconsistent datasets
    datasets_maks = [ds not in datasets_to_remove for ds in ls_datasets]
    filtered_datasets = [ds for ds in ls_datasets if ds not in datasets_to_remove]
    print(
        f"Keeping {len(filtered_datasets)} datasets. Removed {len(datasets_to_remove)} inconsistent ones. Inconsistent datasets:\n{datasets_to_remove}"
    )
    return filtered_datasets, datasets_maks


def aggregate_stats_per_robot_type(ls_datasets) -> dict[str, dict[str, torch.Tensor]]:
    """Aggregate stats of multiple LeRobot datasets into multiple set of stats per robot type.

    The final stats will have the union of all data keys from each of the datasets.

    The final stats will have the union of all data keys from each of the datasets. For instance:
    - new_max = max(max_dataset_0, max_dataset_1, ...)
    - new_min = min(min_dataset_0, min_dataset_1, ...)
    - new_mean = (mean of all data)
    - new_std = (std of all data)
    """

    robot_types = {ds.meta.info["robot_type"] for ds in ls_datasets}
    stats = {robot_type: {} for robot_type in robot_types}
    for robot_type in robot_types:
        robot_type_datasets = []
        for ds in ls_datasets:
            if ds.meta.info["robot_type"] == robot_type:
                robot_type_datasets.extend(list(episode_stats_values(ds.meta)))
        # robot_type_datasets = [list(ds.episodes_stats.values()) for ds in ls_datasets if ds.meta.info["robot_type"] == robot_type]
        stat = aggregate_stats(robot_type_datasets)
        stats[robot_type] = stat
    return stats

def reshape_features_to_max_dim(features: dict, reshape_dim: int = -1, keys_to_max_dim: dict = {}) -> dict:
    """Reshape features to have a maximum dimension of `max_dim`."""
    reshaped_features = {}
    for key in features:
        if key in keys_to_max_dim and keys_to_max_dim[key] is not None:
            reshaped_features[key] = features[key]
            shape = list(features[key]["shape"])
            if any([k in key for k in [OBS_IMAGE, OBS_IMAGE_2, OBS_IMAGE_3]]):  # Assume square images
                shape[-3] = keys_to_max_dim[key]
                shape[-2] = keys_to_max_dim[key]
            else:
                shape[reshape_dim] = keys_to_max_dim[key]
            reshaped_features[key]["shape"] = tuple(shape)
        else:
            reshaped_features[key] = features[key]
    return reshaped_features

def create_padded_features(item: dict, features: dict = {}):
    for key, ft in features.items():
        if any([k in key for k in ["cam", "effort", "absolute"]]):  # FIXME(mshukor): temporary hack
            continue
        shape = ft["shape"]
        if len(shape) == 3:  # images to torch format (C, H, W)
            shape = (shape[2], shape[0], shape[1])
        if len(shape) == 1 and shape[0] == 1:  # ft with shape are actually tensor(ele)
            shape = []
        if key not in item:
            dtype = str_to_torch_dtype(ft["dtype"])
            item[key] = torch.zeros(shape, dtype=dtype)
            item[f"{key}_padding_mask"] = torch.tensor(0, dtype=torch.int64)
            if "image" in key:  # FIXME(mshukor): support other observations
                item[f"{key}_is_pad"] = torch.BoolTensor([False])
        else:
            item[f"{key}_padding_mask"] = torch.tensor(1, dtype=torch.int64)
    return item

def str_to_torch_dtype(dtype_str):
    """Convert a dtype string to a torch dtype."""
    mapping = {
        "float32": torch.float32,
        "int64": torch.int64,
        "int16": torch.int16,
        "bool": torch.bool,
        "video": torch.float32,  # Assuming video is stored as uint8 images
    }
    return mapping.get(dtype_str, torch.float32)  # Default to float32

def episode_stats_values(meta):
    episodes = meta.episodes.to_pandas().to_dict(orient="records")
    return [
        {k: v for k, v in ep.items() if isinstance(v, dict) and "mean" in v}
        for ep in episodes
    ]
