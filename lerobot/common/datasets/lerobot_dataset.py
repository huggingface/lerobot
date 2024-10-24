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
import logging
import os
import shutil
from functools import cached_property
from pathlib import Path
from typing import Callable

import datasets
import pyarrow.parquet as pq
import torch
import torch.utils
from datasets import load_dataset
from huggingface_hub import snapshot_download, upload_folder

from lerobot.common.datasets.compute_stats import aggregate_stats, compute_stats
from lerobot.common.datasets.image_writer import ImageWriter
from lerobot.common.datasets.utils import (
    EPISODES_PATH,
    INFO_PATH,
    STATS_PATH,
    TASKS_PATH,
    _get_info_from_robot,
    append_jsonl,
    check_delta_timestamps,
    check_timestamps_sync,
    check_version_compatibility,
    create_branch,
    create_empty_dataset_info,
    get_delta_indices,
    get_episode_data_index,
    get_hub_safe_version,
    hf_transform_to_torch,
    load_episode_dicts,
    load_info,
    load_stats,
    load_tasks,
    write_json,
    write_stats,
)
from lerobot.common.datasets.video_utils import (
    VideoFrame,
    decode_video_frames_torchvision,
    encode_video_frames,
)
from lerobot.common.robot_devices.robots.utils import Robot

# For maintainers, see lerobot/common/datasets/push_dataset_to_hub/CODEBASE_VERSION.md
CODEBASE_VERSION = "v2.0"
LEROBOT_HOME = Path(os.getenv("LEROBOT_HOME", "~/.cache/huggingface/lerobot")).expanduser()


class LeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        root: Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        download_videos: bool = True,
        local_files_only: bool = False,
        video_backend: str | None = None,
        image_writer: ImageWriter | None = None,
    ):
        """LeRobotDataset encapsulates 3 main things:
            - metadata:
                - info contains various information about the dataset like shapes, keys, fps etc.
                - stats stores the dataset statistics of the different modalities for normalization
                - tasks contains the prompts for each task of the dataset, which can be used for
                  task-conditionned training.
            - hf_dataset (from datasets.Dataset), which will read any values from parquet files.
            - (optional) videos from which frames are loaded to be synchronous with data from parquet files.

        3 modes are available for this class, depending on 3 different use cases:

        1. Your dataset already exists on the Hugging Face Hub at the address
        https://huggingface.co/datasets/{repo_id} and is not on your local disk in the 'root' folder:
            Instantiating this class with this 'repo_id' will download the dataset from that address and load
            it, pending your dataset is compliant with codebase_version v2.0. If your dataset has been created
            before this new format, you will be prompted to convert it using our conversion script from v1.6
            to v2.0, which you can find at lerobot/common/datasets/v2/convert_dataset_v1_to_v2.py.

        2. Your dataset already exists on your local disk in the 'root' folder:
            This is typically the case when you recorded your dataset locally and you may or may not have
            pushed it to the hub yet. Instantiating this class with 'root' will load your dataset directly
            from disk. This can happen while you're offline (no internet connection).

        3. Your dataset doesn't already exists (either on local disk or on the Hub):
            [TODO(aliberts): add classmethod for this case?]


        In terms of files, a typical LeRobotDataset looks like this from its root path:
        .
        ├── data
        │   ├── chunk-000
        │   │   ├── episode_000000.parquet
        │   │   ├── episode_000001.parquet
        │   │   ├── episode_000002.parquet
        │   │   └── ...
        │   ├── chunk-001
        │   │   ├── episode_001000.parquet
        │   │   ├── episode_001001.parquet
        │   │   ├── episode_001002.parquet
        │   │   └── ...
        │   └── ...
        ├── meta
        │   ├── episodes.jsonl
        │   ├── info.json
        │   ├── stats.json
        │   └── tasks.jsonl
        └── videos (optional)
            ├── chunk-000
            │   ├── observation.images.laptop
            │   │   ├── episode_000000.mp4
            │   │   ├── episode_000001.mp4
            │   │   ├── episode_000002.mp4
            │   │   └── ...
            │   ├── observation.images.phone
            │   │   ├── episode_000000.mp4
            │   │   ├── episode_000001.mp4
            │   │   ├── episode_000002.mp4
            │   │   └── ...
            ├── chunk-001
            └── ...

        Note that this file-based structure is designed to be as versatile as possible. The files are split by
        episodes which allows a more granular control over which episodes one wants to use and download. The
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
            download_videos (bool, optional): Flag to download the videos. Note that when set to True but the
                video files are already present on local disk, they won't be downloaded again. Defaults to
                True.
            video_backend (str | None, optional): Video backend to use for decoding videos. There is currently
                a single option which is the pyav decoder used by Torchvision. Defaults to pyav.
        """
        super().__init__()
        self.repo_id = repo_id
        self.root = root if root is not None else LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.video_backend = video_backend if video_backend is not None else "pyav"
        self.delta_indices = None
        self.local_files_only = local_files_only
        self.consolidated = True

        # Unused attributes
        self.image_writer = None
        self.episode_buffer = {}

        # Load metadata
        self.root.mkdir(exist_ok=True, parents=True)
        self.pull_from_repo(allow_patterns="meta/")
        self.info = load_info(self.root)
        self.stats = load_stats(self.root)
        self.tasks = load_tasks(self.root)
        self.episode_dicts = load_episode_dicts(self.root)

        # Check version
        check_version_compatibility(self.repo_id, self._version, CODEBASE_VERSION)

        # Load actual data
        self.download_episodes(download_videos)
        self.hf_dataset = self.load_hf_dataset()
        self.episode_data_index = get_episode_data_index(self.episodes, self.episode_dicts)

        # Check timestamps
        check_timestamps_sync(self.hf_dataset, self.episode_data_index, self.fps, self.tolerance_s)

        # Setup delta_indices
        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

        # TODO(aliberts):
        # - [X] Move delta_timestamp logic outside __get_item__
        # - [X] Update __get_item__
        # - [/] Add doc
        # - [ ] Add self.add_frame()
        # - [ ] Add self.consolidate() for:
        #     - [X] Check timestamps sync
        #     - [ ] Sanity checks (episodes num, shapes, files, etc.)
        #     - [ ] Update episode_index (arg update=True)
        #     - [ ] Update info.json (arg update=True)

    @cached_property
    def _hub_version(self) -> str | None:
        return None if self.local_files_only else get_hub_safe_version(self.repo_id, CODEBASE_VERSION)

    @property
    def _version(self) -> str:
        """Codebase version used to create this dataset."""
        return self.info["codebase_version"]

    def push_to_hub(self, push_videos: bool = True) -> None:
        if not self.consolidated:
            raise RuntimeError(
                "You are trying to upload to the hub a LeRobotDataset that has not been consolidated yet."
                "Please call the dataset 'consolidate()' method first."
            )
        ignore_patterns = ["images/"]
        if not push_videos:
            ignore_patterns.append("videos/")

        upload_folder(
            repo_id=self.repo_id,
            folder_path=self.root,
            repo_type="dataset",
            ignore_patterns=ignore_patterns,
        )
        create_branch(repo_id=self.repo_id, branch=CODEBASE_VERSION, repo_type="dataset")

    def pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
    ) -> None:
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self._hub_version,
            local_dir=self.root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            local_files_only=self.local_files_only,
        )

    def download_episodes(self, download_videos: bool = True) -> None:
        """Downloads the dataset from the given 'repo_id' at the provided version. If 'episodes' is given, this
        will only download those episodes (selected by their episode_index). If 'episodes' is None, the whole
        dataset will be downloaded. Thanks to the behavior of snapshot_download, if the files are already present
        in 'local_dir', they won't be downloaded again.
        """
        # TODO(rcadene, aliberts): implement faster transfer
        # https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads
        files = None
        ignore_patterns = None if download_videos else "videos/"
        if self.episodes is not None:
            files = [str(self.get_data_file_path(ep_idx)) for ep_idx in self.episodes]
            if len(self.video_keys) > 0 and download_videos:
                video_files = [
                    str(self.get_video_file_path(ep_idx, vid_key))
                    for vid_key in self.video_keys
                    for ep_idx in self.episodes
                ]
                files += video_files

        self.pull_from_repo(allow_patterns=files, ignore_patterns=ignore_patterns)

    def load_hf_dataset(self) -> datasets.Dataset:
        """hf_dataset contains all the observations, states, actions, rewards, etc."""
        if self.episodes is None:
            path = str(self.root / "data")
            hf_dataset = load_dataset("parquet", data_dir=path, split="train")
        else:
            files = [str(self.root / self.get_data_file_path(ep_idx)) for ep_idx in self.episodes]
            hf_dataset = load_dataset("parquet", data_files=files, split="train")

        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def get_data_file_path(self, ep_index: int) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.data_path.format(episode_chunk=ep_chunk, episode_index=ep_index)
        return Path(fpath)

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.videos_path.format(episode_chunk=ep_chunk, video_key=vid_key, episode_index=ep_index)
        return Path(fpath)

    def get_episode_chunk(self, ep_index: int) -> int:
        return ep_index // self.chunks_size

    @property
    def data_path(self) -> str:
        """Formattable string for the parquet files."""
        return self.info["data_path"]

    @property
    def videos_path(self) -> str | None:
        """Formattable string for the video files."""
        return self.info["videos"]["videos_path"] if len(self.video_keys) > 0 else None

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.info["fps"]

    @property
    def keys(self) -> list[str]:
        """Keys to access non-image data (state, actions etc.)."""
        return self.info["keys"]

    @property
    def image_keys(self) -> list[str]:
        """Keys to access visual modalities stored as images."""
        return self.info["image_keys"]

    @property
    def video_keys(self) -> list[str]:
        """Keys to access visual modalities stored as videos."""
        return self.info["video_keys"]

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access visual modalities (regardless of their storage method)."""
        return self.image_keys + self.video_keys

    @property
    def names(self) -> dict[list[str]]:
        """Names of the various dimensions of vector modalities."""
        return self.info["names"]

    @property
    def num_samples(self) -> int:
        """Number of samples/frames in selected episodes."""
        return len(self.hf_dataset)

    @property
    def num_episodes(self) -> int:
        """Number of episodes selected."""
        return len(self.episodes) if self.episodes is not None else self.total_episodes

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
    def total_chunks(self) -> int:
        """Total number of chunks (groups of episodes)."""
        return self.info["total_chunks"]

    @property
    def chunks_size(self) -> int:
        """Max number of episodes per chunk."""
        return self.info["chunks_size"]

    @property
    def shapes(self) -> dict:
        """Shapes for the different features."""
        return self.info["shapes"]

    @property
    def features(self) -> datasets.Features:
        """Features of the hf_dataset."""
        if self.hf_dataset is not None:
            return self.hf_dataset.features
        elif self.episode_buffer is None:
            raise NotImplementedError(
                "Dataset features must be infered from an existing hf_dataset or episode_buffer."
            )

        features = {}
        for key in self.episode_buffer:
            if key in ["episode_index", "frame_index", "index", "task_index"]:
                features[key] = datasets.Value(dtype="int64")
            elif key in ["next.done", "next.success"]:
                features[key] = datasets.Value(dtype="bool")
            elif key in ["timestamp", "next.reward"]:
                features[key] = datasets.Value(dtype="float32")
            elif key in self.image_keys:
                features[key] = datasets.Image()
            elif key in self.keys:
                features[key] = datasets.Sequence(
                    length=self.shapes[key], feature=datasets.Value(dtype="float32")
                )

        return datasets.Features(features)

    @property
    def task_to_task_index(self) -> dict:
        return {task: task_idx for task_idx, task in self.tasks.items()}

    def get_task_index(self, task: str) -> int:
        """
        Given a task in natural language, returns its task_index if the task already exists in the dataset,
        otherwise creates a new task_index.
        """
        task_index = self.task_to_task_index.get(task, None)
        return task_index if task_index is not None else self.total_tasks

    def current_episode_index(self, idx: int) -> int:
        episode_index = self.hf_dataset["episode_index"][idx]
        if self.episodes is not None:
            # get episode_index from selected episodes
            episode_index = self.episodes.index(episode_index)

        return episode_index

    def episode_length(self, episode_index) -> int:
        """Number of samples/frames for given episode."""
        return self.info["episodes"][episode_index]["length"]

    def _get_query_indices(self, idx: int, ep_idx: int) -> tuple[dict[str, list[int | bool]]]:
        ep_start = self.episode_data_index["from"][ep_idx]
        ep_end = self.episode_data_index["to"][ep_idx]
        query_indices = {
            key: [max(ep_start.item(), min(ep_end.item() - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {  # Pad values outside of current episode range
            f"{key}_is_pad": torch.BoolTensor(
                [(idx + delta < ep_start.item()) | (idx + delta >= ep_end.item()) for delta in delta_idx]
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
        for key in self.video_keys:
            if query_indices is not None and key in query_indices:
                timestamps = self.hf_dataset.select(query_indices[key])["timestamp"]
                query_timestamps[key] = torch.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]

        return query_timestamps

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        return {
            key: torch.stack(self.hf_dataset.select(q_idx)[key])
            for key, q_idx in query_indices.items()
            if key not in self.video_keys
        }

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault. This probably happens because a memory reference to the video loader is created in
        the main process and a subprocess fails to access it.
        """
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            video_path = self.root / self.get_video_file_path(ep_idx, vid_key)
            frames = decode_video_frames_torchvision(
                video_path, query_ts, self.tolerance_s, self.video_backend
            )
            item[vid_key] = frames.squeeze(0)

        return item

    def _add_padding_keys(self, item: dict, padding: dict[str, list[bool]]) -> dict:
        for key, val in padding.items():
            item[key] = torch.BoolTensor(val)
        return item

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx) -> dict:
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()

        query_indices = None
        if self.delta_indices is not None:
            current_ep_idx = self.episodes.index(ep_idx) if self.episodes is not None else ep_idx
            query_indices, padding = self._get_query_indices(idx, current_ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if len(self.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if self.image_transforms is not None:
            image_keys = self.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])

        return item

    def __repr__(self):
        return (
            f"{self.__class__.__name__}\n"
            f"  Repository ID: '{self.repo_id}',\n"
            f"  Selected episodes: {self.episodes},\n"
            f"  Number of selected episodes: {self.num_episodes},\n"
            f"  Number of selected samples: {self.num_samples},\n"
            f"\n{json.dumps(self.info, indent=4)}\n"
        )

    def _create_episode_buffer(self, episode_index: int | None = None) -> dict:
        # TODO(aliberts): Handle resume
        return {
            "size": 0,
            "episode_index": self.total_episodes if episode_index is None else episode_index,
            "task_index": None,
            "frame_index": [],
            "timestamp": [],
            "next.done": [],
            **{key: [] for key in self.keys},
            **{key: [] for key in self.image_keys},
        }

    def add_frame(self, frame: dict) -> None:
        """
        This function only adds the frame to the episode_buffer. Apart from images — which are written in a
        temporary directory — nothing is written to disk. To save those frames, the 'add_episode()' method
        then needs to be called.
        """
        frame_index = self.episode_buffer["size"]
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(frame_index / self.fps)
        self.episode_buffer["next.done"].append(False)

        # Save all observed modalities except images
        for key in self.keys:
            self.episode_buffer[key].append(frame[key])

        self.episode_buffer["size"] += 1

        if self.image_writer is None:
            return

        # Save images
        for cam_key in self.camera_keys:
            img_path = self.image_writer.get_image_file_path(
                episode_index=self.episode_buffer["episode_index"], image_key=cam_key, frame_index=frame_index
            )
            if frame_index == 0:
                img_path.parent.mkdir(parents=True, exist_ok=True)

            self.image_writer.async_save_image(
                image=frame[cam_key],
                file_path=img_path,
            )
            if cam_key in self.image_keys:
                self.episode_buffer[cam_key].append(str(img_path))

    def add_episode(self, task: str, encode_videos: bool = False) -> None:
        """
        This will save to disk the current episode in self.episode_buffer. Note that since it affects files on
        disk, it sets self.consolidated to False to ensure proper consolidation later on before uploading to
        the hub.

        Use 'encode_videos' if you want to encode videos during the saving of each episode. Otherwise,
        you can do it later with dataset.consolidate(). This is to give more flexibility on when to spend
        time for video encoding.
        """
        episode_length = self.episode_buffer.pop("size")
        episode_index = self.episode_buffer["episode_index"]
        if episode_index != self.total_episodes:
            # TODO(aliberts): Add option to use existing episode_index
            raise NotImplementedError()

        task_index = self.get_task_index(task)
        self.episode_buffer["next.done"][-1] = True

        for key in self.episode_buffer:
            if key in self.image_keys:
                continue
            if key in self.keys:
                self.episode_buffer[key] = torch.stack(self.episode_buffer[key])
            elif key == "episode_index":
                self.episode_buffer[key] = torch.full((episode_length,), episode_index)
            elif key == "task_index":
                self.episode_buffer[key] = torch.full((episode_length,), task_index)
            else:
                self.episode_buffer[key] = torch.tensor(self.episode_buffer[key])

        self.episode_buffer["index"] = torch.arange(self.total_frames, self.total_frames + episode_length)
        self._save_episode_to_metadata(episode_index, episode_length, task, task_index)
        self._save_episode_table(episode_index)

        if encode_videos and len(self.video_keys) > 0:
            self.encode_videos()

        # Reset the buffer
        self.episode_buffer = self._create_episode_buffer()
        self.consolidated = False

    def _save_episode_table(self, episode_index: int) -> None:
        ep_dataset = datasets.Dataset.from_dict(self.episode_buffer, features=self.features, split="train")
        ep_table = ep_dataset._data.table
        ep_data_path = self.root / self.get_data_file_path(ep_index=episode_index)
        ep_data_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(ep_table, ep_data_path)

    def _save_episode_to_metadata(
        self, episode_index: int, episode_length: int, task: str, task_index: int
    ) -> None:
        self.info["total_episodes"] += 1
        self.info["total_frames"] += episode_length

        if task_index not in self.tasks:
            self.info["total_tasks"] += 1
            self.tasks[task_index] = task
            task_dict = {
                "task_index": task_index,
                "task": task,
            }
            append_jsonl(task_dict, self.root / TASKS_PATH)

        chunk = self.get_episode_chunk(episode_index)
        if chunk >= self.total_chunks:
            self.info["total_chunks"] += 1

        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}
        self.info["total_videos"] += len(self.video_keys)
        write_json(self.info, self.root / INFO_PATH)

        episode_dict = {
            "episode_index": episode_index,
            "tasks": [task],
            "length": episode_length,
        }
        self.episode_dicts.append(episode_dict)
        append_jsonl(episode_dict, self.root / EPISODES_PATH)

    def clear_episode_buffer(self) -> None:
        episode_index = self.episode_buffer["episode_index"]
        if self.image_writer is not None:
            for cam_key in self.camera_keys:
                img_dir = self.image_writer.get_episode_dir(episode_index, cam_key)
                if img_dir.is_dir():
                    shutil.rmtree(img_dir)

        # Reset the buffer
        self.episode_buffer = self._create_episode_buffer()

    def start_image_writter(self, num_processes: int = 0, num_threads: int = 1) -> None:
        if isinstance(self.image_writer, ImageWriter):
            logging.warning(
                "You are starting a new ImageWriter that is replacing an already exising one in the dataset."
            )

        self.image_writer = ImageWriter(
            write_dir=self.root / "images",
            num_processes=num_processes,
            num_threads=num_threads,
        )

    def stop_image_writter(self) -> None:
        """
        Whenever wrapping this dataset inside a parallelized DataLoader, this needs to be called first to
        remove the image_write in order for the LeRobotDataset object to be pickleable and parallelized.
        """
        if self.image_writer is not None:
            self.image_writer.stop()
            self.image_writer = None

    def encode_videos(self) -> None:
        # Use ffmpeg to convert frames stored as png into mp4 videos
        for episode_index in range(self.num_episodes):
            for key in self.video_keys:
                # TODO: create video_buffer to store the state of encoded/unencoded videos and remove the need
                # to call self.image_writer here
                tmp_imgs_dir = self.image_writer.get_episode_dir(episode_index, key)
                video_path = self.root / self.get_video_file_path(episode_index, key)
                if video_path.is_file():
                    # Skip if video is already encoded. Could be the case when resuming data recording.
                    continue
                # note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
                # since video encoding with ffmpeg is already using multithreading.
                encode_video_frames(tmp_imgs_dir, video_path, self.fps, overwrite=True)

    def consolidate(self, run_compute_stats: bool = True, keep_image_files: bool = False) -> None:
        self.hf_dataset = self.load_hf_dataset()
        self.episode_data_index = get_episode_data_index(self.episodes, self.episode_dicts)
        check_timestamps_sync(self.hf_dataset, self.episode_data_index, self.fps, self.tolerance_s)

        if len(self.video_keys) > 0:
            self.encode_videos()

        if not keep_image_files and self.image_writer is not None:
            shutil.rmtree(self.image_writer.dir)

        if run_compute_stats:
            self.stop_image_writter()
            self.stats = compute_stats(self)
            write_stats(self.stats, self.root / STATS_PATH)
            self.consolidated = True
        else:
            logging.warning(
                "Skipping computation of the dataset statistics, dataset is not fully consolidated."
            )

        # TODO(aliberts)
        # - [ ] add video info in info.json
        # Sanity checks:
        # - [ ] shapes
        # - [ ] ep_lenghts
        # - [ ] number of files

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        root: Path | None = None,
        robot: Robot | None = None,
        robot_type: str | None = None,
        keys: list[str] | None = None,
        image_keys: list[str] | None = None,
        video_keys: list[str] = None,
        shapes: dict | None = None,
        names: dict | None = None,
        tolerance_s: float = 1e-4,
        image_writer_processes: int = 0,
        image_writer_threads_per_camera: int = 0,
        use_videos: bool = True,
        video_backend: str | None = None,
    ) -> "LeRobotDataset":
        """Create a LeRobot Dataset from scratch in order to record data."""
        obj = cls.__new__(cls)
        obj.repo_id = repo_id
        obj.root = root if root is not None else LEROBOT_HOME / repo_id
        obj.tolerance_s = tolerance_s
        obj.image_writer = None

        if robot is not None:
            robot_type, keys, image_keys, video_keys, shapes, names = _get_info_from_robot(robot, use_videos)
            if not all(cam.fps == fps for cam in robot.cameras.values()):
                logging.warning(
                    f"Some cameras in your {robot.robot_type} robot don't have an fps matching the fps of your dataset."
                    "In this case, frames from lower fps cameras will be repeated to fill in the blanks"
                )
            if len(robot.cameras) > 0 and (image_writer_processes or image_writer_threads_per_camera):
                obj.start_image_writter(
                    image_writer_processes, image_writer_threads_per_camera * robot.num_cameras
                )
        elif (
            robot_type is None
            or keys is None
            or image_keys is None
            or video_keys is None
            or shapes is None
            or names is None
        ):
            raise ValueError(
                "Dataset info (robot_type, keys, shapes...) must either come from a Robot or explicitly passed upon creation."
            )

        if len(video_keys) > 0 and not use_videos:
            raise ValueError

        obj.tasks, obj.stats, obj.episode_dicts = {}, {}, []
        obj.info = create_empty_dataset_info(
            CODEBASE_VERSION, fps, robot_type, keys, image_keys, video_keys, shapes, names
        )
        write_json(obj.info, obj.root / INFO_PATH)

        # TODO(aliberts, rcadene, alexander-soare): Merge this with OnlineBuffer/DataBuffer
        obj.episode_buffer = obj._create_episode_buffer()

        # This bool indicates that the current LeRobotDataset instance is in sync with the files on disk. It
        # is used to know when certain operations are need (for instance, computing dataset statistics). In
        # order to be able to push the dataset to the hub, it needs to be consolidated first by calling
        # self.consolidate().
        obj.consolidated = True

        obj.episodes = None
        obj.hf_dataset = None
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.delta_indices = None
        obj.local_files_only = True
        obj.episode_data_index = None
        obj.video_backend = video_backend if video_backend is not None else "pyav"
        return obj


class MultiLeRobotDataset(torch.utils.data.Dataset):
    """A dataset consisting of multiple underlying `LeRobotDataset`s.

    The underlying `LeRobotDataset`s are effectively concatenated, and this class adopts much of the API
    structure of `LeRobotDataset`.
    """

    def __init__(
        self,
        repo_ids: list[str],
        root: Path | None = None,
        episodes: dict | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        video_backend: str | None = None,
    ):
        super().__init__()
        self.repo_ids = repo_ids
        # Construct the underlying datasets passing everything but `transform` and `delta_timestamps` which
        # are handled by this class.
        self._datasets = [
            LeRobotDataset(
                repo_id,
                root=root / repo_id if root is not None else None,
                episodes=episodes[repo_id] if episodes is not None else None,
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                video_backend=video_backend,
            )
            for repo_id in repo_ids
        ]
        # Check that some properties are consistent across datasets. Note: We may relax some of these
        # consistency requirements in future iterations of this class.
        for repo_id, dataset in zip(self.repo_ids, self._datasets, strict=True):
            if dataset.info != self._datasets[0].info:
                raise ValueError(
                    f"Detected a mismatch in dataset info between {self.repo_ids[0]} and {repo_id}. This is "
                    "not yet supported."
                )
        # Disable any data keys that are not common across all of the datasets. Note: we may relax this
        # restriction in future iterations of this class. For now, this is necessary at least for being able
        # to use PyTorch's default DataLoader collate function.
        self.disabled_data_keys = set()
        intersection_data_keys = set(self._datasets[0].hf_dataset.features)
        for dataset in self._datasets:
            intersection_data_keys.intersection_update(dataset.hf_dataset.features)
        if len(intersection_data_keys) == 0:
            raise RuntimeError(
                "Multiple datasets were provided but they had no keys common to all of them. The "
                "multi-dataset functionality currently only keeps common keys."
            )
        for repo_id, dataset in zip(self.repo_ids, self._datasets, strict=True):
            extra_keys = set(dataset.hf_dataset.features).difference(intersection_data_keys)
            logging.warning(
                f"keys {extra_keys} of {repo_id} were disabled as they are not contained in all the "
                "other datasets."
            )
            self.disabled_data_keys.update(extra_keys)

        self.root = root
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.stats = aggregate_stats(self._datasets)

    @property
    def repo_id_to_index(self):
        """Return a mapping from dataset repo_id to a dataset index automatically created by this class.

        This index is incorporated as a data key in the dictionary returned by `__getitem__`.
        """
        return {repo_id: i for i, repo_id in enumerate(self.repo_ids)}

    @property
    def repo_index_to_id(self):
        """Return the inverse mapping if repo_id_to_index."""
        return {v: k for k, v in self.repo_id_to_index}

    @property
    def fps(self) -> int:
        """Frames per second used during data collection.

        NOTE: Fow now, this relies on a check in __init__ to make sure all sub-datasets have the same info.
        """
        return self._datasets[0].info["fps"]

    @property
    def video(self) -> bool:
        """Returns True if this dataset loads video frames from mp4 files.

        Returns False if it only loads images from png files.

        NOTE: Fow now, this relies on a check in __init__ to make sure all sub-datasets have the same info.
        """
        return self._datasets[0].info.get("video", False)

    @property
    def features(self) -> datasets.Features:
        features = {}
        for dataset in self._datasets:
            features.update({k: v for k, v in dataset.features.items() if k not in self.disabled_data_keys})
        return features

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access image and video stream from cameras."""
        keys = []
        for key, feats in self.features.items():
            if isinstance(feats, (datasets.Image, VideoFrame)):
                keys.append(key)
        return keys

    @property
    def video_frame_keys(self) -> list[str]:
        """Keys to access video frames that requires to be decoded into images.

        Note: It is empty if the dataset contains images only,
        or equal to `self.cameras` if the dataset contains videos only,
        or can even be a subset of `self.cameras` in a case of a mixed image/video dataset.
        """
        video_frame_keys = []
        for key, feats in self.features.items():
            if isinstance(feats, VideoFrame):
                video_frame_keys.append(key)
        return video_frame_keys

    @property
    def num_samples(self) -> int:
        """Number of samples/frames."""
        return sum(d.num_samples for d in self._datasets)

    @property
    def num_episodes(self) -> int:
        """Number of episodes."""
        return sum(d.num_episodes for d in self._datasets)

    @property
    def tolerance_s(self) -> float:
        """Tolerance in seconds used to discard loaded frames when their timestamps
        are not close enough from the requested frames. It is only used when `delta_timestamps`
        is provided or when loading video frames from mp4 files.
        """
        # 1e-4 to account for possible numerical error
        return 1 / self.fps - 1e-4

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        # Determine which dataset to get an item from based on the index.
        start_idx = 0
        dataset_idx = 0
        for dataset in self._datasets:
            if idx >= start_idx + dataset.num_samples:
                start_idx += dataset.num_samples
                dataset_idx += 1
                continue
            break
        else:
            raise AssertionError("We expect the loop to break out as long as the index is within bounds.")
        item = self._datasets[dataset_idx][idx - start_idx]
        item["dataset_index"] = torch.tensor(dataset_idx)
        for data_key in self.disabled_data_keys:
            if data_key in item:
                del item[data_key]

        return item

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  Repository IDs: '{self.repo_ids}',\n"
            f"  Number of Samples: {self.num_samples},\n"
            f"  Number of Episodes: {self.num_episodes},\n"
            f"  Type: {'video (.mp4)' if self.video else 'image (.png)'},\n"
            f"  Recorded Frames per Second: {self.fps},\n"
            f"  Camera Keys: {self.camera_keys},\n"
            f"  Video Frame Keys: {self.video_frame_keys if self.video else 'N/A'},\n"
            f"  Transformations: {self.image_transforms},\n"
            f")"
        )
