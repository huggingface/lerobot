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
from pathlib import Path
from typing import Callable

import datasets
import torch
import torch.utils
from huggingface_hub import snapshot_download

from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.utils import (
    check_delta_timestamps,
    check_timestamps_sync,
    create_dataset_info,
    get_delta_indices,
    get_episode_data_index,
    get_hub_safe_version,
    load_hf_dataset,
    load_metadata,
)
from lerobot.common.datasets.video_utils import VideoFrame, decode_video_frames_torchvision
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
        video_backend: str | None = None,
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
        │   │   ├── train-00000-of-03603.parquet
        │   │   ├── train-00001-of-03603.parquet
        │   │   ├── train-00002-of-03603.parquet
        │   │   └── ...
        │   ├── chunk-001
        │   │   ├── train-01000-of-03603.parquet
        │   │   ├── train-01001-of-03603.parquet
        │   │   ├── train-01002-of-03603.parquet
        │   │   └── ...
        │   └── ...
        ├── meta
        │   ├── episodes.jsonl
        │   ├── info.json
        │   ├── stats.json
        │   └── tasks.json
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
        self.download_videos = download_videos
        self.video_backend = video_backend if video_backend is not None else "pyav"
        self.delta_indices = None

        # Load metadata
        self.root.mkdir(exist_ok=True, parents=True)
        self._version = get_hub_safe_version(repo_id, CODEBASE_VERSION)
        self.download_metadata()
        self.info, self.episode_dicts, self.stats, self.tasks = load_metadata(self.root)

        # Load actual data
        self.download_episodes()
        self.hf_dataset = load_hf_dataset(self.root, self.data_path, self.total_episodes, self.episodes)
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

    def download_metadata(self) -> None:
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self._version,
            local_dir=self.root,
            allow_patterns="meta/",
        )

    def download_episodes(self) -> None:
        """Downloads the dataset from the given 'repo_id' at the provided version. If 'episodes' is given, this
        will only download those episodes (selected by their episode_index). If 'episodes' is None, the whole
        dataset will be downloaded. Thanks to the behavior of snapshot_download, if the files are already present
        in 'local_dir', they won't be downloaded again.
        """
        # TODO(rcadene, aliberts): implement faster transfer
        # https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads
        files = None
        ignore_patterns = None if self.download_videos else "videos/"
        if self.episodes is not None:
            files = [
                self.data_path.format(episode_index=ep_idx, total_episodes=self.total_episodes)
                for ep_idx in self.episodes
            ]
            if len(self.video_keys) > 0 and self.download_videos:
                video_files = [
                    self.videos_path.format(video_key=vid_key, episode_index=ep_idx)
                    for vid_key in self.video_keys
                    for ep_idx in self.episodes
                ]
                files += video_files

        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self._version,
            local_dir=self.root,
            allow_patterns=files,
            ignore_patterns=ignore_patterns,
        )

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
        """Number of samples/frames."""
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
        self.info.get("shapes")

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
            video_path = self.root / self.videos_path.format(video_key=vid_key, episode_index=ep_idx)
            frames = decode_video_frames_torchvision(
                video_path, query_ts, self.tolerance_s, self.video_backend
            )
            item[vid_key] = frames

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
            image_keys = self.camera_keys if self.download_videos else self.image_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])

        return item

    def write_info(self) -> None:
        with open(self.root / "meta/info.json", "w") as f:
            json.dump(self.info, f, indent=4, ensure_ascii=False)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  Repository ID: '{self.repo_id}',\n"
            f"  Number of Samples: {self.num_samples},\n"
            f"  Number of Episodes: {self.num_episodes},\n"
            f"  Type: {'video (.mp4)' if self.video else 'image (.png)'},\n"
            f"  Recorded Frames per Second: {self.fps},\n"
            f"  Camera Keys: {self.camera_keys},\n"
            f"  Video Frame Keys: {self.camera_keys if self.video else 'N/A'},\n"
            f"  Transformations: {self.image_transforms},\n"
            f"  Codebase Version: {self.info.get('codebase_version', '< v1.6')},\n"
            f")"
        )

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        robot: Robot,
        root: Path | None = None,
        tolerance_s: float = 1e-4,
    ) -> "LeRobotDataset":
        """Create a LeRobot Dataset from scratch in order to record data."""
        obj = cls.__new__(cls)
        obj.repo_id = repo_id
        obj.root = root if root is not None else LEROBOT_HOME / repo_id
        obj._version = CODEBASE_VERSION

        obj.root.mkdir(exist_ok=True, parents=True)
        obj.info = create_dataset_info(obj._version, fps, robot)
        obj.write_info()
        obj.fps = fps

        # obj.episodes = None
        # obj.image_transforms = None
        # obj.delta_timestamps = None
        # obj.episode_data_index = episode_data_index
        # obj.stats = stats
        # obj.info = info if info is not None else {}
        # obj.videos_dir = videos_dir
        # obj.video_backend = video_backend if video_backend is not None else "pyav"
        return obj


class MultiLeRobotDataset(torch.utils.data.Dataset):
    """A dataset consisting of multiple underlying `LeRobotDataset`s.

    The underlying `LeRobotDataset`s are effectively concatenated, and this class adopts much of the API
    structure of `LeRobotDataset`.
    """

    def __init__(
        self,
        repo_ids: list[str],
        root: Path | None = LEROBOT_HOME,
        split: str = "train",
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
                root=root,
                split=split,
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
        self.split = split
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
            f"  Split: '{self.split}',\n"
            f"  Number of Samples: {self.num_samples},\n"
            f"  Number of Episodes: {self.num_episodes},\n"
            f"  Type: {'video (.mp4)' if self.video else 'image (.png)'},\n"
            f"  Recorded Frames per Second: {self.fps},\n"
            f"  Camera Keys: {self.camera_keys},\n"
            f"  Video Frame Keys: {self.video_frame_keys if self.video else 'N/A'},\n"
            f"  Transformations: {self.image_transforms},\n"
            f")"
        )
