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
import logging
import os
from itertools import accumulate
from pathlib import Path
from typing import Callable

import datasets
import torch
import torch.utils

from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.utils import (
    download_episodes,
    get_hub_safe_version,
    load_hf_dataset,
    load_info,
    load_previous_and_future_frames,
    load_stats,
    load_tasks,
)
from lerobot.common.datasets.video_utils import VideoFrame, load_from_videos

# For maintainers, see lerobot/common/datasets/push_dataset_to_hub/CODEBASE_VERSION.md
CODEBASE_VERSION = "v2.0"
LEROBOT_HOME = Path(os.getenv("LEROBOT_HOME", "~/.cache/huggingface/lerobot")).expanduser()


class LeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        root: Path | None = None,
        episodes: list[int] | None = None,
        split: str = "train",
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        video_backend: str | None = None,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.root = root if root is not None else LEROBOT_HOME / repo_id
        self.split = split
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.video_backend = video_backend if video_backend is not None else "pyav"

        # Load metadata
        self.root.mkdir(exist_ok=True, parents=True)
        self._version = get_hub_safe_version(repo_id, CODEBASE_VERSION)
        self.info = load_info(repo_id, self._version, self.root)
        self.stats = load_stats(repo_id, self._version, self.root)
        self.tasks = load_tasks(repo_id, self._version, self.root)

        # Load actual data
        download_episodes(
            repo_id,
            self._version,
            self.root,
            self.data_path,
            self.video_keys,
            self.num_episodes,
            self.episodes,
            self.videos_path,
        )
        self.hf_dataset = load_hf_dataset(self.root, self.data_path, self.total_episodes, self.episodes)
        self.episode_data_index = self.get_episode_data_index()

        # TODO(aliberts):
        # - [ ] Update __get_item__
        # - [ ] Add self.consolidate() for:
        #     - [ ] Sanity checks (episodes num, shapes, files, etc.)
        #     - [ ] Update episode_index (arg update=True)
        #     - [ ] Update info.json (arg update=True)

        # TODO(aliberts): remove (deprecated)
        # if split == "train":
        #     self.episode_data_index = load_episode_data_index(self.episodes, self.episode_list)
        # else:
        #     self.episode_data_index = calculate_episode_data_index(self.hf_dataset)
        #     self.hf_dataset = reset_episode_index(self.hf_dataset)
        # if self.video:
        #     self.videos_dir = load_videos(repo_id, CODEBASE_VERSION, root)

    @property
    def data_path(self) -> str:
        """Formattable string for the parquet files."""
        return self.info["data_path"]

    @property
    def videos_path(self) -> str | None:
        """Formattable string for the video files."""
        return self.info["videos"]["videos_path"] if len(self.video_keys) > 0 else None

    @property
    def episode_dicts(self) -> list[dict]:
        """List of dictionary containing information for each episode, indexed by episode_index."""
        return self.info["episodes"]

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
        """Keys to access image and video streams from cameras."""
        return self.image_keys + self.video_keys

    @property
    def video_frame_keys(self) -> list[str]:
        """Keys to access video frames that requires to be decoded into images.

        Note: It is empty if the dataset contains images only,
        or equal to `self.cameras` if the dataset contains videos only,
        or can even be a subset of `self.cameras` in a case of a mixed image/video dataset.
        """
        video_frame_keys = []
        for key, feats in self.hf_dataset.features.items():
            if isinstance(feats, VideoFrame):
                video_frame_keys.append(key)
        return video_frame_keys

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
    def tolerance_s(self) -> float:
        """Tolerance in seconds used to discard loaded frames when their timestamps
        are not close enough from the requested frames. It is only used when `delta_timestamps`
        is provided or when loading video frames from mp4 files.
        """
        # 1e-4 to account for possible numerical error
        return 1 / self.fps - 1e-4

    @property
    def shapes(self) -> dict:
        """Shapes for the different features."""
        self.info.get("shapes")

    def get_episode_data_index(self) -> dict[str, torch.Tensor]:
        episode_lengths = {ep_idx: ep_dict["length"] for ep_idx, ep_dict in enumerate(self.episode_dicts)}
        if self.episodes is not None:
            episode_lengths = {ep_idx: episode_lengths[ep_idx] for ep_idx in self.episodes}

        cumulative_lenghts = list(accumulate(episode_lengths.values()))
        return {
            "from": torch.LongTensor([0] + cumulative_lenghts[:-1]),
            "to": torch.LongTensor(cumulative_lenghts),
        }

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]

        if self.delta_timestamps is not None:
            item = load_previous_and_future_frames(
                item,
                self.hf_dataset,
                self.episode_data_index,
                self.delta_timestamps,
                self.tolerance_s,
            )

        if self.video:
            item = load_from_videos(
                item,
                self.video_keys,
                self.videos_dir,
                self.tolerance_s,
                self.video_backend,
            )

        if self.image_transforms is not None:
            for cam in self.camera_keys:
                item[cam] = self.image_transforms(item[cam])

        return item

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  Repository ID: '{self.repo_id}',\n"
            f"  Split: '{self.split}',\n"
            f"  Number of Samples: {self.num_samples},\n"
            f"  Number of Episodes: {self.num_episodes},\n"
            f"  Type: {'video (.mp4)' if self.video else 'image (.png)'},\n"
            f"  Recorded Frames per Second: {self.fps},\n"
            f"  Camera Keys: {self.camera_keys},\n"
            f"  Video Frame Keys: {self.video_frame_keys if self.video else 'N/A'},\n"
            f"  Transformations: {self.image_transforms},\n"
            f"  Codebase Version: {self.info.get('codebase_version', '< v1.6')},\n"
            f")"
        )

    @classmethod
    def from_preloaded(
        cls,
        repo_id: str = "from_preloaded",
        root: Path | None = None,
        split: str = "train",
        transform: callable = None,
        delta_timestamps: dict[list[float]] | None = None,
        # additional preloaded attributes
        hf_dataset=None,
        episode_data_index=None,
        stats=None,
        info=None,
        videos_dir=None,
        video_backend=None,
    ) -> "LeRobotDataset":
        """Create a LeRobot Dataset from existing data and attributes instead of loading from the filesystem.

        It is especially useful when converting raw data into LeRobotDataset before saving the dataset
        on the filesystem or uploading to the hub.

        Note: Meta-data attributes like `repo_id`, `version`, `root`, etc are optional and potentially
        meaningless depending on the downstream usage of the return dataset.
        """
        # create an empty object of type LeRobotDataset
        obj = cls.__new__(cls)
        obj.repo_id = repo_id
        obj.root = root
        obj.split = split
        obj.image_transforms = transform
        obj.delta_timestamps = delta_timestamps
        obj.hf_dataset = hf_dataset
        obj.episode_data_index = episode_data_index
        obj.stats = stats
        obj.info = info if info is not None else {}
        obj.videos_dir = videos_dir
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
