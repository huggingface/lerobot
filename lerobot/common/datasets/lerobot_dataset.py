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
from typing import Any, Callable

import datasets
import numpy as np
import torch
import torch.utils
from safetensors.torch import save_file

from lerobot.common.datasets.compute_stats import aggregate_stats
from lerobot.common.datasets.utils import (
    calculate_episode_data_index,
    flatten_dict,
    hf_transform_to_torch,
    load_episode_data_index,
    load_hf_dataset,
    load_info,
    load_previous_and_future_frames,
    load_stats,
    load_videos,
    reset_episode_index,
)
from lerobot.common.datasets.video_utils import VideoFrame, load_from_videos

DATA_DIR = Path(os.environ["DATA_DIR"]) if "DATA_DIR" in os.environ else None
CODEBASE_VERSION = "v1.4"


def make_memmap_safe(**kwargs) -> np.memmap:
    """Make a numpy memmap with checks on available disk space first.

    Expected kwargs are: "filename", "dtype" (must by np.dtype), "mode" and "shape"

    For information on dtypes:
    https://numpy.org/doc/stable/reference/arrays.dtypes.html#arrays-dtypes-constructing
    """
    required_space = kwargs["dtype"].itemsize * np.prod(kwargs["shape"])  # bytes
    stats = os.statvfs(Path(kwargs["filename"]).parent)
    available_space = stats.f_bavail * stats.f_frsize  # bytes
    if required_space >= available_space * 0.8:
        raise RuntimeError(f"You're about to take up {required_space} of {available_space} bytes available.")
    return np.memmap(**kwargs)


class OnlineLeRobotDataset(torch.utils.data.Dataset):
    """
    The actual underlying data structure will have data inserted in a rolling fashion. Always insert
    after the last index, and when you reach the end, wrap around to the start.
    """

    def __init__(
        self,
        write_dir: str | Path,
        data_shapes: dict[str, tuple[int, ...]],
        buffer_capacity: int,
        fps: float,
        delta_timestamps: dict[str, list[float]] | dict[str, torch.Tensor] | None = None,
        use_cache: bool = False,
    ):
        super().__init__()
        self._buffer_capacity = buffer_capacity
        self.delta_timestamps = delta_timestamps
        self._fps = fps
        data_spec = self._make_data_spec(data_shapes, buffer_capacity)
        os.makedirs(write_dir, exist_ok=True)
        self._data = {
            k: make_memmap_safe(
                filename=Path(write_dir) / k,
                dtype=v["dtype"],
                mode="r+",
                shape=tuple(v["shape"]),
            )
            for k, v in data_spec.items()
        }
        self.cache: dict[str, np.ndarray] = {} if use_cache else None

    def add_data(self, data: dict[str, np.ndarray]):
        """Add new data to the end of the buffer.

        Remove data from the start of the buffer if necessary.

        Shift the incoming index and episode_index if necessary. Note that this will be done in place!
        """
        if not set(data) == set(self.data_keys):
            raise ValueError("Missing data keys")
        new_data_length = len(data[self.data_keys[0]])
        if not all(len(data[k]) == new_data_length for k in self.data_keys):
            raise ValueError("All data items should have the same length")

        next_index = self._data["_next_index"][0]

        # Shift the incoming indices if necessary.
        if self.num_samples > 0:
            last_episode_index = self._data["episode_index"][next_index - 1]
            last_data_index = self._data["index"][next_index - 1]
            data["episode_index"] += last_episode_index + 1
            data["index"] += last_data_index + 1

        # Insert the new data starting from next_index. It may be necessary to wrap around to the start.
        n_surplus = max(0, new_data_length - (self._buffer_capacity - next_index))
        for k in self.data_keys:
            if n_surplus == 0:
                slc = slice(next_index, next_index + new_data_length)
                self._data[k][slc] = data[k]
                self._data["_occupancy_mask"][slc] = True
            else:
                self._data[k][next_index:] = data[k][:-n_surplus]
                self._data["_occupancy_mask"][next_index:] = True
                self._data[k][:n_surplus] = data[k][-n_surplus:]
                # Also invalidate overwritten cache indices.
                if self.cache is not None:
                    for i in range(n_surplus):
                        if i in self.cache:
                            del self.cache[i]
        if n_surplus == 0:
            self._data["_next_index"][0] = next_index + new_data_length
        else:
            self._data["_next_index"][0] = n_surplus

    @property
    def data_keys(self) -> list[str]:
        keys = set(self._data)
        keys.remove("_occupancy_mask")
        keys.remove("_next_index")
        return sorted(keys)

    def _make_data_spec(
        self, data_shapes: dict[str, tuple[int, ...]], buffer_capacity: int
    ) -> dict[str, dict[str, Any]]:
        data_spec = {
            "_next_index": {"dtype": np.dtype("int64"), "shape": (1,)},
            "_occupancy_mask": {"dtype": np.dtype("?"), "shape": (buffer_capacity,)},
            "index": {"dtype": np.dtype("int64"), "shape": (buffer_capacity,)},
            "frame_index": {"dtype": np.dtype("int64"), "shape": (buffer_capacity,)},
            "episode_index": {"dtype": np.dtype("int64"), "shape": (buffer_capacity,)},
            "timestamp": {"dtype": np.dtype("float64"), "shape": (buffer_capacity,)},
            **{
                k: {"dtype": np.dtype("float32"), "shape": (buffer_capacity, *v)}
                for k, v in data_shapes.items()
            },
        }
        return data_spec

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def tolerance_s(self) -> float:
        """Tolerance in seconds used to discard loaded frames when their timestamps
        are not close enough from the requested frames. It is only used when `delta_timestamps`
        is provided or when loading video frames from mp4 files.
        """
        # 1e-4 to account for possible numerical error
        return 1 / self.fps - 1e-4

    @property
    def num_episodes(self) -> int:
        return len(np.unique(self._data["episode_index"][self._data["_occupancy_mask"]]))

    @property
    def num_samples(self) -> int:
        return np.count_nonzero(self._data["_occupancy_mask"])

    def __len__(self):
        return self.num_samples

    def _item_to_tensors(self, item: dict) -> dict:
        item_ = {}
        for k, v in item.items():
            if isinstance(v, torch.Tensor):
                item_[k] = v
            elif isinstance(v, np.ndarray):
                item_[k] = torch.from_numpy(v)
            else:
                item_[k] = torch.tensor(v)
        return item_

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {}

        if self.cache is not None and idx in self.cache:
            item = self.cache[idx]
            if self.delta_timestamps is None:
                return self._item_to_tensors(item)

        if self.delta_timestamps is None:
            item = {k: v[idx] for k, v in self._data.items() if not k.startswith("_")}
            if self.cache is not None:
                self.cache[idx] = item
            return self._item_to_tensors(item)

        delta_timestamps_keys = set(self.delta_timestamps)
        delta_timestamps_image_keys = {k for k in self.delta_timestamps if k.startswith("observation.image")}

        if len(item) == 0:
            item = {
                k: v[idx]
                for k, v in self._data.items()
                if not k.startswith("_") and k not in delta_timestamps_keys
            }

        episode_index = item["episode_index"]
        current_ts = item["timestamp"]
        episode_data_indices = np.where(
            np.bitwise_and(self._data["episode_index"] == episode_index, self._data["_occupancy_mask"])
        )[0]
        episode_timestamps = self._data["timestamp"][episode_data_indices]

        for data_key in self.delta_timestamps:
            if data_key in item:
                continue
            # get timestamps used as query to retrieve data of previous/future frames
            query_ts = current_ts + np.array(self.delta_timestamps[data_key])

            # compute distances between each query timestamp and all timestamps of all the frames belonging to the episode
            dist = np.abs(query_ts[:, None] - episode_timestamps[None, :])
            argmin_ = np.argmin(dist, axis=1)
            min_ = dist[np.arange(dist.shape[0]), argmin_]

            is_pad = min_ > self.tolerance_s

            # check violated query timestamps are all outside the episode range
            assert (
                (query_ts[is_pad] < episode_timestamps[0]) | (episode_timestamps[-1] < query_ts[is_pad])
            ).all(), (
                f"One or several timestamps unexpectedly violate the tolerance ({min_} > {self.tolerance_s=}"
                ") inside episode range. This might be due to synchronization issues with timestamps during "
                "data collection."
            )

            # load frames for this data key.
            item[data_key] = self._data[data_key][episode_data_indices[argmin_]]

            item[f"{data_key}_is_pad"] = is_pad

        if self.cache is not None and idx not in self.cache:
            self.cache[idx] = {k: v.copy() for k, v in item.items() if k not in delta_timestamps_image_keys}

        return self._item_to_tensors(item)


# class OnlineLeRobotDataset(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         data: dict[str, torch.Tensor],
#         fps: float,
#         delta_timestamps: dict[str, list[float]] | dict[str, torch.Tensor] | None = None,
#         use_cache: bool = False,
#     ):
#         super().__init__()
#         self.delta_timestamps = delta_timestamps
#         self._fps = fps
#         self.data = data
#         self.cache = {} if use_cache else None

#     def save_data(self, path: str):
#         torch.save(self.data, path)

#     @property
#     def fps(self) -> float:
#         return self._fps

#     @property
#     def tolerance_s(self) -> float:
#         """Tolerance in seconds used to discard loaded frames when their timestamps
#         are not close enough from the requested frames. It is only used when `delta_timestamps`
#         is provided or when loading video frames from mp4 files.
#         """
#         # 1e-4 to account for possible numerical error
#         return 1 / self.fps - 1e-4

#     @property
#     def delta_timestamps(self) -> dict[str, torch.Tensor] | None:
#         return self._delta_timestamps

#     @delta_timestamps.setter
#     def delta_timestamps(self, delta_timestamps: dict[str, torch.Tensor] | dict[str, list[float]] | None):
#         if delta_timestamps is None:
#             self._delta_timestamps = delta_timestamps
#         else:
#             self._delta_timestamps = {
#                 k: v if isinstance(v, torch.Tensor) else torch.tensor(v) for k, v in delta_timestamps.items()
#             }

#     @property
#     def num_episodes(self) -> int:
#         if len(self.data) > 0:
#             return len(torch.unique(self.data["episode_index"]))

#     @property
#     def num_samples(self) -> int:
#         if len(self.data) > 0:
#             return len(next(iter(self.data.values())))
#         else:
#             return 0

#     def __len__(self):
#         return self.num_samples

#     def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
#         item = {}

#         if self.cache is not None and idx in self.cache:
#             item = self.cache[idx]
#             if self.delta_timestamps is None:
#                 return item

#         if self.delta_timestamps is None:
#             item = {k: v[idx] for k, v in self.data.items()}
#             if self.cache is not None:
#                 self.cache[idx] = deepcopy(item)
#             return item

#         delta_timestamps_keys = set(self.delta_timestamps)
#         delta_timestamps_image_keys = {k for k in self.delta_timestamps if k.startswith("observation.image")}

#         if len(item) == 0:
#             item = {k: v[idx] for k, v in self.data.items() if k not in delta_timestamps_keys}

#         episode_index = item["episode_index"].item()
#         current_ts = item["timestamp"].item()
#         episode_data_indices = torch.where(self.data["episode_index"] == episode_index)[0]
#         episode_timestamps = self.data["timestamp"][self.data["episode_index"] == episode_index]

#         for data_key in self.delta_timestamps:
#             if data_key in item:
#                 continue
#             # get timestamps used as query to retrieve data of previous/future frames
#             query_ts = current_ts + self.delta_timestamps[data_key]

#             # compute distances between each query timestamp and all timestamps of all the frames belonging to the episode
#             dist = torch.cdist(query_ts[:, None], episode_timestamps[:, None], p=1)
#             min_, argmin_ = dist.min(1)

#             is_pad = min_ > self.tolerance_s

#             # check violated query timestamps are all outside the episode range
#             assert (
#                 (query_ts[is_pad] < episode_timestamps[0]) | (episode_timestamps[-1] < query_ts[is_pad])
#             ).all(), (
#                 f"One or several timestamps unexpectedly violate the tolerance ({min_} > {self.tolerance_s=}"
#                 ") inside episode range. This might be due to synchronization issues with timestamps during "
#                 "data collection."
#             )

#             # load frames for this data key.
#             item[data_key] = self.data[data_key][episode_data_indices[argmin_]]

#             item[f"{data_key}_is_pad"] = is_pad

#         if self.cache is not None and idx not in self.cache:
#             self.cache[idx] = {k: v.clone() for k, v in item.items() if k not in delta_timestamps_image_keys}
#         return item


class LeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        version: str | None = CODEBASE_VERSION,
        root: Path | None = DATA_DIR,
        split: str = "train",
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        video_backend: str | None = None,
        use_cache: bool = False,
    ):
        """
        Args:
            use_cache: Enable this to cache all items as tensors for faster data loading after the first
                epoch. Useful if you have a small enough dataset to fit into memory. You may set multiple
                workers for the PyTorch Dataloader but remember to set persistent_workers=True.
        """
        super().__init__()
        self.repo_id = repo_id
        self.version = version
        self.root = root
        self.split = split
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        # load data from hub or locally when root is provided
        # TODO(rcadene, aliberts): implement faster transfer
        # https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads
        self.hf_dataset: datasets.Dataset = load_hf_dataset(repo_id, version, root, split)
        if split == "train":
            self.episode_data_index = load_episode_data_index(repo_id, version, root)
        else:
            self.episode_data_index = calculate_episode_data_index(self.hf_dataset)
            self.hf_dataset = reset_episode_index(self.hf_dataset)
        self.stats = load_stats(repo_id, version, root)
        self.info = load_info(repo_id, version, root)
        if self.video:
            self.videos_dir = load_videos(repo_id, version, root)
            self.video_backend = video_backend if video_backend is not None else "pyav"
        self.cache = {} if use_cache else None

    def save(self, save_dir: str | Path):
        save_dir = Path(save_dir)
        self.hf_dataset.set_transform(None)
        os.makedirs(save_dir / "train", exist_ok=True)
        self.hf_dataset.save_to_disk(str(save_dir / "train"))
        os.makedirs(save_dir / "meta_data", exist_ok=True)
        save_file(self.episode_data_index, save_dir / "meta_data" / "episode_data_index.safetensors")
        save_file(flatten_dict(self.stats), save_dir / "meta_data" / "stats.safetensors")
        with open(save_dir / "meta_data" / "info.json", "w") as f:
            json.dump(self.info, f, indent=2)
        self.hf_dataset.set_transform(hf_transform_to_torch)

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.info["fps"]

    @property
    def video(self) -> bool:
        """Returns True if this dataset loads video frames from mp4 files.
        Returns False if it only loads images from png files.
        """
        return self.info.get("video", False)

    @property
    def features(self) -> datasets.Features:
        return self.hf_dataset.features

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access image and video stream from cameras."""
        keys = []
        for key, feats in self.hf_dataset.features.items():
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
        """Number of episodes."""
        return len(self.hf_dataset.unique("episode_index"))

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

    def __getitem__(self, idx):
        if self.cache is not None and idx in self.cache:
            item = self.cache[idx]
        else:
            item = self.hf_dataset[idx]

            if self.delta_timestamps is not None:
                item = load_previous_and_future_frames(
                    item,
                    self.hf_dataset,
                    self.episode_data_index,
                    self.delta_timestamps,
                    self.tolerance_s,
                    self.hf_dataset["episode_index"][0].item(),
                )

            if self.video:
                item = load_from_videos(
                    item,
                    self.video_frame_keys,
                    self.videos_dir,
                    self.tolerance_s,
                    self.video_backend,
                )
            if self.cache is not None:
                self.cache[idx] = item

        if self.image_transforms is not None:
            for cam in self.camera_keys:
                item[cam] = self.image_transforms(item[cam])

        return item

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  Repository ID: '{self.repo_id}',\n"
            f"  Version: '{self.version}',\n"
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

    @classmethod
    def from_preloaded(
        cls,
        repo_id: str = "from_preloaded",
        version: str | None = CODEBASE_VERSION,
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
        obj.version = version
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
        version: str | None = CODEBASE_VERSION,
        root: Path | None = DATA_DIR,
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
                version=version,
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

        self.version = version
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
            f"  Version: '{self.version}',\n"
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
