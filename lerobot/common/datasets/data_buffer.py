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
"""A data buffer for efficient data management during offline and online training."""

import json
import os
import shutil
from itertools import chain
from pathlib import Path
from typing import Callable

import datasets
import numpy as np
import torch

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, DATA_DIR, LeRobotDataset
from lerobot.common.datasets.utils import load_hf_dataset, load_info, load_videos
from lerobot.common.datasets.video_utils import VideoFrame, decode_video_frames_torchvision, load_from_videos

# TODO(alexander-soare): Move somewhere more appropriate once the DataBuffer class permeates more of the coe
# base.
MAX_VIDEO_PATH_LENGTH = 100


def _make_memmap_safe(**kwargs) -> np.memmap:
    """Make a numpy memmap with checks on available disk space first.

    Expected kwargs are: "filename", "dtype" (must by np.dtype), "mode" and "shape"

    For information on dtypes:
    https://numpy.org/doc/stable/reference/arrays.dtypes.html#arrays-dtypes-constructing
    """
    if kwargs["mode"].startswith("w"):
        required_space = kwargs["dtype"].itemsize * np.prod(kwargs["shape"])  # bytes
        stats = os.statvfs(Path(kwargs["filename"]).parent)
        available_space = stats.f_bavail * stats.f_frsize  # bytes
        if required_space >= available_space * 0.8:
            raise RuntimeError(
                f"You're about to take up {required_space} of {available_space} bytes available. This "
                "exception has been raised to protect your storage device."
                ""
            )
    return np.memmap(**kwargs)


class TimestampOutsideToleranceError(Exception):
    pass


class DataBuffer(torch.utils.data.Dataset):
    """Data buffer and training data item getter.

    Data is considered to come in the form of "episodes" (an instance of a robot performing a task). Episodes
    are made up of "frames", which are chronoligically ordered and contain timestamp aligned data, potentially
    including environment observations, and robot actions. NOTE: for the time being, we require all data
    modalities to be timestamp aligned. This constraint may be relaxed in the future.

    The data is stored in a mapping from data keys to arrays with shape (total_number_of_frames, *data_dim).
    The compulsory data keys are:
        - "index": A sequential integer index per frame.
        - "episode_index": A sequential integer index per episode.
        - "frame_index": A sequential integer index per frame within an episode (it resets for each episode).
        - "timestamp": The relative timestamp of the frame within the episode in units of seconds. The choice.
            of reference time is not important.

    Under the hood, the data is stored in `numpy.memmap`s, one for each data key. Loosely speaking, memory
    mapping (https://en.wikipedia.org/wiki/Memory-mapped_file) allows us to treat a portion of disk space as
    virtual memory. This allows us to work with more data than can fit in our physical memory, while treating
    the data as if it were just standard numpy arrays. The associated files are saved in the file system under
    what we call the "storage directory", and the Python object that allows us to treat them as virtual memory
    is called the "buffer". The storage directory also contains a "meta.json" file which includes information
    about the date types and shapes for each memmap. This allows us to load the data without having to specify
    the data specifications at runtime.

    The `add_episodes` method can be used to insert more data in the form of integral episodes (starting from
    frame 0 and with the frames ordered). The buffer has a compulsory size limit, which must be provided when
    creating a new one. Data is inserted in a circular fashion, inserting after the most recently added frame,
    and wrapping around to the start when necessary (in which case older episodes are overwritten).

    This class is also a PyTorch Dataset and can be used as such in a dataloader for a training loop. The item
    getter returns either a single frame, or a slice of a single episode if `delta_timestamps` is set.
    """

    # TODO(now): What should we do about implicit data key specs, like observations.image?

    # Special key for a (1,) array storing a pointer to the next index to fill from when adding data.
    NEXT_INDEX_KEY = "_next_index"
    # Since the data buffer is pre-allocated, this boolean mask is used to indicate which frames have "real"
    # data.
    OCCUPANCY_MASK_KEY = "_occupancy_mask"
    # This is not a data key used in the buffer. It is used to indicate that a frame is padding, added by the
    # __getitem__ method.
    IS_PAD_POSTFIX = "_is_pad"
    INDEX_KEY = "index"
    FRAME_INDEX_KEY = "frame_index"
    EPISODE_INDEX_KEY = "episode_index"
    TIMESTAMP_KEY = "timestamp"
    PRESET_KEYS = {INDEX_KEY, FRAME_INDEX_KEY, EPISODE_INDEX_KEY, TIMESTAMP_KEY}

    METADATA_FILE_NAME = "meta.json"

    def __init__(
        self,
        storage_dir: str | Path,
        buffer_capacity: int | None = None,
        image_transform: Callable[[np.ndarray], np.ndarray] | None = None,
        delta_timestamps: dict[str, list[float]] | dict[str, np.ndarray] | None = None,
        fps: float | None = None,
    ):
        """
        Args:
            storage_dir: Where to keep the numpy memmap files and metadata files. One memmap file will be
                stored for each data key. Note that if the storage directory already exist, the memmap files
                are opened in read-write mode. If the storage directory does not exist, it will be lazily
                created with the first call to `add_episodes`.
            buffer_capacity: How many frames should be stored in the buffer as a maximum. Be aware of your
                system's available disk space when choosing this. Note that if `storage_dir` references an
                existing storage directory, `buffer_capacity` should not be provided, as it is already
                included in "meta.json".
            image_transform: Transforms to apply in the item getter to all image data (any data whose key
                starts with "observation.image").
            delta_timestamps: TODO(alexander-soare): Document this somewhere when
                `load_previous_and_future_frames` is refactored.
            fps: TODO(alexander-soare): Document this somewhere when `load_previous_and_future_frames` is
                refactored.

        """
        # Argument validation.
        if (delta_timestamps is None) ^ (fps is None):
            raise ValueError("`delta_timestamps` and `fps` should be provided together, or not at all.")

        # Parameters for the data structure.
        self._storage_dir = Path(storage_dir)
        self._data: dict[str, np.memmap] = {}
        self._is_video_dataset = False
        # If the storage directory already exists, load the memmaps.
        if self._storage_dir.exists():
            if buffer_capacity is not None:
                raise ValueError(
                    "The storage directory already exists, which means you should not provide a "
                    "buffer_capacity explicitly. Instead, it will be read from 'meta.json' in the storage "
                    "directory."
                )
            data_spec = self._load_data_spec()
            self._make_memmaps(data_spec, mode="r+")
            self._buffer_capacity = len(self._data[self.INDEX_KEY])
        else:
            if buffer_capacity is None:
                raise ValueError(
                    "The storage directory does not exist, which means you need to provide a buffer_capacity."
                )
            self._buffer_capacity = buffer_capacity

        # Parameters for the item getter.
        self.set_delta_timestamps(delta_timestamps)
        self._fps = fps
        # Tolerance (in seconds) used to discard loaded frames when their timestamps are not close enough to
        # the requested frames. It is only used when `delta_timestamps` is provided. The -1e-4 accounts for
        # possible numerical error.
        self.tolerance_s = 1 / self.fps - 1e-4 if fps is not None else None
        self.image_transform = image_transform

    @property
    def storage_dir(self) -> Path:
        return self._storage_dir

    @property
    def is_video_dataset(self) -> bool:
        return self._is_video_dataset

    @property
    def data_keys(self) -> list[str]:
        keys = set(self._data)
        keys.remove(self.OCCUPANCY_MASK_KEY)
        keys.remove(self.NEXT_INDEX_KEY)
        return sorted(keys)

    @property
    def fps(self) -> float | None:
        return self._fps

    @property
    def num_episodes(self) -> int:
        """Total number of unique episode indices in the dataset."""
        if len(self._data) == 0:
            # Buffers not created yet.
            return 0
        return len(np.unique(self._data[self.EPISODE_INDEX_KEY][self._data[self.OCCUPANCY_MASK_KEY]]))

    @property
    def num_samples(self) -> int:
        """Total number of unique samples (aka frames) in the dataset.

        TODO(alexander-soare): Rename to num_frames once LeRobotDataset is deprecated.
        """
        if len(self._data) == 0:
            # Buffers not created yet.
            return 0
        return np.count_nonzero(self._data[self.OCCUPANCY_MASK_KEY])

    @property
    def camera_keys(self) -> list[str]:
        """Return the names of all data keys pertaining to camera observations.

        By convention, this is all the keys starting with "observation.image".
        """
        return [k for k in self._data if k.startswith("observation.image")]

    def _save_data_spec(self, data_spec: dict[str, dict]):
        """Save the data type and shape specifications to the storage directory."""
        meta_file = self._storage_dir / self.METADATA_FILE_NAME
        with open(meta_file, "w") as f:
            for k in data_spec:
                data_spec[k]["dtype"] = str(data_spec[k]["dtype"])
            json.dump(data_spec, f)

    def _load_data_spec(self) -> dict[str, dict]:
        """Load the data type and shape specifications from the storage directory."""
        meta_file = self._storage_dir / self.METADATA_FILE_NAME
        with open(meta_file) as f:
            data_spec = json.load(f)
        for k in data_spec:
            data_spec[k]["dtype"] = np.dtype(data_spec[k]["dtype"])
        return data_spec

    def _make_storage_dir(self, episode_data: dict[str, np.ndarray]):
        """Create the storage directory based on example episode data from the first `add_episodes` call."""
        assert (
            not self.storage_dir.exists()
        ), "This method should only be called before the storage directory has been created."

        self._storage_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Make the data spec for np.memmap
            data_spec = {
                self.NEXT_INDEX_KEY: {"dtype": np.dtype("int64"), "shape": (1,)},
                self.OCCUPANCY_MASK_KEY: {"dtype": np.dtype("?"), "shape": (self._buffer_capacity,)},
                self.INDEX_KEY: {"dtype": np.dtype("int64"), "shape": (self._buffer_capacity,)},
                self.FRAME_INDEX_KEY: {"dtype": np.dtype("int64"), "shape": (self._buffer_capacity,)},
                self.EPISODE_INDEX_KEY: {"dtype": np.dtype("int64"), "shape": (self._buffer_capacity,)},
                # TODO(now): Should this just be float32?
                self.TIMESTAMP_KEY: {"dtype": np.dtype("float64"), "shape": (self._buffer_capacity,)},
            }
            for k, v in episode_data.items():
                if k in data_spec:
                    continue
                data_spec[k] = {"dtype": v.dtype, "shape": (self._buffer_capacity, *v.shape[1:])}

            self._make_memmaps(data_spec, "w+")
            self._save_data_spec(data_spec)
        except Exception as e:
            # Attempt to clean up by removing the empty storage directory.
            shutil.rmtree(self._storage_dir)
            raise e

    def _make_memmaps(self, data_spec: dict[str, dict], mode: str):
        """Create the memmap buffer objects.

        The underlying storage directory may or may not already exist. Provide the file opening `mode`
        accordingly.
        """
        for k, v in data_spec.items():
            self._data[k] = _make_memmap_safe(
                filename=self._storage_dir / k,
                dtype=v["dtype"] if v is not None else None,
                mode=mode,
                shape=tuple(v["shape"]) if v is not None else None,
            )

    @property
    def delta_timestamps(self) -> dict[str, np.ndarray] | None:
        return self._delta_timestamps

    def set_delta_timestamps(self, value: dict[str, list[float]] | None):
        """Set delta_timestamps converting the values to numpy arrays.

        Note: The conversion is for an optimization in the __getitem__. The loop is much slower if lists need
        to be converted into numpy arrays.
        """
        if value is not None:
            self._delta_timestamps = {k: np.array(v) for k, v in value.items()}
        else:
            self._delta_timestamps = None

    def add_episodes(self, data: dict[str, np.ndarray]):
        """Add data to the buffer.

        `data` should have the same key, array mapping as the buffer. It should contain at least one episode.
        The episodes should have frame indices that start from 0 and step up in increments of 1.

        Episodes are added to the buffer one-by-one. If an episode has more frames then are available till the
        end of the buffer, the pointer is reset to the start of the buffer and the episode is inserted there,
        overwriting existing episode frames.

        When episode frames are overwritten by a new episode, by default, any remaining frames belonging to
        the existing episode are left in place (meaning not all episodes will be guaranteed to start from
        their frame 0).

        After adding the episodes to the buffer, the buffer is flushed to disk.
        """
        for episode_index in np.unique(data[self.EPISODE_INDEX_KEY]):
            where_episode = np.where(data[self.EPISODE_INDEX_KEY] == episode_index)[0]
            episode_data = {k: data[k][where_episode] for k in data}
            self._add_episode(episode_data)

        self.flush()

    def _add_episode(self, data: dict[str, np.ndarray]):
        """Add data for a single episode to the buffer."""
        if len(self._data) == 0:
            self._make_storage_dir(data)

        if len(missing_keys := (set(self.data_keys).difference(set(data)))) > 0:
            raise ValueError(f"Missing data keys: {missing_keys}")
        new_data_length = len(data[self.data_keys[0]])
        if new_data_length <= 0:
            raise ValueError("The episode has 0 frames")
        if new_data_length > self._buffer_capacity:
            raise ValueError("The episode length is larger than the buffer capacity.")
        if not all(len(data[k]) == new_data_length for k in self.data_keys):
            raise ValueError("All data items should have the same length")
        if not np.all(data[self.EPISODE_INDEX_KEY] == data[self.EPISODE_INDEX_KEY][0]):
            raise ValueError(
                "New data should only contain one episode but there is more than one unique episode index."
            )
        if not np.array_equal(data[self.FRAME_INDEX_KEY], np.arange(new_data_length)):
            raise ValueError(
                "Expected frame indices to start from 0 and step up in increments of 1 per frame."
            )

        # Figure out where we need to start filling data next, and make sure we continue data and episode
        # indices.
        next_index = self._data[self.NEXT_INDEX_KEY][0]
        if self.num_samples > 0:
            last_episode_index = self._data[self.EPISODE_INDEX_KEY][next_index - 1]
            last_data_index = self._data[self.INDEX_KEY][next_index - 1]
        else:
            last_episode_index = -1
            last_data_index = -1
        # If there aren't enough slots in the buffer left to accommodate the episode, wrap to the start.
        if max(0, new_data_length - (self._buffer_capacity - next_index)) > 0:
            next_index = 0

        # Insert the new data starting from next_index.
        for k in self.data_keys:
            slc = slice(next_index, next_index + new_data_length)
            if k == self.EPISODE_INDEX_KEY:
                self._data[k][slc] = last_episode_index + 1
            elif k == self.INDEX_KEY:
                self._data[k][slc] = np.arange(last_data_index + 1, last_data_index + 1 + new_data_length)
            else:
                self._data[k][slc] = data[k]
            self._data[self.OCCUPANCY_MASK_KEY][slc] = True

        # Update the data pointer.
        self._data[self.NEXT_INDEX_KEY][0] = next_index + new_data_length

    def flush(self):
        """Save the data to disk.

        `np.memmap`s keep a portion of the data mirrored in memory. Updates to the in-memory data are not
        immediately reflected on disk. Call this method to explicitly save the updates to disk.
        """
        for k in self._data:
            self._data[k].flush()

    @staticmethod
    def _item_to_tensors(item: dict) -> dict:
        item_ = {}
        for k, v in item.items():
            item_[k] = torch.from_numpy(v)
        return item_

    def _optimized_advanced_slice(self, data_key: str, indices: np.ndarray) -> np.ndarray:
        """Convert advanced slicing to basic slicing by finding contiguous ranges in the requested indices.

        TODO(now): Is this needed?
        """
        indices_diff = np.diff(indices, prepend=indices[0] - 1)
        where_not_1 = np.where(indices_diff != 1)[0]
        ptr = 0
        ret = []
        for ix in chain(where_not_1, [len(indices)]):
            ret.append(self._data[data_key][indices[ptr] : indices[ix - 1] + 1])
            ptr = ix

        # Avoid creating a copy with concatenate if possible.
        return np.concatenate(ret) if len(ret) > 1 else ret[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx >= len(self) or idx < -len(self):
            raise IndexError

        item = {k: v[idx] for k, v in self._data.items() if not k.startswith("_")}

        if self.delta_timestamps is not None:
            episode_index = item[self.EPISODE_INDEX_KEY]
            current_ts = item[self.TIMESTAMP_KEY]
            episode_data_indices = np.where(
                np.bitwise_and(
                    self._data[self.EPISODE_INDEX_KEY] == episode_index,
                    self._data[self.OCCUPANCY_MASK_KEY],
                )
            )[0]
            episode_timestamps = self._data[self.TIMESTAMP_KEY][episode_data_indices]

            if self.is_video_dataset:
                video_timestamps = {}  # TODO(now): HACK

            for data_key in self.delta_timestamps:
                # Get timestamps used as query to retrieve data of previous/future frames.
                query_ts = current_ts + self.delta_timestamps[data_key]

                # Compute distances between each query timestamp and all timestamps of all the frames
                # belonging to the episode.
                dist = np.abs(query_ts[:, None] - episode_timestamps[None, :])
                argmin_ = np.argmin(dist, axis=1)
                min_ = dist[np.arange(dist.shape[0]), argmin_]

                is_pad = min_ > self.tolerance_s

                # Check violated query timestamps are all outside the episode range.
                if not (
                    (query_ts[is_pad] < episode_timestamps[0]) | (episode_timestamps[-1] < query_ts[is_pad])
                ).all():
                    raise TimestampOutsideToleranceError(
                        f"One or several timestamps unexpectedly violate the tolerance ({min_} > "
                        f"{self.tolerance_s=}) inside the episode range."
                    )

                if self.is_video_dataset and data_key.startswith("observation.image"):
                    video_timestamps[data_key] = self._data[self.TIMESTAMP_KEY][episode_data_indices[argmin_]]

                # Load frames for this data key.
                if np.any(np.diff(argmin_) != 1):
                    item[data_key] = self._data[data_key][episode_data_indices[argmin_]]
                    # item[data_key] = self._optimized_advanced_slice(data_key, episode_data_indices[argmin_])
                else:
                    # Do basic slicing where possible
                    item[data_key] = self._data[data_key][
                        episode_data_indices[argmin_.min()] : episode_data_indices[argmin_.max()] + 1
                    ]

                item[f"{data_key}{self.IS_PAD_POSTFIX}"] = is_pad

        if self.is_video_dataset:
            item_ = dict(item)
            for k in self.video_frame_keys:  # TODO(now): HACK
                if self.delta_timestamps is None:
                    item_[k] = {"path": item[k].decode(), "timestamp": float(item[self.TIMESTAMP_KEY])}
                else:
                    query_ts = current_ts + self.delta_timestamps[k]
                    item_[k] = [
                        {"path": item[k][i].decode(), "timestamp": float(video_timestamps[k][i])}
                        for i in range(len(item[k]))
                    ]
            item = load_from_videos(
                item_,
                self.video_frame_keys,
                self.videos_dir,
                self.tolerance_s,
                self.video_backend,
            )

        if self.image_transform is not None:
            for cam in self.camera_keys:
                item[cam] = self.image_transform(item[cam])

        return self._item_to_tensors(item)

    @classmethod
    def from_huggingface_hub(
        cls,
        repo_id: str,
        decode_video: bool = False,
        root: Path | None = DATA_DIR,
        **kwargs,
    ) -> "DataBuffer":
        """Create a DataBuffer from a data repository on the Hugging Face Hub.

        NOTE: If the DataBuffer already exists in /tmp, this function will reuse it rather than creating a new
        one.

        Args:
            repo_id: The dataset repository ID.
            decode_video: If repo_id refers to a video dataset (the image observations are encoded as videos),
                decode the videos and store the frames in a numpy memmap.
            root: (will be deprecated) Directory to load the dataset from, instead of the hub.
            **kwargs: Other arguments to `self.__init__` except for the `data_spec` and
                `buffer_capacity` arguments which are inferred automatically. `storage_dir` is set to
                `/tmp/{repo_id}_{hf_dataset._fingerprint}_{decoded?}` unless provided explicitly.
        Returns:
            The resulting DataBuffer object.

        TODO(now): Consider populating the buffer one episode at a time in order not to kill RAM.
        TODO(now): Reuse previously saved? Maybe I need a hash.
        """
        for k in ["data_spec", "buffer_capacity"]:
            if k in kwargs:
                raise ValueError(f"`{k}` should not be provided as it is inferred from the hub dataset.")

        hf_dataset = load_hf_dataset(repo_id, version=CODEBASE_VERSION, root=root, split="train")
        hf_dataset.set_transform(lambda x: x)
        kwargs.setdefault(
            "storage_dir",
            DataBuffer._default_storage_dir_from_huggingface_hub(
                repo_id, hf_dataset._fingerprint, decode_video
            ),
        )

        buffer_already_on_disk = False
        if kwargs["storage_dir"].exists():
            buffer_already_on_disk = True

        obj = cls(
            **kwargs,
            buffer_capacity=len(hf_dataset) if not buffer_already_on_disk else None,
        )
        # If we have accessed an existing cached data buffer, just return the object as is.
        if buffer_already_on_disk:
            return obj

        # Get some metadata necessary for processing videos.
        lerobot_dataset_info = load_info(repo_id, version=CODEBASE_VERSION, root=root)
        is_video_dataset = lerobot_dataset_info.get("video", False)
        if not is_video_dataset and decode_video:
            raise ValueError(f"The provided dataset is not a video dataset but you have {decode_video=}")
        if is_video_dataset:
            videos_path = load_videos(repo_id, version=CODEBASE_VERSION, root=root)

        # Populate the buffer with the data from the dataset.
        data_dict = {}
        for k, feature in hf_dataset.features.items():
            if isinstance(feature, datasets.features.Image):
                data_dict[k] = np.stack([np.array(pil_img) for pil_img in hf_dataset[k]])
            elif isinstance(feature, VideoFrame):
                if decode_video:
                    # Decode all videos into images.
                    episode_indices = np.array(hf_dataset["episode_index"])
                    timestamps = np.array(hf_dataset["timestamp"])
                    all_imgs = []
                    for episode_index in np.unique(episode_indices):
                        episode_data_indices = np.where(episode_indices == episode_index)[0]
                        episode_timestamps = timestamps[episode_indices == episode_index]
                        episode_imgs = decode_video_frames_torchvision(
                            videos_path.parent / hf_dataset[k][episode_data_indices[0]]["path"],
                            episode_timestamps,
                            1 / lerobot_dataset_info["fps"] - 1e-4,
                            to_pytorch_format=False,
                        )
                        all_imgs.extend(episode_imgs)
                    data_dict[k] = np.stack(all_imgs)
                else:
                    data_dict[k] = np.stack(
                        [np.array(dct["path"], dtype=f"S{MAX_VIDEO_PATH_LENGTH}") for dct in hf_dataset[k]]
                    )
            else:
                data_dict[k] = np.array(hf_dataset[k])
        obj.add_episodes(data_dict)
        if is_video_dataset:
            obj._is_video_dataset = True  # TODO(now): HACK
            # Symlink videos if needed.
            obj.videos_dir = kwargs["storage_dir"] / "videos"
            if not obj.videos_dir.exists():
                os.symlink(videos_path.absolute(), obj.videos_dir)
            obj.videos_dir = kwargs["storage_dir"] / "videos"  # TODO(now): HACK
            obj.video_backend = "pyav"  # TODO(now): HACK

        return obj

    @staticmethod
    def _default_storage_dir_from_huggingface_hub(repo_id: str, fingerprint: str, decode_video: bool) -> Path:
        """Create the default storage directory used for the `from_huggingface_hub` method.

        Note: This method is really meant for development / testing.
        """
        return Path(f"/tmp/{repo_id}_{fingerprint}{'_decoded' if decode_video else ''}")


def compute_sampler_weights(
    offline_dataset: LeRobotDataset,
    offline_drop_n_last_frames: int = 0,
    online_dataset: DataBuffer | None = None,
    online_sampling_ratio: float | None = None,
    online_drop_n_last_frames: int = 0,
) -> torch.Tensor:
    """Compute the sampling weights for the online training dataloader in train.py.

    Args:
        offline_dataset: The LeRobotDataset used for offline pre-training.
        online_drop_n_last_frames: Number of frames to drop from the end of each offline dataset episode.
        online_dataset: The DataBuffer used in online training.
        online_sampling_ratio: The proportion of data that should be sampled from the online dataset. If an
            online dataset is provided, this value must also be provided.
        online_drop_n_first_frames: See `offline_drop_n_last_frames`. This is the same, but for the online
            dataset.
    Returns:
        Tensor of weights for [offline_dataset; online_dataset], normalized to 1.

    Notes to maintainers:
        - This duplicates some logic from EpisodeAwareSampler. We should consider converging to one approach.
        - When used with `torch.utils.data.WeightedRandomSampler`, it could completely replace
          `EpisodeAwareSampler` as the online dataset related arguments are optional. The only missing feature
          is the ability to turn shuffling off.
        - Options `drop_first_n_frames` and `episode_indices_to_use` can be added easily. They were not
          included here to avoid adding complexity.
    """
    if len(offline_dataset) == 0 and (online_dataset is None or len(online_dataset) == 0):
        raise ValueError("At least one of `offline_dataset` or `online_dataset` should be contain data.")
    if (online_dataset is None) ^ (online_sampling_ratio is None):
        raise ValueError(
            "`online_dataset` and `online_sampling_ratio` must be provided together or not at all."
        )
    offline_sampling_ratio = 0 if online_sampling_ratio is None else 1 - online_sampling_ratio

    weights = []

    if len(offline_dataset) > 0:
        offline_data_mask_indices = []
        for start_index, end_index in zip(
            offline_dataset.episode_data_index["from"],
            offline_dataset.episode_data_index["to"],
            strict=True,
        ):
            offline_data_mask_indices.extend(
                range(start_index.item(), end_index.item() - offline_drop_n_last_frames)
            )
        offline_data_mask = torch.zeros(len(offline_dataset), dtype=torch.bool)
        offline_data_mask[torch.tensor(offline_data_mask_indices)] = True
        weights.append(
            torch.full(
                size=(len(offline_dataset),),
                fill_value=offline_sampling_ratio / offline_data_mask.sum(),
            )
            * offline_data_mask
        )

    if online_dataset is not None and len(online_dataset) > 0:
        online_data_mask_indices = []
        episode_indices = online_dataset.get_data_by_key("episode_index")
        for episode_idx in torch.unique(episode_indices):
            where_episode = torch.where(episode_indices == episode_idx)
            start_index = where_episode[0][0]
            end_index = where_episode[0][-1] + 1
            online_data_mask_indices.extend(
                range(start_index.item(), end_index.item() - online_drop_n_last_frames)
            )
        online_data_mask = torch.zeros(len(online_dataset), dtype=torch.bool)
        online_data_mask[torch.tensor(online_data_mask_indices)] = True
        weights.append(
            torch.full(
                size=(len(online_dataset),),
                fill_value=online_sampling_ratio / online_data_mask.sum(),
            )
            * online_data_mask
        )

    weights = torch.cat(weights)

    if weights.sum() == 0:
        weights += 1 / len(weights)
    else:
        weights /= weights.sum()

    return weights
