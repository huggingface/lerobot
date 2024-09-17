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

import logging
import os
from itertools import chain
from pathlib import Path
from typing import Any, Callable

import datasets
import numpy as np
import torch

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
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


class DataBuffer(torch.utils.data.Dataset):
    """Data buffer for efficient dataset management during offline and online training.

    Data is considered to come in the form of "episodes" (an instance of a robot performing a task). Episodes
    are made up of "frames", which are chronoligically ordered and contain timestamp aligned data, potentially
    including environment observations, and robot actions. NOTE: for the time being, we require all data
    modalities are timestamp aligned. This constraint may be relaxed in the future.

    The data is stored in a mapping from data keys to arrays with shape (total_number_of_frames, *data_dim).
    The compulsory data keys are:
        - "index": A sequential integer index per frame.
        - "episode_index": A sequential integer index per episode.
        - "frame_index": A sequential integer index per frame within an episode (it resets for each episode).
        - "timestamp": The relative timestamp of the frame within the episode in units of seconds. The choice.
            of reference time is not important.

    The `add_data` and `add_episodes` functions can be used to insert more data in the form of integral
    episodes (starting from frame 0 and with the frames ordered). The buffer has a compulsory size limit,
    which must be provided. Data is inserted in a circular fashion, inserting after the most recently added
    frame, and wrapping around to the start when necessary (in which case older episodes are overwritten).

    This class is also a PyTorch Dataset and can be used as such in a dataloader for a training loop. The item
    getter returns either a single from, or a slice of a single episode based on the requested data index.
    """

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

    def __init__(
        self,
        storage_dir: str | Path,
        data_spec: dict[str, Any] | None,
        buffer_capacity: int | None,
        image_transform: Callable[[np.ndarray], np.ndarray] | None = None,
        delta_timestamps: dict[str, list[float]] | dict[str, np.ndarray] | None = None,
        fps: float | None = None,
    ):
        """Create a data buffer including reserving the underlying on-disk storage.

        Args:
            storage_dir: Where to keep the numpy memmap files. One memmap file will be stored for each data
                key. Note that if the files already exist, they are opened in read-write mode.
            data_spec: A mapping from data key to data specification, like {data_key: {"shape": tuple[int],
                "dtype": np.dtype}}. This should include all the data that you wish to record into the buffer,
                but note that "index", "frame_index", "episode_index", and "timestamp", are already handled
                internally, so you don't need to include them.
            buffer_capacity: How many frames should be stored in the buffer as a maximum. Be aware of your
                system's available disk space when choosing this.
            image_transform: Transforms to apply in the item getter to all image data (any data whose key
                starts with "observation.image").
            delta_timestamps: TODO(alexander-soare): Document this somewhere when
                `load_previous_and_future_frames` is refactored.
            fps: TODO(alexander-soare): Document this somewhere when `load_previous_and_future_frames` is
                refactored.

        """
        if (delta_timestamps is None) ^ (fps is None):
            raise ValueError("`delta_timestamps` and `fps` should be provided together, or not at all.")
        self.set_delta_timestamps(delta_timestamps)
        self._fps = fps
        # Tolerance in seconds used to discard loaded frames when their timestamps are not close enough from
        # the requested frames. It is only used when `delta_timestamps` is provided.
        # minus 1e-4 to account for possible numerical error
        self.tolerance_s = 1 / self.fps - 1e-4 if fps is not None else None
        self._buffer_capacity = buffer_capacity
        Path(storage_dir).mkdir(parents=True, exist_ok=True)
        data_spec = self._make_data_spec(data_spec, buffer_capacity)
        self._data: dict[str, np.memmap] = {}
        for k, v in data_spec.items():
            self._data[k] = _make_memmap_safe(
                filename=Path(storage_dir) / k,
                dtype=v["dtype"] if v is not None else None,
                mode="r+" if (Path(storage_dir) / k).exists() else "w+",
                shape=tuple(v["shape"]) if v is not None else None,
            )
        self.image_transform = image_transform

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

    def _make_data_spec(self, data_spec: dict[str, Any], buffer_capacity: int) -> dict[str, dict[str, Any]]:
        """Makes the necessary data specification keyword arguments for np.memmap."""
        if any(k.startswith("_") for k in data_spec):
            raise ValueError(
                "data_spec keys should not start with '_'. This prefix is reserved for internal logic."
            )
        if len(intersection := set(data_spec).intersection(DataBuffer.PRESET_KEYS)) > 0:
            raise ValueError(
                f"`data_spec` should not contain any of {DataBuffer.PRESET_KEYS} as these are handled "
                f"internally. The provided data_spec has {intersection}."
            )
        complete_data_spec = {
            DataBuffer.NEXT_INDEX_KEY: {"dtype": np.dtype("int64"), "shape": (1,)},
            DataBuffer.OCCUPANCY_MASK_KEY: {"dtype": np.dtype("?"), "shape": (buffer_capacity,)},
            DataBuffer.INDEX_KEY: {"dtype": np.dtype("int64"), "shape": (buffer_capacity,)},
            DataBuffer.FRAME_INDEX_KEY: {"dtype": np.dtype("int64"), "shape": (buffer_capacity,)},
            DataBuffer.EPISODE_INDEX_KEY: {"dtype": np.dtype("int64"), "shape": (buffer_capacity,)},
            DataBuffer.TIMESTAMP_KEY: {"dtype": np.dtype("float64"), "shape": (buffer_capacity,)},
        }
        for k, v in data_spec.items():
            complete_data_spec[k] = {"dtype": v["dtype"], "shape": (buffer_capacity, *v["shape"])}
        return complete_data_spec

    def add_data(self, data: dict[str, np.ndarray]):
        """Add data to the buffer.

        This calls `add_episode` for each unique episode index. See the documentation there for more
        information.
        """
        for episode_index in np.unique(data[DataBuffer.EPISODE_INDEX_KEY]):
            where_episode = np.where(data[DataBuffer.EPISODE_INDEX_KEY] == episode_index)[0]
            episode_data = {k: data[k][where_episode] for k in data}
            self.add_episode(episode_data)

    def add_episode(self, data: dict[str, np.ndarray]):
        """Add data for a single episode to the buffer.

        `data` should have the same key, array mapping as the buffer. It should contain exactly one episode.
        The episode should have frame indices that start from 0 and step up in increments of 1.

        If the episode has more frames then are available till the end of the buffer, the pointer is reset
        to the start of the buffer and the episode is inserted there, overwriting existing episode frames.

        When episode frames are overwritten by a new episode, by default, any remaining frames belonging to
        the existing episode are left in place (meaning not all episodes will be guaranteed to start from
        their frame 0).
        """
        if len(missing_keys := (set(self.data_keys).difference(set(data)))) > 0:
            raise ValueError(f"Missing data keys: {missing_keys}")
        new_data_length = len(data[self.data_keys[0]])
        if new_data_length <= 0:
            raise ValueError("The episode has 0 frames")
        if new_data_length > self._buffer_capacity:
            raise ValueError("The episode length is larger than the buffer capacity.")
        if not all(len(data[k]) == new_data_length for k in self.data_keys):
            raise ValueError("All data items should have the same length")
        if not np.all(data[DataBuffer.EPISODE_INDEX_KEY] == data[DataBuffer.EPISODE_INDEX_KEY][0]):
            raise ValueError(
                "New data should only contain one episode but there is more than one unique episode index."
            )
        if not np.array_equal(data[DataBuffer.FRAME_INDEX_KEY], np.arange(new_data_length)):
            raise ValueError(
                "Expected frame indices to start from 0 and step up in increments of 1 per frame."
            )

        # Figure out where we need to start filling data next, and make sure we continue data and episode
        # indices.
        next_index = self._data[DataBuffer.NEXT_INDEX_KEY][0]
        if self.num_samples > 0:
            last_episode_index = self._data[DataBuffer.EPISODE_INDEX_KEY][next_index - 1]
            last_data_index = self._data[DataBuffer.INDEX_KEY][next_index - 1]
        else:
            last_episode_index = -1
            last_data_index = -1
        # If there aren't enough slots in the buffer left to accommodate the episode, wrap to the start.
        if max(0, new_data_length - (self._buffer_capacity - next_index)) > 0:
            next_index = 0

        # Insert the new data starting from next_index.
        for k in self.data_keys:
            slc = slice(next_index, next_index + new_data_length)
            if k == DataBuffer.EPISODE_INDEX_KEY:
                self._data[k][slc] = last_episode_index + 1
            elif k == DataBuffer.INDEX_KEY:
                self._data[k][slc] = np.arange(last_data_index + 1, last_data_index + 1 + new_data_length)
            else:
                self._data[k][slc] = data[k]
            self._data[DataBuffer.OCCUPANCY_MASK_KEY][slc] = True

        # Update the data pointer.
        self._data[DataBuffer.NEXT_INDEX_KEY][0] = next_index + new_data_length

    @property
    def data_keys(self) -> list[str]:
        keys = set(self._data)
        keys.remove(DataBuffer.OCCUPANCY_MASK_KEY)
        keys.remove(DataBuffer.NEXT_INDEX_KEY)
        return sorted(keys)

    @property
    def fps(self) -> float | None:
        return self._fps

    @property
    def num_episodes(self) -> int:
        return len(
            np.unique(self._data[DataBuffer.EPISODE_INDEX_KEY][self._data[DataBuffer.OCCUPANCY_MASK_KEY]])
        )

    @property
    def num_samples(self) -> int:
        return np.count_nonzero(self._data[DataBuffer.OCCUPANCY_MASK_KEY])

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

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access image and video stream from cameras."""
        return [k for k in self._data if k.startswith("observation.image")]

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

    def flush(self):
        """Save the data to disk.

        `np.memmap`s keep a portion of the data mirrored in memory. Updates to the in-memory data are not
        immediately reflected on disk. Call this method to explicitly save the updates to disk.
        """
        for k in self._data:
            self._data[k].flush()

    def get_data_by_key(self, key: str) -> torch.Tensor:
        """Returns all data for a given data key as a Tensor."""
        return torch.from_numpy(self._data[key][self._data[DataBuffer.OCCUPANCY_MASK_KEY]])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx >= len(self) or idx < -len(self):
            raise IndexError

        item = {k: v[idx] for k, v in self._data.items() if not k.startswith("_")}

        if self.delta_timestamps is not None:
            episode_index = item[DataBuffer.EPISODE_INDEX_KEY]
            current_ts = item[DataBuffer.TIMESTAMP_KEY]
            episode_data_indices = np.where(
                np.bitwise_and(
                    self._data[DataBuffer.EPISODE_INDEX_KEY] == episode_index,
                    self._data[DataBuffer.OCCUPANCY_MASK_KEY],
                )
            )[0]
            episode_timestamps = self._data[DataBuffer.TIMESTAMP_KEY][episode_data_indices]

            if self.video:
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
                err_msg = (
                    f"One or several timestamps unexpectedly violate the tolerance ({min_} > "
                    f"{self.tolerance_s=}) inside the episode range."
                )
                try:
                    assert (
                        (query_ts[is_pad] < episode_timestamps[0])
                        | (episode_timestamps[-1] < query_ts[is_pad])
                    ).all(), err_msg
                except AssertionError:
                    logging.warning(err_msg)
                    return self.__getitem__(np.random.choice(len(self)))

                if self.video and data_key in self.video_frame_keys:  # TODO(now): HACK
                    video_timestamps[data_key] = self._data[DataBuffer.TIMESTAMP_KEY][
                        episode_data_indices[argmin_]
                    ]

                # Load frames for this data key.
                if np.any(np.diff(argmin_) != 1):
                    item[data_key] = self._data[data_key][episode_data_indices[argmin_]]
                    # item[data_key] = self._optimized_advanced_slice(data_key, episode_data_indices[argmin_])
                else:
                    # Do basic slicing where possible
                    item[data_key] = self._data[data_key][
                        episode_data_indices[argmin_.min()] : episode_data_indices[argmin_.max()] + 1
                    ]

                item[f"{data_key}{DataBuffer.IS_PAD_POSTFIX}"] = is_pad

        if self.video:
            item_ = dict(item)
            for k in self.video_frame_keys:  # TODO(now): HACK
                if self.delta_timestamps is None:
                    item_[k] = {"path": item[k].decode(), "timestamp": float(item[DataBuffer.TIMESTAMP_KEY])}
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
    def from_hf_dataset(
        cls,
        repo_id: str,
        decode_video: bool = False,
        **kwargs,
    ) -> "DataBuffer":
        hf_dataset = load_hf_dataset(repo_id, version=CODEBASE_VERSION, root=None, split="train")
        lerobot_dataset_info = load_info(repo_id, version=CODEBASE_VERSION, root=None)
        is_video_dataset = lerobot_dataset_info.get("video", False)
        if not is_video_dataset and decode_video:
            raise ValueError(f"The provided dataset is not a video dataset but you have {decode_video=}")
        if is_video_dataset:
            videos_path = load_videos(repo_id, version=CODEBASE_VERSION, root=None)

        data_spec = {}
        video_frame_keys = []
        for k, feature in hf_dataset.features.items():
            if k in DataBuffer.PRESET_KEYS:
                continue
            if isinstance(feature, datasets.features.Image):
                example_img = np.array(hf_dataset[0][k])
                data_spec[k] = {"shape": example_img.shape, "dtype": np.dtype("uint8")}
            elif isinstance(feature, VideoFrame):
                if decode_video:
                    video_dct = hf_dataset[0][k]
                    example_img = decode_video_frames_torchvision(
                        videos_path.parent / video_dct["path"],
                        [video_dct["timestamp"]],
                        1 / lerobot_dataset_info["fps"] - 1e-4,
                    )[0]
                    data_spec[k] = {"shape": example_img.shape, "dtype": np.dtype("uint8")}
                else:
                    video_frame_keys.append(k)
                    data_spec[k] = {
                        "shape": (),
                        "dtype": np.dtype(f"S{MAX_VIDEO_PATH_LENGTH}"),
                    }
            elif isinstance(feature, datasets.features.Sequence):
                data_spec[k] = {"shape": (feature.length,), "dtype": np.dtype(feature.feature.dtype)}
            elif isinstance(feature, datasets.features.Value):
                data_spec[k] = {"shape": (), "dtype": np.dtype(feature.dtype)}
            else:
                raise NotImplementedError(f"Dataset feature type {type(feature)} is not handled.")
        obj = cls(
            **kwargs,
            data_spec=data_spec,
            buffer_capacity=len(hf_dataset),
        )
        data_dict = {}
        for k, feature in hf_dataset.features.items():
            if isinstance(feature, datasets.features.Image):
                data_dict[k] = np.stack(
                    [np.array(pil_img).astype(np.float32) / 255 for pil_img in hf_dataset[k]]
                )
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
                        )
                        all_imgs.extend(episode_imgs.numpy())
                    data_dict[k] = np.stack(all_imgs)
                else:
                    data_dict[k] = np.stack(
                        [np.array(dct["path"], dtype=f"S{MAX_VIDEO_PATH_LENGTH}") for dct in hf_dataset[k]]
                    )
            else:
                data_dict[k] = np.array(hf_dataset[k])
        obj.add_data(data_dict)
        obj.video = False  # TODO(now): HACK
        if len(video_frame_keys) > 0:
            obj.video = True  # TODO(now): HACK
            obj.video_frame_keys = video_frame_keys  # TODO(now): HACK
            # Symlink videos if needed.
            obj.videos_dir = kwargs["storage_dir"] / "videos"
            if not obj.videos_dir.exists():
                os.symlink(videos_path.absolute(), kwargs["storage_dir"])
            obj.videos_dir = Path(kwargs["storage_dir"]) / "videos"  # TODO(now): HACK
            obj.video_backend = "pyav"  # TODO(now): HACK

        return obj


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
