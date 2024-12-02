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
"""An online buffer for the online training loop in train.py

Note to maintainers: This duplicates some logic from LeRobotDataset and EpisodeAwareSampler. We should
consider converging to one approach. Here we have opted to use numpy.memmap to back the data buffer. It's much
faster than using HuggingFace Datasets as there's no conversion to an intermediate non-python object. Also it
supports in-place slicing and mutation which is very handy for a dynamic buffer.
"""

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


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
                f"You're about to take up {required_space} of {available_space} bytes available."
            )
    return np.memmap(**kwargs)


class OnlineBuffer(torch.utils.data.Dataset):
    """FIFO data buffer for the online training loop in train.py.

    Follows the protocol of LeRobotDataset as much as is required to have it be used by the online training
    loop in the same way that a LeRobotDataset would be used.

    The underlying data structure will have data inserted in a circular fashion. Always insert after the
    last index, and when you reach the end, wrap around to the start.

    The data is stored in a numpy memmap.
    """

    NEXT_INDEX_KEY = "_next_index"
    OCCUPANCY_MASK_KEY = "_occupancy_mask"
    INDEX_KEY = "index"
    FRAME_INDEX_KEY = "frame_index"
    EPISODE_INDEX_KEY = "episode_index"
    TIMESTAMP_KEY = "timestamp"
    IS_PAD_POSTFIX = "_is_pad"

    def __init__(
        self,
        write_dir: str | Path,
        data_spec: dict[str, Any] | None,
        buffer_capacity: int | None,
        fps: float | None = None,
        delta_timestamps: dict[str, list[float]] | dict[str, np.ndarray] | None = None,
    ):
        """
        The online buffer can be provided from scratch or you can load an existing online buffer by passing
        a `write_dir` associated with an existing buffer.

        Args:
            write_dir: Where to keep the numpy memmap files. One memmap file will be stored for each data key.
                Note that if the files already exist, they are opened in read-write mode (used for training
                resumption.)
            data_spec: A mapping from data key to data specification, like {data_key: {"shape": tuple[int],
                "dtype": np.dtype}}. This should include all the data that you wish to record into the buffer,
                but note that "index", "frame_index" and "episode_index" are already accounted for by this
                class, so you don't need to include them.
            buffer_capacity: How many frames should be stored in the buffer as a maximum. Be aware of your
                system's available disk space when choosing this.
            fps: Same as the fps concept in LeRobot dataset. Here it needs to be provided for the
                 delta_timestamps logic. You can pass None if you are not using delta_timestamps.
            delta_timestamps: Same as the delta_timestamps concept in LeRobotDataset. This is internally
                converted to dict[str, np.ndarray] for optimization purposes.

        """
        self.set_delta_timestamps(delta_timestamps)
        self._fps = fps
        # Tolerance in seconds used to discard loaded frames when their timestamps are not close enough from
        # the requested frames. It is only used when `delta_timestamps` is provided.
        # minus 1e-4 to account for possible numerical error
        self.tolerance_s = 1 / self.fps - 1e-4 if fps is not None else None
        self._buffer_capacity = buffer_capacity
        data_spec = self._make_data_spec(data_spec, buffer_capacity)
        Path(write_dir).mkdir(parents=True, exist_ok=True)
        self._data = {}
        for k, v in data_spec.items():
            self._data[k] = _make_memmap_safe(
                filename=Path(write_dir) / k,
                dtype=v["dtype"] if v is not None else None,
                mode="r+" if (Path(write_dir) / k).exists() else "w+",
                shape=tuple(v["shape"]) if v is not None else None,
            )

    @property
    def delta_timestamps(self) -> dict[str, np.ndarray] | None:
        return self._delta_timestamps

    def set_delta_timestamps(self, value: dict[str, list[float]] | None):
        """Set delta_timestamps converting the values to numpy arrays.

        The conversion is for an optimization in the __getitem__. The loop is much slower if the arrays
        need to be converted into numpy arrays.
        """
        if value is not None:
            self._delta_timestamps = {k: np.array(v) for k, v in value.items()}
        else:
            self._delta_timestamps = None

    def _make_data_spec(self, data_spec: dict[str, Any], buffer_capacity: int) -> dict[str, dict[str, Any]]:
        """Makes the data spec for np.memmap."""
        if any(k.startswith("_") for k in data_spec):
            raise ValueError(
                "data_spec keys should not start with '_'. This prefix is reserved for internal logic."
            )
        preset_keys = {
            OnlineBuffer.INDEX_KEY,
            OnlineBuffer.FRAME_INDEX_KEY,
            OnlineBuffer.EPISODE_INDEX_KEY,
            OnlineBuffer.TIMESTAMP_KEY,
        }
        if len(intersection := set(data_spec).intersection(preset_keys)) > 0:
            raise ValueError(
                f"data_spec should not contain any of {preset_keys} as these are handled internally. "
                f"The provided data_spec has {intersection}."
            )
        complete_data_spec = {
            # _next_index will be a pointer to the next index that we should start filling from when we add
            # more data.
            OnlineBuffer.NEXT_INDEX_KEY: {"dtype": np.dtype("int64"), "shape": ()},
            # Since the memmap is initialized with all-zeros, this keeps track of which indices are occupied
            # with real data rather than the dummy initialization.
            OnlineBuffer.OCCUPANCY_MASK_KEY: {"dtype": np.dtype("?"), "shape": (buffer_capacity,)},
            OnlineBuffer.INDEX_KEY: {"dtype": np.dtype("int64"), "shape": (buffer_capacity,)},
            OnlineBuffer.FRAME_INDEX_KEY: {"dtype": np.dtype("int64"), "shape": (buffer_capacity,)},
            OnlineBuffer.EPISODE_INDEX_KEY: {"dtype": np.dtype("int64"), "shape": (buffer_capacity,)},
            OnlineBuffer.TIMESTAMP_KEY: {"dtype": np.dtype("float64"), "shape": (buffer_capacity,)},
        }
        for k, v in data_spec.items():
            complete_data_spec[k] = {"dtype": v["dtype"], "shape": (buffer_capacity, *v["shape"])}
        return complete_data_spec

    def add_data(self, data: dict[str, np.ndarray]):
        """Add new data to the buffer, which could potentially mean shifting old data out.

        The new data should contain all the frames (in order) of any number of episodes. The indices should
        start from 0 (note to the developer: this can easily be generalized). See the `rollout` and
        `eval_policy` functions in `eval.py` for more information on how the data is constructed.

        Shift the incoming data index and episode_index to continue on from the last frame. Note that this
        will be done in place!
        """
        if len(missing_keys := (set(self.data_keys).difference(set(data)))) > 0:
            raise ValueError(f"Missing data keys: {missing_keys}")
        new_data_length = len(data[self.data_keys[0]])
        if not all(len(data[k]) == new_data_length for k in self.data_keys):
            raise ValueError("All data items should have the same length")

        next_index = self._data[OnlineBuffer.NEXT_INDEX_KEY]

        # Sanity check to make sure that the new data indices start from 0.
        assert data[OnlineBuffer.EPISODE_INDEX_KEY][0].item() == 0
        assert data[OnlineBuffer.INDEX_KEY][0].item() == 0

        # Shift the incoming indices if necessary.
        if self.num_frames > 0:
            last_episode_index = self._data[OnlineBuffer.EPISODE_INDEX_KEY][next_index - 1]
            last_data_index = self._data[OnlineBuffer.INDEX_KEY][next_index - 1]
            data[OnlineBuffer.EPISODE_INDEX_KEY] += last_episode_index + 1
            data[OnlineBuffer.INDEX_KEY] += last_data_index + 1

        # Insert the new data starting from next_index. It may be necessary to wrap around to the start.
        n_surplus = max(0, new_data_length - (self._buffer_capacity - next_index))
        for k in self.data_keys:
            if n_surplus == 0:
                slc = slice(next_index, next_index + new_data_length)
                self._data[k][slc] = data[k]
                self._data[OnlineBuffer.OCCUPANCY_MASK_KEY][slc] = True
            else:
                self._data[k][next_index:] = data[k][:-n_surplus]
                self._data[OnlineBuffer.OCCUPANCY_MASK_KEY][next_index:] = True
                self._data[k][:n_surplus] = data[k][-n_surplus:]
        if n_surplus == 0:
            self._data[OnlineBuffer.NEXT_INDEX_KEY] = next_index + new_data_length
        else:
            self._data[OnlineBuffer.NEXT_INDEX_KEY] = n_surplus

    @property
    def data_keys(self) -> list[str]:
        keys = set(self._data)
        keys.remove(OnlineBuffer.OCCUPANCY_MASK_KEY)
        keys.remove(OnlineBuffer.NEXT_INDEX_KEY)
        return sorted(keys)

    @property
    def fps(self) -> float | None:
        return self._fps

    @property
    def num_episodes(self) -> int:
        return len(
            np.unique(self._data[OnlineBuffer.EPISODE_INDEX_KEY][self._data[OnlineBuffer.OCCUPANCY_MASK_KEY]])
        )

    @property
    def num_frames(self) -> int:
        return np.count_nonzero(self._data[OnlineBuffer.OCCUPANCY_MASK_KEY])

    def __len__(self):
        return self.num_frames

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
        if idx >= len(self) or idx < -len(self):
            raise IndexError

        item = {k: v[idx] for k, v in self._data.items() if not k.startswith("_")}

        if self.delta_timestamps is None:
            return self._item_to_tensors(item)

        episode_index = item[OnlineBuffer.EPISODE_INDEX_KEY]
        current_ts = item[OnlineBuffer.TIMESTAMP_KEY]
        episode_data_indices = np.where(
            np.bitwise_and(
                self._data[OnlineBuffer.EPISODE_INDEX_KEY] == episode_index,
                self._data[OnlineBuffer.OCCUPANCY_MASK_KEY],
            )
        )[0]
        episode_timestamps = self._data[OnlineBuffer.TIMESTAMP_KEY][episode_data_indices]

        for data_key in self.delta_timestamps:
            # Note: The logic in this loop is copied from `load_previous_and_future_frames`.
            # Get timestamps used as query to retrieve data of previous/future frames.
            query_ts = current_ts + self.delta_timestamps[data_key]

            # Compute distances between each query timestamp and all timestamps of all the frames belonging to
            # the episode.
            dist = np.abs(query_ts[:, None] - episode_timestamps[None, :])
            argmin_ = np.argmin(dist, axis=1)
            min_ = dist[np.arange(dist.shape[0]), argmin_]

            is_pad = min_ > self.tolerance_s

            # Check violated query timestamps are all outside the episode range.
            assert (
                (query_ts[is_pad] < episode_timestamps[0]) | (episode_timestamps[-1] < query_ts[is_pad])
            ).all(), (
                f"One or several timestamps unexpectedly violate the tolerance ({min_} > {self.tolerance_s=}"
                ") inside the episode range."
            )

            # Load frames for this data key.
            item[data_key] = self._data[data_key][episode_data_indices[argmin_]]

            item[f"{data_key}{OnlineBuffer.IS_PAD_POSTFIX}"] = is_pad

        return self._item_to_tensors(item)

    def get_data_by_key(self, key: str) -> torch.Tensor:
        """Returns all data for a given data key as a Tensor."""
        return torch.from_numpy(self._data[key][self._data[OnlineBuffer.OCCUPANCY_MASK_KEY]])


def compute_sampler_weights(
    offline_dataset: LeRobotDataset,
    offline_drop_n_last_frames: int = 0,
    online_dataset: OnlineBuffer | None = None,
    online_sampling_ratio: float | None = None,
    online_drop_n_last_frames: int = 0,
) -> torch.Tensor:
    """Compute the sampling weights for the online training dataloader in train.py.

    Args:
        offline_dataset: The LeRobotDataset used for offline pre-training.
        online_drop_n_last_frames: Number of frames to drop from the end of each offline dataset episode.
        online_dataset: The OnlineBuffer used in online training.
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
