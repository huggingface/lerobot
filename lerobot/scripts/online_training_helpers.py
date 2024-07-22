"""A collection of helper functions and classes for the online training loop in train.py"""

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


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
        super().__init__()
        self.set_delta_timestamps(delta_timestamps)
        self._fps = fps
        self._buffer_capacity = buffer_capacity
        data_spec = self._make_data_spec(data_spec, buffer_capacity)
        os.makedirs(write_dir, exist_ok=True)
        self._data = {
            k: _make_memmap_safe(
                filename=Path(write_dir) / k,
                dtype=v["dtype"] if v is not None else None,
                mode="r+" if (Path(write_dir) / k).exists() else "w+",
                shape=tuple(v["shape"]) if v is not None else None,
            )
            for k, v in data_spec.items()
        }

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
        preset_keys = {"index", "frame_index", "episode_index", "timestamp"}
        if len(intersection := set(data_spec).intersection(preset_keys)) > 0:
            raise ValueError(
                f"data_spec should not contain any of {preset_keys} as these are handled internally. "
                f"The provided data_spec has {intersection}."
            )
        data_spec = {
            # _next_index will be a pointer to the next index that we should start filling from when we add
            # more data.
            "_next_index": {"dtype": np.dtype("int64"), "shape": (1,)},
            # Since the memmap is initialized with all-zeros, this keeps track of which indices are occupied
            # with real data rather than the dummy initialization.
            "_occupancy_mask": {"dtype": np.dtype("?"), "shape": (buffer_capacity,)},
            "index": {"dtype": np.dtype("int64"), "shape": (buffer_capacity,)},
            "frame_index": {"dtype": np.dtype("int64"), "shape": (buffer_capacity,)},
            "episode_index": {"dtype": np.dtype("int64"), "shape": (buffer_capacity,)},
            "timestamp": {"dtype": np.dtype("float64"), "shape": (buffer_capacity,)},
            **{
                k: {"dtype": v["dtype"], "shape": (buffer_capacity, *v["shape"])}
                for k, v in data_spec.items()
            },
        }
        return data_spec

    def add_data(self, data: dict[str, np.ndarray]):
        """Add new data to the buffer, which could potentially mean shifting old data out.

        Shift the incoming data index and episode_index to continue on from the last frame. Note that this
        will be done in place!
        """
        if not set(data) == set(self.data_keys):
            raise ValueError("Missing data keys")
        new_data_length = len(data[self.data_keys[0]])
        if not all(len(data[k]) == new_data_length for k in self.data_keys):
            raise ValueError("All data items should have the same length")

        next_index = self._data["_next_index"][0]

        # Sanity check to make sure that the new data indices start from 0.
        assert data["episode_index"][0].item() == 0
        assert data["index"][0].item() == 0

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

    @property
    def fps(self) -> float | None:
        return self._fps

    @property
    def tolerance_s(self) -> float:
        """
        Tolerance in seconds used to discard loaded frames when their timestamps are not close enough from the
        requested frames. It is only used when `delta_timestamps` is provided.
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
        item = {k: v[idx] for k, v in self._data.items() if not k.startswith("_")}

        if self.delta_timestamps is None:
            return self._item_to_tensors(item)

        episode_index = item["episode_index"]
        current_ts = item["timestamp"]
        episode_data_indices = np.where(
            np.bitwise_and(self._data["episode_index"] == episode_index, self._data["_occupancy_mask"])
        )[0]
        episode_timestamps = self._data["timestamp"][episode_data_indices]

        for data_key in self.delta_timestamps:
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
                ") inside episode range. This might be due to synchronization issues with timestamps during "
                "data collection."
            )

            # Load frames for this data key.
            item[data_key] = self._data[data_key][episode_data_indices[argmin_]]

            item[f"{data_key}_is_pad"] = is_pad

        return self._item_to_tensors(item)


def update_online_buffer(
    online_dataset: OnlineBuffer,
    concat_dataset: torch.utils.data.ConcatDataset,
    sampler: torch.utils.data.WeightedRandomSampler,
    new_data_dict: dict[str, torch.Tensor],
    online_sampling_ratio: float,
):
    """
    Modifies the online_dataset, concat_dataset, and sampler in place by integrating new episodes from
    new_data_dict into the online_dataset, updating the concatenated dataset's structure and adjusting the
    sampling strategy based on the specified percentage of online samples.

    Args:
        online_dataset: The existing online dataset to be updated.
        concat_dataset: The concatenated PyTorch Dataset that combines offline and online datasets (in that
            order), used for sampling purposes.
        sampler: A sampler that will be updated to reflect changes in the dataset sizes and specified sampling
            weights.
        new_data_dict: A mapping from data key to data tensor containing the new episodes to be added.
        online_sampling_ratio: The target percentage of samples that should come from the online dataset
            during sampling operations.
    """
    online_dataset.add_data(new_data_dict)

    # Update the concatenated dataset length used during sampling.
    concat_dataset.cumulative_sizes = concat_dataset.cumsum(concat_dataset.datasets)

    # Update the sampling weights for each frame.
    len_online = len(online_dataset)
    len_offline = len(concat_dataset) - len_online
    sampler.weights = torch.tensor(
        [(1 - online_sampling_ratio) / len_offline] * len_offline
        + [online_sampling_ratio / len_online] * len_online
    )

    # Update the total number of samples used during sampling
    sampler.num_samples = len(concat_dataset)
