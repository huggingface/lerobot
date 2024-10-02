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
"""A dataset class for efficient data management during offline and online training."""

import json
import logging
import os
import shutil
import tempfile
from copy import copy
from enum import Enum
from pathlib import Path
from typing import Callable

import datasets
import einops
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, DATA_DIR, LeRobotDataset
from lerobot.common.datasets.utils import load_episode_data_index, load_hf_dataset, load_info, load_videos
from lerobot.common.datasets.video_utils import (
    VideoFrame,
    decode_video_frames_torchvision,
    encode_video_frames,
)

# Take no more than 80% of the available storage space when creating memmaps.
MEMMAP_STORAGE_PCT_CAP = 0.8


# Some descriptive custom error classes for testing purposes and to some degree for informing the user.
class TimestampOutsideToleranceError(Exception):
    pass


class DiskSpaceError(Exception):
    pass


class StorageDirCorruptError(Exception):
    pass


# Custom exception for testing purpose only. The user will never see this. See unit-tests.
class _TestRollBackError(Exception):
    pass


class LeRobotDatasetV2ImageMode(Enum):
    MEMMAP = "memmap"
    PNG = "png"
    VIDEO = "video"

    @staticmethod
    def needs_decoding(mode: "LeRobotDatasetV2ImageMode") -> bool:
        """Return true if the provided image mode means decoding is needed, else False."""
        return mode in [LeRobotDatasetV2ImageMode.PNG, LeRobotDatasetV2ImageMode.VIDEO]


class LeRobotDatasetV2(torch.utils.data.Dataset):
    """
    This class is intended for the following main use cases:
    - Downloading and using datasets from the Hugging Face Hub.
    - Creating new datasets and potentially uploading them to the Hugging Face Hub.
    - Use as an online experience replay buffer during training.

    Conceptually, data is considered to come in the form of "episodes" (an instance of a robot performing a
    task). Episodes are made up of "frames", which are chronologically ordered and contain timestamp aligned
    data, potentially including environment observations, and robot actions. NOTE: for the time being, we
    require all data modalities to be timestamp aligned. This constraint may be relaxed in the future.

    Physically, data is stored on disk in one or more of three forms: video files, PNG files, `numpy.memmap`s.
    Image data can be stored in any of these forms, while non-image data is only stored in `numpy.memmaps`.

    A quick note on `numpy.memmap`s: memory mapping (https://en.wikipedia.org/wiki/Memory-mapped_file)
    allows us to treat a portion of disk space as virtual memory. This allows us to work with more data than
    can fit in our physical memory, while treating the data as if it were just standard numpy arrays.

    All of the on-disk storage goes in one directory on your file system. In here you will find:
      - One memmap file for each non-image data key, and perhaps one for each image data key.
      - Perhaps a directory called "videos".
      - Perhaps a directory called "images"
      - A "metadata.json" file which contains human-readable information about the dataset and some
        configuration parameters.

    `numpy.memmap` data is stored in a mapping from data keys to arrays with shape (total_number_of_frames,
    *data_dim). The compulsory data keys are:
        - "index": A sequential integer index per frame.
        - "episode_index": A sequential integer index per episode.
        - "frame_index": A sequential integer index per frame within an episode (it resets for each episode).
        - "timestamp": The relative timestamp of the frame within the episode in units of seconds. The choice.
            of reference time is not important.
    Other data keys may be included, and when using LeRobot policies, one should note LeRobot's implicit
    naming conventions for data keys:
        - "action": A 1D vector for the robot command.
        - "observation.state": A 1D vector for the proprioceptive state.
        - "observation.environment_state": A 1D vector encoding the environment state (for example: the
            poses of objects in the environment).
        - Any key starting with "observation.image" is considered to be a camera image. Note that this doesn't
          necessarily get stored in this data structure. Instead, we might have video files or PNG files to
          decode (more on that below).

    Image data may be stored in at least one of the following formats: video files, PNG image files, and as
    raw pixel arrays (in which case they also have a data key to array mapping as discussed above). For each
    of these methods, we care about storage capacity, random access data loading speed, and in some use cases,
    how fast we can add new data.
        - Video files: These are the most compact in terms of storage space. Video decoding can be faster
            than PNG image file decoding when we need to access multiple sequential frames at a time,
            otherwise it is generally slower. Video encoding time, when adding new data, is also something to
            consider.
        - PNG files: These are less compact than videos in terms of storage space. When randomly accessing
            individual image frames from the dataset, decoding PNG files can be significantly faster than
            decoding video files. PNG encoding time, when adding new data, is also something to consider.
        - Numpy memmaps (more on memmaps below): These are by far the least compact in terms of storage space.
            They are also the fastest option for adding new episodes to the dataset and data loading. But
            under certain settings, video files and PNG images can get surprisingly close when using a PyTorch
            DataLoader with multiple workers.

    This class is also a PyTorch Dataset and can be used as such in a dataloader for a training loop. The item
    getter returns either a single frame, or a slice of a single episode if `delta_timestamps` is set. It also
    converts the numpy data to torch tensors, and handles converting images to channel-first, float32
    normalized to the range [0, 1].

    USE CASES AND EXAMPLES:

    The most common use case will be to download a dataset from Hugging Face Hub for training a policy.
    The following examples downloads the PushT dataset from the hub (or uses the locally cached dataset if it
    exists) and passes it to a PyTorch dataloader.

    ```python
    dataset = LeRobotDatasetV2.from_huggingface_hub("lerobot/pusht")
    dataloader = torch.utils.data.DataLoader(dataset)
    ```

    The next most common use case is to create a dataset from scratch, and push it to the hub in video format.
    For example:

    ```python
    dataset = LeRobotDatasetV2("path/to/new/dataset")
    # OR, if you know the size of the dataset (in number of frames) up front, provide a buffer capacity up
    # front for more efficient handling of the underlying memmaps.
    dataset = LeRobotDatasetV2("path/to/new/dataset", buffer_capacity=25000)

    # Add episodes to the dataset.
    for _ in range(num_episodes):
        # Create a dictionary mapping data keys to arrays of shape (num_frames_in_episode, *).
        dataset.add_episodes(data_dict)
        TODO(alexander-soare): Push to hub
    ```

    Finally, one may also use LeRobotDatasetV2 as an experience replay buffer for online RL algorithms.

    ```python
    # Note: Other image modes could be used although if you can fit the buffer on disk, you should probably
    # stick to memmap mode.
    dataset = LeRobotDatasetV2(
        "online_buffer", image_mode="memmap", buffer_capacity=10000, use_as_filo_buffer=True
    )
    iter_dataloader = iter(torch.utils.data.DataLoader(dataset))

    # training loop
    while True:
        data_dict = do_online_rollouts()
        # Here, if the new frames exceed the capacity of the buffer, the oldest frames are shifted out to
        # make space.
        dataset.add_episodes(data_dict)
        batch = next(iter_dataloader)
        # Policy forward, backward, gradient step.
    ```
    """

    # Special key for a (1,) array storing a pointer to the next index to fill from when adding data.
    NEXT_INDEX_KEY = "_next_index"
    # Since the numpy.memmap storage is pre-allocated, this boolean mask is used to indicate which frames have
    # "real" data.
    OCCUPANCY_MASK_KEY = "_occupancy_mask"
    # IS_PAD_POSTFIX is added by the __getitem__ method, and is not stored in the numpy.memmap. It is used to
    # indicate that a frame is padding.
    IS_PAD_POSTFIX = "_is_pad"
    INDEX_KEY = "index"
    FRAME_INDEX_KEY = "frame_index"
    EPISODE_INDEX_KEY = "episode_index"
    TIMESTAMP_KEY = "timestamp"
    PRESET_KEYS = {INDEX_KEY, FRAME_INDEX_KEY, EPISODE_INDEX_KEY, TIMESTAMP_KEY}
    # By convention, all images should be stored under a key with this prefix.
    IMAGE_KEY_PREFIX = "observation.image"

    METADATA_FILE_NAME = "metadata.json"

    VIDEOS_DIR = "videos"  # directory (relative to storage directory), to store videos
    VIDEO_NAME_FSTRING = "{data_key}_episode_{episode_index:06d}.mp4"
    PNGS_DIR = "images"  # directory (relative to storage directory), to store images
    PNG_NAME_FSTRING = "{data_key}_episode_{episode_index:06d}_frame_{frame_index:06d}.png"

    def __init__(
        self,
        storage_dir: str | Path,
        fps: float | None = None,
        buffer_capacity: int | None = None,
        use_as_filo_buffer: bool = False,
        image_mode: LeRobotDatasetV2ImageMode | str = LeRobotDatasetV2ImageMode.VIDEO,
        image_transform: Callable[[np.ndarray], np.ndarray] | None = None,
        delta_timestamps: dict[str, list[float]] | dict[str, np.ndarray] | None = None,
    ):
        """
        Args:
            storage_dir: Where to keep the numpy memmap files, metadata file, video files, and/or image files.
                One memmap file will be stored for each data key. Note that if the storage directory already
                exists, the memmap files are opened in read-write mode. If the storage directory does not
                exist, it will be lazily created with the first call to `add_episodes`.
            fps: Frames rate (frames per second) for all of the episodes in this dataset. This is used in two
                places: video encoding, and computing property `tolerance_s` for retrieving frames via their
                timestamp. Doesn't have to be provided if an existing dataset is being loaded (
                that is, the storage_directory already exists).
            buffer_capacity: Sets the size of the preallocated storage space for the memmaps in terms of
                frames. If not provided, the memmap storage space is dynamically expanded as episodes are
                added (doubling every time extra space is needed). If you know the number of frames you plan
                to add in advance, it is recommended that this parameter be provided for efficiency. If
                provided, attempts to add data beyond the capacity will result in an exception (unless
                `use_as_filo_buffer` is set). If provided with an existing storage directory, the existing
                memmaps will be expanded to the provided capacity if needed.
            use_as_filo_buffer: Set this to use the dataset as an online replay buffer. Calls to `add_episode`
                beyond the buffer_capacity, will result in the oldest episodes being pushed out of the buffer
                to make way for the new ones (first-in-last-out aka FILO).
            image_mode: The image storage mode used for the item getter. Options are: "video", "png",
                "memmap". See notes above for more information on the various options. If not provided it
                defaults to video mode.
            image_transform: Transforms to apply in the item getter to all image data (any data whose key
                starts with "observation.image").
            delta_timestamps: TODO(alexander-soare): Document this somewhere when
                `load_previous_and_future_frames` is refactored.
        """
        if use_as_filo_buffer and buffer_capacity is None:
            raise ValueError(f"A buffer_capacity must be provided if {use_as_filo_buffer=}.")

        self._storage_dir = Path(storage_dir)
        self._data: dict[str, np.memmap] = {}
        self._use_as_filo_buffer = use_as_filo_buffer
        self._videos_dir: str | None = None
        self._images_dir: str | None = None

        # If the storage directory and already exists, load the memmaps.
        if self._storage_dir.exists():
            # Try to catch corrupt data ahead of time.
            LeRobotDatasetV2.check_storage_dir_integrity(self._storage_dir)
            metadata = self._load_metadata()
            self._fps = metadata["fps"]
            self._make_memmaps(metadata["_data_spec"], mode="r+")
            if buffer_capacity is not None:
                current_capacity = len(self._data[self.INDEX_KEY])
                if buffer_capacity < current_capacity:
                    raise ValueError(
                        f"The storage directory already exists and contains a memmaps with capacity: "
                        f"{current_capacity} frames. But you have provided "
                        f"{buffer_capacity=}. It is required that buffer_capacity >= {current_capacity}"
                    )
                if buffer_capacity > current_capacity:
                    self._extend_memmaps(new_length=buffer_capacity)
                self._buffer_capacity = buffer_capacity
            else:
                self._buffer_capacity = buffer_capacity
            # Set image mode based on what's available in the storage directory and/or the user's selection.
            if any(k.startswith(LeRobotDatasetV2.IMAGE_KEY_PREFIX) for k in metadata["data_keys"]):
                possible_image_modes = self._infer_image_modes()
                if image_mode not in possible_image_modes:
                    raise ValueError(
                        f"Provided image_mode {str(image_mode)} not available with this storage directory. "
                        f"Modes available: {[str(m) for m in possible_image_modes]}"
                    )
        else:
            if fps is None:
                raise ValueError("fps must be provided when creating a new dataset")
            self._fps = fps
            self._buffer_capacity = buffer_capacity

        self._image_mode = LeRobotDatasetV2ImageMode(image_mode)

        self.set_delta_timestamps(delta_timestamps)
        self.image_transform = image_transform

        # For unit testing purposes.
        self.__test_roll_back: int = -1

    @property
    def storage_dir(self) -> Path:
        return self._storage_dir

    @property
    def data_keys(self) -> list[str]:
        """Return the names of all data keys in the dataset.

        Exclude "private" internal keys.
        """
        metadata = self._load_metadata()
        return list(metadata["data_keys"])

    @property
    def camera_keys(self) -> list[str]:
        """Return the names of all data keys pertaining to camera observations.

        By convention, this is all the keys starting with "observation.image".
        """
        metadata = self._load_metadata()
        return [k for k in metadata["data_keys"] if k.startswith(self.IMAGE_KEY_PREFIX)]

    @property
    def fps(self) -> float | None:
        return self._fps

    @property
    def num_episodes(self) -> int:
        """Total number of unique episode indices in the dataset."""
        if len(self._data) == 0:
            return 0
        return len(np.unique(self._data[self.EPISODE_INDEX_KEY][self._data[self.OCCUPANCY_MASK_KEY]]))

    @property
    def num_samples(self) -> int:
        """Total number of unique samples (aka frames) in the dataset.

        TODO(alexander-soare): Rename to num_frames once LeRobotDataset is deprecated.
        """
        if len(self._data) == 0:
            return 0
        return np.count_nonzero(self._data[self.OCCUPANCY_MASK_KEY])

    def get_unique_episode_indices(self) -> np.ndarray:
        return np.unique(self._data[self.EPISODE_INDEX_KEY][self._data[self.OCCUPANCY_MASK_KEY]])

    def _infer_image_modes(self) -> list[LeRobotDatasetV2ImageMode]:
        """Infer which image modes are available according to what is in the storage directory"""
        image_modes = []
        if (self.storage_dir / self.VIDEOS_DIR).exists():
            image_modes.append(LeRobotDatasetV2ImageMode.VIDEO)
        if (self.storage_dir / self.PNGS_DIR).exists():
            image_modes.append(LeRobotDatasetV2ImageMode.PNG)
        data_spec = self._load_metadata()["_data_spec"]
        if any(k.startswith(self.IMAGE_KEY_PREFIX) for k in data_spec):
            image_modes.append(LeRobotDatasetV2ImageMode.MEMMAP)
        return image_modes

    def _save_metadata(
        self,
        data_keys: list[str] | None = None,
        data_spec: dict | None = None,
    ):
        """Save or update the metadata file in the storage directory.

        If the metadata file already exists, it is updated with the provided parameters, otherwise a new
        metadata file is created. There is no mechanism for clearing a field in the metadata file.

        Args:
            data_keys: All the data keys of the data set. Used for human readability.
            data_spec: `numpy.memmap` data spec used for loading the memmaps.
        """
        metadata_file = self._storage_dir / self.METADATA_FILE_NAME
        if not metadata_file.exists() and any([data_spec is None, data_keys is None]):
            raise AssertionError("The first time _save_metadata is called, all arguments should be provided.")
        metadata = self._load_metadata() if metadata_file.exists() else {}

        # Go through each provided argument and internal parameter, updating the metadata.
        metadata["fps"] = self._fps
        if data_keys is not None:
            metadata["data_keys"] = copy(data_keys)
        if data_spec is not None:
            data_spec = dict(data_spec)
            metadata["_data_spec"] = copy(data_spec)

        # Custom serialization.
        for k in metadata["_data_spec"]:
            metadata["_data_spec"][k]["dtype"] = str(metadata["_data_spec"][k]["dtype"])

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def _load_metadata(self) -> dict:
        """Load the data type and shape specifications from the storage directory."""
        with open(self._storage_dir / self.METADATA_FILE_NAME) as f:
            metadata = json.load(f)
        # Custom deserialization.
        for k in metadata["_data_spec"]:
            metadata["_data_spec"][k]["dtype"] = np.dtype(metadata["_data_spec"][k]["dtype"])
        return metadata

    def _make_storage_dir(self, episode_data: dict[str, np.ndarray], from_huggingface_hub: bool = False):
        """Create the storage directory based on example episode data from the first `add_episodes` call."""
        # Note: from_huggingface_hub=True allows the `from_huggingface_hub` method to do the hack of copying
        # video/png files over in advance. Therefore, we'll first make sure that it is indeed the case that
        # all that exists in the storage directory is either the video or png files.
        if from_huggingface_hub:
            if self._image_mode == LeRobotDatasetV2ImageMode.VIDEO:
                assert os.listdir(self._storage_dir) == [LeRobotDatasetV2.VIDEOS_DIR]
            elif self._image_mode == LeRobotDatasetV2ImageMode.PNG:
                assert os.listdir(self._storage_dir) == [LeRobotDatasetV2.PNGS_DIR]
        else:
            assert not self._storage_dir.exists()
        self._storage_dir.mkdir(parents=True, exist_ok=from_huggingface_hub)

        if self._buffer_capacity is None:
            # Reserve enough storage for one episode. Storage will be extended as needed.
            num_frames = len(episode_data[self.INDEX_KEY])
        else:
            num_frames = self._buffer_capacity

        # Make the data spec for np.memmap
        data_spec = {
            self.NEXT_INDEX_KEY: {"dtype": np.dtype("int64"), "shape": (1,)},
            self.OCCUPANCY_MASK_KEY: {"dtype": np.dtype("?"), "shape": (num_frames,)},
            self.INDEX_KEY: {"dtype": np.dtype("int64"), "shape": (num_frames,)},
            self.FRAME_INDEX_KEY: {"dtype": np.dtype("int64"), "shape": (num_frames,)},
            self.EPISODE_INDEX_KEY: {"dtype": np.dtype("int64"), "shape": (num_frames,)},
            self.TIMESTAMP_KEY: {"dtype": np.dtype("float32"), "shape": (num_frames,)},
        }
        for k, v in episode_data.items():
            if k in data_spec:
                continue
            is_image_key = k.startswith(self.IMAGE_KEY_PREFIX)
            if is_image_key and self._image_mode == LeRobotDatasetV2ImageMode.VIDEO:
                (self._storage_dir / self.VIDEOS_DIR).mkdir()
            elif is_image_key and self._image_mode == LeRobotDatasetV2ImageMode.PNG:
                (self._storage_dir / self.PNGS_DIR).mkdir()
            else:
                data_spec[k] = {"dtype": v.dtype, "shape": (num_frames, *v.shape[1:])}

        try:
            self._make_memmaps(data_spec, "w+")
            self._save_metadata(data_spec=data_spec, data_keys=list(episode_data))
        except Exception as e:
            # Clean up (except if `from_huggingface_hub` is set, in which case we let that method take
            # responsibility for the cleanup).
            if not from_huggingface_hub:
                shutil.rmtree(self._storage_dir)
            raise RuntimeError(
                "An exception was caught while attempting to create the storage directory for "
                f"{self.__class__.__name__}. As part of the cleanup, the storage directory was removed."
            ) from e

    def _make_memmaps(self, data_spec: dict[str, dict], mode: str):
        """Create the memmap objects.

        The underlying storage directory may or may not already exist. Provide the file opening `mode`
        accordingly.
        """
        # First check that this will not use up most or all of the storage space.
        required_space = 0
        for spec in data_spec.values():
            required_space += spec["dtype"].itemsize * np.prod(spec["shape"])  # bytes
        stats = os.statvfs(self._storage_dir)
        available_space = stats.f_bavail * stats.f_frsize  # bytes
        if required_space >= available_space * MEMMAP_STORAGE_PCT_CAP:
            raise DiskSpaceError(
                f"You're about to take up {required_space} of {available_space} bytes available. This "
                "exception has been raised to protect your storage device."
                ""
            )

        for k, v in data_spec.items():
            self._data[k] = np.memmap(
                filename=self._storage_dir / k,
                dtype=v["dtype"] if v is not None else None,
                mode=mode,
                shape=tuple(v["shape"]) if v is not None else None,
            )

    def _extend_memmaps(self, new_length: int):
        """Increase the frame capacity of the memmaps to new_length."""
        assert (
            len(self._data[self.INDEX_KEY]) < new_length
        ), "new_length must be more than the current capacity of the memmaps"
        data_spec = self._load_metadata()["_data_spec"]
        for k in data_spec:
            if k == self.NEXT_INDEX_KEY:
                continue
            data_spec[k]["shape"][0] = new_length
        self._make_memmaps(data_spec=data_spec, mode="r+")
        self._save_metadata(data_spec=data_spec)

    @property
    def delta_timestamps(self) -> dict[str, np.ndarray] | None:
        return self._delta_timestamps

    @property
    def tolerance_s(self) -> float | None:
        """
        Tolerance (in seconds) used to discard loaded frames when their timestamps are not close enough to
        the requested frames. It is only used when `delta_timestamps` is provided. The -1e-4 accounts for
        possible numerical error.
        """
        return 1 / self._fps - 1e-4

    def set_delta_timestamps(self, delta_timestamps: dict[str, list[float]] | None):
        """Set delta_timestamps converting the values to numpy arrays.

        Note: The conversion is for an optimization in the __getitem__. The loop is much slower if lists need
        to be converted into numpy arrays.
        """
        if delta_timestamps is not None:
            self._delta_timestamps = {k: np.array(v) for k, v in delta_timestamps.items()}
        else:
            self._delta_timestamps = None

    def add_episodes(self, data: dict[str, np.ndarray], no_flush: bool = False):
        """Add episodes to the dataset.

        `data` should be a mapping of data key to data in array format. It should contain at least one
        episode. The episodes should have frame indices that start from 0 and step up in increments of 1.
        All image data should be np.uint8, channel-last.

        Episodes are added to the dataset one-by-one. If an episode has more frames then are available till
        the end of the numpy.memmap buffer, there are several possibilities:
            - If `buffer_capacity` was not provided at initialization, the memmaps are doubled in size.
            - If `buffer_capacity` was provided and `use_as_filo_buffer=False`, an exception will be raised.
            - If `buffer_capacity` was provided and `use_as_filo_buffer=True`, the data insertion pointer is
                reset to the start of the memmap and the episode is inserted there, overwriting existing
                episode frames. When episode frames are overwritten by a new episode, any remaining frames
                belonging to the existing episode are left in place (meaning not all episodes will be
                guaranteed to start from their frame 0).

        After adding the episodes to the dataset, the numpy.memmap buffer is flushed to disk, unless
        `no_flush=True` is provided. In a loop where one episode is added at a time, providing `no_flush=True`
        may provide speed advantages. The user should just remember to call `flush()` after finishing the
        loop.
        """
        for episode_index in np.unique(data[self.EPISODE_INDEX_KEY]):
            where_episode = np.where(data[self.EPISODE_INDEX_KEY] == episode_index)[0]
            episode_data = {k: data[k][where_episode] for k in data}
            self._add_episode(episode_data)

        if not no_flush:
            self.flush()

    def _add_episode(self, data: dict[str, np.ndarray], from_huggingface_hub: bool = False):
        """Add a single episode to the dataset.

        Also manages FILO logic.

        Also attempt to roll back any changes if an exception occurs mid way.

        Setting `from_huggingface_hub` is only intended for calls from that method. This is a hack that allows
        `from_huggingface_hub` to create the video or png directories ahead of time.
        """
        if len(self._data) == 0:
            self._make_storage_dir(data, from_huggingface_hub=from_huggingface_hub)

        if len(missing_keys := (set(self.data_keys).difference(set(data)))) > 0:
            raise ValueError(f"Missing data keys: {missing_keys}")
        new_data_length = len(data[self.data_keys[0]])
        if new_data_length <= 0:
            raise ValueError("The episode has 0 frames")
        if self._buffer_capacity is not None and new_data_length > self._buffer_capacity:
            raise ValueError("The episode length is larger than the total buffer capacity.")
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
        # Special checks on image keys.
        for k in data:
            if not k.startswith(self.IMAGE_KEY_PREFIX):
                continue
            _, h, w, c = data[k].shape
            if data[k].dtype is not np.dtype("uint8") or c >= min(h, w):
                raise ValueError(
                    f"Any data key starting with '{self.IMAGE_KEY_PREFIX}' is assumed to be an image, "
                    "and should be of type np.uint8, with channel-last format."
                )
        if (
            not from_huggingface_hub
            and LeRobotDatasetV2ImageMode.needs_decoding(self._image_mode)
            and not any(k.startswith(self.IMAGE_KEY_PREFIX) for k in data)
        ):
            raise ValueError(
                f"Since the image mode is {str(self._image_mode)}, all added episodes are expected to have "
                f"image data (that is, at least one key that starts with {self.IMAGE_KEY_PREFIX})."
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

        new_episode_index = last_episode_index + 1

        # Handle situation if there aren't enough empty frames left in the memmaps to handle the new episode.
        capacity = len(self._data[self.INDEX_KEY]) if self._buffer_capacity is None else self._buffer_capacity
        n_excess = max(0, new_data_length - (capacity - next_index))
        if n_excess > 0:
            if self._buffer_capacity is None:
                # A buffer capacity was not explicitly provided, so dynamically resize the memmaps (double the
                # capacity).
                self._extend_memmaps(capacity * 2)
            elif self._use_as_filo_buffer:
                # A buffer capacity was provided and we wish to use the dataset as a FILO buffer. Wrap to the
                # start.
                next_index = 0
            else:
                # A buffer capacity was provided meaning we intend to fix the dataset size.
                raise ValueError(
                    "Can't add this episode as it would exceed the provided buffer_capacity "
                    f"({self._buffer_capacity} frames) by {n_excess} frames."
                )

        # Insert the new data starting from next_index. If any exception occurs here, we need to roll back.
        # We'll restore the occupancy mask if needed.
        prior_occupancy_mask = np.array(self._data[self.OCCUPANCY_MASK_KEY])

        try:
            for i, k in enumerate(self.data_keys):
                # See unit-test for more information on what this is.
                if self.__test_roll_back == i:
                    raise _TestRollBackError("This test was raised for a unit test.")
                # Special treatment of image keys depending on the image mode.
                if self._image_mode == LeRobotDatasetV2ImageMode.VIDEO and k.startswith(
                    self.IMAGE_KEY_PREFIX
                ):
                    # Encode all frames of the episode into a video and save to disk.
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        for frame_index, img in zip(data[self.FRAME_INDEX_KEY], data[k], strict=True):
                            Image.fromarray(img).save(Path(tmpdirname) / f"frame_{frame_index:06d}.png")
                        encode_video_frames(
                            Path(tmpdirname),
                            self._storage_dir
                            / self.VIDEOS_DIR
                            / self.VIDEO_NAME_FSTRING.format(data_key=k, episode_index=new_episode_index),
                            self._fps,
                            overwrite=True,
                        )
                elif self._image_mode == LeRobotDatasetV2ImageMode.PNG and k.startswith(
                    self.IMAGE_KEY_PREFIX
                ):
                    # Encode images to PNG and save to disk.
                    for frame_index, img in zip(data[self.FRAME_INDEX_KEY], data[k], strict=True):
                        Image.fromarray(img).save(
                            self._storage_dir
                            / self.PNGS_DIR
                            / self.PNG_NAME_FSTRING.format(
                                data_key=k, episode_index=new_episode_index, frame_index=frame_index
                            )
                        )
                else:
                    # Insertion into the memmap for non-image keys or image keys in memmap mode.
                    slc = slice(next_index, next_index + new_data_length)
                    if k == self.EPISODE_INDEX_KEY:
                        self._data[k][slc] = new_episode_index
                    elif k == self.INDEX_KEY:
                        self._data[k][slc] = np.arange(
                            last_data_index + 1, last_data_index + 1 + new_data_length
                        )
                    else:
                        self._data[k][slc] = data[k]
                    self._data[self.OCCUPANCY_MASK_KEY][slc] = True
        except Exception as e:
            if self._use_as_filo_buffer:
                # Roll back is not implemented for this scenario.
                logging.warning(
                    "An exception was caught while adding an episode to the dataset. If you have issues "
                    "loading the dataset again, please check the dataset integrity with "
                    f"{self.__class__.__name__}.check_storage_dir_integrity."
                )
                raise e
            logging.warning(
                "An exception was caught while adding an episode to the dataset. Rolling back. Please do not "
                "interrupt."
            )
            self._data[self.OCCUPANCY_MASK_KEY][: len(prior_occupancy_mask)] = prior_occupancy_mask
            self._remove_stale_image_files()
            # As a last step, try to see if the data is corrupted.
            LeRobotDatasetV2.check_storage_dir_integrity(self._storage_dir)
            raise e

        # Update the data pointer.
        self._data[self.NEXT_INDEX_KEY][0] = next_index + new_data_length

        # Remove stale videos or PNG files if needed.
        if self._use_as_filo_buffer and LeRobotDatasetV2ImageMode.needs_decoding(self._image_mode):
            self._remove_stale_image_files()

    def _remove_stale_image_files(self):
        """Remove image files that are not aligned with the episode / frame indices in the memmaps."""
        if self._use_as_filo_buffer and LeRobotDatasetV2ImageMode.needs_decoding(self._image_mode):
            relevant_file_names = []
            if self._image_mode == LeRobotDatasetV2ImageMode.VIDEO:
                for k in self.camera_keys:
                    relevant_file_names += [
                        self.VIDEO_NAME_FSTRING.format(data_key=k, episode_index=ep_ix)
                        for ep_ix in self.get_unique_episode_indices()
                    ]
                files_dir = self._storage_dir / self.VIDEOS_DIR
                found_file_names = set(os.listdir(files_dir))

            elif self._image_mode == LeRobotDatasetV2ImageMode.PNG:
                for k in self.camera_keys:
                    for ep_ix in self.get_unique_episode_indices():
                        frame_indices = self.get_data_by_key(LeRobotDatasetV2.FRAME_INDEX_KEY)[
                            self.get_data_by_key(LeRobotDatasetV2.EPISODE_INDEX_KEY) == ep_ix
                        ]
                        for frame_ix in frame_indices:
                            relevant_file_names.append(
                                self.PNG_NAME_FSTRING.format(
                                    data_key=k, episode_index=ep_ix, frame_index=frame_ix
                                )
                            )
                files_dir = self._storage_dir / self.PNGS_DIR
                found_file_names = set(os.listdir(self._storage_dir / self.PNGS_DIR))
            else:
                raise AssertionError("All decodable image modes should be handled")

            relevant_file_names = set(relevant_file_names)
            # Sanity check. All relevant file names should exist.
            assert len(relevant_file_names.difference(found_file_names)) == 0
            file_names_to_remove = found_file_names.difference(relevant_file_names)
            if len(file_names_to_remove) > 0:
                # Sanity check: the file names to remove should match the camera keys that we used to
                # construct the relevant files. Adds a layer of protection against deleting files that
                # shouldn't be deleted.
                assert {f.split("_episode", 1)[0] for f in file_names_to_remove} == set(self.camera_keys)
                # Now remove all irrelevant files.
                for file_name in found_file_names.difference(relevant_file_names):
                    os.remove(files_dir / file_name)

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
            if isinstance(v, np.ndarray):
                item_[k] = torch.from_numpy(v)
            elif isinstance(v, torch.Tensor):
                item_[k] = v
            elif isinstance(v, np.bool_):
                # Note: This is not necessary vs just doing torch.tensor(v), but it dodges a
                # DeprecationWarning from torch.
                item_[k] = torch.tensor(bool(v))
            else:
                item_[k] = torch.tensor(v)
        return item_

    def __len__(self):
        return self.num_samples

    def get_data_by_key(self, key: str) -> np.ndarray:
        """Returns all data for a given data key in the numpy.memmap data."""
        if key.startswith(self.IMAGE_KEY_PREFIX) and LeRobotDatasetV2ImageMode.needs_decoding(
            self._image_mode
        ):
            raise ValueError(
                f"Can't get all data for image keys when {self._image_mode=}. This would require decoding "
                "the whole dataset!"
            )
        return self._data[key][self._data[self.OCCUPANCY_MASK_KEY]]

    def get_episode(self, episode_index: int) -> dict[str, np.ndarray]:
        """Returns a whole episode as a key-array mapping.

        Includes decoding videos or PNG files where necessary.
        """
        episode_data = {}
        data_mask = np.bitwise_and(
            self._data[self.EPISODE_INDEX_KEY] == episode_index, self._data[self.OCCUPANCY_MASK_KEY]
        )
        if np.count_nonzero(data_mask) == 0:
            raise IndexError(f"Episode index {episode_index} is not in the dataset.")
        for k in self.data_keys:
            if self._image_mode == LeRobotDatasetV2ImageMode.VIDEO and k in self.camera_keys:
                episode_data[k] = decode_video_frames_torchvision(
                    video_path=self.storage_dir
                    / self.VIDEOS_DIR
                    / self.VIDEO_NAME_FSTRING.format(data_key=k, episode_index=episode_index),
                    timestamps=self._data[self.TIMESTAMP_KEY][data_mask],
                    tolerance_s=1e-8,
                    backend="pyav",
                    to_pytorch_format=False,
                )
            elif self._image_mode == LeRobotDatasetV2ImageMode.PNG and k in self.camera_keys:
                imgs = []
                for frame_index in self._data[self.FRAME_INDEX_KEY][data_mask]:
                    img_path = (
                        self.storage_dir
                        / self.PNGS_DIR
                        / self.PNG_NAME_FSTRING.format(
                            data_key=k,
                            episode_index=episode_index,
                            frame_index=frame_index,
                        )
                    )
                    imgs.append(np.array(Image.open(img_path)))
                episode_data[k] = np.stack(imgs)
            else:
                episode_data[k] = self._data[k][data_mask]
        return episode_data

    @staticmethod
    def _numpy_img_to_tensor(img: np.ndarray) -> torch.Tensor:
        """
        Converts img from numpy uint8, in range [0, 255], channel-first to torch float32, in range [0, 1],
        channel-last.
        """
        return torch.from_numpy(einops.rearrange(img.astype(np.float32) / 255.0, "... h w c -> ... c h w"))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Gets an item or slice from the dataset and returns it in PyTorch format.

        Images (any data key starting with "observation.image") get converted from numpy uint8, in range
        [0, 255], channel-first to torch float32, in range [0, 1], channel-last.

        If `delta_timestamps` is set... TODO(alexander-soare): Document this somewhere when
        `load_previous_and_future_frames` is refactored.
        """
        if idx >= len(self) or idx < -len(self):
            raise IndexError

        # Grab the relevant frame from the memmap data. At this point, this may be incomplete for one of two
        # reasons:
        # 1. We are using delta_timestamps, so some of these will act as key frames for a temporal chunk that
        #    wish to extract.
        # 2. The image mode is either "video" or "png", in which case we will need to decode the image frames
        #    further down.
        item = {}
        for k in self.data_keys:
            if LeRobotDatasetV2ImageMode.needs_decoding(self._image_mode) and k in self.camera_keys:
                continue
            item[k] = self._data[k][idx]

        if self.delta_timestamps is not None:
            # We are using delta timestamps, so extract the appropriate temporal chunks of data.
            episode_index = item[self.EPISODE_INDEX_KEY]
            current_ts = item[self.TIMESTAMP_KEY]
            episode_data_indices = np.where(
                np.bitwise_and(
                    self._data[self.EPISODE_INDEX_KEY] == episode_index,
                    self._data[self.OCCUPANCY_MASK_KEY],
                )
            )[0]
            episode_timestamps = self._data[self.TIMESTAMP_KEY][episode_data_indices]

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

                # For image data, either decode video frames, load PNG files, or slice the memmap. For all
                # other data, slice the memmap.
                episode_data_indices_to_retrieve = episode_data_indices[argmin_]
                if (
                    data_key.startswith(self.IMAGE_KEY_PREFIX)
                    and self._image_mode == LeRobotDatasetV2ImageMode.VIDEO
                ):
                    item[data_key] = decode_video_frames_torchvision(
                        video_path=self.storage_dir
                        / self.VIDEOS_DIR
                        / self.VIDEO_NAME_FSTRING.format(data_key=data_key, episode_index=episode_index),
                        timestamps=self._data[self.TIMESTAMP_KEY][episode_data_indices_to_retrieve],
                        tolerance_s=self.tolerance_s,
                        backend="pyav",
                        to_pytorch_format=True,
                    )
                elif (
                    data_key.startswith(self.IMAGE_KEY_PREFIX)
                    and self._image_mode == LeRobotDatasetV2ImageMode.PNG
                ):
                    imgs = []
                    for frame_index in self._data[self.FRAME_INDEX_KEY][episode_data_indices_to_retrieve]:
                        img_path = (
                            self.storage_dir
                            / self.PNGS_DIR
                            / self.PNG_NAME_FSTRING.format(
                                data_key=data_key,
                                episode_index=episode_index,
                                frame_index=frame_index,
                            )
                        )
                        imgs.append(self._numpy_img_to_tensor(np.array(Image.open(img_path))))
                    item[data_key] = torch.stack(imgs)
                else:
                    item[data_key] = self._data[data_key][episode_data_indices_to_retrieve]

                item[f"{data_key}{self.IS_PAD_POSTFIX}"] = is_pad

        # For video and png images, we may still need to load frames if we have not already done it in the
        # delta timestamps logic. For memmap images, we just need to finalize by converting the images to
        # pytorch format.
        if self._image_mode == LeRobotDatasetV2ImageMode.VIDEO:
            for k in self.camera_keys:
                if k in item:
                    # We have already populated it in the delta_timestamps logic.
                    assert self.delta_timestamps is not None and k in self.delta_timestamps  # sanity check
                    continue
                item[k] = decode_video_frames_torchvision(
                    video_path=self.storage_dir
                    / self.VIDEOS_DIR
                    / self.VIDEO_NAME_FSTRING.format(data_key=k, episode_index=item[self.EPISODE_INDEX_KEY]),
                    timestamps=[item[self.TIMESTAMP_KEY]],
                    tolerance_s=1e-8,  # we expect the timestamp to match exactly
                    backend="pyav",
                    to_pytorch_format=True,
                )[0]
        elif self._image_mode == LeRobotDatasetV2ImageMode.PNG:
            for k in self.camera_keys:
                if k in item:
                    # We have already populated it in the delta_timestamps logic.
                    assert self.delta_timestamps is not None and k in self.delta_timestamps  # sanity check
                    continue
                img_path = (
                    self.storage_dir
                    / self.PNGS_DIR
                    / self.PNG_NAME_FSTRING.format(
                        data_key=k,
                        episode_index=item[self.EPISODE_INDEX_KEY],
                        frame_index=item[self.FRAME_INDEX_KEY],
                    )
                )
                item[k] = self._numpy_img_to_tensor(np.array(Image.open(img_path)))
        elif self._image_mode == LeRobotDatasetV2ImageMode.MEMMAP:
            for k in self.camera_keys:
                item[k] = self._numpy_img_to_tensor(item[k])

        if self.image_transform is not None:
            for k in self.camera_keys:
                item[k] = self.image_transform(item[k])

        return self._item_to_tensors(item)

    @classmethod
    def from_huggingface_hub(
        cls,
        repo_id: str,
        decode_images: bool = False,
        root: Path | None = DATA_DIR,
        **kwargs,
    ) -> "LeRobotDatasetV2":
        """Create a LeRobotDatasetV2 from a data repository on the Hugging Face Hub.

        NOTE: If the LeRobotDatasetV2 already exists in the storage directory, this function will reuse it
        rather than creating a new one.

        Args:
            repo_id: The dataset repository ID.
            decode_images: Optionally decode videos files or image files into a numpy memmap up front. This
                provides large speed benefits for data access but only if your storage can handle it (decoded
                images take up a lot more storage space than videos or png files).
            root: (will be deprecated) Directory to load the dataset from, instead of the hub.
            **kwargs: Other arguments to `self.__init__` except for the `data_spec`, `buffer_capacity` and
                `fps` arguments which are inferred automatically. `storage_dir` is set to
                `/tmp/{repo_id}_{hf_dataset._fingerprint}_{decoded?}` unless provided explicitly.
        Returns:
            The resulting LeRobotDatasetV2 object.
        """
        for k in ["data_spec", "buffer_capacity", "fps"]:
            if k in kwargs:
                raise ValueError(f"`{k}` should not be provided as it is inferred from the hub dataset.")

        hf_dataset = load_hf_dataset(repo_id, version=CODEBASE_VERSION, root=root, split="train")
        hf_dataset_camera_keys = [
            k for k in hf_dataset.features if k.startswith(LeRobotDatasetV2.IMAGE_KEY_PREFIX)
        ]
        episode_data_index = load_episode_data_index(repo_id, version=CODEBASE_VERSION, root=root)
        hf_dataset.set_transform(lambda x: x)  # there is a default transform in place. reset it
        # Get some metadata necessary for processing videos.
        lerobot_dataset_info = load_info(repo_id, version=CODEBASE_VERSION, root=root)
        if lerobot_dataset_info.get("video", False):
            lerobot_dataset_videos_path = load_videos(repo_id, version=CODEBASE_VERSION, root=root)

        kwargs.setdefault(
            "storage_dir",
            LeRobotDatasetV2._default_storage_dir_from_huggingface_hub(
                repo_id, hf_dataset._fingerprint, decode_images
            ),
        )

        dataset_already_on_disk = False
        if Path(kwargs["storage_dir"]).exists():
            dataset_already_on_disk = True

        # Set the image mode based on the provided HF dataset and whether we are decoding images.
        if len(hf_dataset_camera_keys) == 0 or decode_images:
            image_mode = LeRobotDatasetV2ImageMode.MEMMAP
        elif lerobot_dataset_info.get("video", False):
            image_mode = LeRobotDatasetV2ImageMode.VIDEO
        else:
            image_mode = LeRobotDatasetV2ImageMode.PNG
            for k, feature in hf_dataset.features.items():
                if isinstance(feature, datasets.features.Image):
                    hf_dataset = hf_dataset.cast_column(k, datasets.features.Image(decode=False))

        # Create the LeRobotDatasetV2 object. Reminder: if the storage directory already exists, this reads
        # it. Otherwise, the storage directory is not created until later when we make the first call to
        # `add_episodes`.
        obj = cls(
            **kwargs,
            buffer_capacity=len(hf_dataset) if not dataset_already_on_disk else None,
            fps=lerobot_dataset_info["fps"],
            image_mode=image_mode,
        )

        # If we have accessed an existing cached dataset, just return the object as is.
        if dataset_already_on_disk:
            if len(obj) == len(hf_dataset):
                # All episodes are already in the storage directory.
                return obj
            else:
                # Only some episodes are in the storage directory. Reset the data pointer and start from
                # scratch.
                obj._data[LeRobotDatasetV2.NEXT_INDEX_KEY][0] = 0

        # Siphon the data from the Hugging Face dataset into our dataset object.
        # If the image mode is either "png" or "video" we will apply a small hack. Rather than passing image
        # arrays to `add_episodes` to be encoded, we will reuse the pre-existing files. This means that we'll
        # need to manually update metadata["data_keys"] afterwards.
        episode_indices = np.unique(hf_dataset["episode_index"])
        for episode_index in tqdm(episode_indices, desc="Siphoning episodes into local data structure"):
            data_dict = {}
            hf_episode_data = hf_dataset[
                episode_data_index["from"][episode_index].item() : episode_data_index["to"][
                    episode_index
                ].item()
            ]

            for k, feature in hf_dataset.features.items():
                if isinstance(feature, datasets.features.Image):
                    if decode_images:
                        data_dict[k] = np.stack([np.array(pil_img) for pil_img in hf_episode_data[k]])
                    else:
                        for i in range(len(hf_episode_data[k])):
                            frame_index = hf_episode_data[LeRobotDatasetV2.FRAME_INDEX_KEY][i]
                            img_path = (
                                obj.storage_dir
                                / LeRobotDatasetV2.PNGS_DIR
                                / LeRobotDatasetV2.PNG_NAME_FSTRING.format(
                                    data_key=k, episode_index=episode_index, frame_index=frame_index
                                )
                            )
                            img_path.parent.mkdir(parents=True, exist_ok=True)
                            img_bytes = hf_episode_data[k][i]["bytes"]
                            with open(img_path, "wb") as f:
                                f.write(img_bytes)
                elif isinstance(feature, VideoFrame):
                    if decode_images:
                        # Decode all frames of the episode.
                        data_dict[k] = decode_video_frames_torchvision(
                            lerobot_dataset_videos_path.parent / hf_episode_data[k][0]["path"],
                            timestamps=np.array(hf_episode_data["timestamp"]),
                            tolerance_s=1e-8,
                            to_pytorch_format=False,
                        )
                    else:
                        # Copy video to storage directory.
                        new_video_path = (
                            obj.storage_dir
                            / LeRobotDatasetV2.VIDEOS_DIR
                            / LeRobotDatasetV2.VIDEO_NAME_FSTRING.format(
                                data_key=k, episode_index=episode_index
                            )
                        )
                        new_video_path.parent.mkdir(parents=True, exist_ok=True)
                        os.symlink(
                            (lerobot_dataset_videos_path / new_video_path.name).absolute(), new_video_path
                        )
                elif isinstance(feature, datasets.features.Sequence):
                    data_dict[k] = np.array(hf_episode_data[k], dtype=np.dtype(feature.feature.dtype))
                elif isinstance(feature, datasets.features.Value):
                    data_dict[k] = np.array(hf_episode_data[k], dtype=np.dtype(feature.dtype))
                else:
                    raise NotImplementedError(f"feature type {type(feature)} is not handled.")

            obj._add_episode(data_dict, from_huggingface_hub=True)

        obj.flush()

        if LeRobotDatasetV2ImageMode.needs_decoding(obj._image_mode):
            # We didn't pass the image keys into the first call to `add_episodes` so manually update
            # metadata["data_keys"].
            data_keys: list[str] = obj._load_metadata()["data_keys"]
            data_keys.extend(hf_dataset_camera_keys)
            obj._save_metadata(data_keys=data_keys)

        return obj

    @staticmethod
    def _default_storage_dir_from_huggingface_hub(repo_id: str, fingerprint: str, decode_video: bool) -> Path:
        """Create the default storage directory used for the `from_huggingface_hub` method.

        Note: This method is really meant for development / testing.
        """
        return Path(f"/tmp/{repo_id}_{fingerprint}{'_decoded' if decode_video else ''}")

    @staticmethod
    def check_storage_dir_integrity(
        storage_dir: Path | str, image_mode: LeRobotDatasetV2ImageMode | None = None
    ):
        """Checks if the dataset, as physically stored on disk, is valid.

        The checks are not-exhaustive, but attempt to cover as much as is feasible without actually creating
        the dataset object.

        Optionally pass the image_mode argument to refine the check for a particular image mode.

        Returns None if the storage directory is valid, or raises a StorageDirCorruptError of the storage
        directory is corrupt.
        """
        storage_dir = Path(storage_dir)

        # Check that the directory exists.
        if not (storage_dir).exists():
            raise ValueError(f"{storage_dir} not found.")

        # Check that the metadata file exists.
        if not (storage_dir / LeRobotDatasetV2.METADATA_FILE_NAME).exists():
            raise StorageDirCorruptError(f"{storage_dir / LeRobotDatasetV2.METADATA_FILE_NAME} not found.")

        # Check that the metadata file can be loaded.
        try:
            with open(storage_dir / LeRobotDatasetV2.METADATA_FILE_NAME) as f:
                metadata = json.load(f)
        except json.JSONDecodeError as e:
            raise StorageDirCorruptError("Exception encountered when attempting to load metadata file") from e

        # Check all mandatory keys are in the metadata.
        for k in ["_data_spec", "data_keys", "fps"]:
            if k not in metadata:
                raise StorageDirCorruptError(
                    f"Metadata from {(storage_dir / LeRobotDatasetV2.METADATA_FILE_NAME)} is missing key {k}"
                )

        # Check that there are memmap files for all the keys in the data spec.
        data_spec = metadata["_data_spec"]
        for k in data_spec:
            if not (storage_dir / k).exists():
                raise StorageDirCorruptError(f"Missing memmap for key {k}")

        # Check that all of the memmaps have the same first array dimension (as per the data_spec).
        for k, v in data_spec.items():
            if k == LeRobotDatasetV2.NEXT_INDEX_KEY:
                continue
            if (n_frames := v["shape"][0]) != (
                n_index_frames := data_spec[LeRobotDatasetV2.INDEX_KEY]["shape"][0]
            ):
                raise StorageDirCorruptError(
                    f"There is an inconsistency in the memmap shapes. Got {n_frames} frames for data key "
                    f"'{k}' but {n_index_frames} frames for data key '{LeRobotDatasetV2.INDEX_KEY}'"
                )

        # Load up some keys needed to check the file directories.
        def load_memmap(key):
            return np.memmap(
                storage_dir / key,
                mode="readonly",
                shape=tuple(data_spec[key]["shape"]),
                dtype=np.dtype(data_spec[key]["dtype"]),
            )

        occupancy_mask = load_memmap(LeRobotDatasetV2.OCCUPANCY_MASK_KEY)
        frame_indices = load_memmap(LeRobotDatasetV2.FRAME_INDEX_KEY)
        valid_frame_indices = frame_indices[occupancy_mask]
        episode_indices = load_memmap(LeRobotDatasetV2.EPISODE_INDEX_KEY)
        valid_episode_indices = episode_indices[occupancy_mask]

        # Check that the file directory exists for the decodable image modes, and that the expected files are
        # present.
        if image_mode == LeRobotDatasetV2ImageMode.VIDEO:
            if not (storage_dir / LeRobotDatasetV2.VIDEOS_DIR).exists():
                raise StorageDirCorruptError(
                    f"Couldn't find video directory: {storage_dir / LeRobotDatasetV2.VIDEOS_DIR}."
                )
            file_names = set(os.listdir(storage_dir / LeRobotDatasetV2.VIDEOS_DIR))
            for k in metadata["data_keys"]:
                if not k.startswith(LeRobotDatasetV2.IMAGE_KEY_PREFIX):
                    continue
                for episode_index in np.unique(valid_episode_indices):
                    expected_name = LeRobotDatasetV2.VIDEO_NAME_FSTRING.format(
                        data_key=k, episode_index=episode_index
                    )
                    if expected_name not in file_names:
                        raise StorageDirCorruptError(
                            f"Memmap data indicates the existence of "
                            f"{storage_dir / LeRobotDatasetV2.VIDEOS_DIR / expected_name}, but this file "
                            "could not be found."
                        )
        elif image_mode == LeRobotDatasetV2ImageMode.PNG:
            if not (storage_dir / LeRobotDatasetV2.PNGS_DIR).exists():
                raise StorageDirCorruptError(
                    f"Couldn't find png directory: {storage_dir / LeRobotDatasetV2.PNGS_DIR}."
                )
            file_names = set(os.listdir(storage_dir / LeRobotDatasetV2.PNGS_DIR))
            for k in metadata["data_keys"]:
                if not k.startswith(LeRobotDatasetV2.IMAGE_KEY_PREFIX):
                    continue
                for episode_index in np.unique(valid_episode_indices):
                    for frame_index in valid_frame_indices[valid_episode_indices == episode_index]:
                        expected_name = LeRobotDatasetV2.PNG_NAME_FSTRING.format(
                            data_key=k, episode_index=episode_index, frame_index=frame_index
                        )
                        if expected_name not in file_names:
                            raise StorageDirCorruptError(
                                f"Memmap data indicates the existence of "
                                f"{storage_dir / LeRobotDatasetV2.PNGS_DIR / expected_name}, but this file "
                                "could not be found."
                            )
        elif image_mode == LeRobotDatasetV2ImageMode.MEMMAP and not any(
            k.startswith(LeRobotDatasetV2.IMAGE_KEY_PREFIX) for k in metadata["_data_spec"]
        ):
            raise StorageDirCorruptError(
                f"Image mode is {str(image_mode)} but no memmap files for images (starting with "
                f"{LeRobotDatasetV2.IMAGE_KEY_PREFIX}) could be found"
            )


def compute_sampler_weights(
    offline_dataset: LeRobotDataset,
    offline_drop_n_last_frames: int = 0,
    online_dataset: LeRobotDatasetV2 | None = None,
    online_sampling_ratio: float | None = None,
    online_drop_n_last_frames: int = 0,
) -> torch.Tensor:
    """Compute the sampling weights for the online training dataloader in train.py.

    Args:
        offline_dataset: The LeRobotDataset used for offline pre-training.
        online_drop_n_last_frames: Number of frames to drop from the end of each offline dataset episode.
        online_dataset: The LeRobotDatasetV2 used in online training.
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
        episode_indices = online_dataset.get_data_by_key(LeRobotDatasetV2.EPISODE_INDEX_KEY)
        for episode_idx in np.unique(episode_indices):
            where_episode = np.where(episode_indices == episode_idx)
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
