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

import os
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import datasets
import numpy as np
import pytest
import torch

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, DATA_DIR, LeRobotDataset
from lerobot.common.datasets.online_buffer import (
    DiskSpaceError,
    LeRobotDatasetV2,
    LeRobotDatasetV2ImageMode,
    TimestampOutsideToleranceError,
    _TestRollBackError,
    compute_sampler_weights,
)
from lerobot.common.datasets.utils import hf_transform_to_torch, load_hf_dataset, load_info, load_videos
from lerobot.common.datasets.video_utils import (
    VideoFrame,
    decode_video_frames_torchvision,
    encode_video_frames,
)
from lerobot.common.utils.utils import seeded_context
from tests.utils import DevTestingError


def make_spoof_data_frames(
    n_episodes: int, n_frames_per_episode: int, seed: int = 0
) -> dict[str, np.ndarray]:
    fps = 10
    proprio_dim = 6
    action_dim = 4
    # This is the minimum size allowed for encoding.
    img_size = (64, 64)
    n_frames = n_frames_per_episode * n_episodes
    with seeded_context(seed):
        new_data = {
            # "observation.image": np.random.randint(0, 256, size=(n_frames, *img_size, 3)).astype(np.uint8),
            "observation.image": np.random.randint(118, 128, size=(n_frames, *img_size, 3)).astype(np.uint8),
            "observation.state": np.random.normal(size=(n_frames, proprio_dim)).astype(np.float32),
            "action": np.random.normal(size=(n_frames, action_dim)).astype(np.float32),
            LeRobotDatasetV2.INDEX_KEY: np.arange(n_frames),
            LeRobotDatasetV2.EPISODE_INDEX_KEY: np.repeat(np.arange(n_episodes), n_frames_per_episode),
            LeRobotDatasetV2.FRAME_INDEX_KEY: np.tile(np.arange(n_frames_per_episode), n_episodes),
            LeRobotDatasetV2.TIMESTAMP_KEY: np.tile(
                np.arange(n_frames_per_episode, dtype=np.float32) / fps, n_episodes
            ),
        }
    return new_data


def test_cleanup_storage_dir(tmp_path: Path):
    """Check that if an exception is raised when creating the storage directory, it is cleaned up."""
    storage_dir = tmp_path / f"dataset_{uuid4().hex}"
    dataset = LeRobotDatasetV2(storage_dir, fps=10)  # arbitrary fps
    # Note: choices for args to make_spoof_data_frames are totally arbitrary.
    new_episodes = make_spoof_data_frames(n_episodes=2, n_frames_per_episode=25)
    with patch(
        f"{LeRobotDatasetV2.__module__}.{LeRobotDatasetV2.__qualname__}._make_memmaps"
    ) as mock_make_memmaps:
        # Mock a runtime error in _make_memmaps. This should trigger the clean up.
        mock_make_memmaps.side_effect = RuntimeError()
        with pytest.raises(RuntimeError):
            dataset.add_episodes(new_episodes)
    assert not storage_dir.exists()


@pytest.mark.parametrize("image_mode", list(LeRobotDatasetV2ImageMode))
def test_get_data_keys(tmp_path: Path, image_mode: LeRobotDatasetV2ImageMode):
    dataset = LeRobotDatasetV2(
        tmp_path / f"dataset_{uuid4().hex}", fps=10, image_mode=image_mode
    )  # arbitrary fps
    # Note: choices for args to make_spoof_data_frames are totally arbitrary.
    new_episodes = make_spoof_data_frames(n_episodes=2, n_frames_per_episode=25)
    # Patch encode_video_frames for CI where we don't have libsvtav1.
    with patch(
        f"{LeRobotDatasetV2.__module__}.{encode_video_frames.__qualname__}",
        new=partial(encode_video_frames, vcodec="libx264"),
    ):
        dataset.add_episodes(new_episodes)
    assert set(dataset.data_keys) == set(new_episodes)


@pytest.mark.parametrize("image_mode", list(LeRobotDatasetV2ImageMode))
def test_get_data_by_key(tmp_path: Path, image_mode: LeRobotDatasetV2ImageMode):
    """Tests that data can be added to a dataset and all data for a specific key can be read back.

    Also checks that get_data_by_key does NOT work with image keys when we are working with a decodable image
    mode.
    """
    dataset = LeRobotDatasetV2(
        tmp_path / f"dataset_{uuid4().hex}", fps=10, image_mode=image_mode
    )  # arbitrary fps
    # Note: choices for args to make_spoof_data_frames are mostly arbitrary. The only intentional aspect is to
    # make sure the memmaps are not full, in order to check that `get_data_by_key` only returns the parts of
    # the memmaps that are occupied.
    n_episodes = 2
    n_frames_per_episode = 25
    new_episodes = make_spoof_data_frames(n_episodes, n_frames_per_episode)
    # Patch encode_video_frames for CI where we don't have libsvtav1.
    with patch(
        f"{LeRobotDatasetV2.__module__}.{encode_video_frames.__qualname__}",
        new=partial(encode_video_frames, vcodec="libx264"),
    ):
        dataset.add_episodes(new_episodes)

    for k in new_episodes:
        if LeRobotDatasetV2ImageMode.needs_decoding(image_mode) and k.startswith(
            LeRobotDatasetV2.IMAGE_KEY_PREFIX
        ):
            with pytest.raises(ValueError):
                dataset.get_data_by_key(k)
        else:
            assert np.array_equal(new_episodes[k], dataset.get_data_by_key(k))


def test_get_unique_episode_indices(tmp_path: Path):
    dataset = LeRobotDatasetV2(
        tmp_path / f"dataset_{uuid4().hex}", fps=10, image_mode=LeRobotDatasetV2ImageMode.MEMMAP
    )  # arbitrary fps
    # Note: choices for args to make_spoof_data_frames are mostly arbitrary. Just make sure there is more
    # than one episode.
    n_episodes = 2
    new_episodes = make_spoof_data_frames(n_episodes=n_episodes, n_frames_per_episode=25)
    dataset.add_episodes(new_episodes)
    assert np.array_equal(np.sort(dataset.get_unique_episode_indices()), np.arange(n_episodes))


def test_non_mutate(tmp_path: Path):
    """Checks that the data added to the memmap with `add_episodes` is copied rather than passed by reference.

    This means that mutating the data in the memmap does not mutate the original data.

    NOTE: If this test fails, it means some of the other tests may be compromised. For example, we can't trust
    a success case for `test_write_read`.
    """
    dataset = LeRobotDatasetV2(
        tmp_path / f"dataset_{uuid4().hex}", fps=10, image_mode=LeRobotDatasetV2ImageMode.MEMMAP
    )  # arbitrary fps
    # Note: choices for args to make_spoof_data_frames are arbitrary.
    new_data = make_spoof_data_frames(2, 25)
    new_data_copy = deepcopy(new_data)
    dataset.add_episodes(new_data)
    dataset.get_data_by_key("action")[:] += 1
    assert all(np.array_equal(new_data[k], new_data_copy[k]) for k in new_data)


def test_index_error_no_data(tmp_path: Path):
    dataset = LeRobotDatasetV2(
        tmp_path / f"dataset_{uuid4().hex}", fps=10, image_mode=LeRobotDatasetV2ImageMode.MEMMAP
    )  # arbitrary fps
    with pytest.raises(IndexError):
        dataset[0]


def test_index_error_with_data(tmp_path: Path):
    dataset = LeRobotDatasetV2(
        tmp_path / f"dataset_{uuid4().hex}", fps=10, image_mode=LeRobotDatasetV2ImageMode.MEMMAP
    )  # arbitrary fps
    n_frames = 50
    # Note: choices for args to make_spoof_data_frames are arbitrary.
    new_data = make_spoof_data_frames(1, n_frames)
    dataset.add_episodes(new_data)
    with pytest.raises(IndexError):
        dataset[n_frames]
    with pytest.raises(IndexError):
        dataset[-n_frames - 1]


@pytest.mark.parametrize("image_mode", list(LeRobotDatasetV2ImageMode))
def test_value_error_no_image(tmp_path: Path, image_mode: LeRobotDatasetV2ImageMode):
    """
    Check that trying to add an episode without image data when the image mode is decodable raises a
    ValueError.
    """
    dataset = LeRobotDatasetV2(
        tmp_path / f"dataset_{uuid4().hex}", fps=10, image_mode=image_mode
    )  # arbitrary fps
    # Note: choices for args to make_spoof_data_frames are totally arbitrary.
    new_episodes = make_spoof_data_frames(n_episodes=2, n_frames_per_episode=25)
    # Drop any image data keys.
    image_data_keys = [k for k in new_episodes if k.startswith(LeRobotDatasetV2.IMAGE_KEY_PREFIX)]
    for k in image_data_keys:
        del new_episodes[k]
    # Patch encode_video_frames for CI where we don't have libsvtav1.
    with patch(
        f"{LeRobotDatasetV2.__module__}.{encode_video_frames.__qualname__}",
        new=partial(encode_video_frames, vcodec="libx264"),
    ):
        if LeRobotDatasetV2ImageMode.needs_decoding(image_mode):
            with pytest.raises(ValueError):
                dataset.add_episodes(new_episodes)
        else:
            dataset.add_episodes(new_episodes)


@pytest.mark.parametrize("do_reload", [False, True])
@pytest.mark.parametrize("image_mode", list(LeRobotDatasetV2ImageMode))
def test_write_read_and_get_episode(tmp_path: Path, do_reload: bool, image_mode: str):
    """Checks that data can be added to the dataset and read back.

    If do_reload we delete the dataset object and load the dataset back from disk before reading.

    This also tests that the get_episode returns the correct data.
    """
    storage_dir = tmp_path / f"dataset_{uuid4().hex}"
    fps = 10
    dataset = LeRobotDatasetV2(storage_dir, fps=fps, image_mode=image_mode)
    # Note: choices for args to make_spoof_data_frames are arbitrary.
    n_episodes = 2
    n_frames_per_episode = 25
    new_data = make_spoof_data_frames(n_episodes, n_frames_per_episode)
    with patch(
        f"{LeRobotDatasetV2.__module__}.{encode_video_frames.__qualname__}",
        new=partial(encode_video_frames, vcodec="libx264"),
    ):
        dataset.add_episodes(new_data)

    if do_reload:
        del dataset
        # Note: Don't provide fps parameter as we are also testing this is loaded up from the metadata.
        dataset = LeRobotDatasetV2(storage_dir, image_mode=image_mode)

    assert dataset.fps == fps
    assert len(dataset) == n_frames_per_episode * n_episodes
    for episode_index in range(n_episodes):
        data_mask = new_data[LeRobotDatasetV2.EPISODE_INDEX_KEY] == episode_index
        episode_data = dataset.get_episode(episode_index)
        for k in dataset.data_keys:
            if image_mode == LeRobotDatasetV2ImageMode.VIDEO and k.startswith(
                LeRobotDatasetV2.IMAGE_KEY_PREFIX
            ):
                # Because of lossy compression for videos (which is exacerbated by the fact that we are
                # working with random noise images), we can't check for pixel-perfect equality here (even
                # with crf=0). See `test_write_read_and_get_episode_video_mode` for the follow up test.
                assert episode_data[k].shape == new_data[k][data_mask].shape
            else:
                assert np.array_equal(episode_data[k], new_data[k][data_mask]), f"Mismatch for {k=}"


@pytest.mark.parametrize("do_reload", [False, True])
def test_write_read_and_get_episode_video_mode(tmp_path: Path, do_reload: bool):
    """
    See `test_write_read_and_get_episode`. There, we had to get by with just checking that the image shapes
    match for video mode. We couldn't check pixel equality because of lossy compression (which is especially
    bad for random noise images as we used there). Here we use the PushT dataset and check that a high
    percentage of pixel values are within tolerance.
    """
    dataset_repo_id = "lerobot/pusht"
    # This is not the dataset we will be testing. We are just getting frames for it for the test.
    dataset = LeRobotDatasetV2.from_huggingface_hub(
        dataset_repo_id, storage_dir=tmp_path / f"{dataset_repo_id}_{uuid4().hex}", decode_images=True
    )
    fps = dataset.fps
    # Since `test_write_read_and_get_episode` already tested `get_episode` in "memmap" mode, it is fair to
    # do it here as part of setting up for this test.
    provided_episode_data = dataset.get_episode(0)
    episode_length = len(provided_episode_data[LeRobotDatasetV2.INDEX_KEY])
    # This is the dataset we will be testing.
    storage_dir = tmp_path / f"dataset_{uuid4().hex}"
    dataset = LeRobotDatasetV2(storage_dir, fps=fps, image_mode=LeRobotDatasetV2ImageMode.VIDEO)
    # To construct the new dataset, just repeat the episode some number of times.
    n_episodes = 2
    for _ in range(n_episodes):
        # Patch encode_video_frames for CI where we don't have libsvtav1, and use minimal compression for the
        # array equality check to succeed.
        with patch(
            f"{LeRobotDatasetV2.__module__}.{encode_video_frames.__qualname__}",
            new=partial(encode_video_frames, vcodec="libx264", crf=0),
        ):
            dataset.add_episodes(provided_episode_data)

    if do_reload:
        del dataset
        # Note: Don't provide fps parameter as we are also testing this is loaded up from the metadata.
        dataset = LeRobotDatasetV2(storage_dir, image_mode=LeRobotDatasetV2ImageMode.VIDEO)

    assert len(dataset) == episode_length * n_episodes
    for episode_index in range(n_episodes):
        retrieved_episode_data = dataset.get_episode(episode_index)
        for k in dataset.data_keys:
            if k.startswith(LeRobotDatasetV2.IMAGE_KEY_PREFIX):
                # Check that >99% of array values are within 10 units of the original.
                diff = np.abs(
                    retrieved_episode_data[k].astype("int") - provided_episode_data[k].astype("int")
                )
                assert np.count_nonzero(diff >= 10) / np.prod(provided_episode_data[k].shape) <= 0.01
            elif k == LeRobotDatasetV2.INDEX_KEY:
                # Account for shift.
                assert np.array_equal(
                    retrieved_episode_data[k], provided_episode_data[k] + episode_length * episode_index
                )
            elif k == LeRobotDatasetV2.EPISODE_INDEX_KEY:
                # Account for shift.
                assert np.all(retrieved_episode_data[k] == episode_index)
            else:
                assert np.array_equal(
                    retrieved_episode_data[k], provided_episode_data[k]
                ), f"Mismatch for {k=}"


@pytest.mark.parametrize("image_mode", list(LeRobotDatasetV2ImageMode))
def test_roll_back(tmp_path: Path, image_mode: LeRobotDatasetV2ImageMode):
    """Tests the roll back feature of add_episodes."""
    dataset = LeRobotDatasetV2(
        tmp_path / f"dataset_{uuid4().hex}", fps=10, image_mode=image_mode
    )  # arbitrary fps
    # Note: choices for args to make_spoof_data_frames are mostly arbitrary. Just make sure there is more
    # than one episode.
    new_episodes = make_spoof_data_frames(n_episodes=2, n_frames_per_episode=25)
    for i in range(-1, len(new_episodes.keys())):
        # In `add_episodes` there is a loop over data keys. It adds the data to the dataset one key at a time.
        # Here, we control at which point in the loop an exception should be raised. For i=-1, an exception
        # is not raised, so we add the first batch of episode data to the dataset. For i >= 0, exceptions are
        # raised after a portion of the data has been added to the dataset. And we are checking that in every
        # such case, the dataset is rolled back to its prior state.
        dataset._LeRobotDatasetV2__test_roll_back = i
        with (
            pytest.raises(_TestRollBackError) if i >= 0 else nullcontext(),
            # patch encode_video_frames for CI where we don't have libsvtav1.
            patch(
                f"{LeRobotDatasetV2.__module__}.{encode_video_frames.__qualname__}",
                new=partial(encode_video_frames, vcodec="libx264"),
            ),
        ):
            dataset.add_episodes(new_episodes)
        for episode_index in dataset.get_unique_episode_indices():
            episode_data = dataset.get_episode(episode_index)
            episode_mask = new_episodes[LeRobotDatasetV2.EPISODE_INDEX_KEY] == episode_index
            for k in dataset.data_keys:
                expected_data = new_episodes[k][episode_mask]
                if image_mode == LeRobotDatasetV2ImageMode.VIDEO:
                    # Because of lossy compression for videos (which is exacerbated by the fact that we are
                    # working with random noise images), we can't check for pixel-perfect equality here (even
                    # with crf=0). See `test_write_read_and_get_episode_video_mode`.
                    assert episode_data[k].shape == expected_data.shape
                else:
                    assert np.array_equal(episode_data[k], expected_data)
            LeRobotDatasetV2.check_storage_dir_integrity(dataset.storage_dir)


@pytest.mark.parametrize("pct", [79.0, 81.0])
@pytest.mark.parametrize("via", ["first_call_to_add_episodes", "extend_memmaps"])
def test_disk_space_error(tmp_path: Path, pct: float, via: str):
    """
    Check that attempting to create or extend memmaps beyond 80% of the remaining disk capacity, raises an
    error.
    """
    # First we'll create a dummy dataset to back out the disk usage.
    dummy_dataset = LeRobotDatasetV2(
        tmp_path / f"dataset_{uuid4().hex}", fps=10, image_mode=LeRobotDatasetV2ImageMode.MEMMAP
    )  # arbitrary fps
    new_episodes = make_spoof_data_frames(n_episodes=10, n_frames_per_episode=25)
    dummy_dataset.add_episodes(new_episodes)
    data_spec = dummy_dataset._load_metadata()["_data_spec"]
    # Required space in bytes.
    required_space = sum(spec["dtype"].itemsize * np.prod(spec["shape"]) for spec in data_spec.values())
    dataset = LeRobotDatasetV2(
        tmp_path / f"dataset_{uuid4().hex}", fps=10, image_mode=LeRobotDatasetV2ImageMode.MEMMAP
    )
    if via == "extend_memmaps":
        # The first time we add_episodes, there is no issue with disk space, so we should expect an error.
        dataset.add_episodes(new_episodes)
        required_space *= 2  # we will add another episode so we need 2x the space
    with patch("os.statvfs") as mock_statvfs:
        mock_statvfs.return_value.f_bavail = 1
        mock_statvfs.return_value.f_frsize = int(required_space / (pct / 100))
        # If via == "first_call_to_add_episodes", we are simulating calling `add_episodes` for the first time
        # with not enough space on disk to do it.
        # If via == "extend_memmaps", we are simulating calling `add_episodes` the second time. The first time
        # (above) was successful, but now there is not enough disk space to extend the memmaps for the new
        # episode.
        with pytest.raises(DiskSpaceError) if pct > 80 else nullcontext():
            dataset.add_episodes(new_episodes)


def test_get_episode_index_error(tmp_path: Path):
    """Test that get_episode raises an IndexError is raised with an invalid episode index."""
    dataset = LeRobotDatasetV2(
        tmp_path / f"dataset_{uuid4().hex}", fps=10, image_mode=LeRobotDatasetV2ImageMode.MEMMAP
    )  # arbitrary fps
    # Note: args to make_spoof_data_frames are mostly arbitrary.
    n_episodes = 1
    new_episodes = make_spoof_data_frames(n_episodes=n_episodes, n_frames_per_episode=25)
    dataset.add_episodes(new_episodes)
    with pytest.raises(IndexError):
        dataset.get_episode(n_episodes)


def test_buffer_capacity(tmp_path: Path):
    """Check that explicitly providing a buffer capacity, then overflowing, causes an exception."""
    dataset = LeRobotDatasetV2(
        tmp_path / f"dataset_{uuid4().hex}",
        fps=10,
        image_mode=LeRobotDatasetV2ImageMode.MEMMAP,
        buffer_capacity=60,
    )
    # Note: choices for args to make_spoof_data_frames are totally arbitrary.
    new_episodes = make_spoof_data_frames(n_episodes=3, n_frames_per_episode=25)
    with pytest.raises(ValueError):
        dataset.add_episodes(new_episodes)


def test_dynamic_memmap_size(tmp_path: Path):
    """Check that the memmap is dynamically resized when trying to add episodes over the capacity.

    Check that the existing data is not corrupted.
    """
    dataset = LeRobotDatasetV2(
        tmp_path / f"dataset_{uuid4().hex}", fps=10, image_mode=LeRobotDatasetV2ImageMode.MEMMAP
    )  # arbitrary fps
    n_frames_per_episode = 5
    new_episodes = make_spoof_data_frames(n_episodes=1, n_frames_per_episode=n_frames_per_episode, seed=0)
    all_episodes = new_episodes
    # This should trigger the creation of memmaps with 10 frame capacity.
    dataset.add_episodes(new_episodes)
    expected_capacity = n_frames_per_episode
    used_capacity = n_frames_per_episode
    assert len(dataset._data[LeRobotDatasetV2.INDEX_KEY]) == expected_capacity
    for i in range(1, 51):  # 50 iterations makes the capacity grow to 320
        new_episodes = make_spoof_data_frames(n_episodes=1, n_frames_per_episode=n_frames_per_episode, seed=i)
        # This should trigger the memmaps to double in size.
        dataset.add_episodes(new_episodes)
        # Track the data we expect to see in the memmaps.
        for k in new_episodes:
            # Shift indices the same we we expect `add_episodes` to.
            if k == LeRobotDatasetV2.INDEX_KEY:
                new_episodes[k] += all_episodes[k][-1] + 1
            if k == LeRobotDatasetV2.EPISODE_INDEX_KEY:
                new_episodes[k] += all_episodes[k][-1] + 1
            all_episodes[k] = np.concatenate([all_episodes[k], new_episodes[k]], axis=0)
        used_capacity += n_frames_per_episode
        if used_capacity > expected_capacity:
            expected_capacity *= 2
        assert len(dataset._data[LeRobotDatasetV2.INDEX_KEY]) == expected_capacity
        for k in dataset.data_keys:
            assert np.array_equal(dataset.get_data_by_key(k), all_episodes[k])


def test_filo_needs_buffer_capacity(tmp_path: Path):
    with pytest.raises(ValueError):
        LeRobotDatasetV2(
            tmp_path / f"dataset_{uuid4().hex}",
            fps=10,
            use_as_filo_buffer=True,
            image_mode=LeRobotDatasetV2ImageMode.MEMMAP,
        )


@pytest.mark.parametrize("image_mode", list(LeRobotDatasetV2ImageMode))
def test_filo(tmp_path: Path, image_mode: LeRobotDatasetV2ImageMode):
    """Checks that if data is added beyond the buffer capacity, we discard the oldest data first."""
    buffer_capacity = 100
    dataset = LeRobotDatasetV2(
        tmp_path / f"dataset_{uuid4().hex}",
        fps=10,
        buffer_capacity=buffer_capacity,
        use_as_filo_buffer=True,
        image_mode=image_mode,
    )
    # Note: choices for args to make_spoof_data_frames are mostly arbitrary. Of interest is:
    #   - later we need `n_more_episodes` to cause an overflow.
    #   - we need that overflow to happen *within* an episode such that we're testing the behavior whereby the
    #     whole episode is wrapped to the start of the buffer, even if part of it can fit.
    n_frames_per_episode = buffer_capacity // 4 + 1  # 26
    n_episodes = 3
    if buffer_capacity - n_frames_per_episode * n_episodes >= n_frames_per_episode:
        raise DevTestingError("Make sure to set this up such that adding another episode causes an overflow.")
    new_episodes = make_spoof_data_frames(n_episodes, n_frames_per_episode)
    # Patch encode_video_frames for CI where we don't have libsvtav1.
    with patch(
        f"{LeRobotDatasetV2.__module__}.{encode_video_frames.__qualname__}",
        new=partial(encode_video_frames, vcodec="libx264"),
    ):
        dataset.add_episodes(new_episodes)
    # Note `n_more_episodes` is chosen to result in an overflow on the first episode.
    n_more_episodes = 2
    # Make this slightly larger than the prior episodes to test that there is no issue with overwriting the
    # start of an existing episode.
    n_frames_per_more_episodes = n_frames_per_episode + 1  # 27
    more_new_episodes = make_spoof_data_frames(n_more_episodes, n_frames_per_more_episodes)
    # Patch encode_video_frames for CI where we don't have libsvtav1.
    with patch(
        f"{LeRobotDatasetV2.__module__}.{encode_video_frames.__qualname__}",
        new=partial(encode_video_frames, vcodec="libx264"),
    ):
        dataset.add_episodes(more_new_episodes)
    assert (
        len(dataset) == n_frames_per_episode * n_episodes
    ), "The new episode should have wrapped around to the start"

    # Shift indices to match what the `add_episodes` method would do.
    more_new_episodes[LeRobotDatasetV2.INDEX_KEY] += n_episodes * n_frames_per_episode
    more_new_episodes[LeRobotDatasetV2.EPISODE_INDEX_KEY] += n_episodes
    expected_data = {}
    for k in new_episodes:
        expected_data[k] = new_episodes[k]
        # The extra new episode should overwrite the start of the buffer.
        expected_data[k][: len(more_new_episodes[k])] = more_new_episodes[k]

    for episode_index in dataset.get_unique_episode_indices():
        episode_data = dataset.get_episode(episode_index)
        episode_mask = expected_data[LeRobotDatasetV2.EPISODE_INDEX_KEY] == episode_index
        for k in dataset.data_keys:
            if image_mode == LeRobotDatasetV2ImageMode.VIDEO:
                # Because of lossy compression for videos (which is exacerbated by the fact that we are
                # working with random noise images), we can't check for pixel-perfect equality here (even
                # with crf=0). See `test_write_read_and_get_episode_video_mode`.
                assert episode_data[k].shape == expected_data[k][episode_mask].shape
            else:
                assert np.array_equal(episode_data[k], expected_data[k][episode_mask])

    # Test that videos and PNG files are removed/added as needed.
    # Note: since we have verified above that the memmap data is correct, we can make use of it to figure out
    # which video/PNG files we expect to see.
    if image_mode == LeRobotDatasetV2ImageMode.VIDEO:
        expected_video_file_names = []
        for k in dataset.camera_keys:
            expected_video_file_names += [
                dataset.VIDEO_NAME_FSTRING.format(data_key=k, episode_index=ep_ix)
                for ep_ix in dataset.get_unique_episode_indices()
            ]
        assert set(os.listdir(dataset.storage_dir / dataset.VIDEOS_DIR)) == set(expected_video_file_names)
    elif image_mode == LeRobotDatasetV2ImageMode.PNG:
        expected_png_file_names = []
        for k in dataset.camera_keys:
            for ep_ix in dataset.get_unique_episode_indices():
                frame_indices = dataset.get_data_by_key(LeRobotDatasetV2.FRAME_INDEX_KEY)[
                    dataset.get_data_by_key(LeRobotDatasetV2.EPISODE_INDEX_KEY) == ep_ix
                ]
                for frame_ix in frame_indices:
                    expected_png_file_names.append(
                        dataset.PNG_NAME_FSTRING.format(data_key=k, episode_index=ep_ix, frame_index=frame_ix)
                    )
        assert set(os.listdir(dataset.storage_dir / dataset.PNGS_DIR)) == set(expected_png_file_names)


@pytest.mark.parametrize("image_mode", list(LeRobotDatasetV2ImageMode))
def test_delta_timestamps_within_tolerance(tmp_path: Path, image_mode: LeRobotDatasetV2ImageMode):
    """Check that getting an item with delta_timestamps within tolerance succeeds."""
    dataset = LeRobotDatasetV2(
        tmp_path / f"dataset_{uuid4().hex}",
        fps=10,
        image_mode=image_mode,
        delta_timestamps={"index": [-0.2, 0, 0.1]},
    )
    new_data = make_spoof_data_frames(n_episodes=1, n_frames_per_episode=5)
    # Patch encode_video_frames for CI where we don't have libsvtav1.
    with patch(
        f"{LeRobotDatasetV2.__module__}.{encode_video_frames.__qualname__}",
        new=partial(encode_video_frames, vcodec="libx264"),
    ):
        dataset.add_episodes(new_data)
    item = dataset[2]
    indices, is_pad = item["index"], item[f"index{LeRobotDatasetV2.IS_PAD_POSTFIX}"]
    assert torch.allclose(indices, torch.tensor([0, 2, 3])), "Data does not match expected values"
    assert not is_pad.any(), "Unexpected padding detected"


def test_delta_timestamps_outside_tolerance_inside_episode_range(tmp_path: Path):
    """Check that getting an item with delta_timestamps outside of tolerance fails.

    We expect it to fail if and only if the requested timestamps are within the episode range.
    """
    dataset = LeRobotDatasetV2(
        tmp_path / f"dataset_{uuid4().hex}",
        fps=10,
        image_mode=LeRobotDatasetV2ImageMode.MEMMAP,
        delta_timestamps={"index": [-0.2, 0, 0.1]},
    )
    new_data = make_spoof_data_frames(n_episodes=1, n_frames_per_episode=5)
    # Hack the timestamps to invoke a tolerance error.
    new_data["timestamp"][3] += 0.1
    dataset.add_episodes(new_data)
    with pytest.raises(TimestampOutsideToleranceError):
        dataset[2]


def test_delta_timestamps_outside_tolerance_outside_episode_range(tmp_path):
    """Check that copy-padding of timestamps outside of the episode range works."""
    dataset = LeRobotDatasetV2(
        tmp_path / f"dataset_{uuid4().hex}",
        fps=10,
        image_mode=LeRobotDatasetV2ImageMode.MEMMAP,
        delta_timestamps={"index": [-0.3, -0.2, 0, 0.2, 0.3]},
    )
    new_data = make_spoof_data_frames(n_episodes=1, n_frames_per_episode=5)
    dataset.add_episodes(new_data)
    item = dataset[2]
    data, is_pad = item["index"], item["index_is_pad"]
    assert torch.equal(data, torch.tensor([0, 0, 2, 4, 4])), "Data does not match expected values"
    assert torch.equal(
        is_pad, torch.tensor([True, False, False, False, True])
    ), "Padding does not match expected values"


@pytest.mark.parametrize(
    ("dataset_repo_id", "decode_images"),
    (
        # choose unitreeh1_two_robot_greeting to have multiple image keys (with minimal video data)
        ("lerobot/unitreeh1_two_robot_greeting", True),
        ("lerobot/unitreeh1_two_robot_greeting", False),
        ("lerobot/pusht", False),
        ("lerobot/pusht_image", False),
        ("lerobot/pusht_image", True),
    ),
)
def test_camera_keys(tmp_path: Path, dataset_repo_id: str, decode_images: bool):
    """Check that the camera_keys property returns all relevant keys."""
    storage_dir = tmp_path / f"{dataset_repo_id}_{uuid4().hex}_{decode_images}"
    # Patch encode_video_frames for CI where we don't have libsvtav1.
    dataset = LeRobotDatasetV2.from_huggingface_hub(dataset_repo_id, decode_images, storage_dir=storage_dir)
    lerobot_dataset = LeRobotDataset(dataset_repo_id)
    assert set(dataset.camera_keys) == set(lerobot_dataset.camera_keys)


@pytest.mark.parametrize(
    ("dataset_repo_id", "decode_images"),
    (
        # choose unitreeh1_two_robot_greeting to have multiple image keys
        ("lerobot/unitreeh1_two_robot_greeting", True),
        ("lerobot/unitreeh1_two_robot_greeting", False),
        ("lerobot/pusht", False),
        ("lerobot/pusht_image", False),
        ("lerobot/pusht_image", True),
    ),
)
def test_getter_returns_pytorch_format(tmp_path: Path, dataset_repo_id: str, decode_images: bool):
    """Checks that tensors are returned and that images are torch.float32, in range [0, 1], channel-first."""
    # Note: storage_dir specified explicitly in order to make use of pytest's temporary file fixture.
    storage_dir = tmp_path / f"{dataset_repo_id}_{uuid4().hex}_{decode_images}"
    dataset = LeRobotDatasetV2.from_huggingface_hub(dataset_repo_id, decode_images, storage_dir=storage_dir)
    # Just iterate through the start and end of the dataset to make the test faster.
    for i in np.concatenate([np.arange(0, 10), np.arange(-10, 0)]):
        item = dataset[i]
        for k in item:
            assert isinstance(item[k], torch.Tensor)
        for k in dataset.camera_keys:
            assert item[k].dtype == torch.float32, "images aren't float32"
            assert item[k].max() <= 1, "image values go above max of range [0, 1]"
            assert item[k].min() >= 0, "image values go below min of range [0, 1]"
            c, h, w = item[k].shape
            assert c < min(h, w), "images are not channel-first"


@pytest.mark.parametrize("decode_images", [True, False])
def test_getter_images_with_delta_timestamps(tmp_path: Path, decode_images: bool):
    """Checks a basic delta_timestamps use case with images.

    Specifically, makes sure that the items returned by the getter have the correct number of frames.

    Note: images deserve particular attention because they are not covered by the basic tests above that use
    simple spoof data.
    """
    dataset_repo_id = "lerobot/pusht_image"
    lerobot_dataset_info = load_info(dataset_repo_id, version=CODEBASE_VERSION, root=DATA_DIR)
    fps = lerobot_dataset_info["fps"]
    # Note: storage_dir specified explicitly in order to make use of pytest's temporary file fixture.
    storage_dir = tmp_path / f"{dataset_repo_id}_{uuid4().hex}_{decode_images}"
    dataset = LeRobotDatasetV2.from_huggingface_hub(
        dataset_repo_id, storage_dir=storage_dir, decode_images=decode_images
    )
    delta_timestamps = [-1 / fps, 0.0, 1 / fps]
    dataset.set_delta_timestamps({k: delta_timestamps for k in dataset.camera_keys})

    # Just iterate through the start and end of the dataset to make the test faster.
    for i in np.concatenate([np.arange(0, 10), np.arange(-10, 0)]):
        item = dataset[i]
        for k in dataset.camera_keys:
            assert item[k].shape[0] == len(delta_timestamps)


def test_getter_video_images_with_delta_timestamps(tmp_path: Path):
    """Checks that images from the video dataset and decoded video dataset are the same.

    Adds to `test_getter_images_with_delta_timestamps` by testing with a video dataset.

    Note: images deserve particular attention because video decoding is involved, and because they are not
    covered by the basic tests above that use simple spoof data.
    """
    # choose unitreeh1_two_robot_greeting to have multiple image keys
    dataset_repo_id = "lerobot/unitreeh1_two_robot_greeting"
    lerobot_dataset_info = load_info(dataset_repo_id, version=CODEBASE_VERSION, root=DATA_DIR)
    fps = lerobot_dataset_info["fps"]
    # Note: storage_dir specified explicitly in order to make use of pytest's temporary file fixture.
    dataset_video = LeRobotDatasetV2.from_huggingface_hub(
        dataset_repo_id, decode_images=False, storage_dir=tmp_path / f"{dataset_repo_id}_{uuid4().hex}_False"
    )
    dataset_memmap = LeRobotDatasetV2.from_huggingface_hub(
        dataset_repo_id, decode_images=True, storage_dir=tmp_path / f"{dataset_repo_id}_{uuid4().hex}_True"
    )

    assert set(dataset_video.camera_keys) == set(dataset_memmap.camera_keys)

    delta_timestamps = [-1 / fps, 0.0, 1 / fps]
    dataset_video.set_delta_timestamps({k: delta_timestamps for k in dataset_video.camera_keys})
    dataset_memmap.set_delta_timestamps({k: delta_timestamps for k in dataset_memmap.camera_keys})

    # Just iterate through the start and end of the datasets to make the test faster.
    for i in np.concatenate([np.arange(0, 10), np.arange(-10, 0)]):
        item_video = dataset_video[i]
        item_decode = dataset_memmap[i]
        for k in dataset_video.camera_keys:
            assert item_video[k].shape[0] == len(delta_timestamps)
            assert item_decode[k].shape[0] == len(delta_timestamps)
            assert torch.equal(item_video[k], item_decode[k])


@pytest.mark.parametrize(
    ("dataset_repo_id", "decode_images"),
    (
        ("lerobot/pusht", True),
        ("lerobot/pusht", False),
        ("lerobot/pusht_image", True),
        ("lerobot/pusht_image", False),
    ),
)
def test_from_huggingface_hub(tmp_path: Path, dataset_repo_id: str, decode_images: bool):
    """
    Check that:
        - We can make a dataset from a Hugging Face Hub dataset repository.
        - The dataset we make, accurately reflects the Hugging Face dataset.
        - We can get an item from the dataset.
        - If we try to make it a second time, everything still works as expected.
    """
    for iteration in range(2):  # do it twice to check that running with an existing cached dataset also works
        hf_dataset = load_hf_dataset(dataset_repo_id, version=CODEBASE_VERSION, root=DATA_DIR, split="train")
        hf_dataset.set_transform(lambda x: x)
        # Note: storage_dir specified explicitly in order to make use of pytest's temporary file fixture.
        # This ensures that the first time this loop is run, the storage directory does not already exist.
        storage_dir = tmp_path / LeRobotDatasetV2._default_storage_dir_from_huggingface_hub(
            dataset_repo_id, hf_dataset._fingerprint, decode_images
        ).relative_to("/tmp")
        if iteration == 0 and storage_dir.exists():
            raise DevTestingError("The storage directory should not exist for the first pass of this test.")
        dataset = LeRobotDatasetV2.from_huggingface_hub(
            dataset_repo_id,
            decode_images,
            storage_dir=storage_dir,
        )
        assert len(dataset) == len(hf_dataset)
        for k, feature in hf_dataset.features.items():
            if isinstance(feature, datasets.features.Image):
                if decode_images:
                    assert np.array_equal(
                        dataset.get_data_by_key(k), np.stack([np.array(pil_img) for pil_img in hf_dataset[k]])
                    )
                else:
                    # Check that all the image files are there.
                    for episode_index in np.unique(hf_dataset[LeRobotDatasetV2.EPISODE_INDEX_KEY]):
                        for frame_index in range(
                            np.count_nonzero(hf_dataset[LeRobotDatasetV2.EPISODE_INDEX_KEY] == episode_index)
                        ):
                            assert Path(
                                dataset.storage_dir
                                / dataset.PNGS_DIR
                                / LeRobotDatasetV2.PNG_NAME_FSTRING.format(
                                    data_key=k, episode_index=episode_index, frame_index=frame_index
                                )
                            ).exists()
            elif isinstance(feature, VideoFrame):
                if decode_images:
                    # Decode the video here.
                    lerobot_dataset_info = load_info(dataset_repo_id, version=CODEBASE_VERSION, root=DATA_DIR)
                    videos_path = load_videos(dataset_repo_id, version=CODEBASE_VERSION, root=DATA_DIR)
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
                    assert np.array_equal(dataset.get_data_by_key(k), all_imgs)
                else:
                    # Check that all the video files are there.
                    for episode_index in np.unique(hf_dataset[LeRobotDatasetV2.EPISODE_INDEX_KEY]):
                        assert Path(
                            dataset.storage_dir
                            / dataset.VIDEOS_DIR
                            / LeRobotDatasetV2.VIDEO_NAME_FSTRING.format(
                                data_key=k, episode_index=episode_index
                            )
                        ).exists()
            elif isinstance(feature, (datasets.features.Sequence, datasets.features.Value)):
                assert np.array_equal(dataset.get_data_by_key(k), hf_dataset[k])
            else:
                raise DevTestingError(f"Tests not implemented for this feature type: {type(feature)=}.")
        # Check that we can get an item.
        _ = dataset[0]


# Arbitrarily set small dataset sizes, making sure to have uneven sizes.
@pytest.mark.parametrize("offline_dataset_size", [0, 6])
@pytest.mark.parametrize("online_dataset_size", [0, 4])
@pytest.mark.parametrize("online_sampling_ratio", [0.0, 1.0])
def test_compute_sampler_weights_trivial(
    tmp_path: Path, offline_dataset_size: int, online_dataset_size: int, online_sampling_ratio: float
):
    # Pass/skip the test if both datasets sizes are zero.
    if offline_dataset_size + online_dataset_size == 0:
        return
    # Create spoof offline dataset.
    offline_dataset = LeRobotDataset.from_preloaded(
        hf_dataset=datasets.Dataset.from_dict({"data": list(range(offline_dataset_size))})
    )
    offline_dataset.hf_dataset.set_transform(hf_transform_to_torch)
    if offline_dataset_size == 0:
        offline_dataset.episode_data_index = {}
    else:
        # Set up an episode_data_index with at least two episodes.
        offline_dataset.episode_data_index = {
            "from": torch.tensor([0, offline_dataset_size // 2]),
            "to": torch.tensor([offline_dataset_size // 2, offline_dataset_size]),
        }
    # Create spoof online datset.
    online_dataset = LeRobotDatasetV2(
        tmp_path / f"dataset_{uuid4().hex}", fps=10, image_mode=LeRobotDatasetV2ImageMode.MEMMAP
    )  # arbitrary fps
    if online_dataset_size > 0:
        online_dataset.add_episodes(
            make_spoof_data_frames(n_episodes=2, n_frames_per_episode=online_dataset_size // 2)
        )

    weights = compute_sampler_weights(
        offline_dataset, online_dataset=online_dataset, online_sampling_ratio=online_sampling_ratio
    )
    if offline_dataset_size == 0 or online_dataset_size == 0:
        expected_weights = torch.ones(offline_dataset_size + online_dataset_size)
    elif online_sampling_ratio == 0:
        expected_weights = torch.cat([torch.ones(offline_dataset_size), torch.zeros(online_dataset_size)])
    elif online_sampling_ratio == 1:
        expected_weights = torch.cat([torch.zeros(offline_dataset_size), torch.ones(online_dataset_size)])
    expected_weights /= expected_weights.sum()
    assert torch.allclose(weights, expected_weights)


def test_compute_sampler_weights_nontrivial_ratio(tmp_path: Path):
    # Arbitrarily set small dataset sizes, making sure to have uneven sizes.
    # Create spoof offline dataset.
    offline_dataset = LeRobotDataset.from_preloaded(
        hf_dataset=datasets.Dataset.from_dict({"data": list(range(4))})
    )
    offline_dataset.hf_dataset.set_transform(hf_transform_to_torch)
    offline_dataset.episode_data_index = {
        "from": torch.tensor([0, 2]),
        "to": torch.tensor([2, 4]),
    }
    # Create spoof online datset.
    online_dataset = LeRobotDatasetV2(
        tmp_path / f"dataset_{uuid4().hex}", fps=10, image_mode=LeRobotDatasetV2ImageMode.MEMMAP
    )  # arbitrary fps
    online_dataset.add_episodes(make_spoof_data_frames(n_episodes=4, n_frames_per_episode=2))
    online_sampling_ratio = 0.8
    weights = compute_sampler_weights(
        offline_dataset, online_dataset=online_dataset, online_sampling_ratio=online_sampling_ratio
    )
    assert torch.allclose(
        weights, torch.tensor([0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    )


def test_compute_sampler_weights_drop_n_last_frames(tmp_path: Path):
    """Check that the drop_n_last_frames feature works as intended."""
    data_dict = {
        "timestamp": [0, 0.1],
        "index": [0, 1],
        "episode_index": [0, 0],
        "frame_index": [0, 1],
    }
    offline_dataset = LeRobotDataset.from_preloaded(hf_dataset=datasets.Dataset.from_dict(data_dict))
    offline_dataset.hf_dataset.set_transform(hf_transform_to_torch)
    offline_dataset.episode_data_index = {"from": torch.tensor([0]), "to": torch.tensor([2])}

    online_dataset = LeRobotDatasetV2(
        tmp_path / f"dataset_{uuid4().hex}", fps=10, image_mode=LeRobotDatasetV2ImageMode.MEMMAP
    )  # arbitrary fps
    online_dataset.add_episodes(make_spoof_data_frames(n_episodes=4, n_frames_per_episode=2))

    weights = compute_sampler_weights(
        offline_dataset,
        offline_drop_n_last_frames=1,
        online_dataset=online_dataset,
        online_sampling_ratio=0.5,
        online_drop_n_last_frames=1,
    )
    assert torch.allclose(weights, torch.tensor([0.5, 0, 0.125, 0, 0.125, 0, 0.125, 0, 0.125, 0]))
