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
# limitations under the License.d
from copy import deepcopy
from uuid import uuid4

import numpy as np
import pytest
import torch

from lerobot.common.datasets.online_buffer import OnlineBuffer, compute_sampler_weights

# Some constants for OnlineBuffer tests.
data_key = "data"
data_shape = (2, 3)  # just some arbitrary > 1D shape
buffer_capacity = 100
fps = 10


def make_new_buffer(
    write_dir: str | None = None, delta_timestamps: dict[str, list[float]] | None = None
) -> tuple[OnlineBuffer, str]:
    if write_dir is None:
        write_dir = f"/tmp/online_buffer_{uuid4().hex}"
    buffer = OnlineBuffer(
        write_dir,
        data_spec={data_key: {"shape": data_shape, "dtype": np.dtype("float32")}},
        buffer_capacity=buffer_capacity,
        fps=fps,
        delta_timestamps=delta_timestamps,
    )
    return buffer, write_dir


def make_spoof_data_frames(n_episodes: int, n_frames_per_episode: int) -> dict[str, np.ndarray]:
    new_data = {
        data_key: np.arange(n_frames_per_episode * n_episodes * np.prod(data_shape)).reshape(-1, *data_shape),
        OnlineBuffer.INDEX_KEY: np.arange(n_frames_per_episode * n_episodes),
        OnlineBuffer.EPISODE_INDEX_KEY: np.repeat(np.arange(n_episodes), n_frames_per_episode),
        OnlineBuffer.FRAME_INDEX_KEY: np.tile(np.arange(n_frames_per_episode), n_episodes),
        OnlineBuffer.TIMESTAMP_KEY: np.tile(np.arange(n_frames_per_episode) / fps, n_episodes),
    }
    return new_data


def test_non_mutate():
    """Checks that the data provided to the add_data method is copied rather than passed by reference.

    This means that mutating the data in the buffer does not mutate the original data.

    NOTE: If this test fails, it means some of the other tests may be compromised. For example, we can't trust
    a success case for `test_write_read`.
    """
    buffer, _ = make_new_buffer()
    new_data = make_spoof_data_frames(2, buffer_capacity // 4)
    new_data_copy = deepcopy(new_data)
    buffer.add_data(new_data)
    buffer._data[data_key][:] += 1
    assert all(np.array_equal(new_data[k], new_data_copy[k]) for k in new_data)


def test_index_error_no_data():
    buffer, _ = make_new_buffer()
    with pytest.raises(IndexError):
        buffer[0]


def test_index_error_with_data():
    buffer, _ = make_new_buffer()
    n_frames = buffer_capacity // 2
    new_data = make_spoof_data_frames(1, n_frames)
    buffer.add_data(new_data)
    with pytest.raises(IndexError):
        buffer[n_frames]
    with pytest.raises(IndexError):
        buffer[-n_frames - 1]


@pytest.mark.parametrize("do_reload", [False, True])
def test_write_read(do_reload: bool):
    """Checks that data can be added to the buffer and read back.

    If do_reload we delete the buffer object and load the buffer back from disk before reading.
    """
    buffer, write_dir = make_new_buffer()
    n_episodes = 2
    n_frames_per_episode = buffer_capacity // 4
    new_data = make_spoof_data_frames(n_episodes, n_frames_per_episode)
    buffer.add_data(new_data)

    if do_reload:
        del buffer
        buffer, _ = make_new_buffer(write_dir)

    assert len(buffer) == n_frames_per_episode * n_episodes
    for i, item in enumerate(buffer):
        assert all(isinstance(item[k], torch.Tensor) for k in item)
        assert np.array_equal(item[data_key].numpy(), new_data[data_key][i])


def test_read_data_key():
    """Tests that data can be added to a buffer and all data for a. specific key can be read back."""
    buffer, _ = make_new_buffer()
    n_episodes = 2
    n_frames_per_episode = buffer_capacity // 4
    new_data = make_spoof_data_frames(n_episodes, n_frames_per_episode)
    buffer.add_data(new_data)

    data_from_buffer = buffer.get_data_by_key(data_key)
    assert isinstance(data_from_buffer, torch.Tensor)
    assert np.array_equal(data_from_buffer.numpy(), new_data[data_key])


def test_fifo():
    """Checks that if data is added beyond the buffer capacity, we discard the oldest data first."""
    buffer, _ = make_new_buffer()
    n_frames_per_episode = buffer_capacity // 4
    n_episodes = 3
    new_data = make_spoof_data_frames(n_episodes, n_frames_per_episode)
    buffer.add_data(new_data)
    n_more_episodes = 2
    # Developer sanity check (in case someone changes the global `buffer_capacity`).
    assert (n_episodes + n_more_episodes) * n_frames_per_episode > buffer_capacity, (
        "Something went wrong with the test code."
    )
    more_new_data = make_spoof_data_frames(n_more_episodes, n_frames_per_episode)
    buffer.add_data(more_new_data)
    assert len(buffer) == buffer_capacity, "The buffer should be full."

    expected_data = {}
    for k in new_data:
        # Concatenate, left-truncate, then roll, to imitate the cyclical FIFO pattern in OnlineBuffer.
        expected_data[k] = np.roll(
            np.concatenate([new_data[k], more_new_data[k]])[-buffer_capacity:],
            shift=len(new_data[k]) + len(more_new_data[k]) - buffer_capacity,
            axis=0,
        )

    for i, item in enumerate(buffer):
        assert all(isinstance(item[k], torch.Tensor) for k in item)
        assert np.array_equal(item[data_key].numpy(), expected_data[data_key][i])


def test_delta_timestamps_within_tolerance():
    """Check that getting an item with delta_timestamps within tolerance succeeds.

    Note: Copied from `test_datasets.py::test_load_previous_and_future_frames_within_tolerance`.
    """
    # Sanity check on global fps as we are assuming it is 10 here.
    assert fps == 10, "This test assumes fps==10"
    buffer, _ = make_new_buffer(delta_timestamps={"index": [-0.2, 0, 0.139]})
    new_data = make_spoof_data_frames(n_episodes=1, n_frames_per_episode=5)
    buffer.add_data(new_data)
    buffer.tolerance_s = 0.04
    item = buffer[2]
    data, is_pad = item["index"], item[f"index{OnlineBuffer.IS_PAD_POSTFIX}"]
    torch.testing.assert_close(data, torch.tensor([0, 2, 3]), msg="Data does not match expected values")
    assert not is_pad.any(), "Unexpected padding detected"


def test_delta_timestamps_outside_tolerance_inside_episode_range():
    """Check that getting an item with delta_timestamps outside of tolerance fails.

    We expect it to fail if and only if the requested timestamps are within the episode range.

    Note: Copied from
    `test_datasets.py::test_load_previous_and_future_frames_outside_tolerance_inside_episode_range`
    """
    # Sanity check on global fps as we are assuming it is 10 here.
    assert fps == 10, "This test assumes fps==10"
    buffer, _ = make_new_buffer(delta_timestamps={"index": [-0.2, 0, 0.141]})
    new_data = make_spoof_data_frames(n_episodes=1, n_frames_per_episode=5)
    buffer.add_data(new_data)
    buffer.tolerance_s = 0.04
    with pytest.raises(AssertionError):
        buffer[2]


def test_delta_timestamps_outside_tolerance_outside_episode_range():
    """Check that copy-padding of timestamps outside of the episode range works.

    Note: Copied from
    `test_datasets.py::test_load_previous_and_future_frames_outside_tolerance_outside_episode_range`
    """
    # Sanity check on global fps as we are assuming it is 10 here.
    assert fps == 10, "This test assumes fps==10"
    buffer, _ = make_new_buffer(delta_timestamps={"index": [-0.3, -0.24, 0, 0.26, 0.3]})
    new_data = make_spoof_data_frames(n_episodes=1, n_frames_per_episode=5)
    buffer.add_data(new_data)
    buffer.tolerance_s = 0.04
    item = buffer[2]
    data, is_pad = item["index"], item["index_is_pad"]
    assert torch.equal(data, torch.tensor([0, 0, 2, 4, 4])), "Data does not match expected values"
    assert torch.equal(is_pad, torch.tensor([True, False, False, True, True])), (
        "Padding does not match expected values"
    )


# Arbitrarily set small dataset sizes, making sure to have uneven sizes.
@pytest.mark.parametrize("offline_dataset_size", [1, 6])
@pytest.mark.parametrize("online_dataset_size", [0, 4])
@pytest.mark.parametrize("online_sampling_ratio", [0.0, 1.0])
def test_compute_sampler_weights_trivial(
    lerobot_dataset_factory,
    tmp_path,
    offline_dataset_size: int,
    online_dataset_size: int,
    online_sampling_ratio: float,
):
    offline_dataset = lerobot_dataset_factory(tmp_path, total_episodes=1, total_frames=offline_dataset_size)
    online_dataset, _ = make_new_buffer()
    if online_dataset_size > 0:
        online_dataset.add_data(
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
    torch.testing.assert_close(weights, expected_weights)


def test_compute_sampler_weights_nontrivial_ratio(lerobot_dataset_factory, tmp_path):
    # Arbitrarily set small dataset sizes, making sure to have uneven sizes.
    offline_dataset = lerobot_dataset_factory(tmp_path, total_episodes=1, total_frames=4)
    online_dataset, _ = make_new_buffer()
    online_dataset.add_data(make_spoof_data_frames(n_episodes=4, n_frames_per_episode=2))
    online_sampling_ratio = 0.8
    weights = compute_sampler_weights(
        offline_dataset, online_dataset=online_dataset, online_sampling_ratio=online_sampling_ratio
    )
    torch.testing.assert_close(
        weights, torch.tensor([0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    )


def test_compute_sampler_weights_nontrivial_ratio_and_drop_last_n(lerobot_dataset_factory, tmp_path):
    # Arbitrarily set small dataset sizes, making sure to have uneven sizes.
    offline_dataset = lerobot_dataset_factory(tmp_path, total_episodes=1, total_frames=4)
    online_dataset, _ = make_new_buffer()
    online_dataset.add_data(make_spoof_data_frames(n_episodes=4, n_frames_per_episode=2))
    weights = compute_sampler_weights(
        offline_dataset, online_dataset=online_dataset, online_sampling_ratio=0.8, online_drop_n_last_frames=1
    )
    torch.testing.assert_close(
        weights, torch.tensor([0.05, 0.05, 0.05, 0.05, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0, 0.2, 0.0])
    )


def test_compute_sampler_weights_drop_n_last_frames(lerobot_dataset_factory, tmp_path):
    """Note: test copied from test_sampler."""
    offline_dataset = lerobot_dataset_factory(tmp_path, total_episodes=1, total_frames=2)
    online_dataset, _ = make_new_buffer()
    online_dataset.add_data(make_spoof_data_frames(n_episodes=4, n_frames_per_episode=2))

    weights = compute_sampler_weights(
        offline_dataset,
        offline_drop_n_last_frames=1,
        online_dataset=online_dataset,
        online_sampling_ratio=0.5,
        online_drop_n_last_frames=1,
    )
    torch.testing.assert_close(weights, torch.tensor([0.5, 0, 0.125, 0, 0.125, 0, 0.125, 0, 0.125, 0]))
