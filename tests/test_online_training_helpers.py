from copy import deepcopy
from uuid import uuid4

import numpy as np
import pytest
import torch

from lerobot.scripts.online_training_helpers import OnlineBuffer

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
        OnlineBuffer.EPISODE_INDEX_KEY: np.zeros(n_frames_per_episode * n_episodes),
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
        assert np.array_equal(item[data_key].numpy(), new_data[data_key][i])


def test_fifo():
    """Checks that if data is added beyond the buffer capacity, we discard the oldest data first."""
    buffer, _ = make_new_buffer()
    n_frames_per_episode = buffer_capacity // 4
    n_episodes = 3
    new_data = make_spoof_data_frames(n_episodes, n_frames_per_episode)
    buffer.add_data(new_data)
    n_more_episodes = 2
    # Developer sanity check (in case someone changes the global `buffer_capacity`).
    assert (
        n_episodes + n_more_episodes
    ) * n_frames_per_episode > buffer_capacity, "Something went wrong with the test code."
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
    assert torch.equal(data, torch.tensor([0, 2, 3])), "Data does not match expected values"
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
    assert torch.equal(
        is_pad, torch.tensor([True, False, False, True, True])
    ), "Padding does not match expected values"
