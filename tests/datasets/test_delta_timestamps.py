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
from itertools import accumulate

import datasets
import numpy as np
import pyarrow.compute as pc
import pytest
import torch

from lerobot.datasets.utils import (
    check_delta_timestamps,
    check_timestamps_sync,
    get_delta_indices,
)
from tests.fixtures.constants import DUMMY_MOTOR_FEATURES


def calculate_total_episode(
    hf_dataset: datasets.Dataset, raise_if_not_contiguous: bool = True
) -> dict[str, torch.Tensor]:
    episode_indices = sorted(hf_dataset.unique("episode_index"))
    total_episodes = len(episode_indices)
    if raise_if_not_contiguous and episode_indices != list(range(total_episodes)):
        raise ValueError("episode_index values are not sorted and contiguous.")
    return total_episodes


def calculate_episode_data_index(hf_dataset: datasets.Dataset) -> dict[str, np.ndarray]:
    episode_lengths = []
    table = hf_dataset.data.table
    total_episodes = calculate_total_episode(hf_dataset)
    for ep_idx in range(total_episodes):
        ep_table = table.filter(pc.equal(table["episode_index"], ep_idx))
        episode_lengths.insert(ep_idx, len(ep_table))

    cumulative_lengths = list(accumulate(episode_lengths))
    return {
        "from": np.array([0] + cumulative_lengths[:-1], dtype=np.int64),
        "to": np.array(cumulative_lengths, dtype=np.int64),
    }


@pytest.fixture(scope="module")
def synced_timestamps_factory(hf_dataset_factory):
    def _create_synced_timestamps(fps: int = 30) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hf_dataset = hf_dataset_factory(fps=fps)
        timestamps = torch.stack(hf_dataset["timestamp"]).numpy()
        episode_indices = torch.stack(hf_dataset["episode_index"]).numpy()
        episode_data_index = calculate_episode_data_index(hf_dataset)
        return timestamps, episode_indices, episode_data_index

    return _create_synced_timestamps


@pytest.fixture(scope="module")
def unsynced_timestamps_factory(synced_timestamps_factory):
    def _create_unsynced_timestamps(
        fps: int = 30, tolerance_s: float = 1e-4
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        timestamps, episode_indices, episode_data_index = synced_timestamps_factory(fps=fps)
        timestamps[30] += tolerance_s * 1.1  # Modify a single timestamp just outside tolerance
        return timestamps, episode_indices, episode_data_index

    return _create_unsynced_timestamps


@pytest.fixture(scope="module")
def slightly_off_timestamps_factory(synced_timestamps_factory):
    def _create_slightly_off_timestamps(
        fps: int = 30, tolerance_s: float = 1e-4
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        timestamps, episode_indices, episode_data_index = synced_timestamps_factory(fps=fps)
        timestamps[30] += tolerance_s * 0.9  # Modify a single timestamp just inside tolerance
        return timestamps, episode_indices, episode_data_index

    return _create_slightly_off_timestamps


@pytest.fixture(scope="module")
def valid_delta_timestamps_factory():
    def _create_valid_delta_timestamps(
        fps: int = 30, keys: list = DUMMY_MOTOR_FEATURES, min_max_range: tuple[int, int] = (-10, 10)
    ) -> dict:
        delta_timestamps = {key: [i * (1 / fps) for i in range(*min_max_range)] for key in keys}
        return delta_timestamps

    return _create_valid_delta_timestamps


@pytest.fixture(scope="module")
def invalid_delta_timestamps_factory(valid_delta_timestamps_factory):
    def _create_invalid_delta_timestamps(
        fps: int = 30, tolerance_s: float = 1e-4, keys: list = DUMMY_MOTOR_FEATURES
    ) -> dict:
        delta_timestamps = valid_delta_timestamps_factory(fps, keys)
        # Modify a single timestamp just outside tolerance
        for key in keys:
            delta_timestamps[key][3] += tolerance_s * 1.1
        return delta_timestamps

    return _create_invalid_delta_timestamps


@pytest.fixture(scope="module")
def slightly_off_delta_timestamps_factory(valid_delta_timestamps_factory):
    def _create_slightly_off_delta_timestamps(
        fps: int = 30, tolerance_s: float = 1e-4, keys: list = DUMMY_MOTOR_FEATURES
    ) -> dict:
        delta_timestamps = valid_delta_timestamps_factory(fps, keys)
        # Modify a single timestamp just inside tolerance
        for key in delta_timestamps:
            delta_timestamps[key][3] += tolerance_s * 0.9
            delta_timestamps[key][-3] += tolerance_s * 0.9
        return delta_timestamps

    return _create_slightly_off_delta_timestamps


@pytest.fixture(scope="module")
def delta_indices_factory():
    def _delta_indices(keys: list = DUMMY_MOTOR_FEATURES, min_max_range: tuple[int, int] = (-10, 10)) -> dict:
        return {key: list(range(*min_max_range)) for key in keys}

    return _delta_indices


def test_check_timestamps_sync_synced(synced_timestamps_factory):
    fps = 30
    tolerance_s = 1e-4
    timestamps, ep_idx, ep_data_index = synced_timestamps_factory(fps)
    result = check_timestamps_sync(
        timestamps=timestamps,
        episode_indices=ep_idx,
        episode_data_index=ep_data_index,
        fps=fps,
        tolerance_s=tolerance_s,
    )
    assert result is True


def test_check_timestamps_sync_unsynced(unsynced_timestamps_factory):
    fps = 30
    tolerance_s = 1e-4
    timestamps, ep_idx, ep_data_index = unsynced_timestamps_factory(fps, tolerance_s)
    with pytest.raises(ValueError):
        check_timestamps_sync(
            timestamps=timestamps,
            episode_indices=ep_idx,
            episode_data_index=ep_data_index,
            fps=fps,
            tolerance_s=tolerance_s,
        )


def test_check_timestamps_sync_unsynced_no_exception(unsynced_timestamps_factory):
    fps = 30
    tolerance_s = 1e-4
    timestamps, ep_idx, ep_data_index = unsynced_timestamps_factory(fps, tolerance_s)
    result = check_timestamps_sync(
        timestamps=timestamps,
        episode_indices=ep_idx,
        episode_data_index=ep_data_index,
        fps=fps,
        tolerance_s=tolerance_s,
        raise_value_error=False,
    )
    assert result is False


def test_check_timestamps_sync_slightly_off(slightly_off_timestamps_factory):
    fps = 30
    tolerance_s = 1e-4
    timestamps, ep_idx, ep_data_index = slightly_off_timestamps_factory(fps, tolerance_s)
    result = check_timestamps_sync(
        timestamps=timestamps,
        episode_indices=ep_idx,
        episode_data_index=ep_data_index,
        fps=fps,
        tolerance_s=tolerance_s,
    )
    assert result is True


def test_check_timestamps_sync_single_timestamp():
    fps = 30
    tolerance_s = 1e-4
    timestamps, ep_idx = np.array([0.0]), np.array([0])
    episode_data_index = {"to": np.array([1]), "from": np.array([0])}
    result = check_timestamps_sync(
        timestamps=timestamps,
        episode_indices=ep_idx,
        episode_data_index=episode_data_index,
        fps=fps,
        tolerance_s=tolerance_s,
    )
    assert result is True


def test_check_delta_timestamps_valid(valid_delta_timestamps_factory):
    fps = 30
    tolerance_s = 1e-4
    valid_delta_timestamps = valid_delta_timestamps_factory(fps)
    result = check_delta_timestamps(
        delta_timestamps=valid_delta_timestamps,
        fps=fps,
        tolerance_s=tolerance_s,
    )
    assert result is True


def test_check_delta_timestamps_slightly_off(slightly_off_delta_timestamps_factory):
    fps = 30
    tolerance_s = 1e-4
    slightly_off_delta_timestamps = slightly_off_delta_timestamps_factory(fps, tolerance_s)
    result = check_delta_timestamps(
        delta_timestamps=slightly_off_delta_timestamps,
        fps=fps,
        tolerance_s=tolerance_s,
    )
    assert result is True


def test_check_delta_timestamps_invalid(invalid_delta_timestamps_factory):
    fps = 30
    tolerance_s = 1e-4
    invalid_delta_timestamps = invalid_delta_timestamps_factory(fps, tolerance_s)
    with pytest.raises(ValueError):
        check_delta_timestamps(
            delta_timestamps=invalid_delta_timestamps,
            fps=fps,
            tolerance_s=tolerance_s,
        )


def test_check_delta_timestamps_invalid_no_exception(invalid_delta_timestamps_factory):
    fps = 30
    tolerance_s = 1e-4
    invalid_delta_timestamps = invalid_delta_timestamps_factory(fps, tolerance_s)
    result = check_delta_timestamps(
        delta_timestamps=invalid_delta_timestamps,
        fps=fps,
        tolerance_s=tolerance_s,
        raise_value_error=False,
    )
    assert result is False


def test_check_delta_timestamps_empty():
    delta_timestamps = {}
    fps = 30
    tolerance_s = 1e-4
    result = check_delta_timestamps(
        delta_timestamps=delta_timestamps,
        fps=fps,
        tolerance_s=tolerance_s,
    )
    assert result is True


def test_delta_indices(valid_delta_timestamps_factory, delta_indices_factory):
    fps = 50
    min_max_range = (-100, 100)
    delta_timestamps = valid_delta_timestamps_factory(fps, min_max_range=min_max_range)
    expected_delta_indices = delta_indices_factory(min_max_range=min_max_range)
    actual_delta_indices = get_delta_indices(delta_timestamps, fps)
    assert expected_delta_indices == actual_delta_indices
