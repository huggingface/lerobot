import pytest
import torch
from datasets import Dataset

from lerobot.common.datasets.utils import (
    calculate_episode_data_index,
    check_delta_timestamps,
    check_timestamps_sync,
    get_delta_indices,
    hf_transform_to_torch,
)
from tests.fixtures.constants import DUMMY_MOTOR_FEATURES


@pytest.fixture(scope="module")
def synced_hf_dataset_factory(hf_dataset_factory):
    def _create_synced_hf_dataset(fps: int = 30) -> Dataset:
        return hf_dataset_factory(fps=fps)

    return _create_synced_hf_dataset


@pytest.fixture(scope="module")
def unsynced_hf_dataset_factory(synced_hf_dataset_factory):
    def _create_unsynced_hf_dataset(fps: int = 30, tolerance_s: float = 1e-4) -> Dataset:
        hf_dataset = synced_hf_dataset_factory(fps=fps)
        features = hf_dataset.features
        df = hf_dataset.to_pandas()
        dtype = df["timestamp"].dtype  # This is to avoid pandas type warning
        # Modify a single timestamp just outside tolerance
        df.at[30, "timestamp"] = dtype.type(df.at[30, "timestamp"] + (tolerance_s * 1.1))
        unsynced_hf_dataset = Dataset.from_pandas(df, features=features)
        unsynced_hf_dataset.set_transform(hf_transform_to_torch)
        return unsynced_hf_dataset

    return _create_unsynced_hf_dataset


@pytest.fixture(scope="module")
def slightly_off_hf_dataset_factory(synced_hf_dataset_factory):
    def _create_slightly_off_hf_dataset(fps: int = 30, tolerance_s: float = 1e-4) -> Dataset:
        hf_dataset = synced_hf_dataset_factory(fps=fps)
        features = hf_dataset.features
        df = hf_dataset.to_pandas()
        dtype = df["timestamp"].dtype  # This is to avoid pandas type warning
        # Modify a single timestamp just inside tolerance
        df.at[30, "timestamp"] = dtype.type(df.at[30, "timestamp"] + (tolerance_s * 0.9))
        unsynced_hf_dataset = Dataset.from_pandas(df, features=features)
        unsynced_hf_dataset.set_transform(hf_transform_to_torch)
        return unsynced_hf_dataset

    return _create_slightly_off_hf_dataset


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


def test_check_timestamps_sync_synced(synced_hf_dataset_factory):
    fps = 30
    tolerance_s = 1e-4
    synced_hf_dataset = synced_hf_dataset_factory(fps)
    episode_data_index = calculate_episode_data_index(synced_hf_dataset)
    result = check_timestamps_sync(
        hf_dataset=synced_hf_dataset,
        episode_data_index=episode_data_index,
        fps=fps,
        tolerance_s=tolerance_s,
    )
    assert result is True


def test_check_timestamps_sync_unsynced(unsynced_hf_dataset_factory):
    fps = 30
    tolerance_s = 1e-4
    unsynced_hf_dataset = unsynced_hf_dataset_factory(fps, tolerance_s)
    episode_data_index = calculate_episode_data_index(unsynced_hf_dataset)
    with pytest.raises(ValueError):
        check_timestamps_sync(
            hf_dataset=unsynced_hf_dataset,
            episode_data_index=episode_data_index,
            fps=fps,
            tolerance_s=tolerance_s,
        )


def test_check_timestamps_sync_unsynced_no_exception(unsynced_hf_dataset_factory):
    fps = 30
    tolerance_s = 1e-4
    unsynced_hf_dataset = unsynced_hf_dataset_factory(fps, tolerance_s)
    episode_data_index = calculate_episode_data_index(unsynced_hf_dataset)
    result = check_timestamps_sync(
        hf_dataset=unsynced_hf_dataset,
        episode_data_index=episode_data_index,
        fps=fps,
        tolerance_s=tolerance_s,
        raise_value_error=False,
    )
    assert result is False


def test_check_timestamps_sync_slightly_off(slightly_off_hf_dataset_factory):
    fps = 30
    tolerance_s = 1e-4
    slightly_off_hf_dataset = slightly_off_hf_dataset_factory(fps, tolerance_s)
    episode_data_index = calculate_episode_data_index(slightly_off_hf_dataset)
    result = check_timestamps_sync(
        hf_dataset=slightly_off_hf_dataset,
        episode_data_index=episode_data_index,
        fps=fps,
        tolerance_s=tolerance_s,
    )
    assert result is True


def test_check_timestamps_sync_single_timestamp():
    single_timestamp_hf_dataset = Dataset.from_dict({"timestamp": [0.0], "episode_index": [0]})
    single_timestamp_hf_dataset.set_transform(hf_transform_to_torch)
    episode_data_index = {"to": torch.tensor([1]), "from": torch.tensor([0])}
    fps = 30
    tolerance_s = 1e-4
    result = check_timestamps_sync(
        hf_dataset=single_timestamp_hf_dataset,
        episode_data_index=episode_data_index,
        fps=fps,
        tolerance_s=tolerance_s,
    )
    assert result is True


# TODO(aliberts): Change behavior of hf_transform_to_torch so that it can work with empty dataset
@pytest.mark.skip("TODO: fix")
def test_check_timestamps_sync_empty_dataset():
    fps = 30
    tolerance_s = 1e-4
    empty_hf_dataset = Dataset.from_dict({"timestamp": [], "episode_index": []})
    empty_hf_dataset.set_transform(hf_transform_to_torch)
    episode_data_index = {
        "to": torch.tensor([], dtype=torch.int64),
        "from": torch.tensor([], dtype=torch.int64),
    }
    result = check_timestamps_sync(
        hf_dataset=empty_hf_dataset,
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
