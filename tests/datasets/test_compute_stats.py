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
from unittest.mock import patch

import numpy as np
import pytest

from lerobot.common.datasets.compute_stats import (
    _assert_type_and_shape,
    aggregate_feature_stats,
    aggregate_stats,
    compute_episode_stats,
    estimate_num_samples,
    get_feature_stats,
    sample_images,
    sample_indices,
)


def mock_load_image_as_numpy(path, dtype, channel_first):
    return np.ones((3, 32, 32), dtype=dtype) if channel_first else np.ones((32, 32, 3), dtype=dtype)


@pytest.fixture
def sample_array():
    return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


def test_estimate_num_samples():
    assert estimate_num_samples(1) == 1
    assert estimate_num_samples(10) == 10
    assert estimate_num_samples(100) == 100
    assert estimate_num_samples(200) == 100
    assert estimate_num_samples(1000) == 177
    assert estimate_num_samples(2000) == 299
    assert estimate_num_samples(5000) == 594
    assert estimate_num_samples(10_000) == 1000
    assert estimate_num_samples(20_000) == 1681
    assert estimate_num_samples(50_000) == 3343
    assert estimate_num_samples(500_000) == 10_000


def test_sample_indices():
    indices = sample_indices(10)
    assert len(indices) > 0
    assert indices[0] == 0
    assert indices[-1] == 9
    assert len(indices) == estimate_num_samples(10)


@patch("lerobot.common.datasets.compute_stats.load_image_as_numpy", side_effect=mock_load_image_as_numpy)
def test_sample_images(mock_load):
    image_paths = [f"image_{i}.jpg" for i in range(100)]
    images = sample_images(image_paths)
    assert isinstance(images, np.ndarray)
    assert images.shape[1:] == (3, 32, 32)
    assert images.dtype == np.uint8
    assert len(images) == estimate_num_samples(100)


def test_get_feature_stats_images():
    data = np.random.rand(100, 3, 32, 32)
    stats = get_feature_stats(data, axis=(0, 2, 3), keepdims=True)
    assert "min" in stats and "max" in stats and "mean" in stats and "std" in stats and "count" in stats
    np.testing.assert_equal(stats["count"], np.array([100]))
    assert stats["min"].shape == stats["max"].shape == stats["mean"].shape == stats["std"].shape


def test_get_feature_stats_axis_0_keepdims(sample_array):
    expected = {
        "min": np.array([[1, 2, 3]]),
        "max": np.array([[7, 8, 9]]),
        "mean": np.array([[4.0, 5.0, 6.0]]),
        "std": np.array([[2.44948974, 2.44948974, 2.44948974]]),
        "count": np.array([3]),
    }
    result = get_feature_stats(sample_array, axis=(0,), keepdims=True)
    for key in expected:
        np.testing.assert_allclose(result[key], expected[key])


def test_get_feature_stats_axis_1(sample_array):
    expected = {
        "min": np.array([1, 4, 7]),
        "max": np.array([3, 6, 9]),
        "mean": np.array([2.0, 5.0, 8.0]),
        "std": np.array([0.81649658, 0.81649658, 0.81649658]),
        "count": np.array([3]),
    }
    result = get_feature_stats(sample_array, axis=(1,), keepdims=False)
    for key in expected:
        np.testing.assert_allclose(result[key], expected[key])


def test_get_feature_stats_no_axis(sample_array):
    expected = {
        "min": np.array(1),
        "max": np.array(9),
        "mean": np.array(5.0),
        "std": np.array(2.5819889),
        "count": np.array([3]),
    }
    result = get_feature_stats(sample_array, axis=None, keepdims=False)
    for key in expected:
        np.testing.assert_allclose(result[key], expected[key])


def test_get_feature_stats_empty_array():
    array = np.array([])
    with pytest.raises(ValueError):
        get_feature_stats(array, axis=(0,), keepdims=True)


def test_get_feature_stats_single_value():
    array = np.array([[1337]])
    result = get_feature_stats(array, axis=None, keepdims=True)
    np.testing.assert_equal(result["min"], np.array(1337))
    np.testing.assert_equal(result["max"], np.array(1337))
    np.testing.assert_equal(result["mean"], np.array(1337.0))
    np.testing.assert_equal(result["std"], np.array(0.0))
    np.testing.assert_equal(result["count"], np.array([1]))


def test_compute_episode_stats():
    episode_data = {
        "observation.image": [f"image_{i}.jpg" for i in range(100)],
        "observation.state": np.random.rand(100, 10),
    }
    features = {
        "observation.image": {"dtype": "image"},
        "observation.state": {"dtype": "numeric"},
    }

    with patch(
        "lerobot.common.datasets.compute_stats.load_image_as_numpy", side_effect=mock_load_image_as_numpy
    ):
        stats = compute_episode_stats(episode_data, features)

    assert "observation.image" in stats and "observation.state" in stats
    assert stats["observation.image"]["count"].item() == 100
    assert stats["observation.state"]["count"].item() == 100
    assert stats["observation.image"]["mean"].shape == (3, 1, 1)


def test_assert_type_and_shape_valid():
    valid_stats = [
        {
            "feature1": {
                "min": np.array([1.0]),
                "max": np.array([10.0]),
                "mean": np.array([5.0]),
                "std": np.array([2.0]),
                "count": np.array([1]),
            }
        }
    ]
    _assert_type_and_shape(valid_stats)


def test_assert_type_and_shape_invalid_type():
    invalid_stats = [
        {
            "feature1": {
                "min": [1.0],  # Not a numpy array
                "max": np.array([10.0]),
                "mean": np.array([5.0]),
                "std": np.array([2.0]),
                "count": np.array([1]),
            }
        }
    ]
    with pytest.raises(ValueError, match="Stats must be composed of numpy array"):
        _assert_type_and_shape(invalid_stats)


def test_assert_type_and_shape_invalid_shape():
    invalid_stats = [
        {
            "feature1": {
                "count": np.array([1, 2]),  # Wrong shape
            }
        }
    ]
    with pytest.raises(ValueError, match=r"Shape of 'count' must be \(1\)"):
        _assert_type_and_shape(invalid_stats)


def test_aggregate_feature_stats():
    stats_ft_list = [
        {
            "min": np.array([1.0]),
            "max": np.array([10.0]),
            "mean": np.array([5.0]),
            "std": np.array([2.0]),
            "count": np.array([1]),
        },
        {
            "min": np.array([2.0]),
            "max": np.array([12.0]),
            "mean": np.array([6.0]),
            "std": np.array([2.5]),
            "count": np.array([1]),
        },
    ]
    result = aggregate_feature_stats(stats_ft_list)
    np.testing.assert_allclose(result["min"], np.array([1.0]))
    np.testing.assert_allclose(result["max"], np.array([12.0]))
    np.testing.assert_allclose(result["mean"], np.array([5.5]))
    np.testing.assert_allclose(result["std"], np.array([2.318405]), atol=1e-6)
    np.testing.assert_allclose(result["count"], np.array([2]))


def test_aggregate_stats():
    all_stats = [
        {
            "observation.image": {
                "min": [1, 2, 3],
                "max": [10, 20, 30],
                "mean": [5.5, 10.5, 15.5],
                "std": [2.87, 5.87, 8.87],
                "count": 10,
            },
            "observation.state": {"min": 1, "max": 10, "mean": 5.5, "std": 2.87, "count": 10},
            "extra_key_0": {"min": 5, "max": 25, "mean": 15, "std": 6, "count": 6},
        },
        {
            "observation.image": {
                "min": [2, 1, 0],
                "max": [15, 10, 5],
                "mean": [8.5, 5.5, 2.5],
                "std": [3.42, 2.42, 1.42],
                "count": 15,
            },
            "observation.state": {"min": 2, "max": 15, "mean": 8.5, "std": 3.42, "count": 15},
            "extra_key_1": {"min": 0, "max": 20, "mean": 10, "std": 5, "count": 5},
        },
    ]

    expected_agg_stats = {
        "observation.image": {
            "min": [1, 1, 0],
            "max": [15, 20, 30],
            "mean": [7.3, 7.5, 7.7],
            "std": [3.5317, 4.8267, 8.5581],
            "count": 25,
        },
        "observation.state": {
            "min": 1,
            "max": 15,
            "mean": 7.3,
            "std": 3.5317,
            "count": 25,
        },
        "extra_key_0": {
            "min": 5,
            "max": 25,
            "mean": 15.0,
            "std": 6.0,
            "count": 6,
        },
        "extra_key_1": {
            "min": 0,
            "max": 20,
            "mean": 10.0,
            "std": 5.0,
            "count": 5,
        },
    }

    # cast to numpy
    for ep_stats in all_stats:
        for fkey, stats in ep_stats.items():
            for k in stats:
                stats[k] = np.array(stats[k], dtype=np.int64 if k == "count" else np.float32)
                if fkey == "observation.image" and k != "count":
                    stats[k] = stats[k].reshape(3, 1, 1)  # for normalization on image channels
                else:
                    stats[k] = stats[k].reshape(1)

    # cast to numpy
    for fkey, stats in expected_agg_stats.items():
        for k in stats:
            stats[k] = np.array(stats[k], dtype=np.int64 if k == "count" else np.float32)
            if fkey == "observation.image" and k != "count":
                stats[k] = stats[k].reshape(3, 1, 1)  # for normalization on image channels
            else:
                stats[k] = stats[k].reshape(1)

    results = aggregate_stats(all_stats)

    for fkey in expected_agg_stats:
        np.testing.assert_allclose(results[fkey]["min"], expected_agg_stats[fkey]["min"])
        np.testing.assert_allclose(results[fkey]["max"], expected_agg_stats[fkey]["max"])
        np.testing.assert_allclose(results[fkey]["mean"], expected_agg_stats[fkey]["mean"])
        np.testing.assert_allclose(
            results[fkey]["std"], expected_agg_stats[fkey]["std"], atol=1e-04, rtol=1e-04
        )
        np.testing.assert_allclose(results[fkey]["count"], expected_agg_stats[fkey]["count"])
