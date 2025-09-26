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

from lerobot.datasets.compute_stats import (
    RunningQuantileStats,
    _assert_type_and_shape,
    aggregate_feature_stats,
    aggregate_stats,
    compute_episode_stats,
    estimate_num_samples,
    get_feature_stats,
    sample_images,
    sample_indices,
)
from lerobot.utils.constants import OBS_IMAGE, OBS_STATE


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


@patch("lerobot.datasets.compute_stats.load_image_as_numpy", side_effect=mock_load_image_as_numpy)
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

    # Check that basic stats are correct (quantiles are also included now)
    assert set(expected.keys()).issubset(set(result.keys()))
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

    # Check that basic stats are correct (quantiles are also included now)
    assert set(expected.keys()).issubset(set(result.keys()))
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
        OBS_IMAGE: [f"image_{i}.jpg" for i in range(100)],
        OBS_STATE: np.random.rand(100, 10),
    }
    features = {
        OBS_IMAGE: {"dtype": "image"},
        OBS_STATE: {"dtype": "numeric"},
    }

    with patch("lerobot.datasets.compute_stats.load_image_as_numpy", side_effect=mock_load_image_as_numpy):
        stats = compute_episode_stats(episode_data, features)

    assert OBS_IMAGE in stats and OBS_STATE in stats
    assert stats[OBS_IMAGE]["count"].item() == 100
    assert stats[OBS_STATE]["count"].item() == 100
    assert stats[OBS_IMAGE]["mean"].shape == (3, 1, 1)


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
            OBS_IMAGE: {
                "min": [1, 2, 3],
                "max": [10, 20, 30],
                "mean": [5.5, 10.5, 15.5],
                "std": [2.87, 5.87, 8.87],
                "count": 10,
            },
            OBS_STATE: {"min": 1, "max": 10, "mean": 5.5, "std": 2.87, "count": 10},
            "extra_key_0": {"min": 5, "max": 25, "mean": 15, "std": 6, "count": 6},
        },
        {
            OBS_IMAGE: {
                "min": [2, 1, 0],
                "max": [15, 10, 5],
                "mean": [8.5, 5.5, 2.5],
                "std": [3.42, 2.42, 1.42],
                "count": 15,
            },
            OBS_STATE: {"min": 2, "max": 15, "mean": 8.5, "std": 3.42, "count": 15},
            "extra_key_1": {"min": 0, "max": 20, "mean": 10, "std": 5, "count": 5},
        },
    ]

    expected_agg_stats = {
        OBS_IMAGE: {
            "min": [1, 1, 0],
            "max": [15, 20, 30],
            "mean": [7.3, 7.5, 7.7],
            "std": [3.5317, 4.8267, 8.5581],
            "count": 25,
        },
        OBS_STATE: {
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
                if fkey == OBS_IMAGE and k != "count":
                    stats[k] = stats[k].reshape(3, 1, 1)  # for normalization on image channels
                else:
                    stats[k] = stats[k].reshape(1)

    # cast to numpy
    for fkey, stats in expected_agg_stats.items():
        for k in stats:
            stats[k] = np.array(stats[k], dtype=np.int64 if k == "count" else np.float32)
            if fkey == OBS_IMAGE and k != "count":
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


def test_running_quantile_stats_initialization():
    """Test proper initialization of RunningQuantileStats."""
    running_stats = RunningQuantileStats()
    assert running_stats._count == 0
    assert running_stats._mean is None
    assert running_stats._num_quantile_bins == 5000

    # Test custom bin size
    running_stats_custom = RunningQuantileStats(num_quantile_bins=1000)
    assert running_stats_custom._num_quantile_bins == 1000


def test_running_quantile_stats_single_batch_update():
    """Test updating with a single batch."""
    np.random.seed(42)
    data = np.random.normal(0, 1, (100, 3))

    running_stats = RunningQuantileStats()
    running_stats.update(data)

    assert running_stats._count == 100
    assert running_stats._mean.shape == (3,)
    assert len(running_stats._histograms) == 3
    assert len(running_stats._bin_edges) == 3

    # Verify basic statistics are reasonable
    np.testing.assert_allclose(running_stats._mean, np.mean(data, axis=0), atol=1e-10)


def test_running_quantile_stats_multiple_batch_updates():
    """Test updating with multiple batches."""
    np.random.seed(42)
    data1 = np.random.normal(0, 1, (100, 2))
    data2 = np.random.normal(1, 1, (150, 2))

    running_stats = RunningQuantileStats()
    running_stats.update(data1)
    running_stats.update(data2)

    assert running_stats._count == 250

    # Verify running mean is correct
    combined_data = np.vstack([data1, data2])
    expected_mean = np.mean(combined_data, axis=0)
    np.testing.assert_allclose(running_stats._mean, expected_mean, atol=1e-10)


def test_running_quantile_stats_get_statistics_basic():
    """Test getting basic statistics without quantiles."""
    np.random.seed(42)
    data = np.random.normal(0, 1, (100, 2))

    running_stats = RunningQuantileStats()
    running_stats.update(data)

    stats = running_stats.get_statistics()

    # Should have basic stats
    expected_keys = {"min", "max", "mean", "std", "count"}
    assert expected_keys.issubset(set(stats.keys()))

    # Verify values
    np.testing.assert_allclose(stats["mean"], np.mean(data, axis=0), atol=1e-10)
    np.testing.assert_allclose(stats["std"], np.std(data, axis=0), atol=1e-6)
    np.testing.assert_equal(stats["count"], np.array([100]))


def test_running_quantile_stats_get_statistics_with_quantiles():
    """Test getting statistics with quantiles."""
    np.random.seed(42)
    data = np.random.normal(0, 1, (1000, 2))

    running_stats = RunningQuantileStats()
    running_stats.update(data)

    stats = running_stats.get_statistics()

    # Should have basic stats plus quantiles
    expected_keys = {"min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"}
    assert expected_keys.issubset(set(stats.keys()))

    # Verify quantile values are reasonable
    from lerobot.datasets.compute_stats import DEFAULT_QUANTILES

    for i, q in enumerate(DEFAULT_QUANTILES):
        q_key = f"q{int(q * 100):02d}"
        assert q_key in stats
        assert stats[q_key].shape == (2,)

        # Check that quantiles are in reasonable order
        if i > 0:
            prev_q_key = f"q{int(DEFAULT_QUANTILES[i - 1] * 100):02d}"
            assert np.all(stats[prev_q_key] <= stats[q_key])


def test_running_quantile_stats_histogram_adjustment():
    """Test that histograms adjust when min/max change."""
    running_stats = RunningQuantileStats()

    # Initial data with small range
    data1 = np.array([[0.0, 1.0], [0.1, 1.1], [0.2, 1.2]])
    running_stats.update(data1)

    initial_edges_0 = running_stats._bin_edges[0].copy()
    initial_edges_1 = running_stats._bin_edges[1].copy()

    # Add data with much larger range
    data2 = np.array([[10.0, -10.0], [11.0, -11.0]])
    running_stats.update(data2)

    # Bin edges should have changed
    assert not np.array_equal(initial_edges_0, running_stats._bin_edges[0])
    assert not np.array_equal(initial_edges_1, running_stats._bin_edges[1])

    # New edges should cover the expanded range
    # First dimension: min should still be ~0.0, max should be ~11.0
    assert running_stats._bin_edges[0][0] <= 0.0
    assert running_stats._bin_edges[0][-1] >= 11.0

    # Second dimension: min should be ~-11.0, max should be ~1.2
    assert running_stats._bin_edges[1][0] <= -11.0
    assert running_stats._bin_edges[1][-1] >= 1.2


def test_running_quantile_stats_insufficient_data_error():
    """Test error when trying to get stats with insufficient data."""
    running_stats = RunningQuantileStats()

    with pytest.raises(ValueError, match="Cannot compute statistics for less than 2 vectors"):
        running_stats.get_statistics()

    # Single vector should also fail
    running_stats.update(np.array([[1.0]]))
    with pytest.raises(ValueError, match="Cannot compute statistics for less than 2 vectors"):
        running_stats.get_statistics()


def test_running_quantile_stats_vector_length_consistency():
    """Test error when vector lengths don't match."""
    running_stats = RunningQuantileStats()
    running_stats.update(np.array([[1.0, 2.0], [3.0, 4.0]]))

    with pytest.raises(ValueError, match="The length of new vectors does not match"):
        running_stats.update(np.array([[1.0, 2.0, 3.0]]))  # Different length


def test_running_quantile_stats_reshape_handling():
    """Test that various input shapes are handled correctly."""
    running_stats = RunningQuantileStats()

    # Test 3D input (e.g., images)
    data_3d = np.random.normal(0, 1, (10, 32, 32))
    running_stats.update(data_3d)

    assert running_stats._count == 10 * 32
    assert running_stats._mean.shape == (32,)

    # Test 1D input
    running_stats_1d = RunningQuantileStats()
    data_1d = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    running_stats_1d.update(data_1d)

    assert running_stats_1d._count == 5
    assert running_stats_1d._mean.shape == (1,)


def test_get_feature_stats_quantiles_enabled_by_default():
    """Test that quantiles are computed by default."""
    data = np.random.normal(0, 1, (100, 5))
    stats = get_feature_stats(data, axis=0, keepdims=False)

    expected_keys = {"min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"}
    assert set(stats.keys()) == expected_keys


def test_get_feature_stats_quantiles_with_vector_data():
    """Test quantile computation with vector data."""
    np.random.seed(42)
    data = np.random.normal(0, 1, (100, 5))

    stats = get_feature_stats(data, axis=0, keepdims=False)

    expected_keys = {"min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"}
    assert set(stats.keys()) == expected_keys

    # Verify shapes
    assert stats["q01"].shape == (5,)
    assert stats["q99"].shape == (5,)

    # Verify quantiles are reasonable
    assert np.all(stats["q01"] < stats["q99"])


def test_get_feature_stats_quantiles_with_image_data():
    """Test quantile computation with image data."""
    np.random.seed(42)
    data = np.random.normal(0, 1, (50, 3, 32, 32))  # batch, channels, height, width

    stats = get_feature_stats(data, axis=(0, 2, 3), keepdims=True)

    expected_keys = {"min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"}
    assert set(stats.keys()) == expected_keys

    # Verify shapes for images (should be (1, channels, 1, 1))
    assert stats["q01"].shape == (1, 3, 1, 1)
    assert stats["q50"].shape == (1, 3, 1, 1)
    assert stats["q99"].shape == (1, 3, 1, 1)


def test_get_feature_stats_fixed_quantiles():
    """Test that fixed quantiles are always computed."""
    data = np.random.normal(0, 1, (200, 3))

    stats = get_feature_stats(data, axis=0, keepdims=False)

    expected_quantile_keys = {"q01", "q10", "q50", "q90", "q99"}
    assert expected_quantile_keys.issubset(set(stats.keys()))


def test_get_feature_stats_unsupported_axis_error():
    """Test error for unsupported axis configuration."""
    data = np.random.normal(0, 1, (10, 5))

    with pytest.raises(ValueError, match="Unsupported axis configuration"):
        get_feature_stats(
            data,
            axis=(1, 2),  # Unsupported axis
            keepdims=False,
        )


def test_compute_episode_stats_backward_compatibility():
    """Test that existing functionality is preserved."""
    episode_data = {
        "action": np.random.normal(0, 1, (100, 7)),
        "observation.state": np.random.normal(0, 1, (100, 10)),
    }
    features = {
        "action": {"dtype": "float32", "shape": (7,)},
        "observation.state": {"dtype": "float32", "shape": (10,)},
    }

    stats = compute_episode_stats(episode_data, features)

    for key in ["action", "observation.state"]:
        expected_keys = {"min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"}
        assert set(stats[key].keys()) == expected_keys


def test_compute_episode_stats_with_custom_quantiles():
    """Test quantile computation with custom quantile values."""
    np.random.seed(42)
    episode_data = {
        "action": np.random.normal(0, 1, (100, 7)),
        "observation.state": np.random.normal(2, 1, (100, 10)),
    }
    features = {
        "action": {"dtype": "float32", "shape": (7,)},
        "observation.state": {"dtype": "float32", "shape": (10,)},
    }

    stats = compute_episode_stats(episode_data, features)

    # Should have quantiles
    for key in ["action", "observation.state"]:
        expected_keys = {"min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"}
        assert set(stats[key].keys()) == expected_keys

        # Verify shapes
        assert stats[key]["q01"].shape == (features[key]["shape"][0],)
        assert stats[key]["q99"].shape == (features[key]["shape"][0],)


def test_compute_episode_stats_with_image_data():
    """Test quantile computation with image features."""
    image_paths = [f"image_{i}.jpg" for i in range(50)]
    episode_data = {
        "observation.image": image_paths,
        "action": np.random.normal(0, 1, (50, 5)),
    }
    features = {
        "observation.image": {"dtype": "image"},
        "action": {"dtype": "float32", "shape": (5,)},
    }

    with patch("lerobot.datasets.compute_stats.load_image_as_numpy", side_effect=mock_load_image_as_numpy):
        stats = compute_episode_stats(episode_data, features)

    # Image quantiles should be normalized and have correct shape
    assert "q01" in stats["observation.image"]
    assert "q50" in stats["observation.image"]
    assert "q99" in stats["observation.image"]
    assert stats["observation.image"]["q01"].shape == (3, 1, 1)
    assert stats["observation.image"]["q50"].shape == (3, 1, 1)
    assert stats["observation.image"]["q99"].shape == (3, 1, 1)

    # Action quantiles should have correct shape
    assert stats["action"]["q01"].shape == (5,)
    assert stats["action"]["q50"].shape == (5,)
    assert stats["action"]["q99"].shape == (5,)


def test_compute_episode_stats_string_features_skipped():
    """Test that string features are properly skipped."""
    episode_data = {
        "task": ["pick_apple"] * 100,  # String feature
        "action": np.random.normal(0, 1, (100, 5)),
    }
    features = {
        "task": {"dtype": "string"},
        "action": {"dtype": "float32", "shape": (5,)},
    }

    stats = compute_episode_stats(
        episode_data,
        features,
    )

    # String features should be skipped
    assert "task" not in stats
    assert "action" in stats
    assert "q01" in stats["action"]


def test_aggregate_feature_stats_with_quantiles():
    """Test aggregating feature stats that include quantiles."""
    stats_ft_list = [
        {
            "min": np.array([1.0]),
            "max": np.array([10.0]),
            "mean": np.array([5.0]),
            "std": np.array([2.0]),
            "count": np.array([100]),
            "q01": np.array([1.5]),
            "q99": np.array([9.5]),
        },
        {
            "min": np.array([2.0]),
            "max": np.array([12.0]),
            "mean": np.array([6.0]),
            "std": np.array([2.5]),
            "count": np.array([150]),
            "q01": np.array([2.5]),
            "q99": np.array([11.5]),
        },
    ]

    result = aggregate_feature_stats(stats_ft_list)

    # Should preserve quantiles
    assert "q01" in result
    assert "q99" in result

    # Verify quantile aggregation (weighted average)
    expected_q01 = (1.5 * 100 + 2.5 * 150) / 250  # ≈ 2.1
    expected_q99 = (9.5 * 100 + 11.5 * 150) / 250  # ≈ 10.7

    np.testing.assert_allclose(result["q01"], np.array([expected_q01]), atol=1e-6)
    np.testing.assert_allclose(result["q99"], np.array([expected_q99]), atol=1e-6)


def test_aggregate_stats_mixed_quantiles():
    """Test aggregating stats where some have quantiles and some don't."""
    stats_with_quantiles = {
        "feature1": {
            "min": np.array([1.0]),
            "max": np.array([10.0]),
            "mean": np.array([5.0]),
            "std": np.array([2.0]),
            "count": np.array([100]),
            "q01": np.array([1.5]),
            "q99": np.array([9.5]),
        }
    }

    stats_without_quantiles = {
        "feature2": {
            "min": np.array([0.0]),
            "max": np.array([5.0]),
            "mean": np.array([2.5]),
            "std": np.array([1.5]),
            "count": np.array([50]),
        }
    }

    all_stats = [stats_with_quantiles, stats_without_quantiles]
    result = aggregate_stats(all_stats)

    # Feature1 should keep its quantiles
    assert "q01" in result["feature1"]
    assert "q99" in result["feature1"]

    # Feature2 should not have quantiles
    assert "q01" not in result["feature2"]
    assert "q99" not in result["feature2"]


def test_assert_type_and_shape_with_quantiles():
    """Test validation works correctly with quantile keys."""
    # Valid stats with quantiles
    valid_stats = [
        {
            "observation.image": {
                "min": np.array([0.0, 0.0, 0.0]).reshape(3, 1, 1),
                "max": np.array([1.0, 1.0, 1.0]).reshape(3, 1, 1),
                "mean": np.array([0.5, 0.5, 0.5]).reshape(3, 1, 1),
                "std": np.array([0.2, 0.2, 0.2]).reshape(3, 1, 1),
                "count": np.array([100]),
                "q01": np.array([0.1, 0.1, 0.1]).reshape(3, 1, 1),
                "q99": np.array([0.9, 0.9, 0.9]).reshape(3, 1, 1),
            }
        }
    ]

    # Should not raise error
    _assert_type_and_shape(valid_stats)

    # Invalid shape for quantile
    invalid_stats = [
        {
            "observation.image": {
                "count": np.array([100]),
                "q01": np.array([0.1, 0.2]),  # Wrong shape for image quantile
            }
        }
    ]

    with pytest.raises(ValueError, match="Shape of quantile 'q01' must be \\(3,1,1\\)"):
        _assert_type_and_shape(invalid_stats)


def test_quantile_integration_single_value_quantiles():
    """Test quantile computation with single repeated value."""
    data = np.ones((100, 3))  # All ones

    running_stats = RunningQuantileStats()
    running_stats.update(data)

    stats = running_stats.get_statistics()

    # All quantiles should be approximately 1.0
    np.testing.assert_allclose(stats["q01"], np.array([1.0, 1.0, 1.0]), atol=1e-6)
    np.testing.assert_allclose(stats["q50"], np.array([1.0, 1.0, 1.0]), atol=1e-6)
    np.testing.assert_allclose(stats["q99"], np.array([1.0, 1.0, 1.0]), atol=1e-6)


def test_quantile_integration_fixed_quantiles():
    """Test that fixed quantiles are computed."""
    np.random.seed(42)
    data = np.random.normal(0, 1, (1000, 2))

    stats = get_feature_stats(data, axis=0, keepdims=False)

    # Check all fixed quantiles are present
    assert "q01" in stats
    assert "q10" in stats
    assert "q50" in stats
    assert "q90" in stats
    assert "q99" in stats


def test_quantile_integration_large_dataset_quantiles():
    """Test quantile computation efficiency with large datasets."""
    np.random.seed(42)
    large_data = np.random.normal(0, 1, (10000, 5))

    running_stats = RunningQuantileStats(num_quantile_bins=1000)  # Reduced bins for speed
    running_stats.update(large_data)

    stats = running_stats.get_statistics()

    # Should complete without issues and produce reasonable results
    assert stats["count"][0] == 10000
    assert len(stats["q01"]) == 5


def test_fixed_quantiles_always_computed():
    """Test that the fixed quantiles [0.01, 0.10, 0.50, 0.90, 0.99] are always computed."""
    np.random.seed(42)
    # Test with vector data
    vector_data = np.random.normal(0, 1, (100, 5))
    vector_stats = get_feature_stats(vector_data, axis=0, keepdims=False)

    # Check all fixed quantiles are present
    expected_quantiles = ["q01", "q10", "q50", "q90", "q99"]
    for q_key in expected_quantiles:
        assert q_key in vector_stats
        assert vector_stats[q_key].shape == (5,)

    # Test with image data
    image_data = np.random.randint(0, 256, (50, 3, 32, 32), dtype=np.uint8)
    image_stats = get_feature_stats(image_data, axis=(0, 2, 3), keepdims=True)

    # Check all fixed quantiles are present for images
    for q_key in expected_quantiles:
        assert q_key in image_stats
        assert image_stats[q_key].shape == (1, 3, 1, 1)

    # Test with episode data
    episode_data = {
        "action": np.random.normal(0, 1, (100, 7)),
        "observation.state": np.random.normal(0, 1, (100, 10)),
    }
    features = {
        "action": {"dtype": "float32", "shape": (7,)},
        "observation.state": {"dtype": "float32", "shape": (10,)},
    }

    episode_stats = compute_episode_stats(episode_data, features)

    # Check all fixed quantiles are present in episode stats
    for key in ["action", "observation.state"]:
        for q_key in expected_quantiles:
            assert q_key in episode_stats[key]
            assert episode_stats[key][q_key].shape == (features[key]["shape"][0],)
