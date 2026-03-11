#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Integration tests for quantile functionality in LeRobotDataset."""

import numpy as np
import pytest

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def mock_load_image_as_numpy(path, dtype, channel_first):
    """Mock image loading for consistent test results."""
    return np.ones((3, 32, 32), dtype=dtype) if channel_first else np.ones((32, 32, 3), dtype=dtype)


@pytest.fixture
def simple_features():
    """Simple feature configuration for testing."""
    return {
        "action": {
            "dtype": "float32",
            "shape": (4,),
            "names": ["arm_x", "arm_y", "arm_z", "gripper"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (10,),
            "names": [f"joint_{i}" for i in range(10)],
        },
    }


def test_create_dataset_with_fixed_quantiles(tmp_path, simple_features):
    """Test creating dataset with fixed quantiles."""
    dataset = LeRobotDataset.create(
        repo_id="test_dataset_fixed_quantiles",
        fps=30,
        features=simple_features,
        root=tmp_path / "create_fixed_quantiles",
    )

    # Dataset should be created successfully
    assert dataset is not None


def test_save_episode_computes_all_quantiles(tmp_path, simple_features):
    """Test that all fixed quantiles are computed when saving an episode."""
    dataset = LeRobotDataset.create(
        repo_id="test_dataset_save_episode",
        fps=30,
        features=simple_features,
        root=tmp_path / "save_episode_quantiles",
    )

    # Add some frames
    for _ in range(10):
        dataset.add_frame(
            {
                "action": np.random.randn(4).astype(np.float32),  # Correct shape for action
                "observation.state": np.random.randn(10).astype(np.float32),
                "task": "test_task",
            }
        )

    dataset.save_episode()

    # Check that all fixed quantiles were computed
    stats = dataset.meta.stats
    for key in ["action", "observation.state"]:
        assert "q01" in stats[key]
        assert "q10" in stats[key]
        assert "q50" in stats[key]
        assert "q90" in stats[key]
        assert "q99" in stats[key]


def test_quantile_values_ordering(tmp_path, simple_features):
    """Test that quantile values are properly ordered."""
    dataset = LeRobotDataset.create(
        repo_id="test_dataset_quantile_ordering",
        fps=30,
        features=simple_features,
        root=tmp_path / "quantile_ordering",
    )

    # Add data with known distribution
    np.random.seed(42)
    for _ in range(100):
        dataset.add_frame(
            {
                "action": np.random.randn(4).astype(np.float32),  # Correct shape for action
                "observation.state": np.random.randn(10).astype(np.float32),
                "task": "test_task",
            }
        )

    dataset.save_episode()
    stats = dataset.meta.stats

    # Verify quantile ordering
    for key in ["action", "observation.state"]:
        assert np.all(stats[key]["q01"] <= stats[key]["q10"])
        assert np.all(stats[key]["q10"] <= stats[key]["q50"])
        assert np.all(stats[key]["q50"] <= stats[key]["q90"])
        assert np.all(stats[key]["q90"] <= stats[key]["q99"])


def test_save_episode_with_fixed_quantiles(tmp_path, simple_features):
    """Test saving episode always computes fixed quantiles."""
    dataset = LeRobotDataset.create(
        repo_id="test_dataset_save_fixed",
        fps=30,
        features=simple_features,
        root=tmp_path / "save_fixed_quantiles",
    )

    # Add frames to episode
    np.random.seed(42)
    for _ in range(50):
        frame = {
            "action": np.random.normal(0, 1, (4,)).astype(np.float32),
            "observation.state": np.random.normal(0, 1, (10,)).astype(np.float32),
            "task": "test_task",
        }
        dataset.add_frame(frame)

    dataset.save_episode()

    # Check that all fixed quantiles are included
    stats = dataset.meta.stats
    for key in ["action", "observation.state"]:
        feature_stats = stats[key]
        expected_keys = {"min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"}
        assert set(feature_stats.keys()) == expected_keys


def test_quantile_aggregation_across_episodes(tmp_path, simple_features):
    """Test quantile aggregation across multiple episodes."""
    dataset = LeRobotDataset.create(
        repo_id="test_dataset_aggregation",
        fps=30,
        features=simple_features,
        root=tmp_path / "quantile_aggregation",
    )

    # Add frames to episode
    np.random.seed(42)
    for _ in range(100):
        frame = {
            "action": np.random.normal(0, 1, (4,)).astype(np.float32),
            "observation.state": np.random.normal(2, 1, (10,)).astype(np.float32),
            "task": "test_task",
        }
        dataset.add_frame(frame)

    dataset.save_episode()

    # Check stats include all fixed quantiles
    stats = dataset.meta.stats
    for key in ["action", "observation.state"]:
        feature_stats = stats[key]
        expected_keys = {"min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"}
        assert set(feature_stats.keys()) == expected_keys
        assert feature_stats["q01"].shape == (simple_features[key]["shape"][0],)
        assert feature_stats["q50"].shape == (simple_features[key]["shape"][0],)
        assert feature_stats["q99"].shape == (simple_features[key]["shape"][0],)
        assert np.all(feature_stats["q01"] <= feature_stats["q50"])
        assert np.all(feature_stats["q50"] <= feature_stats["q99"])


def test_save_multiple_episodes_with_quantiles(tmp_path, simple_features):
    """Test quantile aggregation across multiple episodes."""
    dataset = LeRobotDataset.create(
        repo_id="test_dataset_multiple_episodes",
        fps=30,
        features=simple_features,
        root=tmp_path / "multiple_episodes",
    )

    # Save multiple episodes
    np.random.seed(42)
    for episode_idx in range(3):
        for _ in range(50):
            frame = {
                "action": np.random.normal(episode_idx * 2.0, 1, (4,)).astype(np.float32),
                "observation.state": np.random.normal(-episode_idx * 1.5, 1, (10,)).astype(np.float32),
                "task": f"task_{episode_idx}",
            }
            dataset.add_frame(frame)

        dataset.save_episode()

    # Verify final stats include properly aggregated quantiles
    stats = dataset.meta.stats
    for key in ["action", "observation.state"]:
        feature_stats = stats[key]
        assert "q01" in feature_stats and "q99" in feature_stats
        assert feature_stats["count"][0] == 150  # 3 episodes * 50 frames
