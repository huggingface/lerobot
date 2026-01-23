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
"""Tests for dataset tools utilities."""

from unittest.mock import patch

import numpy as np
import pytest
import torch

from lerobot.datasets.dataset_tools import (
    add_features,
    delete_episodes,
    merge_datasets,
    modify_features,
    remove_feature,
    split_dataset,
)
from lerobot.scripts.lerobot_edit_dataset import convert_image_to_video_dataset


@pytest.fixture
def sample_dataset(tmp_path, empty_lerobot_dataset_factory):
    """Create a sample dataset for testing."""
    features = {
        "action": {"dtype": "float32", "shape": (6,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (4,), "names": None},
        "observation.images.top": {"dtype": "image", "shape": (224, 224, 3), "names": None},
    }

    dataset = empty_lerobot_dataset_factory(
        root=tmp_path / "test_dataset",
        features=features,
    )

    for ep_idx in range(5):
        for _ in range(10):
            frame = {
                "action": np.random.randn(6).astype(np.float32),
                "observation.state": np.random.randn(4).astype(np.float32),
                "observation.images.top": np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8),
                "task": f"task_{ep_idx % 2}",
            }
            dataset.add_frame(frame)
        dataset.save_episode()

    dataset.finalize()
    return dataset


def test_delete_single_episode(sample_dataset, tmp_path):
    """Test deleting a single episode."""
    output_dir = tmp_path / "filtered"

    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(output_dir)

        new_dataset = delete_episodes(
            sample_dataset,
            episode_indices=[2],
            output_dir=output_dir,
        )

    assert new_dataset.meta.total_episodes == 4
    assert new_dataset.meta.total_frames == 40

    episode_indices = {int(idx.item()) for idx in new_dataset.hf_dataset["episode_index"]}
    assert episode_indices == {0, 1, 2, 3}

    assert len(new_dataset) == 40


def test_delete_multiple_episodes(sample_dataset, tmp_path):
    """Test deleting multiple episodes."""
    output_dir = tmp_path / "filtered"

    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(output_dir)

        new_dataset = delete_episodes(
            sample_dataset,
            episode_indices=[1, 3],
            output_dir=output_dir,
        )

    assert new_dataset.meta.total_episodes == 3
    assert new_dataset.meta.total_frames == 30

    episode_indices = {int(idx.item()) for idx in new_dataset.hf_dataset["episode_index"]}
    assert episode_indices == {0, 1, 2}


def test_delete_invalid_episodes(sample_dataset, tmp_path):
    """Test error handling for invalid episode indices."""
    with pytest.raises(ValueError, match="Invalid episode indices"):
        delete_episodes(
            sample_dataset,
            episode_indices=[10, 20],
            output_dir=tmp_path / "filtered",
        )


def test_delete_all_episodes(sample_dataset, tmp_path):
    """Test error when trying to delete all episodes."""
    with pytest.raises(ValueError, match="Cannot delete all episodes"):
        delete_episodes(
            sample_dataset,
            episode_indices=list(range(5)),
            output_dir=tmp_path / "filtered",
        )


def test_delete_empty_list(sample_dataset, tmp_path):
    """Test error when no episodes specified."""
    with pytest.raises(ValueError, match="No episodes to delete"):
        delete_episodes(
            sample_dataset,
            episode_indices=[],
            output_dir=tmp_path / "filtered",
        )


def test_split_by_episodes(sample_dataset, tmp_path):
    """Test splitting dataset by specific episode indices."""
    splits = {
        "train": [0, 1, 2],
        "val": [3, 4],
    }

    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"

        def mock_snapshot(repo_id, **kwargs):
            if "train" in repo_id:
                return str(tmp_path / f"{sample_dataset.repo_id}_train")
            elif "val" in repo_id:
                return str(tmp_path / f"{sample_dataset.repo_id}_val")
            return str(kwargs.get("local_dir", tmp_path))

        mock_snapshot_download.side_effect = mock_snapshot

        result = split_dataset(
            sample_dataset,
            splits=splits,
            output_dir=tmp_path,
        )

    assert set(result.keys()) == {"train", "val"}

    assert result["train"].meta.total_episodes == 3
    assert result["train"].meta.total_frames == 30

    assert result["val"].meta.total_episodes == 2
    assert result["val"].meta.total_frames == 20

    train_episodes = {int(idx.item()) for idx in result["train"].hf_dataset["episode_index"]}
    assert train_episodes == {0, 1, 2}

    val_episodes = {int(idx.item()) for idx in result["val"].hf_dataset["episode_index"]}
    assert val_episodes == {0, 1}


def test_split_by_fractions(sample_dataset, tmp_path):
    """Test splitting dataset by fractions."""
    splits = {
        "train": 0.6,
        "val": 0.4,
    }

    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"

        def mock_snapshot(repo_id, **kwargs):
            for split_name in splits:
                if split_name in repo_id:
                    return str(tmp_path / f"{sample_dataset.repo_id}_{split_name}")
            return str(kwargs.get("local_dir", tmp_path))

        mock_snapshot_download.side_effect = mock_snapshot

        result = split_dataset(
            sample_dataset,
            splits=splits,
            output_dir=tmp_path,
        )

    assert result["train"].meta.total_episodes == 3
    assert result["val"].meta.total_episodes == 2


def test_split_overlapping_episodes(sample_dataset, tmp_path):
    """Test error when episodes appear in multiple splits."""
    splits = {
        "train": [0, 1, 2],
        "val": [2, 3, 4],
    }

    with pytest.raises(ValueError, match="Episodes cannot appear in multiple splits"):
        split_dataset(sample_dataset, splits=splits, output_dir=tmp_path)


def test_split_invalid_fractions(sample_dataset, tmp_path):
    """Test error when fractions sum to more than 1."""
    splits = {
        "train": 0.7,
        "val": 0.5,
    }

    with pytest.raises(ValueError, match="Split fractions must sum to <= 1.0"):
        split_dataset(sample_dataset, splits=splits, output_dir=tmp_path)


def test_split_empty(sample_dataset, tmp_path):
    """Test error with empty splits."""
    with pytest.raises(ValueError, match="No splits provided"):
        split_dataset(sample_dataset, splits={}, output_dir=tmp_path)


def test_merge_two_datasets(sample_dataset, tmp_path, empty_lerobot_dataset_factory):
    """Test merging two datasets."""
    features = {
        "action": {"dtype": "float32", "shape": (6,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (4,), "names": None},
        "observation.images.top": {"dtype": "image", "shape": (224, 224, 3), "names": None},
    }

    dataset2 = empty_lerobot_dataset_factory(
        root=tmp_path / "test_dataset2",
        features=features,
    )

    for ep_idx in range(3):
        for _ in range(10):
            frame = {
                "action": np.random.randn(6).astype(np.float32),
                "observation.state": np.random.randn(4).astype(np.float32),
                "observation.images.top": np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8),
                "task": f"task_{ep_idx % 2}",
            }
            dataset2.add_frame(frame)
        dataset2.save_episode()
    dataset2.finalize()

    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(tmp_path / "merged_dataset")

        merged = merge_datasets(
            [sample_dataset, dataset2],
            output_repo_id="merged_dataset",
            output_dir=tmp_path / "merged_dataset",
        )

    assert merged.meta.total_episodes == 8  # 5 + 3
    assert merged.meta.total_frames == 80  # 50 + 30

    episode_indices = sorted({int(idx.item()) for idx in merged.hf_dataset["episode_index"]})
    assert episode_indices == list(range(8))


def test_merge_empty_list(tmp_path):
    """Test error when merging empty list."""
    with pytest.raises(ValueError, match="No datasets to merge"):
        merge_datasets([], output_repo_id="merged", output_dir=tmp_path)


def test_add_features_with_values(sample_dataset, tmp_path):
    """Test adding a feature with pre-computed values."""
    num_frames = sample_dataset.meta.total_frames
    reward_values = np.random.randn(num_frames, 1).astype(np.float32)

    feature_info = {
        "dtype": "float32",
        "shape": (1,),
        "names": None,
    }
    features = {
        "reward": (reward_values, feature_info),
    }

    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(tmp_path / "with_reward")

        new_dataset = add_features(
            dataset=sample_dataset,
            features=features,
            output_dir=tmp_path / "with_reward",
        )

    assert "reward" in new_dataset.meta.features
    assert new_dataset.meta.features["reward"] == feature_info

    assert len(new_dataset) == num_frames
    sample_item = new_dataset[0]
    assert "reward" in sample_item
    assert isinstance(sample_item["reward"], torch.Tensor)


def test_add_features_with_callable(sample_dataset, tmp_path):
    """Test adding a feature with a callable."""

    def compute_reward(frame_dict, episode_idx, frame_idx):
        return float(episode_idx * 10 + frame_idx)

    feature_info = {
        "dtype": "float32",
        "shape": (1,),
        "names": None,
    }
    features = {
        "reward": (compute_reward, feature_info),
    }
    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(tmp_path / "with_reward")

        new_dataset = add_features(
            dataset=sample_dataset,
            features=features,
            output_dir=tmp_path / "with_reward",
        )

    assert "reward" in new_dataset.meta.features

    items = [new_dataset[i] for i in range(10)]
    first_episode_items = [item for item in items if item["episode_index"] == 0]
    assert len(first_episode_items) == 10

    first_frame = first_episode_items[0]
    assert first_frame["frame_index"] == 0
    assert float(first_frame["reward"]) == 0.0


def test_add_existing_feature(sample_dataset, tmp_path):
    """Test error when adding an existing feature."""
    feature_info = {"dtype": "float32", "shape": (1,)}
    features = {
        "action": (np.zeros(50), feature_info),
    }

    with pytest.raises(ValueError, match="Feature 'action' already exists"):
        add_features(
            dataset=sample_dataset,
            features=features,
            output_dir=tmp_path / "modified",
        )


def test_add_feature_invalid_info(sample_dataset, tmp_path):
    """Test error with invalid feature info."""
    with pytest.raises(ValueError, match="feature_info for 'reward' must contain keys"):
        add_features(
            dataset=sample_dataset,
            features={
                "reward": (np.zeros(50), {"dtype": "float32"}),
            },
            output_dir=tmp_path / "modified",
        )


def test_modify_features_add_and_remove(sample_dataset, tmp_path):
    """Test modifying features by adding and removing simultaneously."""
    feature_info = {"dtype": "float32", "shape": (1,), "names": None}

    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(tmp_path / "modified")

        # First add a feature we'll later remove
        dataset_with_reward = add_features(
            sample_dataset,
            features={"reward": (np.random.randn(50, 1).astype(np.float32), feature_info)},
            output_dir=tmp_path / "with_reward",
        )

        # Now use modify_features to add "success" and remove "reward" in one pass
        modified_dataset = modify_features(
            dataset_with_reward,
            add_features={
                "success": (np.random.randn(50, 1).astype(np.float32), feature_info),
            },
            remove_features="reward",
            output_dir=tmp_path / "modified",
        )

    assert "success" in modified_dataset.meta.features
    assert "reward" not in modified_dataset.meta.features
    assert len(modified_dataset) == 50


def test_modify_features_only_add(sample_dataset, tmp_path):
    """Test that modify_features works with only add_features."""
    feature_info = {"dtype": "float32", "shape": (1,), "names": None}

    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(tmp_path / "modified")

        modified_dataset = modify_features(
            sample_dataset,
            add_features={
                "reward": (np.random.randn(50, 1).astype(np.float32), feature_info),
            },
            output_dir=tmp_path / "modified",
        )

    assert "reward" in modified_dataset.meta.features
    assert len(modified_dataset) == 50


def test_modify_features_only_remove(sample_dataset, tmp_path):
    """Test that modify_features works with only remove_features."""
    feature_info = {"dtype": "float32", "shape": (1,), "names": None}

    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.side_effect = lambda repo_id, **kwargs: str(kwargs.get("local_dir", tmp_path))

        dataset_with_reward = add_features(
            sample_dataset,
            features={"reward": (np.random.randn(50, 1).astype(np.float32), feature_info)},
            output_dir=tmp_path / "with_reward",
        )

        modified_dataset = modify_features(
            dataset_with_reward,
            remove_features="reward",
            output_dir=tmp_path / "modified",
        )

    assert "reward" not in modified_dataset.meta.features


def test_modify_features_no_changes(sample_dataset, tmp_path):
    """Test error when modify_features is called with no changes."""
    with pytest.raises(ValueError, match="Must specify at least one of add_features or remove_features"):
        modify_features(
            sample_dataset,
            output_dir=tmp_path / "modified",
        )


def test_remove_single_feature(sample_dataset, tmp_path):
    """Test removing a single feature."""
    feature_info = {"dtype": "float32", "shape": (1,), "names": None}
    features = {
        "reward": (np.random.randn(50, 1).astype(np.float32), feature_info),
    }
    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.side_effect = lambda repo_id, **kwargs: str(kwargs.get("local_dir", tmp_path))

        dataset_with_reward = add_features(
            dataset=sample_dataset,
            features=features,
            output_dir=tmp_path / "with_reward",
        )

        dataset_without_reward = remove_feature(
            dataset_with_reward,
            feature_names="reward",
            output_dir=tmp_path / "without_reward",
        )

    assert "reward" not in dataset_without_reward.meta.features

    sample_item = dataset_without_reward[0]
    assert "reward" not in sample_item


def test_remove_multiple_features(sample_dataset, tmp_path):
    """Test removing multiple features at once."""
    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.side_effect = lambda repo_id, **kwargs: str(kwargs.get("local_dir", tmp_path))

        dataset = sample_dataset
        features = {}
        for feature_name in ["reward", "success"]:
            feature_info = {"dtype": "float32", "shape": (1,), "names": None}
            features[feature_name] = (
                np.random.randn(dataset.meta.total_frames, 1).astype(np.float32),
                feature_info,
            )

        dataset_with_features = add_features(
            dataset, features=features, output_dir=tmp_path / "with_features"
        )
        dataset_clean = remove_feature(
            dataset_with_features, feature_names=["reward", "success"], output_dir=tmp_path / "clean"
        )

    assert "reward" not in dataset_clean.meta.features
    assert "success" not in dataset_clean.meta.features


def test_remove_nonexistent_feature(sample_dataset, tmp_path):
    """Test error when removing non-existent feature."""
    with pytest.raises(ValueError, match="Feature 'nonexistent' not found"):
        remove_feature(
            sample_dataset,
            feature_names="nonexistent",
            output_dir=tmp_path / "modified",
        )


def test_remove_required_feature(sample_dataset, tmp_path):
    """Test error when trying to remove required features."""
    with pytest.raises(ValueError, match="Cannot remove required features"):
        remove_feature(
            sample_dataset,
            feature_names="timestamp",
            output_dir=tmp_path / "modified",
        )


def test_remove_camera_feature(sample_dataset, tmp_path):
    """Test removing a camera feature."""
    camera_keys = sample_dataset.meta.camera_keys
    if not camera_keys:
        pytest.skip("No camera keys in dataset")

    camera_to_remove = camera_keys[0]

    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(tmp_path / "without_camera")

        dataset_without_camera = remove_feature(
            sample_dataset,
            feature_names=camera_to_remove,
            output_dir=tmp_path / "without_camera",
        )

    assert camera_to_remove not in dataset_without_camera.meta.features
    assert camera_to_remove not in dataset_without_camera.meta.camera_keys

    sample_item = dataset_without_camera[0]
    assert camera_to_remove not in sample_item


def test_complex_workflow_integration(sample_dataset, tmp_path):
    """Test a complex workflow combining multiple operations."""
    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.side_effect = lambda repo_id, **kwargs: str(kwargs.get("local_dir", tmp_path))

        dataset = add_features(
            sample_dataset,
            features={
                "reward": (
                    np.random.randn(50, 1).astype(np.float32),
                    {"dtype": "float32", "shape": (1,), "names": None},
                )
            },
            output_dir=tmp_path / "step1",
        )

        dataset = delete_episodes(
            dataset,
            episode_indices=[2],
            output_dir=tmp_path / "step2",
        )

        splits = split_dataset(
            dataset,
            splits={"train": 0.75, "val": 0.25},
            output_dir=tmp_path / "step3",
        )

        merged = merge_datasets(
            list(splits.values()),
            output_repo_id="final_dataset",
            output_dir=tmp_path / "step4",
        )

    assert merged.meta.total_episodes == 4
    assert merged.meta.total_frames == 40
    assert "reward" in merged.meta.features

    assert len(merged) == 40
    sample_item = merged[0]
    assert "reward" in sample_item


def test_delete_episodes_preserves_stats(sample_dataset, tmp_path):
    """Test that deleting episodes preserves statistics correctly."""
    output_dir = tmp_path / "filtered"

    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(output_dir)

        new_dataset = delete_episodes(
            sample_dataset,
            episode_indices=[2],
            output_dir=output_dir,
        )

    assert new_dataset.meta.stats is not None
    for feature in ["action", "observation.state"]:
        assert feature in new_dataset.meta.stats
        assert "mean" in new_dataset.meta.stats[feature]
        assert "std" in new_dataset.meta.stats[feature]


def test_delete_episodes_preserves_tasks(sample_dataset, tmp_path):
    """Test that tasks are preserved correctly after deletion."""
    output_dir = tmp_path / "filtered"

    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(output_dir)

        new_dataset = delete_episodes(
            sample_dataset,
            episode_indices=[0],
            output_dir=output_dir,
        )

    assert new_dataset.meta.tasks is not None
    assert len(new_dataset.meta.tasks) == 2

    tasks_in_dataset = {str(item["task"]) for item in new_dataset}
    assert len(tasks_in_dataset) > 0


def test_split_three_ways(sample_dataset, tmp_path):
    """Test splitting dataset into three splits."""
    splits = {
        "train": 0.6,
        "val": 0.2,
        "test": 0.2,
    }

    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"

        def mock_snapshot(repo_id, **kwargs):
            for split_name in splits:
                if split_name in repo_id:
                    return str(tmp_path / f"{sample_dataset.repo_id}_{split_name}")
            return str(kwargs.get("local_dir", tmp_path))

        mock_snapshot_download.side_effect = mock_snapshot

        result = split_dataset(
            sample_dataset,
            splits=splits,
            output_dir=tmp_path,
        )

    assert set(result.keys()) == {"train", "val", "test"}
    assert result["train"].meta.total_episodes == 3
    assert result["val"].meta.total_episodes == 1
    assert result["test"].meta.total_episodes == 1

    total_frames = sum(ds.meta.total_frames for ds in result.values())
    assert total_frames == sample_dataset.meta.total_frames


def test_split_preserves_stats(sample_dataset, tmp_path):
    """Test that statistics are preserved when splitting."""
    splits = {"train": [0, 1, 2], "val": [3, 4]}

    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"

        def mock_snapshot(repo_id, **kwargs):
            for split_name in splits:
                if split_name in repo_id:
                    return str(tmp_path / f"{sample_dataset.repo_id}_{split_name}")
            return str(kwargs.get("local_dir", tmp_path))

        mock_snapshot_download.side_effect = mock_snapshot

        result = split_dataset(
            sample_dataset,
            splits=splits,
            output_dir=tmp_path,
        )

    for split_ds in result.values():
        assert split_ds.meta.stats is not None
        for feature in ["action", "observation.state"]:
            assert feature in split_ds.meta.stats
            assert "mean" in split_ds.meta.stats[feature]
            assert "std" in split_ds.meta.stats[feature]


def test_merge_three_datasets(sample_dataset, tmp_path, empty_lerobot_dataset_factory):
    """Test merging three datasets."""
    features = {
        "action": {"dtype": "float32", "shape": (6,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (4,), "names": None},
        "observation.images.top": {"dtype": "image", "shape": (224, 224, 3), "names": None},
    }

    datasets = [sample_dataset]

    for i in range(2):
        dataset = empty_lerobot_dataset_factory(
            root=tmp_path / f"test_dataset{i + 2}",
            features=features,
        )

        for ep_idx in range(2):
            for _ in range(10):
                frame = {
                    "action": np.random.randn(6).astype(np.float32),
                    "observation.state": np.random.randn(4).astype(np.float32),
                    "observation.images.top": np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8),
                    "task": f"task_{ep_idx}",
                }
                dataset.add_frame(frame)
            dataset.save_episode()
        dataset.finalize()

        datasets.append(dataset)

    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(tmp_path / "merged_dataset")

        merged = merge_datasets(
            datasets,
            output_repo_id="merged_dataset",
            output_dir=tmp_path / "merged_dataset",
        )

    assert merged.meta.total_episodes == 9
    assert merged.meta.total_frames == 90


def test_merge_preserves_stats(sample_dataset, tmp_path, empty_lerobot_dataset_factory):
    """Test that statistics are computed for merged datasets."""
    features = {
        "action": {"dtype": "float32", "shape": (6,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (4,), "names": None},
        "observation.images.top": {"dtype": "image", "shape": (224, 224, 3), "names": None},
    }

    dataset2 = empty_lerobot_dataset_factory(
        root=tmp_path / "test_dataset2",
        features=features,
    )

    for ep_idx in range(3):
        for _ in range(10):
            frame = {
                "action": np.random.randn(6).astype(np.float32),
                "observation.state": np.random.randn(4).astype(np.float32),
                "observation.images.top": np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8),
                "task": f"task_{ep_idx % 2}",
            }
            dataset2.add_frame(frame)
        dataset2.save_episode()
    dataset2.finalize()

    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(tmp_path / "merged_dataset")

        merged = merge_datasets(
            [sample_dataset, dataset2],
            output_repo_id="merged_dataset",
            output_dir=tmp_path / "merged_dataset",
        )

    assert merged.meta.stats is not None
    for feature in ["action", "observation.state"]:
        assert feature in merged.meta.stats
        assert "mean" in merged.meta.stats[feature]
        assert "std" in merged.meta.stats[feature]


def test_add_features_preserves_existing_stats(sample_dataset, tmp_path):
    """Test that adding a feature preserves existing stats."""
    num_frames = sample_dataset.meta.total_frames
    reward_values = np.random.randn(num_frames, 1).astype(np.float32)

    feature_info = {
        "dtype": "float32",
        "shape": (1,),
        "names": None,
    }
    features = {
        "reward": (reward_values, feature_info),
    }

    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(tmp_path / "with_reward")

        new_dataset = add_features(
            dataset=sample_dataset,
            features=features,
            output_dir=tmp_path / "with_reward",
        )

    assert new_dataset.meta.stats is not None
    for feature in ["action", "observation.state"]:
        assert feature in new_dataset.meta.stats
        assert "mean" in new_dataset.meta.stats[feature]
        assert "std" in new_dataset.meta.stats[feature]


def test_remove_feature_updates_stats(sample_dataset, tmp_path):
    """Test that removing a feature removes it from stats."""
    feature_info = {"dtype": "float32", "shape": (1,), "names": None}

    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.side_effect = lambda repo_id, **kwargs: str(kwargs.get("local_dir", tmp_path))

        dataset_with_reward = add_features(
            sample_dataset,
            features={
                "reward": (np.random.randn(50, 1).astype(np.float32), feature_info),
            },
            output_dir=tmp_path / "with_reward",
        )

        dataset_without_reward = remove_feature(
            dataset_with_reward,
            feature_names="reward",
            output_dir=tmp_path / "without_reward",
        )

    if dataset_without_reward.meta.stats:
        assert "reward" not in dataset_without_reward.meta.stats


def test_delete_consecutive_episodes(sample_dataset, tmp_path):
    """Test deleting consecutive episodes."""
    output_dir = tmp_path / "filtered"

    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(output_dir)

        new_dataset = delete_episodes(
            sample_dataset,
            episode_indices=[1, 2, 3],
            output_dir=output_dir,
        )

    assert new_dataset.meta.total_episodes == 2
    assert new_dataset.meta.total_frames == 20

    episode_indices = sorted({int(idx.item()) for idx in new_dataset.hf_dataset["episode_index"]})
    assert episode_indices == [0, 1]


def test_delete_first_and_last_episodes(sample_dataset, tmp_path):
    """Test deleting first and last episodes."""
    output_dir = tmp_path / "filtered"

    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(output_dir)

        new_dataset = delete_episodes(
            sample_dataset,
            episode_indices=[0, 4],
            output_dir=output_dir,
        )

    assert new_dataset.meta.total_episodes == 3
    assert new_dataset.meta.total_frames == 30

    episode_indices = sorted({int(idx.item()) for idx in new_dataset.hf_dataset["episode_index"]})
    assert episode_indices == [0, 1, 2]


def test_split_all_episodes_assigned(sample_dataset, tmp_path):
    """Test that all episodes can be explicitly assigned to splits."""
    splits = {
        "split1": [0, 1],
        "split2": [2, 3],
        "split3": [4],
    }

    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"

        def mock_snapshot(repo_id, **kwargs):
            for split_name in splits:
                if split_name in repo_id:
                    return str(tmp_path / f"{sample_dataset.repo_id}_{split_name}")
            return str(kwargs.get("local_dir", tmp_path))

        mock_snapshot_download.side_effect = mock_snapshot

        result = split_dataset(
            sample_dataset,
            splits=splits,
            output_dir=tmp_path,
        )

    total_episodes = sum(ds.meta.total_episodes for ds in result.values())
    assert total_episodes == sample_dataset.meta.total_episodes


def test_modify_features_preserves_file_structure(sample_dataset, tmp_path):
    """Test that modifying features preserves chunk_idx and file_idx from source dataset."""
    feature_info = {"dtype": "float32", "shape": (1,), "names": None}

    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"

        def mock_snapshot(repo_id, **kwargs):
            return str(kwargs.get("local_dir", tmp_path / repo_id.split("/")[-1]))

        mock_snapshot_download.side_effect = mock_snapshot

        # First split the dataset to create a non-zero starting chunk/file structure
        splits = split_dataset(
            sample_dataset,
            splits={"train": [0, 1, 2], "val": [3, 4]},
            output_dir=tmp_path / "splits",
        )

        train_dataset = splits["train"]

        # Get original chunk/file indices from first episode
        if train_dataset.meta.episodes is None:
            from lerobot.datasets.utils import load_episodes

            train_dataset.meta.episodes = load_episodes(train_dataset.meta.root)
        original_chunk_indices = [ep["data/chunk_index"] for ep in train_dataset.meta.episodes]
        original_file_indices = [ep["data/file_index"] for ep in train_dataset.meta.episodes]

        # Now add a feature to the split dataset
        modified_dataset = add_features(
            train_dataset,
            features={
                "reward": (
                    np.random.randn(train_dataset.meta.total_frames, 1).astype(np.float32),
                    feature_info,
                ),
            },
            output_dir=tmp_path / "modified",
        )

        # Check that chunk/file indices are preserved
        if modified_dataset.meta.episodes is None:
            from lerobot.datasets.utils import load_episodes

            modified_dataset.meta.episodes = load_episodes(modified_dataset.meta.root)
        new_chunk_indices = [ep["data/chunk_index"] for ep in modified_dataset.meta.episodes]
        new_file_indices = [ep["data/file_index"] for ep in modified_dataset.meta.episodes]

        assert new_chunk_indices == original_chunk_indices, "Chunk indices should be preserved"
        assert new_file_indices == original_file_indices, "File indices should be preserved"
        assert "reward" in modified_dataset.meta.features


def test_convert_image_to_video_dataset(tmp_path):
    """Test converting lerobot/pusht_image dataset to video format."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Load the actual lerobot/pusht_image dataset (only first 2 episodes for speed)
    source_dataset = LeRobotDataset("lerobot/pusht_image", episodes=[0, 1])

    output_dir = tmp_path / "pusht_video"

    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(output_dir)

        # Verify source dataset has images, not videos
        assert len(source_dataset.meta.video_keys) == 0
        assert "observation.image" in source_dataset.meta.features

        # Convert to video dataset (only first 2 episodes for speed)
        video_dataset = convert_image_to_video_dataset(
            dataset=source_dataset,
            output_dir=output_dir,
            repo_id="lerobot/pusht_video",
            vcodec="libsvtav1",
            pix_fmt="yuv420p",
            g=2,
            crf=30,
            episode_indices=[0, 1],
            num_workers=2,
        )

        # Verify new dataset has videos
        assert len(video_dataset.meta.video_keys) > 0
        assert "observation.image" in video_dataset.meta.video_keys

        # Verify correct number of episodes and frames (2 episodes)
        assert video_dataset.meta.total_episodes == 2
        # Compare against the actual number of frames in the loaded episodes, not metadata total
        assert len(video_dataset) == len(source_dataset)

        # Verify video files exist
        for ep_idx in range(video_dataset.meta.total_episodes):
            for video_key in video_dataset.meta.video_keys:
                video_path = video_dataset.root / video_dataset.meta.get_video_file_path(ep_idx, video_key)
                assert video_path.exists(), f"Video file should exist: {video_path}"

        # Verify we can load the dataset and access it
        assert len(video_dataset) == video_dataset.meta.total_frames

        # Test that we can actually get an item from the video dataset
        item = video_dataset[0]
        assert "observation.image" in item
        assert "action" in item

        # Cleanup
        import shutil

        if output_dir.exists():
            shutil.rmtree(output_dir)


def test_convert_image_to_video_dataset_subset_episodes(tmp_path):
    """Test converting only specific episodes from lerobot/pusht_image to video format."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Load the actual lerobot/pusht_image dataset (only first 3 episodes)
    source_dataset = LeRobotDataset("lerobot/pusht_image", episodes=[0, 1, 2])

    output_dir = tmp_path / "pusht_video_subset"

    with (
        patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
        patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
    ):
        mock_get_safe_version.return_value = "v3.0"
        mock_snapshot_download.return_value = str(output_dir)

        # Convert only episode 0 to video (subset of loaded episodes)
        episode_indices = [0]

        video_dataset = convert_image_to_video_dataset(
            dataset=source_dataset,
            output_dir=output_dir,
            repo_id="lerobot/pusht_video_subset",
            episode_indices=episode_indices,
            num_workers=2,
        )

        # Verify correct number of episodes
        assert video_dataset.meta.total_episodes == len(episode_indices)

        # Verify video files exist for selected episodes
        assert len(video_dataset.meta.video_keys) > 0
        assert "observation.image" in video_dataset.meta.video_keys

        # Cleanup
        import shutil

        if output_dir.exists():
            shutil.rmtree(output_dir)
