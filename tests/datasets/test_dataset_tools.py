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
    add_feature,
    delete_episodes,
    merge_datasets,
    remove_feature,
    split_dataset,
)


@pytest.fixture
def sample_dataset(tmp_path, empty_lerobot_dataset_factory):
    """Create a sample dataset for testing."""
    # Create an empty dataset and add data manually
    features = {
        "action": {"dtype": "float32", "shape": (6,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (4,), "names": None},
        "observation.images.top": {"dtype": "image", "shape": (224, 224, 3), "names": None},
    }

    dataset = empty_lerobot_dataset_factory(
        root=tmp_path / "test_dataset",
        features=features,
    )

    # Add episodes manually
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

    return dataset


class TestDeleteEpisodes:
    def test_delete_single_episode(self, sample_dataset, tmp_path):
        """Test deleting a single episode."""
        output_dir = tmp_path / "filtered"

        # Delete episode 2
        # Mock the revision check and snapshot_download to prevent Hub calls
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

        # Check results
        assert new_dataset.meta.total_episodes == 4
        assert new_dataset.meta.total_frames == 40

        # Check episode indices are renumbered
        episode_indices = {int(idx.item()) for idx in new_dataset.hf_dataset["episode_index"]}
        assert episode_indices == {0, 1, 2, 3}

        # Check data integrity
        assert len(new_dataset) == 40

    def test_delete_multiple_episodes(self, sample_dataset, tmp_path):
        """Test deleting multiple episodes."""
        output_dir = tmp_path / "filtered"

        # Delete episodes 1 and 3
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

        # Check results
        assert new_dataset.meta.total_episodes == 3
        assert new_dataset.meta.total_frames == 30

        # Check episode indices
        episode_indices = {int(idx.item()) for idx in new_dataset.hf_dataset["episode_index"]}
        assert episode_indices == {0, 1, 2}

    def test_delete_invalid_episodes(self, sample_dataset, tmp_path):
        """Test error handling for invalid episode indices."""
        with pytest.raises(ValueError, match="Invalid episode indices"):
            delete_episodes(
                sample_dataset,
                episode_indices=[10, 20],  # Out of range
                output_dir=tmp_path / "filtered",
            )

    def test_delete_all_episodes(self, sample_dataset, tmp_path):
        """Test error when trying to delete all episodes."""
        with pytest.raises(ValueError, match="Cannot delete all episodes"):
            delete_episodes(
                sample_dataset,
                episode_indices=list(range(5)),  # All episodes
                output_dir=tmp_path / "filtered",
            )

    def test_delete_empty_list(self, sample_dataset, tmp_path):
        """Test error when no episodes specified."""
        with pytest.raises(ValueError, match="No episodes to delete"):
            delete_episodes(
                sample_dataset,
                episode_indices=[],
                output_dir=tmp_path / "filtered",
            )


class TestSplitDataset:
    def test_split_by_episodes(self, sample_dataset, tmp_path):
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

            # Mock snapshot_download to return the appropriate directory for each split
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

        # Check we got both splits
        assert set(result.keys()) == {"train", "val"}

        # Check train split
        assert result["train"].meta.total_episodes == 3
        assert result["train"].meta.total_frames == 30

        # Check val split
        assert result["val"].meta.total_episodes == 2
        assert result["val"].meta.total_frames == 20

        # Check episode renumbering
        train_episodes = {int(idx.item()) for idx in result["train"].hf_dataset["episode_index"]}
        assert train_episodes == {0, 1, 2}

        val_episodes = {int(idx.item()) for idx in result["val"].hf_dataset["episode_index"]}
        assert val_episodes == {0, 1}

    def test_split_by_fractions(self, sample_dataset, tmp_path):
        """Test splitting dataset by fractions."""
        splits = {
            "train": 0.6,  # 3 episodes
            "val": 0.4,  # 2 episodes
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

        # Check splits
        assert result["train"].meta.total_episodes == 3
        assert result["val"].meta.total_episodes == 2

    def test_split_overlapping_episodes(self, sample_dataset, tmp_path):
        """Test error when episodes appear in multiple splits."""
        splits = {
            "train": [0, 1, 2],
            "val": [2, 3, 4],  # Episode 2 appears in both
        }

        with pytest.raises(ValueError, match="Episodes cannot appear in multiple splits"):
            split_dataset(sample_dataset, splits=splits, output_dir=tmp_path)

    def test_split_invalid_fractions(self, sample_dataset, tmp_path):
        """Test error when fractions sum to more than 1."""
        splits = {
            "train": 0.7,
            "val": 0.5,  # Sum = 1.2
        }

        with pytest.raises(ValueError, match="Split fractions must sum to <= 1.0"):
            split_dataset(sample_dataset, splits=splits, output_dir=tmp_path)

    def test_split_empty(self, sample_dataset, tmp_path):
        """Test error with empty splits."""
        with pytest.raises(ValueError, match="No splits provided"):
            split_dataset(sample_dataset, splits={}, output_dir=tmp_path)


class TestMergeDatasets:
    def test_merge_two_datasets(self, sample_dataset, tmp_path, empty_lerobot_dataset_factory):
        """Test merging two datasets."""
        # Create a second dataset manually
        features = {
            "action": {"dtype": "float32", "shape": (6,), "names": None},
            "observation.state": {"dtype": "float32", "shape": (4,), "names": None},
            "observation.images.top": {"dtype": "image", "shape": (224, 224, 3), "names": None},
        }

        dataset2 = empty_lerobot_dataset_factory(
            root=tmp_path / "test_dataset2",
            features=features,
        )

        # Add 3 episodes
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

        # Merge datasets
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

        # Check results
        assert merged.meta.total_episodes == 8  # 5 + 3
        assert merged.meta.total_frames == 80  # 50 + 30

        # Check episode indices are sequential
        episode_indices = sorted({int(idx.item()) for idx in merged.hf_dataset["episode_index"]})
        assert episode_indices == list(range(8))

    def test_merge_empty_list(self, tmp_path):
        """Test error when merging empty list."""
        with pytest.raises(ValueError, match="No datasets to merge"):
            merge_datasets([], output_repo_id="merged", output_dir=tmp_path)


class TestAddFeature:
    def test_add_feature_with_values(self, sample_dataset, tmp_path):
        """Test adding a feature with pre-computed values."""
        # Create reward values for all frames
        num_frames = sample_dataset.meta.total_frames
        reward_values = np.random.randn(num_frames, 1).astype(np.float32)

        feature_info = {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        }

        with (
            patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
            patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
        ):
            mock_get_safe_version.return_value = "v3.0"
            mock_snapshot_download.return_value = str(tmp_path / "with_reward")

            new_dataset = add_feature(
                sample_dataset,
                feature_name="reward",
                feature_values=reward_values,
                feature_info=feature_info,
                output_dir=tmp_path / "with_reward",
            )

        # Check feature was added
        assert "reward" in new_dataset.meta.features
        assert new_dataset.meta.features["reward"] == feature_info

        # Check values
        assert len(new_dataset) == num_frames
        sample_item = new_dataset[0]
        assert "reward" in sample_item
        # Scalar features don't have shape, just check it's a tensor
        assert isinstance(sample_item["reward"], torch.Tensor)

    def test_add_feature_with_callable(self, sample_dataset, tmp_path):
        """Test adding a feature with a callable."""

        def compute_reward(frame_dict, episode_idx, frame_idx):
            # Simple reward based on episode and frame indices
            return float(episode_idx * 10 + frame_idx)

        feature_info = {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        }

        with (
            patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
            patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
        ):
            mock_get_safe_version.return_value = "v3.0"
            mock_snapshot_download.return_value = str(tmp_path / "with_reward")

            new_dataset = add_feature(
                sample_dataset,
                feature_name="reward",
                feature_values=compute_reward,
                feature_info=feature_info,
                output_dir=tmp_path / "with_reward",
            )

        # Check feature was added
        assert "reward" in new_dataset.meta.features

        # Check computed values
        # Episode 0, frame 0 should have reward 0
        items = [new_dataset[i] for i in range(10)]
        first_episode_items = [item for item in items if item["episode_index"] == 0]
        assert len(first_episode_items) == 10

        # Check first frame of first episode
        first_frame = first_episode_items[0]
        assert first_frame["frame_index"] == 0
        assert float(first_frame["reward"]) == 0.0

    def test_add_existing_feature(self, sample_dataset, tmp_path):
        """Test error when adding an existing feature."""
        feature_info = {"dtype": "float32", "shape": (1,)}

        with pytest.raises(ValueError, match="Feature 'action' already exists"):
            add_feature(
                sample_dataset,
                feature_name="action",  # Already exists
                feature_values=np.zeros(50),
                feature_info=feature_info,
                output_dir=tmp_path / "modified",
            )

    def test_add_feature_invalid_info(self, sample_dataset, tmp_path):
        """Test error with invalid feature info."""
        with pytest.raises(ValueError, match="feature_info must contain keys"):
            add_feature(
                sample_dataset,
                feature_name="reward",
                feature_values=np.zeros(50),
                feature_info={"dtype": "float32"},  # Missing 'shape'
                output_dir=tmp_path / "modified",
            )


class TestRemoveFeature:
    def test_remove_single_feature(self, sample_dataset, tmp_path):
        """Test removing a single feature."""
        # First add a feature to remove
        feature_info = {"dtype": "float32", "shape": (1,), "names": None}

        with (
            patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
            patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
        ):
            mock_get_safe_version.return_value = "v3.0"
            mock_snapshot_download.side_effect = lambda repo_id, **kwargs: str(
                kwargs.get("local_dir", tmp_path)
            )

            dataset_with_reward = add_feature(
                sample_dataset,
                feature_name="reward",
                feature_values=np.random.randn(50, 1).astype(np.float32),
                feature_info=feature_info,
                output_dir=tmp_path / "with_reward",
            )

            # Now remove it
            dataset_without_reward = remove_feature(
                dataset_with_reward,
                feature_names="reward",
                output_dir=tmp_path / "without_reward",
            )

        # Check feature was removed
        assert "reward" not in dataset_without_reward.meta.features

        # Check data
        sample_item = dataset_without_reward[0]
        assert "reward" not in sample_item

    def test_remove_multiple_features(self, sample_dataset, tmp_path):
        """Test removing multiple features at once."""
        with (
            patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
            patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
        ):
            mock_get_safe_version.return_value = "v3.0"
            mock_snapshot_download.side_effect = lambda repo_id, **kwargs: str(
                kwargs.get("local_dir", tmp_path)
            )

            # Add two features
            dataset = sample_dataset
            for feature_name in ["reward", "success"]:
                feature_info = {"dtype": "float32", "shape": (1,), "names": None}
                dataset = add_feature(
                    dataset,
                    feature_name=feature_name,
                    feature_values=np.random.randn(dataset.meta.total_frames, 1).astype(np.float32),
                    feature_info=feature_info,
                    output_dir=tmp_path / f"with_{feature_name}",
                )

            # Remove both
            dataset_clean = remove_feature(
                dataset,
                feature_names=["reward", "success"],
                output_dir=tmp_path / "clean",
            )

        # Check both were removed
        assert "reward" not in dataset_clean.meta.features
        assert "success" not in dataset_clean.meta.features

    def test_remove_nonexistent_feature(self, sample_dataset, tmp_path):
        """Test error when removing non-existent feature."""
        with pytest.raises(ValueError, match="Feature 'nonexistent' not found"):
            remove_feature(
                sample_dataset,
                feature_names="nonexistent",
                output_dir=tmp_path / "modified",
            )

    def test_remove_required_feature(self, sample_dataset, tmp_path):
        """Test error when trying to remove required features."""
        with pytest.raises(ValueError, match="Cannot remove required features"):
            remove_feature(
                sample_dataset,
                feature_names="timestamp",  # Required feature
                output_dir=tmp_path / "modified",
            )

    def test_remove_camera_feature(self, sample_dataset, tmp_path):
        """Test removing a camera feature."""
        camera_keys = sample_dataset.meta.camera_keys
        if not camera_keys:
            pytest.skip("No camera keys in dataset")

        # Remove first camera
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

        # Check camera was removed
        assert camera_to_remove not in dataset_without_camera.meta.features
        assert camera_to_remove not in dataset_without_camera.meta.camera_keys

        # Check data
        sample_item = dataset_without_camera[0]
        assert camera_to_remove not in sample_item


class TestIntegration:
    def test_complex_workflow(self, sample_dataset, tmp_path):
        """Test a complex workflow combining multiple operations."""
        with (
            patch("lerobot.datasets.lerobot_dataset.get_safe_version") as mock_get_safe_version,
            patch("lerobot.datasets.lerobot_dataset.snapshot_download") as mock_snapshot_download,
        ):
            mock_get_safe_version.return_value = "v3.0"
            mock_snapshot_download.side_effect = lambda repo_id, **kwargs: str(
                kwargs.get("local_dir", tmp_path)
            )

            # 1. Add a reward feature
            dataset = add_feature(
                sample_dataset,
                feature_name="reward",
                feature_values=np.random.randn(50, 1).astype(np.float32),
                feature_info={"dtype": "float32", "shape": (1,), "names": None},
                output_dir=tmp_path / "step1",
            )

            # 2. Delete an episode
            dataset = delete_episodes(
                dataset,
                episode_indices=[2],
                output_dir=tmp_path / "step2",
            )

            # 3. Split into train/val
            splits = split_dataset(
                dataset,
                splits={"train": 0.75, "val": 0.25},
                output_dir=tmp_path / "step3",
            )

            # 4. Merge them back
            merged = merge_datasets(
                list(splits.values()),
                output_repo_id="final_dataset",
                output_dir=tmp_path / "step4",
            )

        # Check final dataset
        assert merged.meta.total_episodes == 4  # Started with 5, deleted 1
        assert merged.meta.total_frames == 40
        assert "reward" in merged.meta.features  # Feature preserved

        # Check data integrity
        assert len(merged) == 40
        sample_item = merged[0]
        assert "reward" in sample_item
