#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""
Tests for subtask functionality in LeRobotDataset.

These tests verify that:
- Subtask information is correctly loaded from datasets that have subtask data
- The __getitem__ method correctly adds subtask strings to returned items
- Subtask handling gracefully handles missing data
"""

import pandas as pd
import pytest
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


class TestSubtaskDataset:
    """Tests for subtask handling in LeRobotDataset."""

    @pytest.fixture
    def subtask_dataset(self):
        """Load the test subtask dataset from the hub."""
        # Use lerobot/pusht-subtask dataset with episode 1
        return LeRobotDataset(
            repo_id="lerobot/pusht-subtask",
            episodes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        )

    def test_subtask_dataset_loads(self, subtask_dataset):
        """Test that the subtask dataset loads successfully."""
        assert subtask_dataset is not None
        assert len(subtask_dataset) > 0

    def test_subtask_metadata_loaded(self, subtask_dataset):
        """Test that subtask metadata is loaded when present in dataset."""
        # The dataset should have subtasks metadata loaded
        assert subtask_dataset.meta.subtasks is not None
        assert isinstance(subtask_dataset.meta.subtasks, pd.DataFrame)

    def test_subtask_index_in_features(self, subtask_dataset):
        """Test that subtask_index is a feature when dataset has subtasks."""
        assert "subtask_index" in subtask_dataset.features

    def test_getitem_returns_subtask_string(self, subtask_dataset):
        """Test that __getitem__ correctly adds subtask string to returned item."""
        item = subtask_dataset[0]

        # Subtask should be present in the returned item
        assert "subtask" in item
        assert isinstance(item["subtask"], str)
        assert len(item["subtask"]) > 0  # Should not be empty

    def test_getitem_has_subtask_index(self, subtask_dataset):
        """Test that __getitem__ includes subtask_index."""
        item = subtask_dataset[0]

        assert "subtask_index" in item
        assert isinstance(item["subtask_index"], torch.Tensor)

    def test_subtask_index_maps_to_valid_subtask(self, subtask_dataset):
        """Test that subtask_index correctly maps to a subtask in metadata."""
        item = subtask_dataset[0]

        subtask_idx = item["subtask_index"].item()
        subtask_from_metadata = subtask_dataset.meta.subtasks.iloc[subtask_idx].name

        assert item["subtask"] == subtask_from_metadata

    def test_all_items_have_subtask(self, subtask_dataset):
        """Test that all items in the dataset have subtask information."""
        for i in range(min(len(subtask_dataset), 5)):  # Check first 5 items
            item = subtask_dataset[i]
            assert "subtask" in item
            assert isinstance(item["subtask"], str)

    def test_task_and_subtask_coexist(self, subtask_dataset):
        """Test that both task and subtask are present in returned items."""
        item = subtask_dataset[0]

        # Both task and subtask should be present
        assert "task" in item
        assert "subtask" in item
        assert isinstance(item["task"], str)
        assert isinstance(item["subtask"], str)


class TestSubtaskDatasetMissing:
    """Tests for graceful handling when subtask data is missing."""

    @pytest.fixture
    def dataset_without_subtasks(self, tmp_path, empty_lerobot_dataset_factory):
        """Create a dataset without subtask information."""
        features = {"state": {"dtype": "float32", "shape": (2,), "names": None}}
        dataset = empty_lerobot_dataset_factory(root=tmp_path / "no_subtask", features=features)

        # Add some frames and save
        for _ in range(5):
            dataset.add_frame({"state": torch.randn(2), "task": "Test task"})
        dataset.save_episode()
        dataset.finalize()

        # Reload the dataset
        return LeRobotDataset(dataset.repo_id, root=dataset.root)

    def test_no_subtask_in_features(self, dataset_without_subtasks):
        """Test that subtask_index is not in features when not provided."""
        assert "subtask_index" not in dataset_without_subtasks.features

    def test_getitem_without_subtask(self, dataset_without_subtasks):
        """Test that __getitem__ works when subtask is not present."""
        item = dataset_without_subtasks[0]

        # Item should still be retrievable
        assert item is not None
        assert "state" in item
        assert "task" in item

        # Subtask should NOT be present
        assert "subtask" not in item

    def test_subtasks_metadata_is_none(self, dataset_without_subtasks):
        """Test that subtasks metadata is None when not present."""
        assert dataset_without_subtasks.meta.subtasks is None


class TestSubtaskEdgeCases:
    """Edge case tests for subtask handling."""

    def test_subtask_with_multiple_episodes(self):
        """Test subtask handling with multiple episodes if available."""
        try:
            dataset = LeRobotDataset(
                repo_id="lerobot/pusht-subtask",
                episodes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            )
        except Exception:
            pytest.skip("Could not load test-subtask dataset")

        # Check first and last items have valid subtasks
        first_item = dataset[0]
        last_item = dataset[len(dataset) - 1]

        assert "subtask" in first_item
        assert "subtask" in last_item
        assert isinstance(first_item["subtask"], str)
        assert isinstance(last_item["subtask"], str)

    def test_subtask_index_consistency(self):
        """Test that same subtask_index returns same subtask string."""
        try:
            dataset = LeRobotDataset(
                repo_id="lerobot/pusht-subtask",
                episodes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            )
        except Exception:
            pytest.skip("Could not load test-subtask dataset")

        if len(dataset) < 2:
            pytest.skip("Dataset too small for this test")

        # Collect subtask_index to subtask mappings
        subtask_map = {}
        for i in range(min(len(dataset), 10)):
            item = dataset[i]
            idx = item["subtask_index"].item()
            subtask = item["subtask"]

            if idx in subtask_map:
                # Same index should always return same subtask
                assert subtask_map[idx] == subtask, (
                    f"Inconsistent subtask for index {idx}: '{subtask_map[idx]}' vs '{subtask}'"
                )
            else:
                subtask_map[idx] = subtask
