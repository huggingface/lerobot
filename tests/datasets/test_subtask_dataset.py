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

import numpy as np
import pandas as pd
import pytest
import torch

from lerobot.data_processing.data_annotations.subtask_annotations import EpisodeSkills, Skill
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import (
    create_subtask_index_array,
    create_subtasks_dataframe,
    save_subtasks,
)


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


class TestCreateSubtasksDataframe:
    """Tests for create_subtasks_dataframe in utils."""

    def test_empty_annotations(self):
        """Empty annotations produce empty DataFrame and empty mapping."""
        subtasks_df, skill_to_subtask_idx = create_subtasks_dataframe({})
        assert len(subtasks_df) == 0
        assert list(subtasks_df.columns) == ["subtask_index"]
        assert skill_to_subtask_idx == {}

    def test_single_episode_single_skill(self):
        """Single episode with one skill produces one row and correct mapping."""
        annotations = {
            0: EpisodeSkills(
                episode_index=0,
                description="Pick",
                skills=[Skill("pick", 0.0, 1.0)],
            ),
        }
        subtasks_df, skill_to_subtask_idx = create_subtasks_dataframe(annotations)
        assert len(subtasks_df) == 1
        assert subtasks_df.index.tolist() == ["pick"]
        assert subtasks_df.loc["pick", "subtask_index"] == 0
        assert skill_to_subtask_idx == {"pick": 0}

    def test_multiple_episodes_overlapping_skills(self):
        """Multiple episodes with overlapping skill names yield unique sorted skills."""
        annotations = {
            0: EpisodeSkills(
                episode_index=0,
                description="Ep0",
                skills=[
                    Skill("place", 0.0, 0.5),
                    Skill("pick", 0.5, 1.0),
                ],
            ),
            1: EpisodeSkills(
                episode_index=1,
                description="Ep1",
                skills=[Skill("pick", 0.0, 1.0)],
            ),
        }
        subtasks_df, skill_to_subtask_idx = create_subtasks_dataframe(annotations)
        # Sorted order: pick, place
        assert subtasks_df.index.tolist() == ["pick", "place"]
        assert int(subtasks_df.loc["pick", "subtask_index"]) == 0
        assert int(subtasks_df.loc["place", "subtask_index"]) == 1
        assert skill_to_subtask_idx["pick"] == 0
        assert skill_to_subtask_idx["place"] == 1

    def test_skills_sorted_alphabetically(self):
        """Subtask rows are in alphabetical order by skill name."""
        annotations = {
            0: EpisodeSkills(
                episode_index=0,
                description="Ep",
                skills=[
                    Skill("z_final", 0.0, 0.33),
                    Skill("a_first", 0.33, 0.66),
                    Skill("m_mid", 0.66, 1.0),
                ],
            ),
        }
        subtasks_df, _ = create_subtasks_dataframe(annotations)
        assert subtasks_df.index.tolist() == ["a_first", "m_mid", "z_final"]
        assert list(subtasks_df["subtask_index"]) == [0, 1, 2]


class TestSaveSubtasks:
    """Tests for save_subtasks in utils."""

    def test_save_subtasks_creates_file(self, tmp_path):
        """save_subtasks writes meta/subtasks.parquet and creates parent dir."""
        subtasks_df = pd.DataFrame(
            [{"subtask": "pick", "subtask_index": 0}, {"subtask": "place", "subtask_index": 1}]
        ).set_index("subtask")
        save_subtasks(subtasks_df, tmp_path)
        out = tmp_path / "meta" / "subtasks.parquet"
        assert out.exists()
        read_df = pd.read_parquet(out)
        pd.testing.assert_frame_equal(read_df.reset_index(), subtasks_df.reset_index())

    def test_save_subtasks_content_matches(self, tmp_path):
        """Saved parquet round-trips with same content."""
        subtasks_df = pd.DataFrame(
            [{"subtask": "a", "subtask_index": 0}, {"subtask": "b", "subtask_index": 1}]
        ).set_index("subtask")
        save_subtasks(subtasks_df, tmp_path)
        read_df = pd.read_parquet(tmp_path / "meta" / "subtasks.parquet")
        assert read_df.index.tolist() == subtasks_df.index.tolist()
        assert list(read_df["subtask_index"]) == list(subtasks_df["subtask_index"])


class TestCreateSubtaskIndexArray:
    """Tests for create_subtask_index_array in utils."""

    @pytest.fixture
    def dataset_with_episodes(self, tmp_path, empty_lerobot_dataset_factory):
        """Dataset with two episodes (10 frames each) for index-array tests."""
        features = {"state": {"dtype": "float32", "shape": (2,), "names": None}}
        dataset = empty_lerobot_dataset_factory(root=tmp_path / "subtask_idx", features=features)
        for _ in range(10):
            dataset.add_frame({"state": torch.randn(2), "task": "Task A"})
        dataset.save_episode()
        for _ in range(10):
            dataset.add_frame({"state": torch.randn(2), "task": "Task B"})
        dataset.save_episode()
        dataset.finalize()
        return LeRobotDataset(dataset.repo_id, root=dataset.root)

    def test_unannotated_all_minus_one(self, dataset_with_episodes):
        """With no annotations, all frame indices are -1."""
        skill_to_subtask_idx = {"pick": 0, "place": 1}
        arr = create_subtask_index_array(dataset_with_episodes, {}, skill_to_subtask_idx)
        assert len(arr) == len(dataset_with_episodes)
        assert arr.dtype == np.int64
        assert np.all(arr == -1)

    def test_annotated_episode_assigns_by_timestamp(self, dataset_with_episodes):
        """Frames in an annotated episode get subtask index from skill time ranges."""
        # Dataset uses DEFAULT_FPS=30. Episode 0: 10 frames -> timestamps 0, 1/30, ..., 9/30 (~0.3s).
        # Skills: "pick" [0, 0.2), "place" [0.2, 0.5). At 30 fps: 0.2s = 6 frames, so frames 0-5 = pick, 6-9 = place.
        annotations = {
            0: EpisodeSkills(
                episode_index=0,
                description="Pick and place",
                skills=[
                    Skill("pick", 0.0, 0.2),  # frames 0-5 at 30 fps
                    Skill("place", 0.2, 0.5),  # frames 6-9 at 30 fps
                ],
            ),
        }
        skill_to_subtask_idx = {"pick": 0, "place": 1}
        arr = create_subtask_index_array(dataset_with_episodes, annotations, skill_to_subtask_idx)
        assert len(arr) == 20
        # Episode 0: from_index=0, to_index=10 at 30 fps
        for i in range(6):
            assert arr[i] == 0, f"frame {i} should be pick"
        for i in range(6, 10):
            assert arr[i] == 1, f"frame {i} should be place"
        # Episode 1 not annotated
        for i in range(10, 20):
            assert arr[i] == -1

    def test_partial_annotations_leave_others_minus_one(self, dataset_with_episodes):
        """Only annotated episodes get non -1 indices; others stay -1."""
        annotations = {
            1: EpisodeSkills(
                episode_index=1,
                description="Place only",
                skills=[Skill("place", 0.0, 1.0)],
            ),
        }
        skill_to_subtask_idx = {"place": 0}
        arr = create_subtask_index_array(dataset_with_episodes, annotations, skill_to_subtask_idx)
        for i in range(10):
            assert arr[i] == -1
        for i in range(10, 20):
            assert arr[i] == 0
