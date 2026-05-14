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
"""Tests for lerobot-edit-dataset CLI functionality."""

from unittest.mock import patch

import numpy as np
import pytest

from lerobot.scripts.lerobot_edit_dataset import (
    AddFeatureConfig,
    EditDatasetConfig,
    handle_add_feature,
)


class TestAddFeatureCLI:
    """Test the handle_add_feature CLI functionality."""

    def test_add_feature_from_safetensors(self, tmp_path):
        """Test adding a feature from a safetensors file."""
        # Create a mock safetensors file
        import safetensors.numpy

        data = np.random.randn(50, 2).astype(np.float32)
        safetensors.numpy.save_file({"reward": data}, tmp_path / "data.safetensors")

        # Mock the dataset and other dependencies
        with (
            patch("lerobot.scripts.lerobot_edit_dataset.LeRobotDataset") as mock_dataset_class,
            patch("lerobot.scripts.lerobot_edit_dataset.add_features") as mock_add_features,
        ):
            mock_dataset = mock_dataset_class.return_value
            mock_dataset.meta.total_frames = 50
            mock_dataset.meta.episodes = [
                {"dataset_to_index": 10},
                {"dataset_to_index": 20},
                {"dataset_to_index": 30},
                {"dataset_to_index": 40},
                {"dataset_to_index": 50},
            ]
            mock_dataset.fps = 30.0

            # Call the function
            cfg = EditDatasetConfig(
                repo_id="test/repo",
                new_repo_id="test/repo_modified",
                root=str(tmp_path / "output"),
                push_to_hub=False,
                operation=AddFeatureConfig(
                    feature_names=["reward"],
                    feature_paths=[str(tmp_path / "data.safetensors")],
                ),
            )
            handle_add_feature(cfg)

            # Verify add_features was called correctly
            mock_add_features.assert_called_once()
            args, kwargs = mock_add_features.call_args
            assert "features" in kwargs
            assert "reward" in kwargs["features"]  # Single key with single name -> just the name
            assert kwargs["features"]["reward"][0].shape == (50, 2)

    def test_add_feature_from_mp4(self, tmp_path):
        """Test adding a video feature from an MP4 file."""
        # Create a mock MP4 file
        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"mock video data")

        with (
            patch("lerobot.scripts.lerobot_edit_dataset.LeRobotDataset") as mock_dataset_class,
            patch("lerobot.scripts.lerobot_edit_dataset.add_features") as mock_add_features,
            patch("lerobot.scripts.lerobot_edit_dataset.get_video_duration_in_s") as mock_get_duration,
            patch("lerobot.scripts.lerobot_edit_dataset.get_video_info") as mock_get_info,
            patch("lerobot.scripts.lerobot_edit_dataset.is_video") as mock_is_video,
        ):
            mock_is_video.return_value = True

            mock_dataset = mock_dataset_class.return_value
            mock_dataset.meta.total_frames = 50
            mock_dataset.meta.episodes = [
                {"dataset_to_index": 10},
                {"dataset_to_index": 20},
                {"dataset_to_index": 30},
                {"dataset_to_index": 40},
                {"dataset_to_index": 50},
            ]
            mock_dataset.fps = 30.0
            mock_get_duration.return_value = 50 / 30
            mock_get_info.return_value = {
                "video.height": 480,
                "video.width": 640,
                "video.channels": 3,
                "video.fps": 30.0,
            }

            handle_add_feature(
                EditDatasetConfig(
                    repo_id="test/repo",
                    new_repo_id="test/repo_modified",
                    root=str(tmp_path / "output"),
                    push_to_hub=False,
                    operation=AddFeatureConfig(
                        feature_names=["observation.camera"],
                        feature_paths=[str(video_path)],
                    ),
                )
            )

            # Verify add_features was called
            mock_add_features.assert_called_once()
            # Verify is_video was actually checked
            mock_is_video.assert_called_with(str(video_path))

    def test_add_multiple_features(self, tmp_path):
        """Test adding multiple features at once."""
        # Create mock safetensors files
        import safetensors.numpy

        data1 = np.random.randn(50, 1).astype(np.float32)
        data2 = np.random.randn(50, 3).astype(np.float32)
        safetensors.numpy.save_file({"reward": data1}, tmp_path / "data1.safetensors")
        safetensors.numpy.save_file({"state": data2}, tmp_path / "data2.safetensors")

        with (
            patch("lerobot.scripts.lerobot_edit_dataset.LeRobotDataset") as mock_dataset_class,
            patch("lerobot.scripts.lerobot_edit_dataset.add_features") as mock_add_features,
        ):
            mock_dataset = mock_dataset_class.return_value
            mock_dataset.meta.total_frames = 50
            mock_dataset.meta.episodes = [
                {"dataset_to_index": 10},
                {"dataset_to_index": 20},
                {"dataset_to_index": 30},
                {"dataset_to_index": 40},
                {"dataset_to_index": 50},
            ]

            handle_add_feature(
                EditDatasetConfig(
                    repo_id="test/repo",
                    new_repo_id="test/repo_modified",
                    root=str(tmp_path / "output"),
                    push_to_hub=False,
                    operation=AddFeatureConfig(
                        feature_names=["reward", "state"],
                        feature_paths=[
                            str(tmp_path / "data1.safetensors"),
                            str(tmp_path / "data2.safetensors"),
                        ],
                    ),
                )
            )

            # Verify add_features was called with both features
            mock_add_features.assert_called_once()
            args, kwargs = mock_add_features.call_args
            assert "features" in kwargs
            assert "reward" in kwargs["features"]  # Single key -> just the name
            assert "state" in kwargs["features"]  # Single key -> just the name

    def test_add_feature_single_name_multiple_keys(self, tmp_path):
        """Test adding features with single name for safetensors file with multiple keys (prefixing)."""
        import safetensors.numpy

        # Create safetensors with multiple keys
        data1 = np.random.randn(50, 1).astype(np.float32)
        data2 = np.random.randn(50, 2).astype(np.float32)
        safetensors.numpy.save_file({"temp": data1, "press": data2}, tmp_path / "multi_data.safetensors")

        with (
            patch("lerobot.scripts.lerobot_edit_dataset.LeRobotDataset") as mock_dataset_class,
            patch("lerobot.scripts.lerobot_edit_dataset.add_features") as mock_add_features,
        ):
            mock_dataset = mock_dataset_class.return_value
            mock_dataset.meta.total_frames = 50
            mock_dataset.meta.episodes = [
                {"dataset_to_index": 10},
                {"dataset_to_index": 20},
                {"dataset_to_index": 30},
                {"dataset_to_index": 40},
                {"dataset_to_index": 50},
            ]

            handle_add_feature(
                EditDatasetConfig(
                    repo_id="test/repo",
                    new_repo_id="test/repo_modified",
                    root=str(tmp_path / "output"),
                    push_to_hub=False,
                    operation=AddFeatureConfig(
                        feature_names=["sensor"],
                        feature_paths=[str(tmp_path / "multi_data.safetensors")],
                    ),
                )
            )

            # Verify add_features was called with prefixed feature names
            mock_add_features.assert_called_once()
            args, kwargs = mock_add_features.call_args
            assert "features" in kwargs
            assert "sensor.temp" in kwargs["features"]
            assert "sensor.press" in kwargs["features"]
            assert kwargs["features"]["sensor.temp"][0].shape == (50, 1)
            assert kwargs["features"]["sensor.press"][0].shape == (50, 2)

    def test_add_feature_list_names_override_keys(self, tmp_path):
        """Test adding features with list of names to override safetensors keys."""
        import safetensors.numpy

        # Create safetensors with multiple keys (in alphabetical order: a_temp, b_press)
        data1 = np.random.randn(50, 1).astype(np.float32)
        data2 = np.random.randn(50, 2).astype(np.float32)
        safetensors.numpy.save_file({"a_temp": data1, "b_press": data2}, tmp_path / "sensor_data.safetensors")

        with (
            patch("lerobot.scripts.lerobot_edit_dataset.LeRobotDataset") as mock_dataset_class,
            patch("lerobot.scripts.lerobot_edit_dataset.add_features") as mock_add_features,
        ):
            mock_dataset = mock_dataset_class.return_value
            mock_dataset.meta.total_frames = 50
            mock_dataset.meta.episodes = [
                {"dataset_to_index": 10},
                {"dataset_to_index": 20},
                {"dataset_to_index": 30},
                {"dataset_to_index": 40},
                {"dataset_to_index": 50},
            ]

            handle_add_feature(
                EditDatasetConfig(
                    repo_id="test/repo",
                    new_repo_id="test/repo_modified",
                    root=str(tmp_path / "output"),
                    push_to_hub=False,
                    operation=AddFeatureConfig(
                        feature_names=[["temp_sensor", "press_sensor"]],
                        feature_paths=[str(tmp_path / "sensor_data.safetensors")],
                    ),
                )
            )

            # Verify add_features was called with overridden feature names
            mock_add_features.assert_called_once()
            args, kwargs = mock_add_features.call_args
            assert "features" in kwargs
            assert "temp_sensor" in kwargs["features"]
            assert "press_sensor" in kwargs["features"]
            assert kwargs["features"]["temp_sensor"][0].shape == (50, 1)  # a_temp data
            assert kwargs["features"]["press_sensor"][0].shape == (50, 2)  # b_press data

    def test_add_feature_list_names_with_empty_fallback(self, tmp_path):
        """Test adding features with list containing empty strings (fallback to original keys)."""
        import safetensors.numpy

        # Create safetensors with multiple keys (in alphabetical order: a_temp, b_press)
        data1 = np.random.randn(50, 1).astype(np.float32)
        data2 = np.random.randn(50, 2).astype(np.float32)
        safetensors.numpy.save_file({"a_temp": data1, "b_press": data2}, tmp_path / "sensor_data.safetensors")

        with (
            patch("lerobot.scripts.lerobot_edit_dataset.LeRobotDataset") as mock_dataset_class,
            patch("lerobot.scripts.lerobot_edit_dataset.add_features") as mock_add_features,
        ):
            mock_dataset = mock_dataset_class.return_value
            mock_dataset.meta.total_frames = 50
            mock_dataset.meta.episodes = [
                {"dataset_to_index": 10},
                {"dataset_to_index": 20},
                {"dataset_to_index": 30},
                {"dataset_to_index": 40},
                {"dataset_to_index": 50},
            ]

            handle_add_feature(
                EditDatasetConfig(
                    repo_id="test/repo",
                    new_repo_id="test/repo_modified",
                    root=str(tmp_path / "output"),
                    push_to_hub=False,
                    operation=AddFeatureConfig(
                        feature_names=[["", "custom_pressure"]],
                        feature_paths=[str(tmp_path / "sensor_data.safetensors")],
                    ),
                )
            )

            # Verify add_features was called with mixed naming (original key + custom)
            mock_add_features.assert_called_once()
            args, kwargs = mock_add_features.call_args
            assert "features" in kwargs
            assert "a_temp" in kwargs["features"]  # Original key due to empty string
            assert "custom_pressure" in kwargs["features"]
            assert kwargs["features"]["a_temp"][0].shape == (50, 1)
            assert kwargs["features"]["custom_pressure"][0].shape == (50, 2)

    def test_add_feature_list_names_mismatch_error(self, tmp_path):
        """Test error when list of names doesn't match number of safetensors keys."""
        import safetensors.numpy

        # Create safetensors with 2 keys
        data1 = np.random.randn(50, 1).astype(np.float32)
        data2 = np.random.randn(50, 2).astype(np.float32)
        safetensors.numpy.save_file({"temp": data1, "press": data2}, tmp_path / "sensor_data.safetensors")

        with (
            patch("lerobot.scripts.lerobot_edit_dataset.LeRobotDataset") as mock_dataset_class,
        ):
            mock_dataset = mock_dataset_class.return_value
            mock_dataset.meta.total_frames = 50
            mock_dataset.meta.episodes = [
                {"dataset_to_index": 10},
                {"dataset_to_index": 20},
                {"dataset_to_index": 30},
                {"dataset_to_index": 40},
                {"dataset_to_index": 50},
            ]

            # Try with list of 3 names for 2 keys
            with pytest.raises(
                ValueError, match="List of feature names has length 3, but the file contains 2 tensors"
            ):
                handle_add_feature(
                    EditDatasetConfig(
                        repo_id="test/repo",
                        new_repo_id="test/repo_modified",
                        root=str(tmp_path / "output"),
                        push_to_hub=False,
                        operation=AddFeatureConfig(
                            feature_names=[["name1", "name2", "name3"]],
                            feature_paths=[str(tmp_path / "sensor_data.safetensors")],
                        ),
                    )
                )

    def test_add_feature_invalid_file_extension(self, tmp_path):
        """Test error with invalid file extension."""
        invalid_path = tmp_path / "data.txt"
        invalid_path.write_text("invalid data")

        with (
            patch("lerobot.scripts.lerobot_edit_dataset.LeRobotDataset") as mock_dataset_class,
            patch("lerobot.scripts.lerobot_edit_dataset.load_file") as mock_load_file,
        ):
            mock_dataset = mock_dataset_class.return_value
            mock_dataset.meta.total_frames = 50
            mock_dataset.meta.episodes = [
                {"dataset_to_index": 10},
                {"dataset_to_index": 20},
                {"dataset_to_index": 30},
                {"dataset_to_index": 40},
                {"dataset_to_index": 50},
            ]
            mock_dataset.fps = 30.0

            # Mock load_file to raise an error for invalid extension
            mock_load_file.side_effect = Exception("Unsupported file format")

            with pytest.raises(Exception, match="Unsupported file format"):
                handle_add_feature(
                    EditDatasetConfig(
                        repo_id="test/repo",
                        new_repo_id="test/repo_modified",
                        root=str(tmp_path / "output"),
                        push_to_hub=False,
                        operation=AddFeatureConfig(
                            feature_names=["reward"],
                            feature_paths=[str(invalid_path)],
                        ),
                    )
                )

    def test_add_feature_mismatched_lengths(self, tmp_path):
        """Test error when data length doesn't match dataset frames."""
        import safetensors.numpy

        # Create data with wrong length (30 instead of 50)
        data = np.random.randn(30, 1).astype(np.float32)
        safetensors.numpy.save_file({"reward": data}, tmp_path / "data.safetensors")

        with (
            patch("lerobot.scripts.lerobot_edit_dataset.LeRobotDataset") as mock_dataset_class,
        ):
            mock_dataset = mock_dataset_class.return_value
            mock_dataset.meta.total_frames = 50
            mock_dataset.meta.episodes = [
                {"dataset_to_index": 10},
                {"dataset_to_index": 20},
                {"dataset_to_index": 30},
                {"dataset_to_index": 40},
                {"dataset_to_index": 50},
            ]

            with pytest.raises(ValueError, match="Feature .* has .* frames, while dataset has .* frames"):
                handle_add_feature(
                    EditDatasetConfig(
                        repo_id="test/repo",
                        new_repo_id="test/repo_modified",
                        root=str(tmp_path / "output"),
                        push_to_hub=False,
                        operation=AddFeatureConfig(
                            feature_names=["reward"],
                            feature_paths=[str(tmp_path / "data.safetensors")],
                        ),
                    )
                )

    def test_add_feature_validation_feature_names_required(self):
        """Test that feature_names is required."""
        cfg = EditDatasetConfig(
            repo_id="test/repo",
            operation=AddFeatureConfig(
                feature_names=None,  # Missing!
                feature_paths=["/some/path"],
            ),
        )

        with pytest.raises(ValueError, match="feature_names must be specified"):
            handle_add_feature(cfg)

    def test_add_feature_validation_feature_paths_required(self):
        """Test that feature_paths is required."""
        cfg = EditDatasetConfig(
            repo_id="test/repo",
            operation=AddFeatureConfig(
                feature_names=["reward"],
                feature_paths=None,  # Missing!
            ),
        )

        with pytest.raises(ValueError, match="feature_paths must be specified"):
            handle_add_feature(cfg)

    def test_add_feature_validation_matching_lengths(self):
        """Test that feature_names and feature_paths must have same length."""
        cfg = EditDatasetConfig(
            repo_id="test/repo",
            operation=AddFeatureConfig(
                feature_names=["reward", "extra"],
                feature_paths=["/path1"],  # Mismatch!
            ),
        )

        with pytest.raises(ValueError, match="must have the same length"):
            handle_add_feature(cfg)

    def test_add_feature_validation_dim_names_length(self):
        """Test that dim_names length matches if provided."""
        cfg = EditDatasetConfig(
            repo_id="test/repo",
            operation=AddFeatureConfig(
                feature_names=["reward"],
                feature_paths=["/path1"],
                dim_names=[[None], [None]],  # Mismatch!
            ),
        )

        with pytest.raises(ValueError, match="dim_names list is provided"):
            handle_add_feature(cfg)

    def test_add_feature_validation_feature_types_length(self):
        """Test that feature_types length matches if provided."""
        cfg = EditDatasetConfig(
            repo_id="test/repo",
            operation=AddFeatureConfig(
                feature_names=["reward"],
                feature_paths=["/path1"],
                feature_types=["reward", "motor"],  # Mismatch!
            ),
        )

        with pytest.raises(ValueError, match="feature_types is provided"):
            handle_add_feature(cfg)

    def test_add_feature_uses_default_feature_types(self, tmp_path):
        """Test that default feature_types are used when not provided."""
        import safetensors.numpy

        data = np.random.randn(50, 1).astype(np.float32)
        safetensors.numpy.save_file({"reward": data}, tmp_path / "data.safetensors")

        with (
            patch("lerobot.scripts.lerobot_edit_dataset.LeRobotDataset") as mock_dataset_class,
            patch("lerobot.scripts.lerobot_edit_dataset.add_features") as mock_add_features,
        ):
            mock_dataset = mock_dataset_class.return_value
            mock_dataset.meta.total_frames = 50
            mock_dataset.meta.episodes = [{"dataset_to_index": 50}]
            mock_dataset.fps = 30.0

            cfg = EditDatasetConfig(
                repo_id="test/repo",
                root=str(tmp_path),
                operation=AddFeatureConfig(
                    feature_names=["reward"],
                    feature_paths=[str(tmp_path / "data.safetensors")],
                    feature_types=None,  # Not provided, should default to "motor"
                ),
            )

            handle_add_feature(cfg)
            mock_add_features.assert_called_once()

    def test_add_feature_uses_default_dim_names(self, tmp_path):
        """Test that default dim_names are used when not provided."""
        import safetensors.numpy

        data = np.random.randn(50, 1).astype(np.float32)
        safetensors.numpy.save_file({"reward": data}, tmp_path / "data.safetensors")

        with (
            patch("lerobot.scripts.lerobot_edit_dataset.LeRobotDataset") as mock_dataset_class,
            patch("lerobot.scripts.lerobot_edit_dataset.add_features") as mock_add_features,
        ):
            mock_dataset = mock_dataset_class.return_value
            mock_dataset.meta.total_frames = 50
            mock_dataset.meta.episodes = [{"dataset_to_index": 50}]
            mock_dataset.fps = 30.0

            cfg = EditDatasetConfig(
                repo_id="test/repo",
                root=str(tmp_path),
                operation=AddFeatureConfig(
                    feature_names=["reward"],
                    feature_paths=[str(tmp_path / "data.safetensors")],
                    dim_names=None,  # Not provided, should default
                ),
            )

            handle_add_feature(cfg)
            mock_add_features.assert_called_once()
