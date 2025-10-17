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

"""
Tests for processor migration detection functionality.
"""

import json
import tempfile
from pathlib import Path

import pytest

from lerobot.processor.pipeline import DataProcessorPipeline, ProcessorMigrationError
from lerobot.utils.constants import ACTION, OBS_STATE


def test_is_processor_config_valid_configs():
    """Test processor config detection with valid configurations."""
    valid_configs = [
        {"steps": []},  # Empty steps
        {"steps": [{"class": "MyClass"}]},  # Class-based step
        {"steps": [{"registry_name": "my_step"}]},  # Registry-based step
        {"steps": [{"class": "A"}, {"registry_name": "B"}]},  # Mixed
        {"name": "Test", "steps": [{"class": "MyClass"}]},  # With name
    ]

    for i, config in enumerate(valid_configs):
        assert DataProcessorPipeline._is_processor_config(config), (
            f"Valid config {i} should be detected as processor config: {config}"
        )


def test_is_processor_config_invalid_configs():
    """Test processor config detection with invalid configurations."""
    invalid_configs = [
        {},  # No steps field
        {"steps": "not a list"},  # Steps is not a list
        {"steps": [{}]},  # Step without class or registry_name
        {"steps": ["not a dict"]},  # Step is not a dict
        {"steps": [{"other_field": "value"}]},  # Step with wrong fields
        {"other_field": "value"},  # Completely different structure
    ]

    for i, config in enumerate(invalid_configs):
        assert not DataProcessorPipeline._is_processor_config(config), (
            f"Invalid config {i} should not be detected as processor config: {config}"
        )


def test_should_suggest_migration_with_processor_config():
    """Test that migration is NOT suggested when processor config exists."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create a valid processor config
        processor_config = {
            "name": "TestProcessor",
            "steps": [
                {
                    "class": "lerobot.processor.normalize.NormalizeStep",
                    "config": {"mean": 0.0, "std": 1.0},
                }
            ],
        }

        with open(tmp_path / "processor.json", "w") as f:
            json.dump(processor_config, f)

        # Should NOT suggest migration (processor config exists)
        result = DataProcessorPipeline._should_suggest_migration(tmp_path)
        assert not result


def test_should_suggest_migration_with_empty_processor_config():
    """Test that migration is NOT suggested when empty processor config exists."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create an empty processor config
        empty_processor_config = {
            "name": "EmptyProcessor",
            "steps": [],  # Empty steps is valid
        }

        with open(tmp_path / "empty_processor.json", "w") as f:
            json.dump(empty_processor_config, f)

        # Should NOT suggest migration (processor config exists, even if empty)
        result = DataProcessorPipeline._should_suggest_migration(tmp_path)
        assert not result


def test_should_suggest_migration_with_model_config_only():
    """Test that migration IS suggested when only model config exists."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create a model config (like old LeRobot format)
        model_config = {
            "type": "act",
            "input_features": {OBS_STATE: {"shape": [7]}},
            "output_features": {ACTION: {"shape": [7]}},
            "hidden_dim": 256,
            "n_obs_steps": 1,
            "n_action_steps": 1,
        }

        with open(tmp_path / "config.json", "w") as f:
            json.dump(model_config, f)

        # SHOULD suggest migration (model config exists but no processor)
        result = DataProcessorPipeline._should_suggest_migration(tmp_path)
        assert result


def test_should_suggest_migration_no_json_files():
    """Test that migration is NOT suggested when no JSON files exist."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create some non-JSON files
        with open(tmp_path / "model.safetensors", "w") as f:
            f.write("fake model data")

        with open(tmp_path / "README.md", "w") as f:
            f.write("# Model README")

        # Should NOT suggest migration (no JSON files)
        result = DataProcessorPipeline._should_suggest_migration(tmp_path)
        assert not result


def test_should_suggest_migration_random_json_files():
    """Test that migration IS suggested when JSON files exist but none are processor configs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create some random JSON file (not a processor config)
        random_config = {"some_field": "some_value", "another_field": 123}

        with open(tmp_path / "random.json", "w") as f:
            json.dump(random_config, f)

        # SHOULD suggest migration (JSON files exist but none are processor configs)
        result = DataProcessorPipeline._should_suggest_migration(tmp_path)
        assert result


def test_should_suggest_migration_mixed_configs():
    """Test that migration is NOT suggested when processor config exists alongside other configs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create both a processor config and a model config
        processor_config = {"name": "TestProcessor", "steps": [{"registry_name": "normalize_step"}]}

        model_config = {"type": "diffusion", "hidden_dim": 512}

        with open(tmp_path / "processor.json", "w") as f:
            json.dump(processor_config, f)

        with open(tmp_path / "config.json", "w") as f:
            json.dump(model_config, f)

        # Should NOT suggest migration (processor config exists)
        result = DataProcessorPipeline._should_suggest_migration(tmp_path)
        assert not result


def test_should_suggest_migration_invalid_json():
    """Test that invalid JSON is handled gracefully."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create an invalid JSON file
        with open(tmp_path / "invalid.json", "w") as f:
            f.write("{ invalid json")

        # Create a valid non-processor config
        model_config = {"type": "act"}
        with open(tmp_path / "model.json", "w") as f:
            json.dump(model_config, f)

        # SHOULD suggest migration (invalid JSON is ignored, but we have non-processor JSON)
        result = DataProcessorPipeline._should_suggest_migration(tmp_path)
        assert result


def test_from_pretrained_multiple_json_files_migration_error():
    """Test that multiple JSON files trigger ProcessorMigrationError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create multiple non-processor configs
        model_config = {"type": "act", "hidden_dim": 128}
        train_config = {"batch_size": 32, "lr": 0.001}

        with open(tmp_path / "config.json", "w") as f:
            json.dump(model_config, f)

        with open(tmp_path / "train_config.json", "w") as f:
            json.dump(train_config, f)

        # Should raise ProcessorMigrationError
        with pytest.raises(ProcessorMigrationError) as exc_info:
            DataProcessorPipeline.from_pretrained(tmp_path, config_filename="config.json")

        # Check the error details
        error = exc_info.value
        assert str(tmp_path) in str(error.model_path)
        assert "migrate_policy_normalization.py" in error.migration_command
        assert "not a valid processor configuration" in error.original_error


def test_from_pretrained_no_processor_config_migration_error():
    """Test that missing processor config triggers ProcessorMigrationError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create a model config but no processor
        model_config = {"type": "diffusion", "hidden_dim": 256}

        with open(tmp_path / "config.json", "w") as f:
            json.dump(model_config, f)

        # Should raise ProcessorMigrationError
        with pytest.raises(ProcessorMigrationError) as exc_info:
            DataProcessorPipeline.from_pretrained(tmp_path, config_filename="config.json")

        # Check the error details
        error = exc_info.value
        assert str(tmp_path) in str(error.model_path)
        assert "migrate_policy_normalization.py" in error.migration_command
        assert "not a valid processor configuration" in error.original_error


def test_from_pretrained_valid_processor_no_migration_error():
    """Test that valid processor config does NOT trigger migration error."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create a valid processor config
        processor_config = {
            "name": "TestProcessor",
            "steps": [],  # Empty is valid
        }

        with open(tmp_path / "processor.json", "w") as f:
            json.dump(processor_config, f)

        # Should succeed and create pipeline
        pipeline = DataProcessorPipeline.from_pretrained(tmp_path, config_filename="processor.json")
        assert pipeline is not None
        assert pipeline.name == "TestProcessor"
        assert len(pipeline) == 0


def test_from_pretrained_no_json_files_no_migration_error():
    """Test that directories with no JSON files don't trigger migration errors."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create some non-JSON files
        with open(tmp_path / "model.safetensors", "w") as f:
            f.write("fake model data")

        # Should raise FileNotFoundError (config file not found)
        with pytest.raises(FileNotFoundError, match="not found in directory"):
            DataProcessorPipeline.from_pretrained(tmp_path, config_filename="processor.json")


def test_processor_migration_error_creation():
    """Test that ProcessorMigrationError is created correctly."""
    model_path = "/path/to/model"
    migration_command = "python migrate.py --path /path/to/model"
    original_error = "Config not found"

    error = ProcessorMigrationError(model_path, migration_command, original_error)

    assert error.model_path == model_path
    assert error.migration_command == migration_command
    assert error.original_error == original_error
    assert model_path in str(error)
    assert migration_command in str(error)
    assert original_error in str(error)


def test_processor_migration_error_attributes():
    """Test that ProcessorMigrationError has correct attributes."""
    model_path = Path("/test/path")
    migration_command = "python test.py"
    original_error = "Test error"

    error = ProcessorMigrationError(model_path, migration_command, original_error)

    # Test that attributes are accessible
    assert hasattr(error, "model_path")
    assert hasattr(error, "migration_command")
    assert hasattr(error, "original_error")

    # Test that it's still an Exception
    assert isinstance(error, Exception)


def test_migration_suggestion_raises_error():
    """Test that migration suggestion always raises ProcessorMigrationError."""
    with pytest.raises(ProcessorMigrationError) as exc_info:
        DataProcessorPipeline._suggest_processor_migration("/test/path", "Test error")

    error = exc_info.value
    assert "/test/path" in str(error.model_path)
    assert "Test error" in error.original_error
    assert "migrate_policy_normalization.py" in error.migration_command


def test_migration_error_always_raised_for_invalid_configs():
    """Test that ProcessorMigrationError is always raised for invalid configs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create a model config
        model_config = {"type": "test", "param": "value"}
        with open(tmp_path / "config.json", "w") as f:
            json.dump(model_config, f)

        # Should always raise ProcessorMigrationError
        with pytest.raises(ProcessorMigrationError):
            DataProcessorPipeline.from_pretrained(tmp_path, config_filename="config.json")
