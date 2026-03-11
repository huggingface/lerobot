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
Tests for DataProcessorPipeline.from_pretrained helper methods.

These tests focus on the individual private methods that were extracted from
the main from_pretrained method to improve modularity and testability.
"""

import json
import tempfile
from pathlib import Path

import pytest

from lerobot.processor.pipeline import DataProcessorPipeline, ProcessorMigrationError

# Simplified Config Loading Tests


def test_load_config_directory():
    """Test loading config from directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create a config file
        config_file = tmp_path / "processor.json"
        test_config = {"name": "TestProcessor", "steps": []}
        config_file.write_text(json.dumps(test_config))

        # Load from directory
        loaded_config, base_path = DataProcessorPipeline._load_config(str(tmp_path), "processor.json", {})

        assert loaded_config == test_config
        assert base_path == tmp_path


def test_load_config_single_file():
    """Test loading config from a single file path."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create a config file
        config_file = tmp_path / "processor.json"
        test_config = {"name": "TestProcessor", "steps": []}
        config_file.write_text(json.dumps(test_config))

        # Load using file path directly
        loaded_config, base_path = DataProcessorPipeline._load_config(
            str(config_file), "any_filename_ignored", {}
        )

        assert loaded_config == test_config
        assert base_path == tmp_path


def test_load_config_directory_file_not_found():
    """Test directory loading when config file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Directory exists but no processor.json
        with pytest.raises(FileNotFoundError, match="not found in directory"):
            DataProcessorPipeline._load_config(str(tmp_path), "processor.json", {})


def test_load_config_directory_with_migration_detection():
    """Test that missing config triggers migration detection."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create old-style config to trigger migration
        (tmp_path / "config.json").write_text(json.dumps({"type": "act"}))

        # Try to load processor.json (doesn't exist), should trigger migration
        with pytest.raises(ProcessorMigrationError):
            DataProcessorPipeline._load_config(str(tmp_path), "processor.json", {})


def test_load_config_nonexistent_path_tries_hub():
    """Test that nonexistent paths try Hub (simplified logic)."""
    # This path doesn't exist locally, should try Hub
    with pytest.raises(FileNotFoundError, match="on the HuggingFace Hub"):
        DataProcessorPipeline._load_config("nonexistent/path", "processor.json", {})


# Config Validation Tests


def test_validate_loaded_config_valid_config():
    """Test validation with valid processor config."""
    valid_config = {"name": "TestProcessor", "steps": []}

    # Should not raise any exception
    DataProcessorPipeline._validate_loaded_config("any-path", valid_config, "processor.json")


def test_validate_loaded_config_invalid_config():
    """Test validation with invalid processor config."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create non-processor config to trigger migration
        (tmp_path / "config.json").write_text(json.dumps({"type": "act"}))

        invalid_config = {"type": "act", "hidden_dim": 256}

        with pytest.raises(ProcessorMigrationError):
            DataProcessorPipeline._validate_loaded_config(str(tmp_path), invalid_config, "config.json")


def test_validate_loaded_config_invalid_config_no_migration():
    """Test validation with invalid config when no migration is detected."""
    # Non-directory path (Hub repo) - no migration detection
    invalid_config = {"type": "act", "hidden_dim": 256}

    with pytest.raises(ValueError, match="not a valid processor configuration"):
        DataProcessorPipeline._validate_loaded_config("user/repo", invalid_config, "config.json")


# Step Class Resolution Tests


def test_resolve_step_class_registry_name():
    """Test resolution using registry name."""
    from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry

    # Register a test step
    @ProcessorStepRegistry.register("test_step")
    class TestStep(ProcessorStep):
        def __call__(self, transition):
            return transition

        def transform_features(self, features):
            return features

    try:
        step_entry = {"registry_name": "test_step"}
        step_class, step_key = DataProcessorPipeline._resolve_step_class(step_entry)

        assert step_class is TestStep
        assert step_key == "test_step"
    finally:
        ProcessorStepRegistry.unregister("test_step")


def test_resolve_step_class_registry_name_not_found():
    """Test resolution with non-existent registry name."""
    step_entry = {"registry_name": "nonexistent_step"}

    with pytest.raises(ImportError, match="Failed to load processor step from registry"):
        DataProcessorPipeline._resolve_step_class(step_entry)


def test_resolve_step_class_import_path():
    """Test resolution using full import path."""
    # Use a valid existing class (this should work)
    step_entry = {"class": "lerobot.processor.pipeline.ProcessorStep"}

    # This should succeed - ProcessorStep can be imported, just not instantiated
    step_class, step_key = DataProcessorPipeline._resolve_step_class(step_entry)

    from lerobot.processor.pipeline import ProcessorStep

    assert step_class is ProcessorStep
    assert step_key == "ProcessorStep"


def test_resolve_step_class_invalid_import_path():
    """Test resolution with invalid import path."""
    step_entry = {"class": "nonexistent.module.ClassName"}

    with pytest.raises(ImportError, match="Failed to load processor step"):
        DataProcessorPipeline._resolve_step_class(step_entry)


# Override Validation Tests


def test_validate_overrides_used_all_used():
    """Test validation when all overrides are used."""
    # Empty set means all overrides were used
    remaining_overrides = set()
    config = {"steps": [{"class": "SomeStep"}]}

    # Should not raise
    DataProcessorPipeline._validate_overrides_used(remaining_overrides, config)


def test_validate_overrides_used_some_unused():
    """Test validation when some overrides are unused."""
    remaining_overrides = {"NonExistentStep", "AnotherMissingStep"}
    config = {
        "steps": [
            {"registry_name": "normalize_step"},
            {"class": "some.module.TransformStep"},
        ]
    }

    with pytest.raises(KeyError, match="Override keys.*do not match any step"):
        DataProcessorPipeline._validate_overrides_used(remaining_overrides, config)


def test_validate_overrides_used_helpful_error_message():
    """Test that error message includes available step keys."""
    remaining_overrides = {"WrongStep"}
    config = {
        "steps": [
            {"registry_name": "correct_step"},
            {"class": "module.path.CorrectClass"},
        ]
    }

    with pytest.raises(KeyError) as exc_info:
        DataProcessorPipeline._validate_overrides_used(remaining_overrides, config)

    error_msg = str(exc_info.value)
    assert "Available step keys" in error_msg
    assert "correct_step" in error_msg
    assert "CorrectClass" in error_msg


# Integration Tests for Simplified Logic


def test_simplified_three_way_loading():
    """Test that the simplified 3-way loading logic works correctly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Test 1: Directory loading
        config_file = tmp_path / "processor.json"
        test_config = {"name": "DirectoryTest", "steps": []}
        config_file.write_text(json.dumps(test_config))

        loaded_config, base_path = DataProcessorPipeline._load_config(str(tmp_path), "processor.json", {})
        assert loaded_config["name"] == "DirectoryTest"
        assert base_path == tmp_path

        # Test 2: Single file loading
        loaded_config, base_path = DataProcessorPipeline._load_config(
            str(config_file), "ignored_filename", {}
        )
        assert loaded_config["name"] == "DirectoryTest"
        assert base_path == tmp_path
