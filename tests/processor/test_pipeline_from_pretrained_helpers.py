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

# Config Source Resolution Tests


def test_resolve_config_source_non_directory():
    """Test resolution for non-directory paths (Hub repos)."""
    config_filename, base_path = DataProcessorPipeline._resolve_config_source("user/repo", "processor.json")
    assert config_filename == "processor.json"
    assert base_path is None


def test_resolve_config_source_specified_filename():
    """Test resolution when user specifies config filename."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        config_filename, base_path = DataProcessorPipeline._resolve_config_source(
            str(tmp_path), "custom.json"
        )
        assert config_filename == "custom.json"
        assert base_path == tmp_path


def test_resolve_config_source_no_json_files():
    """Test resolution when directory has no JSON files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create non-JSON files
        (tmp_path / "model.safetensors").write_text("fake data")

        config_filename, base_path = DataProcessorPipeline._resolve_config_source(str(tmp_path), None)
        assert config_filename is None
        assert base_path == tmp_path


def test_resolve_config_source_single_json_file():
    """Test auto-detection with single JSON file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create single JSON file
        config_path = tmp_path / "processor.json"
        config_path.write_text(json.dumps({"name": "test", "steps": []}))

        config_filename, base_path = DataProcessorPipeline._resolve_config_source(str(tmp_path), None)
        assert config_filename == "processor.json"
        assert base_path == tmp_path


def test_resolve_config_source_multiple_json_valid_processor():
    """Test multiple JSON files where one is a valid processor."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create multiple files including valid processor
        (tmp_path / "config.json").write_text(json.dumps({"type": "act"}))
        (tmp_path / "processor.json").write_text(json.dumps({"name": "test", "steps": []}))

        # Should raise ValueError about multiple files (not migration error)
        with pytest.raises(ValueError, match="Multiple .json files found"):
            DataProcessorPipeline._resolve_config_source(str(tmp_path), None)


def test_resolve_config_source_multiple_json_needs_migration():
    """Test multiple JSON files that need migration."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create multiple non-processor files
        (tmp_path / "config.json").write_text(json.dumps({"type": "act"}))
        (tmp_path / "train.json").write_text(json.dumps({"lr": 0.001}))

        # Should trigger migration error
        with pytest.raises(ProcessorMigrationError):
            DataProcessorPipeline._resolve_config_source(str(tmp_path), None)


# Path Type Detection Tests


def test_looks_like_local_path_hub_repos():
    """Test detection of Hub repository IDs."""
    hub_repos = [
        "user/repo",
        "organization/model-name",
        "simple-repo",
    ]

    for repo in hub_repos:
        assert not DataProcessorPipeline._looks_like_local_path(repo)


def test_looks_like_local_path_local_paths():
    """Test detection of local file paths."""
    local_paths = [
        "/absolute/path/to/model",
        "./relative/path",
        "../parent/dir",
        "user/repo/extra/path",  # Multiple slashes
        "C:\\Windows\\Path",  # Windows path
        "/home/user/models",
    ]

    for path in local_paths:
        assert DataProcessorPipeline._looks_like_local_path(path)


# Config Validation Tests


def test_validate_loaded_config_none_no_migration():
    """Test validation when no config loaded and no migration needed."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Empty directory - no migration needed
        with pytest.raises(RuntimeError, match="Failed to load configuration"):
            DataProcessorPipeline._validate_loaded_config(tmp_dir, None, "processor.json")


def test_validate_loaded_config_none_with_migration():
    """Test validation when no config loaded but migration needed."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create non-processor config to trigger migration
        (tmp_path / "config.json").write_text(json.dumps({"type": "act"}))

        with pytest.raises(ProcessorMigrationError):
            DataProcessorPipeline._validate_loaded_config(tmp_dir, None, "processor.json")


def test_validate_loaded_config_invalid_config():
    """Test validation with invalid processor config."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create non-processor config
        (tmp_path / "config.json").write_text(json.dumps({"type": "act"}))

        invalid_config = {"type": "act", "hidden_dim": 256}

        with pytest.raises(ProcessorMigrationError):
            DataProcessorPipeline._validate_loaded_config(tmp_dir, invalid_config, "config.json")


def test_validate_loaded_config_valid_config():
    """Test validation with valid processor config."""
    valid_config = {"name": "TestProcessor", "steps": []}

    # Should not raise any exception
    DataProcessorPipeline._validate_loaded_config("any-path", valid_config, "processor.json")


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
