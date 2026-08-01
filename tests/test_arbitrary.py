#!/usr/bin/env python3
import json
import os
import tempfile

import pytest

from lerobot.processor.pipeline import DataProcessorPipeline


def _create_temp_model_dir(config_dict: dict) -> str:
    """Helper to create a temporary directory containing processor.json."""
    model_dir = tempfile.mkdtemp(prefix="lerobot_test_model_")
    config_path = os.path.join(model_dir, "processor.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f)
    return model_dir


def test_rce_attempt_blocked():
    """Test 1: Malicious external module import attempt (e.g. subprocess.Popen) is blocked."""
    marker_dir = tempfile.mkdtemp()
    marker_file = os.path.join(marker_dir, "LEROBOT_PWNED")

    malicious_config = {
        "name": "malicious-processor",
        "steps": [
            {"class": "subprocess.Popen", "config": {"args": ["/bin/sh", "-c", f"touch {marker_file}"]}}
        ],
    }

    model_dir = _create_temp_model_dir(malicious_config)

    with pytest.raises((ImportError, ValueError)) as exc_info:
        DataProcessorPipeline.from_pretrained(
            model_dir,
            config_filename="processor.json",
        )

    assert "Unauthorized module path 'subprocess'" in str(exc_info.value)
    assert not os.path.exists(marker_file), (
        "Security vulnerability! Code executed and marker file was created."
    )


def test_invalid_class_path_without_dot():
    """Test 2: Malicious/invalid class path without a dot (e.g. 'Popen' or 'eval') is blocked cleanly."""
    invalid_config = {"name": "invalid-class-processor", "steps": [{"class": "Popen", "config": {}}]}

    model_dir = _create_temp_model_dir(invalid_config)

    with pytest.raises(ValueError) as exc_info:
        DataProcessorPipeline.from_pretrained(
            model_dir,
            config_filename="processor.json",
        )

    assert "Invalid class path 'Popen'" in str(exc_info.value)


def test_unregistered_registry_name():
    """Test 3: Unregistered registry_name raises ImportError."""
    invalid_registry_config = {
        "name": "invalid-registry-processor",
        "steps": [{"registry_name": "non_existent_processor_step", "config": {}}],
    }

    model_dir = _create_temp_model_dir(invalid_registry_config)

    with pytest.raises(ImportError) as exc_info:
        DataProcessorPipeline.from_pretrained(
            model_dir,
            config_filename="processor.json",
        )

    assert "Failed to load processor step from registry" in str(exc_info.value)


def test_valid_registry_name():
    """Test 4: Registry-based step resolution works correctly."""
    valid_registry_config = {
        "name": "valid-registry-processor",
        "steps": [{"registry_name": "rename_observations_processor", "config": {"rename_map": {}}}],
    }

    model_dir = _create_temp_model_dir(valid_registry_config)

    pipeline = DataProcessorPipeline.from_pretrained(
        model_dir,
        config_filename="processor.json",
    )
    assert len(pipeline.steps) == 1
    assert pipeline.steps[0].__class__.__name__ == "RenameObservationsProcessorStep"


def test_valid_legacy_class_path():
    """Test 5: Backward compatibility - valid internal lerobot.processor module class path works."""
    valid_legacy_config = {
        "name": "valid-legacy-processor",
        "steps": [
            {
                "class": "lerobot.processor.rename_processor.RenameObservationsProcessorStep",
                "config": {"rename_map": {}},
            }
        ],
    }

    model_dir = _create_temp_model_dir(valid_legacy_config)

    pipeline = DataProcessorPipeline.from_pretrained(
        model_dir,
        config_filename="processor.json",
    )
    assert len(pipeline.steps) == 1
    assert pipeline.steps[0].__class__.__name__ == "RenameObservationsProcessorStep"
