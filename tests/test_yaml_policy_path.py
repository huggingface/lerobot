"""Tests for policy.path support in YAML config files (issue #2957)."""

import json
import tempfile
from pathlib import Path

import yaml

from lerobot.configs.parser import (
    _config_path_args,
    extract_path_fields_from_config,
    get_path_arg,
)


def test_extract_path_fields_from_yaml():
    """Test that policy.path is extracted from a YAML config and removed."""
    config = {
        "dataset": {"repo_id": "lerobot/pusht"},
        "policy": {"type": "smolvla", "path": "lerobot/smolvla_base", "push_to_hub": False},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name

    _config_path_args.clear()
    cleaned_path = extract_path_fields_from_config(config_path, ["policy"])

    # Path should be extracted and stored
    assert _config_path_args["policy"] == "lerobot/smolvla_base"

    # Cleaned config should not have the path field
    with open(cleaned_path) as f:
        cleaned = yaml.safe_load(f)
    assert "path" not in cleaned["policy"]
    assert cleaned["policy"]["type"] == "smolvla"
    assert cleaned["policy"]["push_to_hub"] is False

    # Original dataset should be untouched
    assert cleaned["dataset"]["repo_id"] == "lerobot/pusht"

    _config_path_args.clear()


def test_extract_path_fields_from_json():
    """Test that policy.path is extracted from a JSON config."""
    config = {
        "policy": {"type": "act", "path": "some/local/path"},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        config_path = f.name

    _config_path_args.clear()
    cleaned_path = extract_path_fields_from_config(config_path, ["policy"])

    assert _config_path_args["policy"] == "some/local/path"

    with open(cleaned_path) as f:
        cleaned = json.load(f)
    assert "path" not in cleaned["policy"]

    _config_path_args.clear()


def test_extract_no_path_returns_original():
    """Test that configs without path fields are returned unchanged."""
    config = {
        "dataset": {"repo_id": "lerobot/pusht"},
        "policy": {"type": "smolvla"},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name

    _config_path_args.clear()
    result = extract_path_fields_from_config(config_path, ["policy"])

    assert result == config_path
    assert "policy" not in _config_path_args

    _config_path_args.clear()


def test_extract_removes_empty_field():
    """Test that the field dict is removed entirely if path was the only key."""
    config = {
        "dataset": {"repo_id": "lerobot/pusht"},
        "policy": {"path": "lerobot/smolvla_base"},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name

    _config_path_args.clear()
    cleaned_path = extract_path_fields_from_config(config_path, ["policy"])

    assert _config_path_args["policy"] == "lerobot/smolvla_base"

    with open(cleaned_path) as f:
        cleaned = yaml.safe_load(f)
    assert "policy" not in cleaned

    _config_path_args.clear()


def test_get_path_arg_fallback():
    """Test that get_path_arg falls back to _config_path_args when CLI has no path."""
    _config_path_args.clear()
    _config_path_args["policy"] = "lerobot/smolvla_base"

    # No CLI args with --policy.path
    result = get_path_arg("policy", args=[])
    assert result == "lerobot/smolvla_base"

    _config_path_args.clear()


def test_get_path_arg_cli_takes_precedence():
    """Test that CLI --policy.path takes precedence over YAML config path."""
    _config_path_args.clear()
    _config_path_args["policy"] = "yaml/path"

    result = get_path_arg("policy", args=["--policy.path=cli/path"])
    assert result == "cli/path"

    _config_path_args.clear()
