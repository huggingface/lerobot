"""Tests for policy.path support in YAML config files (issue #2957)."""

import json
import tempfile

import yaml

from lerobot.configs.parser import (
    _config_path_args,
    _config_yaml_overrides,
    _flatten_to_cli_args,
    extract_path_fields_from_config,
    get_path_arg,
    get_yaml_overrides,
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


def test_yaml_overrides_captured():
    """Test that non-path policy fields are captured as CLI-style overrides."""
    config = {
        "policy": {"path": "lerobot/smolvla_base", "lr": 1e-4, "batch_size": 32},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name

    _config_path_args.clear()
    _config_yaml_overrides.clear()
    extract_path_fields_from_config(config_path, ["policy"])

    overrides = get_yaml_overrides("policy")
    assert "--lr=0.0001" in overrides or any("lr=" in o for o in overrides)
    assert any("batch_size=32" in o for o in overrides)

    _config_path_args.clear()
    _config_yaml_overrides.clear()


def test_yaml_overrides_excludes_type_and_path():
    """Test that type and path fields are not included in YAML overrides."""
    config = {
        "policy": {"path": "lerobot/smolvla_base", "type": "smolvla", "lr": 5e-5},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name

    _config_path_args.clear()
    _config_yaml_overrides.clear()
    extract_path_fields_from_config(config_path, ["policy"])

    overrides = get_yaml_overrides("policy")
    assert not any("path=" in o for o in overrides)
    assert not any("type=" in o for o in overrides)
    assert any("lr=" in o for o in overrides)

    _config_path_args.clear()
    _config_yaml_overrides.clear()


def test_get_yaml_overrides_empty_when_path_only():
    """Test that get_yaml_overrides returns [] when policy had only a path field."""
    config = {
        "policy": {"path": "lerobot/smolvla_base"},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        config_path = f.name

    _config_path_args.clear()
    _config_yaml_overrides.clear()
    extract_path_fields_from_config(config_path, ["policy"])

    assert get_yaml_overrides("policy") == []

    _config_path_args.clear()
    _config_yaml_overrides.clear()


def test_flatten_bool_values():
    """Test that boolean values are serialized as lowercase strings for draccus."""
    d = {"push_to_hub": True, "use_rabc": False, "lr": 0.001, "name": "test"}
    args = _flatten_to_cli_args(d)
    assert "--push_to_hub=true" in args
    assert "--use_rabc=false" in args
    assert "--lr=0.001" in args
    assert "--name=test" in args


def test_flatten_none_values_skipped():
    """Test that None values are not included in flattened args."""
    d = {"lr": 0.001, "path_override": None, "name": "test"}
    args = _flatten_to_cli_args(d)
    assert any("lr=" in a for a in args)
    assert any("name=" in a for a in args)
    assert not any("path_override" in a for a in args)


def test_flatten_nested_with_bools():
    """Test that bools in nested dicts are handled correctly."""
    d = {"optimizer": {"use_warmup": True, "lr": 0.01}}
    args = _flatten_to_cli_args(d)
    assert "--optimizer.use_warmup=true" in args
    assert "--optimizer.lr=0.01" in args
