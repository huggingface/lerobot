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

import inspect
import json
import tempfile
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition
from lerobot.processor.pipeline import (
    DataProcessorPipeline,
    ProcessorMigrationError,
    ProcessorStep,
    ProcessorStepRegistry,
)


class ArtifactOverrideStep(ProcessorStep):
    """Accept replacement values unchanged and keep artifact and tensor state separate."""

    def __init__(self, tokenizer_name: Any = None, auxiliary: Any = None, tokenizer: Any = None) -> None:
        self.tokenizer_name = tokenizer_name
        self.auxiliary = auxiliary
        self.tokenizer = tokenizer
        self.value = torch.tensor(0)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def get_config(self) -> dict[str, Any]:
        return {"tokenizer_name": self.tokenizer_name, "auxiliary": self.auxiliary}

    def save_artifacts(self, save_directory: Path) -> dict[str, str]:
        filename = "saved-tokenizer.txt"
        (save_directory / filename).write_text(Path(self.tokenizer_name).read_text())
        return {"tokenizer_name": filename}

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"value": self.value}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.value = state["value"]


@pytest.fixture
def artifact_step_registry() -> Iterator[str]:
    name = "artifact_override_test_step"
    ProcessorStepRegistry.register(name)(ArtifactOverrideStep)
    try:
        yield name
    finally:
        ProcessorStepRegistry.unregister(name)
        del ArtifactOverrideStep._registry_name


def _artifact_override_entry(selector: str) -> dict[str, Any]:
    return {
        "registry_name": selector,
        "config": {"tokenizer_name": "obsolete"},
        "artifacts": {"tokenizer_name": "obsolete"},
    }


def _write_artifact_checkpoint(path: Path, *entries: dict[str, Any]) -> Path:
    config_path = path / "processor.json"
    config_path.write_text(json.dumps({"name": "ArtifactOverrides", "steps": entries}))
    return config_path


@pytest.mark.parametrize("source", ["directory", "file"])
@pytest.mark.parametrize("replacement", ["replacement", None, False, 0, "", [], {}])
def test_artifact_override_local(
    tmp_path: Path, artifact_step_registry: str, source: str, replacement: Any
) -> None:
    config_path = _write_artifact_checkpoint(tmp_path, _artifact_override_entry(artifact_step_registry))
    original_json = config_path.read_bytes()
    overrides = {artifact_step_registry: {"tokenizer_name": replacement}}
    original_overrides = deepcopy(overrides)

    pipeline = DataProcessorPipeline.from_pretrained(
        tmp_path if source == "directory" else config_path,
        config_filename="processor.json" if source == "directory" else "ignored.json",
        overrides=overrides,
    )

    assert pipeline.steps[0].tokenizer_name is replacement
    assert overrides == original_overrides
    assert config_path.read_bytes() == original_json


@pytest.mark.parametrize("required_artifact", [None, "remaining", "remaining.json"])
def test_artifact_override_hub(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_step_registry: str,
    required_artifact: str | None,
) -> None:
    entry = _artifact_override_entry(artifact_step_registry)
    if required_artifact is not None:
        entry["artifacts"]["auxiliary"] = required_artifact
    config_path = _write_artifact_checkpoint(tmp_path, entry)
    config_calls = []
    artifact_calls = []
    hub_kwargs = {
        "force_download": False,
        "token": None,
        "cache_dir": str(tmp_path / "cache"),
        "local_files_only": True,
        "revision": "test-revision",
    }

    def fake_config_download(**kwargs: Any) -> str:
        inspect.signature(hf_hub_download).bind(**kwargs)
        config_calls.append(kwargs)
        return str(config_path)

    def fake_snapshot_download(**kwargs: Any) -> str:
        artifact_calls.append(kwargs)
        if required_artifact is None:
            pytest.fail("overridden obsolete artifact must not be downloaded")
        expected_pattern = f"nested/{required_artifact}"
        if not Path(required_artifact).suffix:
            expected_pattern += "/**"
        assert kwargs == {
            "repo_id": "test/artifact-overrides",
            "repo_type": "model",
            "allow_patterns": expected_pattern,
            **hub_kwargs,
        }
        target = tmp_path / required_artifact
        if target.suffix:
            target.write_text("required")
        else:
            target.mkdir()
        return str(tmp_path)

    monkeypatch.setattr("lerobot.processor.pipeline.hf_hub_download", fake_config_download)
    monkeypatch.setattr("lerobot.processor.pipeline.snapshot_download", fake_snapshot_download)
    pipeline = DataProcessorPipeline.from_pretrained(
        "test/artifact-overrides",
        config_filename="nested/processor.json",
        overrides={artifact_step_registry: {"tokenizer_name": "replacement"}},
        **hub_kwargs,
    )

    assert pipeline.steps[0].tokenizer_name == "replacement"
    assert config_calls == [
        {
            "repo_id": "test/artifact-overrides",
            "filename": "nested/processor.json",
            "repo_type": "model",
            **hub_kwargs,
        }
    ]
    assert len(artifact_calls) == (required_artifact is not None)
    if required_artifact is not None:
        assert pipeline.steps[0].auxiliary == str(tmp_path / required_artifact)


@pytest.mark.parametrize("present", [False, True])
@pytest.mark.parametrize("shared_path", [False, True])
def test_artifact_override_partial_local(
    tmp_path: Path, artifact_step_registry: str, present: bool, shared_path: bool
) -> None:
    entry = _artifact_override_entry(artifact_step_registry)
    remaining = "obsolete" if shared_path else "remaining"
    entry["artifacts"]["auxiliary"] = remaining
    _write_artifact_checkpoint(tmp_path, entry)
    if present:
        (tmp_path / remaining).write_text("required")
    kwargs = {
        "config_filename": "processor.json",
        "overrides": {artifact_step_registry: {"tokenizer_name": "new"}},
    }
    if not present:
        with pytest.raises(FileNotFoundError, match=f"Missing processor artifact '{remaining}'"):
            DataProcessorPipeline.from_pretrained(tmp_path, **kwargs)
    else:
        pipeline = DataProcessorPipeline.from_pretrained(tmp_path, **kwargs)
        assert pipeline.steps[0].tokenizer_name == "new"
        assert pipeline.steps[0].auxiliary == str(tmp_path / remaining)


@pytest.mark.parametrize(
    "override_kind", ["none", "empty", "step-empty", "object", "wrong-selector", "exact"]
)
@pytest.mark.parametrize("present", [False, True])
def test_artifact_override_controls(
    tmp_path: Path, artifact_step_registry: str, override_kind: str, present: bool
) -> None:
    _write_artifact_checkpoint(tmp_path, _artifact_override_entry(artifact_step_registry))
    tokenizer = object()
    overrides = {
        "none": None,
        "empty": {},
        "step-empty": {artifact_step_registry: {}},
        "object": {artifact_step_registry: {"tokenizer": tokenizer}},
        "wrong-selector": {"unknown_step": {"tokenizer_name": "new"}},
        "exact": {artifact_step_registry: {"tokenizer_name": "new"}},
    }[override_kind]
    if present:
        (tmp_path / "obsolete").write_text("saved")
    kwargs = {"config_filename": "processor.json", "overrides": overrides}
    if not present and override_kind != "exact":
        with pytest.raises(FileNotFoundError, match="Missing processor artifact 'obsolete'"):
            DataProcessorPipeline.from_pretrained(tmp_path, **kwargs)
    elif override_kind == "wrong-selector":
        with pytest.raises(KeyError, match="do not match any step"):
            DataProcessorPipeline.from_pretrained(tmp_path, **kwargs)
    else:
        pipeline = DataProcessorPipeline.from_pretrained(tmp_path, **kwargs)
        assert pipeline.steps[0].tokenizer_name == (
            "new" if override_kind == "exact" else str(tmp_path / "obsolete")
        )
        if override_kind == "object":
            assert pipeline.steps[0].tokenizer is tokenizer


@pytest.mark.parametrize("selector_kind", ["class", "both", "repeated"])
def test_artifact_override_selectors(tmp_path: Path, artifact_step_registry: str, selector_kind: str) -> None:
    entry = _artifact_override_entry(artifact_step_registry)
    selector = artifact_step_registry
    if selector_kind == "class":
        del entry["registry_name"]
        entry["class"] = f"{__name__}.ArtifactOverrideStep"
        selector = "ArtifactOverrideStep"
    elif selector_kind == "both":
        entry["class"] = "must.not.be.imported.ArtifactOverrideStep"
    entries = [entry, deepcopy(entry)] if selector_kind == "repeated" else [entry]
    _write_artifact_checkpoint(tmp_path, *entries)
    pipeline = DataProcessorPipeline.from_pretrained(
        tmp_path, config_filename="processor.json", overrides={selector: {"tokenizer_name": "new"}}
    )
    assert len(pipeline.steps) == len(entries)
    assert all(step.tokenizer_name == "new" for step in pipeline.steps)


@pytest.mark.parametrize("selector", ["unknown_registry", "", None])
def test_artifact_override_registry_presence(tmp_path: Path, selector: str | None) -> None:
    entry = _artifact_override_entry(selector)
    entry["class"] = f"{__name__}.ArtifactOverrideStep"
    _write_artifact_checkpoint(tmp_path, entry)
    with pytest.raises(ImportError, match="Failed to load processor step from registry"):
        DataProcessorPipeline.from_pretrained(
            tmp_path, config_filename="processor.json", overrides={selector: {"tokenizer_name": "new"}}
        )


@pytest.mark.parametrize("present", [False, True])
@pytest.mark.parametrize("invalid_overrides", [None, [], ["tokenizer_name"], "tokenizer_name"])
def test_artifact_override_invalid_container(
    tmp_path: Path, artifact_step_registry: str, invalid_overrides: Any, present: bool
) -> None:
    _write_artifact_checkpoint(tmp_path, _artifact_override_entry(artifact_step_registry))
    if present:
        (tmp_path / "obsolete").write_text("saved")
    error = ValueError if present else FileNotFoundError
    message = "Failed to instantiate processor step" if present else "Missing processor artifact 'obsolete'"
    with pytest.raises(error, match=message):
        DataProcessorPipeline.from_pretrained(
            tmp_path, config_filename="processor.json", overrides={artifact_step_registry: invalid_overrides}
        )


def test_artifact_override_unknown_constructor_key(tmp_path: Path, artifact_step_registry: str) -> None:
    entry = _artifact_override_entry(artifact_step_registry)
    entry["artifacts"] = {"unsupported": "obsolete"}
    _write_artifact_checkpoint(tmp_path, entry)
    with pytest.raises(ValueError, match="unexpected keyword argument 'unsupported'"):
        DataProcessorPipeline.from_pretrained(
            tmp_path,
            config_filename="processor.json",
            overrides={artifact_step_registry: {"unsupported": "new"}},
        )


@pytest.mark.parametrize(
    ("artifacts", "error"),
    [
        ({"tokenizer_name": "/absolute"}, ValueError),
        ({"tokenizer_name": "../escape"}, ValueError),
        ({"tokenizer_name": "safe/../../escape"}, ValueError),
        ({"tokenizer_name": None}, TypeError),
        ({"tokenizer_name": 3}, TypeError),
        ({"tokenizer_name": []}, TypeError),
        ({"tokenizer_name": {}}, TypeError),
        (None, AttributeError),
        ([], AttributeError),
        ("tokenizer_name", AttributeError),
    ],
)
def test_artifact_override_validates_declarations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_step_registry: str,
    artifacts: Any,
    error: type[Exception],
) -> None:
    entry = _artifact_override_entry(artifact_step_registry)
    entry["artifacts"] = artifacts
    _write_artifact_checkpoint(tmp_path, entry)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("invalid declarations must fail before class resolution or Hub access")

    monkeypatch.setattr(DataProcessorPipeline, "_resolve_step_class", forbidden)
    monkeypatch.setattr("lerobot.processor.pipeline.snapshot_download", forbidden)
    monkeypatch.setattr("lerobot.processor.pipeline.hf_hub_download", forbidden)
    with pytest.raises(error):
        DataProcessorPipeline.from_pretrained(
            tmp_path,
            config_filename="processor.json",
            overrides={artifact_step_registry: {"tokenizer_name": "new"}},
        )


@pytest.mark.parametrize("saved_config", [None, []])
def test_artifact_override_preserves_invalid_config_error(
    tmp_path: Path, artifact_step_registry: str, saved_config: Any
) -> None:
    entry = _artifact_override_entry(artifact_step_registry)
    entry["config"] = saved_config
    _write_artifact_checkpoint(tmp_path, entry)
    with pytest.raises(ValueError, match="Failed to instantiate processor step"):
        DataProcessorPipeline.from_pretrained(
            tmp_path,
            config_filename="processor.json",
            overrides={artifact_step_registry: {"tokenizer_name": "new"}},
        )


def test_artifact_override_lifecycle(tmp_path: Path, artifact_step_registry: str) -> None:
    original = tmp_path / "original.txt"
    original.write_text("original")
    step = ArtifactOverrideStep(tokenizer_name=str(original))
    step.value = torch.tensor(7)
    pipeline = DataProcessorPipeline([step], name="ArtifactLifecycle")
    checkpoint = tmp_path / "checkpoint"
    pipeline.save_pretrained(checkpoint, config_filename="processor.json")
    config_path = checkpoint / "processor.json"
    original_json = config_path.read_bytes()
    config = json.loads(original_json)
    original_config = deepcopy(config)
    replacement = tmp_path / "replacement.txt"
    replacement.write_text("replacement")
    overrides = {artifact_step_registry: {"tokenizer_name": str(replacement)}}
    original_overrides = deepcopy(overrides)

    loaded = DataProcessorPipeline.from_pretrained(
        checkpoint, config_filename="processor.json", overrides=overrides
    )
    assert loaded.steps[0].tokenizer_name == str(replacement)
    assert loaded.steps[0].value.item() == 7
    second_checkpoint = tmp_path / "second"
    loaded.save_pretrained(second_checkpoint, config_filename="processor.json")
    reloaded = DataProcessorPipeline.from_pretrained(second_checkpoint, config_filename="processor.json")
    assert Path(reloaded.steps[0].tokenizer_name).read_text() == "replacement"
    assert reloaded.steps[0].value.item() == 7

    # In-memory construction does not resolve checkpoint-relative artifact declarations.
    in_memory = DataProcessorPipeline.from_config(config, state_dict=pipeline.state_dict())
    assert in_memory.steps[0].tokenizer_name == "saved-tokenizer.txt"
    assert in_memory.steps[0].value.item() == 7
    DataProcessorPipeline._build_steps_with_overrides(
        config, overrides, str(checkpoint), checkpoint, "processor.json", {}, is_local_source=True
    )
    assert config == original_config
    assert overrides == original_overrides
    assert config_path.read_bytes() == original_json


def test_artifact_override_preserves_missing_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, artifact_step_registry: str
) -> None:
    entry = _artifact_override_entry(artifact_step_registry)
    entry["state_file"] = "missing.safetensors"
    _write_artifact_checkpoint(tmp_path, entry)

    def forbidden(**kwargs: Any) -> None:
        pytest.fail("local artifact replacement must not download missing tensor state")

    monkeypatch.setattr("lerobot.processor.pipeline.hf_hub_download", forbidden)
    with pytest.raises(FileNotFoundError, match="missing.safetensors.*local processor pipeline"):
        DataProcessorPipeline.from_pretrained(
            tmp_path,
            config_filename="processor.json",
            overrides={artifact_step_registry: {"tokenizer_name": "new"}},
        )


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


def test_load_config_legacy_hub_policy_suggests_revision_aware_migration(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "type": "diffusion",
                "input_features": {"observation.state": {"type": "STATE", "shape": [2]}},
                "output_features": {"action": {"type": "ACTION", "shape": [2]}},
            }
        )
    )

    def fake_hf_hub_download(*, filename, **kwargs):  # noqa: ARG001
        if filename == "config.json":
            return str(config_path)
        raise FileNotFoundError(filename)

    monkeypatch.setattr("lerobot.processor.pipeline.hf_hub_download", fake_hf_hub_download)

    with pytest.raises(ProcessorMigrationError) as exc_info:
        DataProcessorPipeline._load_config(
            "lerobot/diffusion_pusht", "policy_preprocessor.json", {"revision": "legacy"}
        )

    assert "--revision legacy" in exc_info.value.migration_command


def test_hub_migration_detection_is_conservative(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "type": "diffusion",
                "input_features": {},
                "output_features": {"action": {"type": "ACTION", "shape": [2]}},
            }
        )
    )
    monkeypatch.setattr("lerobot.processor.pipeline.hf_hub_download", lambda **kwargs: str(config_path))
    assert not DataProcessorPipeline._hub_model_requires_migration("someone/model", {})

    def fail_lookup(**kwargs):  # noqa: ARG001
        raise ValueError("invalid repository")

    monkeypatch.setattr("lerobot.processor.pipeline.hf_hub_download", fail_lookup)
    assert not DataProcessorPipeline._hub_model_requires_migration("invalid", {})


def test_from_pretrained_local_directory_missing_state_does_not_call_hub(monkeypatch):
    """Local processor dirs must fail locally when a state file is missing."""

    @ProcessorStepRegistry.register("local_missing_state_step")
    class LocalMissingStateStep(ProcessorStep):
        def __call__(self, transition: EnvTransition) -> EnvTransition:
            return transition

        def transform_features(
            self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
        ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
            return features

        def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
            pass

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config = {
                "name": "LocalMissingStatePipeline",
                "steps": [{"registry_name": "local_missing_state_step", "state_file": "missing.safetensors"}],
            }
            (tmp_path / "processor.json").write_text(json.dumps(config))

            def fail_hub_download(*args, **kwargs):
                pytest.fail("local missing processor state should not call hf_hub_download")

            monkeypatch.setattr("lerobot.processor.pipeline.hf_hub_download", fail_hub_download)

            with pytest.raises(FileNotFoundError, match="missing.safetensors.*local processor pipeline"):
                DataProcessorPipeline.from_pretrained(tmp_path, config_filename="processor.json")
    finally:
        ProcessorStepRegistry.unregister("local_missing_state_step")


def test_from_pretrained_local_config_file_missing_state_does_not_call_hub(monkeypatch):
    """Local single-file processor configs must also keep missing state resolution local."""

    @ProcessorStepRegistry.register("local_file_missing_state_step")
    class LocalFileMissingStateStep(ProcessorStep):
        def __call__(self, transition: EnvTransition) -> EnvTransition:
            return transition

        def transform_features(
            self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
        ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
            return features

        def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
            pass

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "processor.json"
            config = {
                "name": "LocalFileMissingStatePipeline",
                "steps": [
                    {"registry_name": "local_file_missing_state_step", "state_file": "missing.safetensors"}
                ],
            }
            config_path.write_text(json.dumps(config))

            def fail_hub_download(*args, **kwargs):
                pytest.fail("local missing processor state should not call hf_hub_download")

            monkeypatch.setattr("lerobot.processor.pipeline.hf_hub_download", fail_hub_download)

            with pytest.raises(FileNotFoundError, match="missing.safetensors.*local processor pipeline"):
                DataProcessorPipeline.from_pretrained(config_path, config_filename="ignored.json")
    finally:
        ProcessorStepRegistry.unregister("local_file_missing_state_step")


def test_from_pretrained_hub_source_missing_local_state_still_calls_hub(monkeypatch, tmp_path):
    """Hub sources still fall back to hf_hub_download for state files."""

    @ProcessorStepRegistry.register("hub_state_step")
    class HubStateStep(ProcessorStep):
        def __init__(self):
            self.value = torch.tensor(0)

        def __call__(self, transition: EnvTransition) -> EnvTransition:
            return transition

        def transform_features(
            self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
        ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
            return features

        def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
            self.value = state["value"]

    try:
        state_path = tmp_path / "downloaded.safetensors"
        save_file({"value": torch.tensor(7)}, state_path)
        loaded_config = {
            "name": "HubStatePipeline",
            "steps": [{"registry_name": "hub_state_step", "state_file": "hub_state.safetensors"}],
        }
        calls = []

        def fake_load_config(cls, model_id, config_filename, hub_download_kwargs):
            return loaded_config, tmp_path / "hub_cache"

        def fake_hub_download(**kwargs):
            inspect.signature(hf_hub_download).bind(**kwargs)
            calls.append(kwargs)
            return str(state_path)

        monkeypatch.setattr(DataProcessorPipeline, "_load_config", classmethod(fake_load_config))
        monkeypatch.setattr("lerobot.processor.pipeline.hf_hub_download", fake_hub_download)

        pipeline = DataProcessorPipeline.from_pretrained(
            "user/repo",
            config_filename="processor.json",
            resume_download=True,
            proxies={"https": "http://localhost:3128"},
        )

        assert calls == [
            {
                "repo_id": "user/repo",
                "filename": "hub_state.safetensors",
                "repo_type": "model",
                "force_download": False,
                "token": None,
                "cache_dir": None,
                "local_files_only": False,
                "revision": None,
            }
        ]
        assert pipeline.steps[0].value.item() == 7
    finally:
        ProcessorStepRegistry.unregister("hub_state_step")


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
