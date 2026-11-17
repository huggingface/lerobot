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
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition
from lerobot.processor import pipeline as pipeline_module
from lerobot.processor.pipeline import (
    DataProcessorPipeline,
    IdentityProcessorStep,
    ProcessorMigrationError,
    ProcessorStep,
    ProcessorStepRegistry,
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
    """Test resolution with invalid import path when trust_remote_code=True."""
    step_entry = {"class": "nonexistent.module.ClassName"}

    with pytest.raises(ImportError, match="Failed to load processor step"):
        DataProcessorPipeline._resolve_step_class(step_entry, trust_remote_code=True)


def test_resolve_step_class_untrusted_import_path():
    """Test resolution with untrusted custom import path raises ValueError."""
    step_entry = {"class": "nonexistent.module.ClassName"}

    with pytest.raises(ValueError, match="requires `trust_remote_code=True`"):
        DataProcessorPipeline._resolve_step_class(step_entry, trust_remote_code=False)


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


# Caller consent for processor imports and construction.


@pytest.fixture(params=["config", "hub"])
def load_pipeline(request, tmp_path, monkeypatch):
    def load(step, **kwargs):
        config = {"steps": [step]}
        if request.param == "config":
            return DataProcessorPipeline.from_config(config, **kwargs)
        path = tmp_path / "processor.json"
        path.write_text(json.dumps(config))
        monkeypatch.setattr(pipeline_module, "hf_hub_download", lambda **_: str(path))
        return DataProcessorPipeline.from_pretrained(
            "user/policy", config_filename="processor.json", **kwargs
        )

    return load


@pytest.mark.parametrize("consent", [False, True])
def test_import_side_effects_require_consent(load_pipeline, tmp_path, monkeypatch, consent):
    marker = tmp_path / "imported"
    (tmp_path / "external_step.py").write_text(
        "from pathlib import Path\nfrom lerobot.processor.pipeline import IdentityProcessorStep\n"
        f"Path({str(marker)!r}).touch()\nclass CustomStep(IdentityProcessorStep): pass\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        if consent:
            assert (
                len(load_pipeline({"class": "external_step.CustomStep"}, trust_remote_code=True).steps) == 1
            )
        else:
            with pytest.raises(ValueError, match="requires `trust_remote_code=True`"):
                load_pipeline({"class": "external_step.CustomStep"})
        assert marker.exists() is consent
    finally:
        sys.modules.pop("external_step", None)


def test_loaded_module_does_not_grant_consent(monkeypatch):
    module = ModuleType("external_step")
    module.__getattr__ = MagicMock(return_value=IdentityProcessorStep)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(ValueError, match="requires `trust_remote_code=True`"):
        DataProcessorPipeline.from_config({"steps": [{"class": "external_step.CustomStep"}]})
    module.__getattr__.assert_not_called()


@pytest.mark.parametrize("consent", [False, True])
def test_popen_never_reaches_construction(load_pipeline, monkeypatch, consent):
    import subprocess

    constructor = MagicMock()
    monkeypatch.setattr(subprocess.Popen, "__init__", constructor)
    with pytest.raises(TypeError if consent else ValueError):
        load_pipeline({"class": "subprocess.Popen"}, trust_remote_code=consent)
    constructor.assert_not_called()


@pytest.mark.parametrize("consent", [False, True])
@pytest.mark.parametrize("requested", ["default", "config", "override"])
def test_step_cannot_grant_its_own_consent(load_pipeline, monkeypatch, consent, requested):
    loader = MagicMock()

    class ConsentStep(IdentityProcessorStep):
        def __init__(self, trust_remote_code=True):
            loader(trust_remote_code=trust_remote_code)

    monkeypatch.setitem(ProcessorStepRegistry._registry, "consent_step", ConsentStep)
    step = {"registry_name": "consent_step", "config": {}}
    overrides = {"consent_step": {"trust_remote_code": True}} if requested == "override" else {}
    if requested == "config":
        step["config"]["trust_remote_code"] = True
    if not consent and requested != "default":
        with pytest.raises(ValueError, match="without pipeline consent"):
            load_pipeline(step, overrides=overrides)
        loader.assert_not_called()
    else:
        load_pipeline(step, overrides=overrides, trust_remote_code=consent)
        loader.assert_called_once_with(trust_remote_code=consent)


@pytest.mark.parametrize("consent", [False, True])
def test_action_tokenizer_loaders_receive_consent(load_pipeline, monkeypatch, consent):
    from lerobot.processor import tokenizer_processor

    monkeypatch.setattr(tokenizer_processor, "_transformers_available", True)
    loaders = [MagicMock(), MagicMock()]
    monkeypatch.setattr(tokenizer_processor, "AutoProcessor", loaders[0])
    monkeypatch.setattr(tokenizer_processor, "AutoTokenizer", loaders[1])
    load_pipeline(
        {
            "registry_name": "action_tokenizer_processor",
            "config": {"action_tokenizer_name": "user/action", "paligemma_tokenizer_name": "user/text"},
        },
        trust_remote_code=consent,
    )
    for loader in loaders:
        assert loader.from_pretrained.call_args.kwargs["trust_remote_code"] is consent


@pytest.mark.parametrize("consent", [False, True])
def test_reward_classifier_pipeline_consent_reaches_encoder(load_pipeline, monkeypatch, tmp_path, consent):
    transformers = pytest.importorskip("transformers")
    from lerobot.configs import rewards as rewards_module
    from lerobot.rewards.classifier.configuration_classifier import RewardClassifierConfig  # noqa: F401

    config_file = tmp_path / "classifier_config.json"
    config_file.write_text(
        json.dumps(
            {
                "type": "reward_classifier",
                "device": "cpu",
                "trust_remote_code": True,
                "model_name": "user/custom-encoder",
                "input_features": {"observation.images.top": {"type": "VISUAL", "shape": [3, 16, 16]}},
            }
        )
    )
    monkeypatch.setattr(rewards_module, "hf_hub_download", lambda **kwargs: str(config_file))
    # Keep pipeline loading, classifier config parsing and model construction real;
    # stop only at the external encoder loader, before downloading weights.
    encoder_loader = MagicMock(side_effect=RuntimeError("reached encoder loader"))
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", encoder_loader)

    with pytest.raises(ValueError, match="reached encoder loader"):
        load_pipeline(
            {
                "registry_name": "reward_classifier_processor",
                "config": {"pretrained_path": "user/classifier"},
            },
            trust_remote_code=consent,
        )

    encoder_loader.assert_called_once_with("user/custom-encoder", trust_remote_code=consent)
