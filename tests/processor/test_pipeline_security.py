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

"""Loading processor metadata must not grant permission to execute custom code."""

import json
import sys
from dataclasses import dataclass
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from lerobot.processor import (
    DataProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    pipeline as pipeline_module,
)


@pytest.fixture(params=["config", "directory", "file", "hub"])
def load_pipeline(request, tmp_path, monkeypatch):
    def load(config, **kwargs):
        if request.param == "config":
            return DataProcessorPipeline.from_config(config, **kwargs)
        config_path = tmp_path / "processor.json"
        config_path.write_text(json.dumps(config))
        if request.param == "hub":
            monkeypatch.setattr(pipeline_module, "hf_hub_download", lambda **_: str(config_path))
            source = "attacker/policy"
        else:
            source = tmp_path if request.param == "directory" else config_path
        return DataProcessorPipeline.from_pretrained(source, config_filename="processor.json", **kwargs)

    return load


@pytest.mark.parametrize("consent", [False, True])
def test_external_module_import_requires_consent(load_pipeline, tmp_path, monkeypatch, consent):
    module_name = "external_processor_security_test"
    marker = tmp_path / "imported"
    (tmp_path / f"{module_name}.py").write_text(
        "from pathlib import Path\n"
        "from lerobot.processor import ProcessorStep\n"
        f"Path({str(marker)!r}).touch()\n"
        "class CustomStep(ProcessorStep):\n"
        "    def __call__(self, transition): return transition\n"
        "    def transform_features(self, features): return features\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    config = {"steps": [{"class": f"{module_name}.CustomStep"}]}
    try:
        if consent:
            result = load_pipeline(config, trust_remote_code=True)
            assert type(result.steps[0]).__name__ == "CustomStep"
            assert marker.exists()
        else:
            with pytest.raises(ValueError, match="requires `trust_remote_code=True`"):
                load_pipeline(config)
            assert not marker.exists()
            assert module_name not in sys.modules
    finally:
        sys.modules.pop(module_name, None)


def test_loaded_external_module_does_not_grant_consent(load_pipeline, tmp_path, monkeypatch):
    marker = tmp_path / "getattr_executed"
    module = ModuleType("already_loaded_processor_security_test")

    def getattr_with_side_effect(name):
        marker.touch()
        return ProcessorStep

    module.__getattr__ = getattr_with_side_effect
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(ValueError, match="requires `trust_remote_code=True`"):
        load_pipeline({"steps": [{"class": f"{module.__name__}.CustomStep"}]})
    assert not marker.exists()


@pytest.mark.parametrize("consent", [False, True])
def test_popen_is_rejected_before_construction(load_pipeline, monkeypatch, consent):
    import subprocess

    constructor = MagicMock()
    monkeypatch.setattr(subprocess.Popen, "__init__", constructor)
    expected = TypeError if consent else ValueError
    with pytest.raises(expected):
        load_pipeline(
            {"steps": [{"class": "subprocess.Popen", "config": {"args": [sys.executable]}}]},
            trust_remote_code=consent,
        )
    constructor.assert_not_called()


@pytest.mark.parametrize("path", [None, 7, [], "", "Popen", ".Step", "pkg.", "pkg..Step", "pkg/evil.Step"])
def test_malformed_class_path_fails_cleanly(path):
    with pytest.raises(ValueError, match="Invalid class path"):
        DataProcessorPipeline.from_config({"steps": [{"class": path}]})


def test_registry_and_builtin_legacy_paths_need_no_consent(load_pipeline):
    for entry in (
        {"registry_name": "rename_observations_processor", "config": {"rename_map": {}}},
        {"class": "lerobot.processor.rename_processor.RenameObservationsProcessorStep", "config": {}},
    ):
        assert len(load_pipeline({"steps": [entry]}).steps) == 1


@pytest.mark.parametrize("target", [dict, lambda: None])
def test_registry_rejects_non_processor_targets(monkeypatch, target):
    monkeypatch.setitem(ProcessorStepRegistry._registry, "invalid_security_step", target)
    with pytest.raises(TypeError, match="not a valid ProcessorStep subclass"):
        DataProcessorPipeline.from_config({"steps": [{"registry_name": "invalid_security_step"}]})


@dataclass
class ConsentStep(ProcessorStep):
    # A future registered step with an unsafe default cannot bypass the outer flag.
    trust_remote_code: bool = True

    def __post_init__(self):
        self.loader(trust_remote_code=self.trust_remote_code)

    def __call__(self, transition):
        return transition

    def transform_features(self, features):
        return features


@pytest.mark.parametrize("via_override", [False, True])
def test_nested_config_cannot_grant_consent(load_pipeline, monkeypatch, via_override):
    loader = MagicMock()
    monkeypatch.setattr(ConsentStep, "loader", loader, raising=False)
    monkeypatch.setitem(ProcessorStepRegistry._registry, "consent_security_step", ConsentStep)
    config = {"steps": [{"registry_name": "consent_security_step", "config": {}}]}
    kwargs = {}
    if via_override:
        kwargs["overrides"] = {"consent_security_step": {"trust_remote_code": True}}
    else:
        config["steps"][0]["config"]["trust_remote_code"] = True
    with pytest.raises(ValueError, match="requested `trust_remote_code=True`"):
        load_pipeline(config, **kwargs)
    loader.assert_not_called()


@pytest.mark.parametrize("consent", [False, True])
def test_step_default_cannot_override_caller_consent(load_pipeline, monkeypatch, consent):
    loader = MagicMock()
    monkeypatch.setattr(ConsentStep, "loader", loader, raising=False)
    monkeypatch.setitem(ProcessorStepRegistry._registry, "consent_security_step", ConsentStep)
    result = load_pipeline({"steps": [{"registry_name": "consent_security_step"}]}, trust_remote_code=consent)
    assert result.steps[0].trust_remote_code is consent
    loader.assert_called_once_with(trust_remote_code=consent)


@pytest.mark.parametrize("consent", [False, True])
def test_action_tokenizer_loaders_receive_caller_consent(load_pipeline, monkeypatch, consent):
    from lerobot.processor import tokenizer_processor

    processor = MagicMock()
    tokenizer = MagicMock()
    monkeypatch.setattr(tokenizer_processor, "_transformers_available", True)
    monkeypatch.setattr(tokenizer_processor, "AutoProcessor", processor)
    monkeypatch.setattr(tokenizer_processor, "AutoTokenizer", tokenizer)
    result = load_pipeline(
        {
            "steps": [
                {
                    "registry_name": "action_tokenizer_processor",
                    "config": {
                        "action_tokenizer_name": "attacker/action",
                        "paligemma_tokenizer_name": "attacker/text",
                    },
                }
            ]
        },
        trust_remote_code=consent,
    )
    assert result.steps[0].trust_remote_code is consent
    assert processor.from_pretrained.call_args.kwargs["trust_remote_code"] is consent
    assert tokenizer.from_pretrained.call_args.kwargs["trust_remote_code"] is consent


@pytest.mark.parametrize("consent", [False, True])
def test_groot_lazy_loaders_honor_pipeline_consent(load_pipeline, monkeypatch, consent):
    from lerobot.policies.groot import processor_groot

    loaders = [MagicMock() for _ in range(3)]
    monkeypatch.setattr(processor_groot, "require_package", lambda *args, **kwargs: None)
    for name, loader in zip(
        ("AutoTokenizer", "Qwen2VLImageProcessor", "Qwen3VLVideoProcessor"), loaders, strict=True
    ):
        monkeypatch.setattr(processor_groot, name, loader)
    monkeypatch.setattr(processor_groot, "Qwen3VLProcessor", MagicMock())
    pipeline = load_pipeline(
        {
            "steps": [
                {"registry_name": "groot_n1_7_vlm_encode_v1", "config": {"model_name": "attacker/model"}}
            ]
        },
        trust_remote_code=consent,
    )
    assert pipeline.steps[0].proc is not None
    for loader in loaders:
        loader.from_pretrained.assert_called_once_with("attacker/model", trust_remote_code=consent)


@pytest.mark.parametrize("consent", [False, True])
def test_wallx_lazy_loaders_honor_pipeline_consent(load_pipeline, monkeypatch, consent):
    from lerobot.policies.wall_x import processor_wall_x

    loader = MagicMock()
    monkeypatch.setattr(processor_wall_x, "require_package", lambda *args, **kwargs: None)
    monkeypatch.setattr(processor_wall_x, "AutoProcessor", loader)
    pipeline = load_pipeline(
        {
            "steps": [
                {
                    "registry_name": "wall_x_tokenizer",
                    "config": {
                        "processor_name": "attacker/model",
                        "image_keys": [],
                        "chunk_size": 1,
                        "max_state_dim": 1,
                        "max_action_dim": 1,
                        "output_action_dim": 1,
                        "use_fast_tokenizer": True,
                        "action_tokenizer_name": "attacker/actions",
                    },
                }
            ]
        },
        trust_remote_code=consent,
    )
    pipeline.steps[0]._get_processors()
    assert loader.from_pretrained.call_count == 2
    assert all(call.kwargs["trust_remote_code"] is consent for call in loader.from_pretrained.call_args_list)
