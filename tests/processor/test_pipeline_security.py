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

"""Untrusted processor configs cannot authorize imports or constructor code."""

import json
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from lerobot.processor import DataProcessorPipeline, ProcessorStepRegistry, pipeline as pipeline_module
from lerobot.processor.pipeline import IdentityProcessorStep


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
