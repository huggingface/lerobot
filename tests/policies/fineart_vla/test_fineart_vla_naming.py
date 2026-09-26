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

"""FineART-VLA is canonical while existing checkpoint artifacts remain loadable."""

import json
from dataclasses import dataclass

import draccus
import pytest

from lerobot.configs import PreTrainedConfig
from lerobot.policies import FineARTVLAConfig, get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.policies.fineart_vla.text_processor_fineart_vla import FineARTVLATextTokenizerStep
from lerobot.processor import PolicyProcessorPipeline


@dataclass
class _CLIConfig:
    policy: PreTrainedConfig


@pytest.mark.parametrize("policy_type", ["fineart_vla", "pi052"])
def test_policy_factory_and_cli_use_canonical_name(policy_type):
    config = make_policy_config(policy_type, device="cpu", enable_fast_action_loss=False)
    parsed = draccus.parse(_CLIConfig, args=[f"--policy.type={policy_type}", "--policy.device=cpu"]).policy
    assert type(config) is type(parsed) is FineARTVLAConfig
    assert config.type == parsed.type == "fineart_vla"
    policy_class = get_policy_class(policy_type)
    assert policy_class.__name__ == "FineARTVLAPolicy"
    assert policy_class.name == "fineart_vla"
    assert policy_class.config_class is FineARTVLAConfig
    preprocessor, _ = make_pre_post_processors(config)
    assert any(isinstance(step, FineARTVLATextTokenizerStep) for step in preprocessor.steps)


@pytest.mark.parametrize("policy_type", ["fineart_vla", "pi052"])
def test_checkpoint_reload_resaves_canonical_type(tmp_path, policy_type):
    config = FineARTVLAConfig(device="cpu", flow_num_repeats=3)
    config.save_pretrained(tmp_path)
    path = tmp_path / "config.json"
    saved = json.loads(path.read_text())
    assert saved["type"] == "fineart_vla"
    saved["type"] = policy_type
    path.write_text(json.dumps(saved))

    restored = PreTrainedConfig.from_pretrained(tmp_path)
    assert type(restored) is FineARTVLAConfig
    assert restored.recipe == config.recipe
    assert restored.flow_num_repeats == 3
    restored.save_pretrained(tmp_path)
    assert json.loads(path.read_text())["type"] == "fineart_vla"


@pytest.mark.parametrize(
    "identifier",
    [
        {"registry_name": "fineart_vla_text_tokenizer"},
        {"registry_name": "pi052_text_tokenizer"},
        {"class": "lerobot.policies.pi052.text_processor_pi052.PI052TextTokenizerStep"},
    ],
)
def test_processor_artifacts_reload_and_resave_canonical_name(tmp_path, identifier):
    path = tmp_path / "policy_preprocessor.json"
    path.write_text(
        json.dumps(
            {
                "name": "policy_preprocessor",
                "steps": [{**identifier, "config": {"max_length": 77, "tokenizer_name": "custom-tokenizer"}}],
            }
        )
    )
    pipeline = PolicyProcessorPipeline.from_pretrained(tmp_path, config_filename=path.name)
    step = pipeline.steps[0]
    assert type(step) is FineARTVLATextTokenizerStep
    assert step.max_length == 77
    assert step.tokenizer_name == "custom-tokenizer"
    pipeline.save_pretrained(tmp_path)
    assert json.loads(path.read_text())["steps"][0]["registry_name"] == "fineart_vla_text_tokenizer"
