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

import torch

from lerobot.policies import factory
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline


class _MinimalPolicy(PreTrainedPolicy):
    config_class = ACTConfig
    name = "minimal_revision_test"

    def get_optim_params(self) -> dict:
        return {}

    def reset(self) -> None:
        pass

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict | None]:
        return torch.tensor(0), None

    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        return torch.tensor(0)

    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        return torch.tensor(0)


def _skip_safetensor_loading(monkeypatch):
    monkeypatch.setattr(
        _MinimalPolicy,
        "_load_as_safetensor",
        classmethod(lambda cls, model, model_file, map_location, strict: model),
    )


def test_policy_weights_use_config_commit_hash(monkeypatch):
    config = ACTConfig(device="cpu")
    config._set_hub_commit_hash("a" * 40, "user/policy")
    calls = []

    def fake_hub_download(**kwargs):
        calls.append(kwargs)
        return "/unused/model.safetensors"

    monkeypatch.setattr("lerobot.policies.pretrained.hf_hub_download", fake_hub_download)
    _skip_safetensor_loading(monkeypatch)

    _MinimalPolicy.from_pretrained("user/policy", config=config, revision="main")

    assert calls[0]["revision"] == "a" * 40


def test_policy_does_not_reuse_commit_hash_for_another_repo(monkeypatch):
    config = ACTConfig(device="cpu")
    config._set_hub_commit_hash("a" * 40, "user/adapter")
    calls = []

    def fake_hub_download(**kwargs):
        calls.append(kwargs)
        return "/unused/model.safetensors"

    monkeypatch.setattr("lerobot.policies.pretrained.hf_hub_download", fake_hub_download)
    _skip_safetensor_loading(monkeypatch)

    _MinimalPolicy.from_pretrained("user/base-policy", config=config, revision="base-tag")

    assert calls[0]["revision"] == "base-tag"


def test_policy_records_weight_commit_for_explicit_config(monkeypatch):
    commit_hash = "a" * 40
    config = ACTConfig(device="cpu")

    def fake_hub_download(**kwargs):
        return f"/cache/models--user--policy/snapshots/{commit_hash}/model.safetensors"

    monkeypatch.setattr("lerobot.policies.pretrained.hf_hub_download", fake_hub_download)
    _skip_safetensor_loading(monkeypatch)

    _MinimalPolicy.from_pretrained("user/policy", config=config, revision="main")

    assert config._commit_hash == commit_hash
    assert config._commit_hash_source == "user/policy"


def test_processor_factory_uses_config_commit_hash(monkeypatch):
    config = ACTConfig(device="cpu")
    config._set_hub_commit_hash("a" * 40, "user/policy")
    calls = []

    def fake_from_pretrained(cls, **kwargs):
        calls.append(kwargs)
        return PolicyProcessorPipeline(steps=[])

    monkeypatch.setattr(
        factory.PolicyProcessorPipeline,
        "from_pretrained",
        classmethod(fake_from_pretrained),
    )

    factory.make_pre_post_processors(
        config,
        pretrained_path="user/policy",
        pretrained_revision="main",
    )

    assert [call["revision"] for call in calls] == ["a" * 40, "a" * 40]
