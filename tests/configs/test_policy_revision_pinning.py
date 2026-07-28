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

import json
from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig


@PreTrainedConfig.register_subclass("revision_pinning_test")
@dataclass
class RevisionPinningTestConfig(PreTrainedConfig):
    @property
    def observation_delta_indices(self) -> list | None:
        return None

    @property
    def action_delta_indices(self) -> list | None:
        return None

    @property
    def reward_delta_indices(self) -> list | None:
        return None

    def get_optimizer_preset(self):
        raise NotImplementedError

    def get_scheduler_preset(self):
        raise NotImplementedError

    def validate_features(self) -> None:
        pass


def test_pretrained_config_pins_resolved_hub_commit(monkeypatch, tmp_path):
    commit_hash = "a" * 40
    snapshot_dir = tmp_path / "models--user--policy" / "snapshots" / commit_hash
    RevisionPinningTestConfig(device="cpu").save_pretrained(snapshot_dir)
    calls = []

    def fake_hub_download(**kwargs):
        calls.append(kwargs)
        return str(snapshot_dir / "config.json")

    monkeypatch.setattr("lerobot.configs.policies.hf_hub_download", fake_hub_download)

    config = PreTrainedConfig.from_pretrained("user/policy", revision="main")

    assert calls[0]["revision"] == "main"
    assert config._commit_hash == commit_hash
    assert config._commit_hash_source == "user/policy"
    assert config.get_hub_revision("user/policy", "main") == commit_hash
    assert config.get_hub_revision("user/base-policy", "base-tag") == "base-tag"


def test_runtime_commit_hash_is_not_serialized(tmp_path):
    config = RevisionPinningTestConfig(device="cpu")
    config._set_hub_commit_hash("a" * 40, "user/policy")

    config.save_pretrained(tmp_path)

    serialized_config = json.loads((tmp_path / "config.json").read_text())
    assert "_commit_hash" not in serialized_config
    assert "_commit_hash_source" not in serialized_config
    assert "_runtime_commit_hash" not in serialized_config
    assert "_runtime_commit_hash_source" not in serialized_config
