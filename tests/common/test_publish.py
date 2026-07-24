#!/usr/bin/env python

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
"""publish_trained_model: commit set, card, log-line contract, PEFT branch (hub fully mocked)."""

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

import lerobot.common.train_utils as train_utils
import lerobot.utils.hub as hub
from lerobot.common.train_utils import generate_model_card, publish_trained_model
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from tests.fixtures.dummy_checkpoint_policy import make_dummy_policy


class FakeHfApi:
    """Records every repo/upload interaction; shared across both HfApi import sites."""

    calls: list[dict] = []

    def __init__(self, *args, **kwargs):
        pass

    def create_repo(self, repo_id, private=None, exist_ok=False, **kwargs):
        return SimpleNamespace(repo_id=repo_id)

    def upload_folder(self, *, repo_id, folder_path, commit_message, **kwargs):
        FakeHfApi.calls.append(
            {
                "repo_id": repo_id,
                "commit_message": commit_message,
                "files": sorted(p.name for p in Path(folder_path).iterdir()),
                "ignore_patterns": kwargs.get("ignore_patterns"),
            }
        )
        return SimpleNamespace(repo_url=SimpleNamespace(url=f"https://huggingface.co/{repo_id}"))


@pytest.fixture
def mocked_hub(monkeypatch):
    FakeHfApi.calls = []
    monkeypatch.setattr(train_utils, "HfApi", FakeHfApi)
    monkeypatch.setattr(hub, "HfApi", FakeHfApi)
    # card.validate() hits the Hub; publishing must work offline in tests
    monkeypatch.setattr(train_utils.ModelCard, "validate", lambda self: None)
    return FakeHfApi


def make_cfg() -> TrainPipelineConfig:
    cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="user/dataset"))
    cfg.parallelism.resolve(1)
    return cfg


class RecordingProcessor:
    def __init__(self):
        self.pushed_to = None

    def push_to_hub(self, repo_id, **kwargs):
        self.pushed_to = repo_id


class TestPublishTrainedModel:
    def test_commit_set_and_log_contract(self, mocked_hub, caplog):
        policy = make_dummy_policy(repo_id="user/policy")
        pre, post = RecordingProcessor(), RecordingProcessor()
        with caplog.at_level(logging.INFO):
            publish_trained_model(make_cfg(), policy, pre, post, dataset_meta=None)

        # commit 1: the model through HubMixin (config.json + model.safetensors in a tmpdir)
        model_commit = mocked_hub.calls[0]
        assert {"config.json", "model.safetensors"} <= set(model_commit["files"])
        # commits 2-3: processors
        assert pre.pushed_to == "user/policy" and post.pushed_to == "user/policy"
        # commit 4: the bundle sidecar
        bundle = mocked_hub.calls[-1]
        assert {"README.md", "train_config.json"} <= set(bundle["files"])
        # DCP resume artifacts are excluded from every publish upload
        for call in (model_commit, bundle):
            assert any("distcp" in p for p in call["ignore_patterns"])
        # the exact line lerobot.jobs.hf watches to end remote runs early
        assert any(
            m.startswith("Model pushed to https://huggingface.co/user/policy") for m in caplog.messages
        )

    def test_peft_branch_skips_model_commit(self, mocked_hub):
        policy = make_dummy_policy(repo_id="user/policy")

        class FakePeftModel:
            def save_pretrained(self, path):
                (Path(path) / "adapter_model.safetensors").write_bytes(b"x")

        publish_trained_model(make_cfg(), policy, None, None, dataset_meta=None, peft_model=FakePeftModel())
        assert len(mocked_hub.calls) == 1  # only the bundle commit
        bundle = mocked_hub.calls[0]
        # adapter weights + the wrapped policy's config + card + train config, no full weights
        assert {"README.md", "adapter_model.safetensors", "config.json", "train_config.json"} <= set(
            bundle["files"]
        )
        assert "model.safetensors" not in bundle["files"]

    def test_missing_repo_id_fails_loudly(self, mocked_hub):
        policy = make_dummy_policy(repo_id=None)
        with pytest.raises(ValueError, match="repo id"):
            publish_trained_model(make_cfg(), policy, None, None, dataset_meta=None)


class TestGenerateModelCard:
    def test_free_function_renders_from_arguments(self, monkeypatch):
        monkeypatch.setattr(train_utils.ModelCard, "validate", lambda self: None)
        policy = make_dummy_policy(repo_id="user/policy")
        card = generate_model_card(policy.config, cfg=make_cfg(), dataset_meta=None)
        assert card.data.library_name == "lerobot"
        assert card.data.datasets == "user/dataset"
        assert "lerobot" in card.data.tags
