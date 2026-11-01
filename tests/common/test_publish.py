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

import httpx
import pytest
import torch
from huggingface_hub.utils import validate_repo_id
from safetensors.torch import load_file

import lerobot.common.train_utils as train_utils
import lerobot.utils.hub as hub
from lerobot.common.train_utils import generate_model_card, publish_trained_model
from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.optim.ema import EMA, ConstantEMAConfig
from tests.fixtures.dummy_checkpoint_policy import make_dummy_policy


class FakeHfApi:
    """Records every repo/upload interaction; shared across both HfApi import sites."""

    calls: list[dict] = []
    branches: list[str] = []

    def __init__(self, *args, **kwargs):
        pass

    def create_repo(self, repo_id, private=None, exist_ok=False, **kwargs):
        return SimpleNamespace(repo_id=repo_id)

    def create_branch(self, repo_id, *, branch, exist_ok=False, **kwargs):
        FakeHfApi.branches.append(branch)

    def upload_folder(self, *, repo_id, folder_path, commit_message, **kwargs):
        weights = Path(folder_path) / "model.safetensors"
        FakeHfApi.calls.append(
            {
                "repo_id": repo_id,
                "commit_message": commit_message,
                "files": sorted(p.name for p in Path(folder_path).iterdir()),
                "ignore_patterns": kwargs.get("ignore_patterns"),
                "revision": kwargs.get("revision"),
                "weights": load_file(weights) if weights.is_file() else None,
            }
        )
        return SimpleNamespace(repo_url=SimpleNamespace(url=f"https://huggingface.co/{repo_id}"))


@pytest.fixture
def mocked_hub(monkeypatch):
    FakeHfApi.calls = []
    FakeHfApi.branches = []
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
        self.branches = []

    def push_to_hub(self, repo_id, branch=None, **kwargs):
        self.pushed_to = repo_id
        self.branches.append(branch)


class TestPublishTrainedModel:
    @pytest.mark.parametrize("with_ema", [False, True])
    def test_commit_set_and_log_contract(self, mocked_hub, caplog, with_ema):
        policy = make_dummy_policy(repo_id="user/policy")
        ema = EMA(policy, ConstantEMAConfig()) if with_ema else None
        if ema is not None:
            torch._foreach_add_(list(ema.averages.values()), 1.0)
        pre, post = RecordingProcessor(), RecordingProcessor()
        with caplog.at_level(logging.INFO):
            publish_trained_model(make_cfg(), policy, pre, post, dataset_meta=None, ema=ema)

        # with EMA, main gets the averages, then a `live` branch created from main gets the live weights
        branches = [None, "live"] if with_ema else [None]
        assert mocked_hub.branches == branches[1:]
        assert [call["revision"] for call in mocked_hub.calls] == [b for b in branches for _ in range(2)]
        assert pre.branches == post.branches == branches
        assert [call["weights"]["net.weight"].unique().item() for call in mocked_hub.calls[::2]] == (
            [1.5, 0.5] if with_ema else [0.5]
        )

        # commit 1: the model through HubMixin (config.json + model.safetensors in a tmpdir)
        model_commit = mocked_hub.calls[0]
        assert {"config.json", "model.safetensors"} <= set(model_commit["files"])
        # commits 2-3: processors
        assert pre.pushed_to == "user/policy" and post.pushed_to == "user/policy"
        # commit 4: the bundle sidecar
        bundle = mocked_hub.calls[-1]
        assert {"README.md", "train_config.json"} <= set(bundle["files"])
        # the exact line lerobot.jobs.hf watches to end remote runs early, once, after every commit
        assert [m for m in caplog.messages if m.startswith("Model pushed to")] == caplog.messages[-1:]
        assert caplog.messages[-1].startswith("Model pushed to https://huggingface.co/user/policy")

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

    def test_hub_parent_becomes_base_model_and_lends_its_license(self, monkeypatch):
        monkeypatch.setattr(train_utils.ModelCard, "validate", lambda self: None)
        monkeypatch.setattr(train_utils, "is_offline_mode", lambda: False)
        looked_up = []

        def fake_model_info(repo_id):
            looked_up.append(repo_id)
            return SimpleNamespace(id=repo_id, tags=["robotics", "license:gemma"])

        monkeypatch.setattr(train_utils, "model_info", fake_model_info)
        policy = make_dummy_policy(repo_id="user/policy")
        policy.config.pretrained_path = Path("user/parent")

        card = generate_model_card(policy.config, cfg=make_cfg(), dataset_meta=None)
        assert looked_up == ["user/parent"]
        assert card.data.base_model == "user/parent"
        assert card.data.license == "gemma"

        policy.config.license = "mit"
        assert generate_model_card(policy.config, cfg=make_cfg(), dataset_meta=None).data.license == "mit"

    def test_peft_policy_is_an_adapter_of_its_parent(self, monkeypatch):
        monkeypatch.setattr(train_utils.ModelCard, "validate", lambda self: None)
        monkeypatch.setattr(train_utils, "is_offline_mode", lambda: False)
        monkeypatch.setattr(train_utils, "model_info", lambda repo_id: SimpleNamespace(id=repo_id, tags=[]))
        policy = make_dummy_policy(repo_id="user/policy")
        policy.config.pretrained_path = Path("user/parent")
        assert "base_model_relation" not in str(
            generate_model_card(policy.config, cfg=make_cfg(), dataset_meta=None)
        )

        policy.config.use_peft = True
        card = generate_model_card(policy.config, cfg=make_cfg(), dataset_meta=None)
        assert card.data.base_model == "user/parent"
        assert card.data.base_model_relation == "adapter"
        assert "base_model_relation: adapter" in str(card)

    @pytest.mark.parametrize(
        "stale_path",
        ["outputs/train/run/checkpoints/000002/pretrained_model", "/home/user/run", "outputs/run"],
    )
    def test_stale_local_parent_is_not_a_base_model(self, monkeypatch, stale_path):
        """A `pretrained_path` from another machine must not reach the card, which the Hub would reject."""
        monkeypatch.setattr(train_utils.ModelCard, "validate", lambda self: None)
        monkeypatch.setattr(train_utils, "is_offline_mode", lambda: False)

        def model_info_without_network(repo_id):
            # Malformed ids fail validation before any request; a well-formed one is a missing repo.
            validate_repo_id(repo_id)
            raise httpx.HTTPStatusError("404", request=None, response=None)

        monkeypatch.setattr(train_utils, "model_info", model_info_without_network)
        policy = make_dummy_policy(repo_id="user/policy")
        policy.config.pretrained_path = Path(stale_path)

        card = generate_model_card(policy.config, cfg=make_cfg(), dataset_meta=None)
        assert card.data.base_model is None
        assert card.data.license is None

    def test_local_parent_is_not_a_base_model(self, monkeypatch, tmp_path):
        monkeypatch.setattr(train_utils.ModelCard, "validate", lambda self: None)
        monkeypatch.setattr(train_utils, "model_info", lambda repo_id: pytest.fail("no Hub lookup expected"))
        policy = make_dummy_policy(repo_id="user/policy")
        policy.config.pretrained_path = tmp_path

        card = generate_model_card(policy.config, cfg=make_cfg(), dataset_meta=None)
        assert card.data.base_model is None
        assert card.data.license is None


class TestDeprecatedPushModelToHub:
    """`push_model_to_hub` stays callable for external scripts, delegating to the publisher."""

    def test_policy_shim_warns_and_publishes(self, mocked_hub):
        policy = make_dummy_policy(repo_id="user/policy")
        with pytest.warns(FutureWarning, match="push_model_to_hub is deprecated"):
            policy.push_model_to_hub(make_cfg())

        # Same artifacts the method produced before: weights + config, then card + train config.
        model_commit = mocked_hub.calls[0]
        assert {"config.json", "model.safetensors"} <= set(model_commit["files"])
        bundle = mocked_hub.calls[-1]
        assert {"README.md", "train_config.json"} <= set(bundle["files"])

    def test_policy_shim_warns_that_state_dict_is_ignored(self, mocked_hub):
        policy = make_dummy_policy(repo_id="user/policy")
        with pytest.warns(FutureWarning, match="`state_dict` argument is ignored"):
            policy.push_model_to_hub(make_cfg(), state_dict=policy.state_dict())
