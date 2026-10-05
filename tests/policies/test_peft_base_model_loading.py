# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Regression tests for PEFT loading when the checkpoint is a base model (see issue #3975).

Starting a fresh LoRA/PEFT fine-tune points ``--policy.path`` at a *base model* (no
``adapter_config.json``) while also setting ``use_peft=True``. This must NOT be mistaken for
loading an existing PEFT adapter. These tests lock in the base-model vs. adapter distinction
made by ``lerobot.policies.factory.has_peft_adapter_config``, the branch it drives in
``make_policy``, and the `--peft.*` / `--policy.use_peft` normalization in
``TrainPipelineConfig.validate``. They are pure/fast (no network, no ``peft``), so they run in CI.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import torch
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError, RepositoryNotFoundError

from lerobot.configs.default import DatasetConfig, PeftConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.factory import has_peft_adapter_config


def test_local_base_model_dir_has_no_adapter_config(tmp_path):
    # A base-model checkpoint directory (only model weights, no adapter config).
    (tmp_path / "model.safetensors").write_bytes(b"")
    (tmp_path / "config.json").write_text("{}")
    assert has_peft_adapter_config(tmp_path) is False


def test_local_adapter_dir_has_adapter_config(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}))
    (tmp_path / "adapter_model.safetensors").write_bytes(b"")
    assert has_peft_adapter_config(tmp_path) is True


def test_hub_adapter_repo_has_adapter_config():
    with patch("lerobot.policies.factory.hf_hub_download", return_value="/cache/adapter_config.json") as dl:
        assert has_peft_adapter_config("some/adapter-repo", revision="main") is True
    dl.assert_called_once_with("some/adapter-repo", "adapter_config.json", revision="main")


@pytest.mark.parametrize(
    "error",
    [
        EntryNotFoundError("404"),  # online: the repo has no adapter_config.json
        LocalEntryNotFoundError("offline"),  # offline: the file is not in the local cache
    ],
)
def test_hub_missing_adapter_config_means_base_model(error):
    with patch("lerobot.policies.factory.hf_hub_download", side_effect=error):
        assert has_peft_adapter_config("lerobot/lingbot_va_base") is False


def test_hub_errors_other_than_missing_file_are_raised():
    # An unknown repo (typo, or private without auth) must not be silently treated as a base model.
    error = RepositoryNotFoundError("not found", response=MagicMock())
    with (
        patch("lerobot.policies.factory.hf_hub_download", side_effect=error),
        pytest.raises(RepositoryNotFoundError),
    ):
        has_peft_adapter_config("some/typo-repo")


@patch("lerobot.policies.factory.validate_visual_features_consistency")
@patch("lerobot.policies.factory.env_to_policy_features", return_value={})
@patch("lerobot.policies.factory.get_policy_class")
def test_make_policy_base_model_with_use_peft_loads_base_not_adapter(
    mock_get_cls, _mock_features, _mock_validate
):
    """`use_peft=True` on a base model must load the base weights, not a PEFT adapter.

    Before the #3975 fix this went down the ``PeftConfig.from_pretrained`` path and failed
    looking for a non-existent ``adapter_config.json``.
    """
    from lerobot.policies import factory

    policy_cls = MagicMock()
    loaded_policy = torch.nn.Linear(1, 1)  # a real nn.Module so make_policy's assert passes
    policy_cls.from_pretrained.return_value = loaded_policy
    mock_get_cls.return_value = policy_cls

    cfg = MagicMock()
    cfg.type = "act"
    cfg.device = "cpu"
    cfg.pretrained_path = "lerobot/lingbot_va_base"
    cfg.pretrained_revision = None
    cfg.use_peft = True
    cfg.input_features = {}
    cfg.output_features = {}

    with patch.object(factory, "has_peft_adapter_config", return_value=False) as mock_has_adapter:
        policy = factory.make_policy(cfg=cfg, env_cfg=MagicMock())

    mock_has_adapter.assert_called_once()
    # Base model is loaded via the normal pretrained path, PEFT adapter loading is not attempted.
    policy_cls.from_pretrained.assert_called_once()
    assert policy_cls.from_pretrained.call_args.kwargs["pretrained_name_or_path"] == (
        "lerobot/lingbot_va_base"
    )
    assert policy is loaded_policy


def _make_train_cfg(pretrained_path="lerobot/some_base", use_peft=False, peft=None):
    policy = ACTConfig(push_to_hub=False, use_peft=use_peft)
    policy.pretrained_path = pretrained_path
    return TrainPipelineConfig(dataset=DatasetConfig(repo_id="lerobot/dummy"), policy=policy, peft=peft)


def test_peft_config_implies_use_peft():
    cfg = _make_train_cfg(peft=PeftConfig(r=8))
    cfg.validate()
    assert cfg.policy.use_peft is True
    assert cfg.peft.r == 8


def test_use_peft_implies_default_peft_config():
    # `--policy.use_peft=true` alone used to fully fine-tune the model while saving a checkpoint
    # marked `use_peft=true`. It now trains with the default PEFT config.
    cfg = _make_train_cfg(use_peft=True)
    cfg.validate()
    assert cfg.policy.use_peft is True
    assert cfg.peft == PeftConfig()


def test_no_peft_leaves_both_unset():
    cfg = _make_train_cfg()
    cfg.validate()
    assert cfg.policy.use_peft is False
    assert cfg.peft is None


@pytest.mark.parametrize("cfg_kwargs", [{"peft": PeftConfig()}, {"use_peft": True}])
def test_peft_without_pretrained_policy_raises(cfg_kwargs):
    cfg = _make_train_cfg(pretrained_path=None, **cfg_kwargs)
    with pytest.raises(ValueError, match="Training from scratch using PEFT"):
        cfg.validate()
