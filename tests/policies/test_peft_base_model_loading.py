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
made by ``lerobot.policies.factory._has_peft_adapter_config`` and the branch it drives in
``make_policy``. They are pure/fast (no network, no ``peft``), so they run in CI.
"""

import json
from unittest.mock import MagicMock, patch

import torch
from huggingface_hub.errors import HfHubHTTPError

from lerobot.policies.factory import _has_peft_adapter_config


def test_local_base_model_dir_has_no_adapter_config(tmp_path):
    # A base-model checkpoint directory (only model weights, no adapter config).
    (tmp_path / "model.safetensors").write_bytes(b"")
    (tmp_path / "config.json").write_text("{}")
    assert _has_peft_adapter_config(str(tmp_path)) is False


def test_local_adapter_dir_has_adapter_config(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}))
    (tmp_path / "adapter_model.safetensors").write_bytes(b"")
    assert _has_peft_adapter_config(str(tmp_path)) is True


def test_hub_base_model_repo_has_no_adapter_config():
    with patch("huggingface_hub.file_exists", return_value=False) as mock_exists:
        assert _has_peft_adapter_config("lerobot/lingbot_va_base") is False
    mock_exists.assert_called_once()
    assert mock_exists.call_args.args[1] == "adapter_config.json"


def test_hub_adapter_repo_has_adapter_config():
    with patch("huggingface_hub.file_exists", return_value=True):
        assert _has_peft_adapter_config("some/adapter-repo", revision="main") is True


def test_hub_lookup_error_falls_back_to_base_model():
    # Offline / private / transient Hub errors must not crash; treat as "not an adapter".
    with patch(
        "huggingface_hub.file_exists",
        side_effect=HfHubHTTPError("boom", response=MagicMock()),
    ):
        assert _has_peft_adapter_config("some/private-repo") is False
    with patch("huggingface_hub.file_exists", side_effect=OSError("offline")):
        assert _has_peft_adapter_config("some/repo") is False


def _make_dummy_policy_cfg(pretrained_path, use_peft):
    cfg = MagicMock()
    cfg.type = "act"
    cfg.device = "cpu"
    cfg.pretrained_path = pretrained_path
    cfg.pretrained_revision = None
    cfg.use_peft = use_peft
    cfg.input_features = {}
    cfg.output_features = {}
    return cfg


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

    cfg = _make_dummy_policy_cfg(pretrained_path="lerobot/lingbot_va_base", use_peft=True)
    env_cfg = MagicMock()

    with patch.object(factory, "_has_peft_adapter_config", return_value=False) as mock_has_adapter:
        policy = factory.make_policy(cfg=cfg, env_cfg=env_cfg)

    mock_has_adapter.assert_called_once()
    # Base model is loaded via the normal pretrained path...
    policy_cls.from_pretrained.assert_called_once()
    assert policy_cls.from_pretrained.call_args.kwargs["pretrained_name_or_path"] == (
        "lerobot/lingbot_va_base"
    )
    # ...and PEFT adapter loading is NOT attempted (would need peft + adapter_config.json).
    assert policy is loaded_policy


@patch("lerobot.policies.factory.validate_visual_features_consistency")
@patch("lerobot.policies.factory.env_to_policy_features", return_value={})
@patch("lerobot.policies.factory.get_policy_class")
def test_make_policy_existing_adapter_uses_peft_loading(mock_get_cls, _mock_features, _mock_validate):
    """A real adapter checkpoint (has ``adapter_config.json``) must go through PEFT loading."""
    from lerobot.policies import factory

    policy_cls = MagicMock()
    mock_get_cls.return_value = policy_cls

    cfg = _make_dummy_policy_cfg(pretrained_path="some/adapter-repo", use_peft=True)
    env_cfg = MagicMock()

    policy_cls.from_pretrained.return_value = torch.nn.Linear(1, 1)

    fake_peft = MagicMock()
    fake_peft_config = MagicMock()
    fake_peft_config.base_model_name_or_path = "lerobot/lingbot_va_base"
    fake_peft.PeftConfig.from_pretrained.return_value = fake_peft_config
    fake_peft.PeftModel.from_pretrained.return_value = torch.nn.Linear(1, 1)

    with (
        patch.object(factory, "_has_peft_adapter_config", return_value=True),
        patch.dict("sys.modules", {"peft": fake_peft}),
    ):
        factory.make_policy(cfg=cfg, env_cfg=env_cfg)

    fake_peft.PeftConfig.from_pretrained.assert_called_once_with("some/adapter-repo")
    fake_peft.PeftModel.from_pretrained.assert_called_once()
