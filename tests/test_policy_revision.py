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

"""Tests for loading policies from a specific Hugging Face Hub revision.

These stay offline: Hub downloads (`hf_hub_download`) and heavy `from_pretrained` calls are mocked,
so we only check that `revision` is resolved and threaded through the loading paths.
"""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import lerobot.policies.factory as factory
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.rewards import RewardModelConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE


@pytest.fixture
def act_config_file(tmp_path):
    """Write a valid ACT ``config.json`` to a temp dir and return its path."""
    ACTConfig()._save_pretrained(tmp_path)
    return tmp_path / "config.json"


def test_from_pretrained_resolves_revision_from_cli_overrides(act_config_file):
    # `--policy.revision` arrives as a CLI override; it must be resolved *before* downloading
    # config.json (not just for the weights) and recorded on the parsed config.
    with patch("lerobot.configs.policies.hf_hub_download", return_value=str(act_config_file)) as mock_dl:
        cfg = PreTrainedConfig.from_pretrained("org/model", cli_overrides=["--revision=abc123"])

    assert mock_dl.call_args.kwargs["revision"] == "abc123"
    assert cfg.revision == "abc123"


def test_from_pretrained_revision_kwarg_is_persisted(act_config_file):
    # When passed programmatically (no CLI override), the revision is still recorded on the config.
    with patch("lerobot.configs.policies.hf_hub_download", return_value=str(act_config_file)) as mock_dl:
        cfg = PreTrainedConfig.from_pretrained("org/model", revision="v1.0")

    assert mock_dl.call_args.kwargs["revision"] == "v1.0"
    assert cfg.revision == "v1.0"


@RewardModelConfig.register_subclass(name="_rev_test_reward")
@dataclass
class _RevRewardConfig(RewardModelConfig):
    def get_optimizer_preset(self):
        return AdamWConfig(lr=1e-4)


def test_reward_config_threads_revision(tmp_path):
    # RewardModelConfig mirrors the policy loading path; verify the copy stays in sync.
    _RevRewardConfig()._save_pretrained(tmp_path)
    with patch(
        "lerobot.configs.rewards.hf_hub_download", return_value=str(tmp_path / "config.json")
    ) as mock_dl:
        loaded = RewardModelConfig.from_pretrained("org/reward", revision="rv2")

    assert mock_dl.call_args.kwargs["revision"] == "rv2"
    assert loaded.revision == "rv2"


def test_load_pretrained_config_from_cli_returns_none_without_path(monkeypatch):
    monkeypatch.setattr(parser, "get_path_arg", lambda field_name: None)
    assert parser.load_pretrained_config_from_cli("policy") is None


def test_load_pretrained_config_from_cli_threads_overrides_and_path(monkeypatch):
    monkeypatch.setattr(parser, "get_path_arg", lambda field_name: "org/model")
    monkeypatch.setattr(parser, "get_yaml_overrides", lambda field_name: ["--n_obs_steps=2"])
    monkeypatch.setattr(parser, "get_cli_overrides", lambda field_name: ["--revision=v2"])

    sentinel = SimpleNamespace(pretrained_path=None)
    with patch("lerobot.configs.policies.PreTrainedConfig.from_pretrained", return_value=sentinel) as mock_fp:
        result = parser.load_pretrained_config_from_cli("policy")

    assert result is sentinel
    assert result.pretrained_path == Path("org/model")
    # YAML overrides come before CLI overrides, and `--revision` rides along for from_pretrained.
    assert mock_fp.call_args.kwargs["cli_overrides"] == ["--n_obs_steps=2", "--revision=v2"]


def test_make_pre_post_processors_passes_revision(monkeypatch):
    cfg = ACTConfig()
    cfg.revision = "v3"

    mock_pipeline = MagicMock()
    monkeypatch.setattr(factory, "PolicyProcessorPipeline", mock_pipeline)
    monkeypatch.setattr(factory, "_reconnect_relative_absolute_steps", lambda *a, **k: None)

    make_pre_post_processors(policy_cfg=cfg, pretrained_path="org/model")

    assert mock_pipeline.from_pretrained.call_count == 2
    for call in mock_pipeline.from_pretrained.call_args_list:
        assert call.kwargs["revision"] == "v3"


@pytest.fixture
def fake_policy_cls(monkeypatch):
    """Patch make_policy's class lookup + feature inference so it can run offline."""
    features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
    }
    fake_cls = MagicMock()
    monkeypatch.setattr(factory, "get_policy_class", lambda _type: fake_cls)
    monkeypatch.setattr(factory, "env_to_policy_features", lambda _env: features)
    monkeypatch.setattr(factory, "validate_visual_features_consistency", lambda *a, **k: None)
    return fake_cls


def test_make_policy_threads_revision_to_weights(fake_policy_cls):
    cfg = ACTConfig()
    cfg.pretrained_path = "org/model"
    cfg.revision = "v9"
    fake_policy_cls.from_pretrained.return_value = MagicMock(spec=torch.nn.Module)

    make_policy(cfg, env_cfg=MagicMock())

    kwargs = fake_policy_cls.from_pretrained.call_args.kwargs
    assert kwargs["pretrained_name_or_path"] == "org/model"
    assert kwargs["revision"] == "v9"


def test_make_policy_peft_revision_applies_to_adapter_not_base_model(fake_policy_cls):
    peft = pytest.importorskip("peft")

    cfg = ACTConfig()
    cfg.pretrained_path = "org/adapter"
    cfg.revision = "v9"
    cfg.use_peft = True

    peft_config = SimpleNamespace(base_model_name_or_path="org/base")
    with (
        patch.object(peft, "PeftConfig") as mock_peft_config,
        patch.object(peft, "PeftModel") as mock_peft_model,
    ):
        mock_peft_config.from_pretrained.return_value = peft_config
        mock_peft_model.from_pretrained.return_value = MagicMock(spec=torch.nn.Module)
        make_policy(cfg, env_cfg=MagicMock())

    # The adapter repo (PeftConfig + PeftModel) is loaded at the requested revision.
    assert mock_peft_config.from_pretrained.call_args.kwargs["revision"] == "v9"
    assert mock_peft_model.from_pretrained.call_args.kwargs["revision"] == "v9"

    # The base model lives in a separate repo and must NOT inherit the adapter's revision.
    base_call = fake_policy_cls.from_pretrained.call_args
    assert base_call.kwargs["pretrained_name_or_path"] == "org/base"
    assert "revision" not in base_call.kwargs
