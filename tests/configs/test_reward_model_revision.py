# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import sys
from types import SimpleNamespace

from lerobot.configs.rewards import RewardModelConfig
from lerobot.configs.train import TrainPipelineConfig


def _bare_train_config() -> TrainPipelineConfig:
    """A TrainPipelineConfig with only the fields `_resolve_pretrained_from_cli` touches.

    `__post_init__`/`validate` pull in a dataset, an optimizer and a device, none of which
    matter here; bypassing them keeps the test focused on the resolution step.
    """
    cfg = object.__new__(TrainPipelineConfig)
    cfg.resume = False
    cfg.policy = None
    cfg.reward_model = None
    return cfg


def test_reward_model_pretrained_revision_reaches_from_pretrained(monkeypatch):
    """`--reward_model.pretrained_revision` must pin the revision the config is loaded from.

    The revision is only present in the de-nested CLI overrides, which draccus applies *after*
    `from_pretrained` has already downloaded `config.json`. Unless the call site forwards it
    explicitly, the download silently resolves `main`.
    """
    captured = {}

    def fake_from_pretrained(pretrained_name_or_path, **kwargs):
        captured["path"] = str(pretrained_name_or_path)
        captured["revision"] = kwargs.get("revision")
        return SimpleNamespace(pretrained_path=None, pretrained_revision=None)

    monkeypatch.setattr(RewardModelConfig, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lerobot-train",
            "--reward_model.path=user/reward-model",
            "--reward_model.pretrained_revision=abc123",
        ],
    )

    cfg = _bare_train_config()
    cfg._resolve_pretrained_from_cli()

    assert captured["path"] == "user/reward-model"
    assert captured["revision"] == "abc123"


def test_reward_model_revision_defaults_to_none_when_not_given(monkeypatch):
    """Without the flag the revision stays `None`, so the Hub default branch is used."""
    captured = {}

    def fake_from_pretrained(pretrained_name_or_path, **kwargs):
        captured["revision"] = kwargs.get("revision")
        return SimpleNamespace(pretrained_path=None, pretrained_revision=None)

    monkeypatch.setattr(RewardModelConfig, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(sys, "argv", ["lerobot-train", "--reward_model.path=user/reward-model"])

    cfg = _bare_train_config()
    cfg._resolve_pretrained_from_cli()

    assert captured["revision"] is None
