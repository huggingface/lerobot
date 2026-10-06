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
"""End-to-end checks that `PreTrainedConfig.dtype` is parsed from config.json, train_config.json and the CLI."""

import json

import draccus
import pytest
import torch

import lerobot.policies
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.eo1.configuration_eo1 import EO1Config

POLICY_CONFIGS = [
    getattr(lerobot.policies, name) for name in lerobot.policies.__all__ if name.endswith("Config")
]


@pytest.mark.parametrize("config_cls", POLICY_CONFIGS, ids=lambda cls: cls.__name__)
def test_policy_config_json_round_trip(tmp_path, config_cls):
    # EO1 would otherwise download the Qwen2.5-VL config from the Hub.
    extra = {"vlm_config": {}} if config_cls is EO1Config else {}
    config_cls(device="cpu", dtype=torch.bfloat16, **extra).save_pretrained(tmp_path)

    assert json.loads((tmp_path / "config.json").read_text())["dtype"] == "bfloat16"
    assert PreTrainedConfig.from_pretrained(tmp_path).dtype is torch.bfloat16
    # What `--policy.path=<dir> --policy.dtype=float32` does.
    overridden = PreTrainedConfig.from_pretrained(tmp_path, cli_overrides=["--dtype=float32"])
    assert overridden.dtype is torch.float32


def test_train_config_cli_and_resume(tmp_path):
    cfg = draccus.parse(
        TrainPipelineConfig, args=["--dataset.repo_id=u/d", "--policy.type=act", "--policy.dtype=bfloat16"]
    )
    assert cfg.policy.dtype is torch.bfloat16

    cfg.save_pretrained(tmp_path)
    assert TrainPipelineConfig.from_pretrained(tmp_path).policy.dtype is torch.bfloat16
    resumed = TrainPipelineConfig.from_pretrained(tmp_path, cli_args=["--policy.dtype=float32"])
    assert resumed.policy.dtype is torch.float32


@pytest.mark.parametrize(
    ("legacy_config", "default"),
    [
        ({"type": "fastwam", "torch_dtype": "float32"}, torch.bfloat16),
        ({"type": "vla_jepa", "torch_dtype": "float32"}, torch.bfloat16),
        ({"type": "evo1", "vlm_dtype": "float32"}, torch.bfloat16),
        ({"type": "groot", "model_params_fp32": False}, torch.float32),
        ({"type": "eo1", "dtype": "auto", "vlm_config": {}}, torch.bfloat16),
    ],
)
def test_legacy_precision_settings_still_load(tmp_path, legacy_config, default):
    # Configs saved before `dtype` existed load with a warning; only `dtype` sets the precision.
    (tmp_path / "config.json").write_text(json.dumps({"device": "cpu", **legacy_config}))
    with pytest.warns(FutureWarning, match="is deprecated"):
        config = PreTrainedConfig.from_pretrained(tmp_path)
    assert config.dtype is default
