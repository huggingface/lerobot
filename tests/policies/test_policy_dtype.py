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
"""Tests for `PreTrainedConfig.dtype`: a `torch.dtype` in Python, a plain name such as "bfloat16" in
config.json and on the command line."""

import json

import draccus
import pytest
import torch
from draccus.utils import DecodingError

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.eo1.configuration_eo1 import EO1Config
from lerobot.policies.evo1.configuration_evo1 import Evo1Config
from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.lingbot_va.configuration_lingbot_va import LingBotVAConfig
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.vla_jepa.configuration_vla_jepa import VLAJEPAConfig


def _write_config_json(directory, **fields):
    (directory / "config.json").write_text(json.dumps({"device": "cpu", **fields}))


def test_dtype_round_trips_through_config_json(tmp_path):
    ACTConfig(device="cpu", dtype=torch.bfloat16).save_pretrained(tmp_path)

    assert json.loads((tmp_path / "config.json").read_text())["dtype"] == "bfloat16"
    assert PreTrainedConfig.from_pretrained(tmp_path).dtype is torch.bfloat16


@pytest.mark.parametrize(
    ("name", "expected"),
    [("float16", torch.float16), ("torch.float16", torch.float16), (None, None)],
)
def test_dtype_names_are_decoded(tmp_path, name, expected):
    _write_config_json(tmp_path, type="act", dtype=name)
    assert PreTrainedConfig.from_pretrained(tmp_path).dtype is expected


def test_policy_dtype_flag_overrides_config_json(tmp_path):
    # `--policy.path=<dir> --policy.dtype=float32` reaches `from_pretrained` as a CLI override.
    _write_config_json(tmp_path, type="act", dtype="bfloat16")
    config = PreTrainedConfig.from_pretrained(tmp_path, cli_overrides=["--dtype=float32"])
    assert config.dtype is torch.float32


def test_policy_dtype_flag_in_a_train_config():
    cfg = draccus.parse(
        TrainPipelineConfig, args=["--dataset.repo_id=u/d", "--policy.type=act", "--policy.dtype=bfloat16"]
    )
    assert cfg.policy.dtype is torch.bfloat16


# "auto" is only accepted by EO1, as a deprecated alias (see the last test).
@pytest.mark.parametrize("name", ["nonsense", "auto"])
def test_names_that_are_not_dtypes_are_rejected(tmp_path, name):
    _write_config_json(tmp_path, type="act", dtype=name)
    with pytest.raises(DecodingError, match="Invalid dtype"):
        PreTrainedConfig.from_pretrained(tmp_path)


def test_python_callers_must_pass_a_torch_dtype():
    with pytest.raises(ValueError, match="must be a torch.dtype"):
        ACTConfig(device="cpu", dtype="bfloat16")


@pytest.mark.parametrize(
    ("config_cls", "supported"),
    [
        (PI0Config, torch.bfloat16),  # same {float32, bfloat16} check in pi05, pi0_fast, xvla, molmoact2
        (FastWAMConfig, torch.float16),
        (LingBotVAConfig, torch.float16),
        (VLAJEPAConfig, torch.float16),
        (GrootConfig, torch.bfloat16),
    ],
)
def test_policies_restrict_dtype_to_the_precisions_they_support(config_cls, supported):
    assert config_cls(device="cpu", dtype=supported).dtype is supported
    with pytest.raises(ValueError, match="dtype"):
        config_cls(device="cpu", dtype=torch.float64)


@pytest.mark.parametrize(
    ("config_cls", "legacy_key", "legacy_value", "default"),
    [
        (FastWAMConfig, "torch_dtype", "float32", torch.bfloat16),
        (VLAJEPAConfig, "torch_dtype", "float32", torch.bfloat16),
        (Evo1Config, "vlm_dtype", "float32", torch.bfloat16),
        (GrootConfig, "model_params_fp32", False, torch.float32),
    ],
)
def test_legacy_precision_keys_are_ignored_with_a_warning(
    tmp_path, config_cls, legacy_key, legacy_value, default
):
    # A config.json written before `dtype` existed still loads, but only `dtype` sets the precision.
    policy_type = PreTrainedConfig.get_choice_name(config_cls)
    _write_config_json(tmp_path, type=policy_type, **{legacy_key: legacy_value})

    with pytest.warns(FutureWarning, match=legacy_key):
        config = PreTrainedConfig.from_pretrained(tmp_path)

    assert config.dtype is default
    assert getattr(config, legacy_key) is None


def test_eo1_auto_dtype_resolves_to_bfloat16_with_a_warning():
    with pytest.warns(FutureWarning, match="auto"):
        config = EO1Config(device="cpu", vlm_config={}, dtype="auto")
    assert config.dtype is torch.bfloat16
