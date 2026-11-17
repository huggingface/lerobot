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

import pytest
import torch
from draccus.utils import DecodingError

from lerobot.configs import PreTrainedConfig
from lerobot.policies import MolmoAct2Config


def _write_config(directory, **fields):
    path = directory / "config.json"
    path.write_text(json.dumps({"type": "molmoact2", "device": "cpu", **fields}), encoding="utf-8")
    return path


@pytest.mark.parametrize(("expert_only", "expected_mode"), [(False, "fft"), (True, "freeze")])
def test_loads_legacy_molmoact2_config(tmp_path, expert_only, expected_mode):
    source = _write_config(
        tmp_path,
        model_dtype="float32",
        action_mode="continuous",
        enable_lora_vlm=False,
        enable_lora_action_expert=False,
        train_action_expert_only=expert_only,
    )
    original = source.read_bytes()
    cfg = PreTrainedConfig.from_pretrained(tmp_path)

    assert isinstance(cfg, MolmoAct2Config)
    assert cfg.dtype is torch.float32
    assert cfg.train_mode_vlm == expected_mode
    assert source.read_bytes() == original

    saved = tmp_path / "saved"
    cfg.save_pretrained(saved)
    config_dict = json.loads((saved / "config.json").read_text(encoding="utf-8"))
    assert "model_dtype" not in config_dict
    assert "enable_lora_vlm" not in config_dict
    reloaded = PreTrainedConfig.from_pretrained(saved)
    assert reloaded.dtype is torch.float32
    assert reloaded.train_mode_vlm == expected_mode


def test_current_molmoact2_fields_and_cli_take_precedence(tmp_path):
    _write_config(
        tmp_path, model_dtype="float32", dtype="bfloat16", enable_lora_vlm=False, train_mode_vlm="lora"
    )
    cfg = PreTrainedConfig.from_pretrained(tmp_path)
    assert cfg.dtype is torch.bfloat16
    assert cfg.train_mode_vlm == "lora"

    cfg = PreTrainedConfig.from_pretrained(
        tmp_path, cli_overrides=["--dtype=float32", "--train_mode_vlm=freeze"]
    )
    assert cfg.dtype is torch.float32
    assert cfg.train_mode_vlm == "freeze"


@pytest.mark.parametrize("expert_lora", [False, True])
def test_legacy_lora_requires_explicit_conversion(tmp_path, expert_lora):
    _write_config(tmp_path, enable_lora_vlm=True, enable_lora_action_expert=expert_lora)
    with pytest.raises(ValueError, match="action-expert LoRA" if expert_lora else "VLM LoRA"):
        PreTrainedConfig.from_pretrained(tmp_path)


def test_unknown_molmoact2_fields_still_fail(tmp_path):
    _write_config(tmp_path, enable_lora_vlm=False, typo=True)
    with pytest.raises(DecodingError, match="typo"):
        PreTrainedConfig.from_pretrained(tmp_path)
