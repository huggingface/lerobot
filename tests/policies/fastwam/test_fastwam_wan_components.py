#!/usr/bin/env python

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

from pathlib import Path

import pytest
from torch import nn

from lerobot.policies.fastwam import modeling_fastwam
from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
from lerobot.policies.fastwam.modeling_fastwam import (
    FastWAMPolicy,
    resolve_wan_component_paths,
)
from lerobot.policies.fastwam.wan_components import (
    WAN_DIT_PATTERN,
    WAN_T5_CHECKPOINT,
    WAN_T5_TOKENIZER,
    WAN_VAE_CHECKPOINT,
    resolve_wan_checkpoint_paths,
)


def _make_wan_component_tree(root: Path) -> None:
    tokenizer = root / WAN_T5_TOKENIZER
    tokenizer.mkdir(parents=True)
    (root / WAN_VAE_CHECKPOINT).touch()
    (root / WAN_T5_CHECKPOINT).touch()
    (root / "diffusion_pytorch_model-00001-of-00001.safetensors").touch()
    (tokenizer / "tokenizer.json").touch()


def test_resolve_wan_component_paths_finds_complete_local_directory(tmp_path):
    _make_wan_component_tree(tmp_path)

    paths = resolve_wan_component_paths(tmp_path)

    assert paths.vae == tmp_path / WAN_VAE_CHECKPOINT
    assert paths.text_encoder == tmp_path / WAN_T5_CHECKPOINT
    assert paths.tokenizer == tmp_path / WAN_T5_TOKENIZER


def test_resolve_wan_component_paths_does_not_require_original_dit_shards(tmp_path):
    _make_wan_component_tree(tmp_path)
    for shard in tmp_path.glob(WAN_DIT_PATTERN):
        shard.unlink()

    paths = resolve_wan_component_paths(tmp_path)

    assert paths.dit == []
    assert paths.vae == tmp_path / WAN_VAE_CHECKPOINT
    assert paths.text_encoder == tmp_path / WAN_T5_CHECKPOINT
    assert paths.tokenizer == tmp_path / WAN_T5_TOKENIZER


def test_resolve_wan_checkpoint_paths_uses_official_wan_layout(tmp_path):
    _make_wan_component_tree(tmp_path)

    paths = resolve_wan_checkpoint_paths(tmp_path)

    assert paths.root == tmp_path
    assert paths.dit == [tmp_path / "diffusion_pytorch_model-00001-of-00001.safetensors"]
    assert paths.vae == tmp_path / WAN_VAE_CHECKPOINT
    assert paths.text_encoder == tmp_path / WAN_T5_CHECKPOINT
    assert paths.tokenizer == tmp_path / WAN_T5_TOKENIZER
    assert WAN_DIT_PATTERN == "diffusion_pytorch_model*.safetensors"


def test_resolve_wan_component_paths_rejects_partial_local_directory(tmp_path):
    _make_wan_component_tree(tmp_path)
    (tmp_path / WAN_T5_CHECKPOINT).unlink()

    with pytest.raises(FileNotFoundError, match="text encoder"):
        resolve_wan_component_paths(tmp_path)


def test_policy_config_construction_loads_wan22_backbone_from_config(monkeypatch):
    class TinyCore(nn.Module):
        def __init__(self):
            super().__init__()
            self.text_encoder = None

    calls = []

    def fake_from_wan22_pretrained(**kwargs):
        calls.append(kwargs)
        return TinyCore()

    monkeypatch.setattr(
        "lerobot.policies.fastwam.modular_fastwam.FastWAM.from_wan22_pretrained",
        fake_from_wan22_pretrained,
    )

    cfg = FastWAMConfig()
    policy = FastWAMPolicy(cfg)

    assert policy.model.text_encoder is None
    assert calls == [
        {
            "device": cfg.device,
            "torch_dtype": modeling_fastwam._dtype_from_name(cfg.torch_dtype),
            "model_id": "Wan-AI/Wan2.2-TI2V-5B",
            "tokenizer_model_id": "Wan-AI/Wan2.2-TI2V-5B",
            "tokenizer_max_len": cfg.tokenizer_max_len,
            "load_text_encoder": cfg.load_text_encoder,
            "proprio_dim": cfg.proprio_dim,
            "video_dit_config": cfg.video_dit_config,
            "action_dit_config": cfg.action_dit_config,
            "mot_checkpoint_mixed_attn": cfg.mot_checkpoint_mixed_attn,
            "video_train_shift": float(cfg.video_scheduler["train_shift"]),
            "video_infer_shift": float(cfg.video_scheduler["infer_shift"]),
            "video_num_train_timesteps": int(cfg.video_scheduler["num_train_timesteps"]),
            "action_train_shift": float(cfg.action_scheduler["train_shift"]),
            "action_infer_shift": float(cfg.action_scheduler["infer_shift"]),
            "action_num_train_timesteps": int(cfg.action_scheduler["num_train_timesteps"]),
            "loss_lambda_video": float(cfg.loss["lambda_video"]),
            "loss_lambda_action": float(cfg.loss["lambda_action"]),
        }
    ]


def test_explicit_local_wan_path_is_preserved(tmp_path):
    cfg = FastWAMConfig(model_id=str(tmp_path), tokenizer_model_id=str(tmp_path))

    assert cfg.model_id == str(tmp_path)
    assert cfg.tokenizer_model_id == str(tmp_path)


def test_other_hub_model_ids_are_rejected():
    with pytest.raises(ValueError, match="model_id"):
        FastWAMConfig(model_id="somebody/other-model")

    with pytest.raises(ValueError, match="tokenizer_model_id"):
        FastWAMConfig(tokenizer_model_id="somebody/other-tokenizer")


def test_resolve_wan_checkpoint_paths_can_skip_text_encoder(tmp_path):
    _make_wan_component_tree(tmp_path)
    (tmp_path / WAN_T5_CHECKPOINT).unlink()
    shutil_tokenizer = tmp_path / WAN_T5_TOKENIZER
    for child in shutil_tokenizer.iterdir():
        child.unlink()
    shutil_tokenizer.rmdir()
    shutil_tokenizer.parent.rmdir()

    paths = resolve_wan_checkpoint_paths(tmp_path, load_text_encoder=False)

    assert paths.text_encoder is None
    assert paths.tokenizer is None
