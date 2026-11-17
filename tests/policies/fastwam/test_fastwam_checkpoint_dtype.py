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

"""Public checkpoint and dtype regression tests with small offline components."""

import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")
pytest.importorskip("diffusers")

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.factory import make_policy
from lerobot.policies.fastwam import modeling_fastwam
from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy
from lerobot.policies.fastwam.wan import LiteKVFusion

DOT_FLAGS = {
    "action_expert_layers": 1,
    "lite_kv_fusion": True,
    "retracted_1d_rope": True,
    "world_model_only_conditioning": True,
}


class ShapeVAE(nn.Module):
    z_dim = 4
    upsampling_factor = 2
    temporal_downsample_factor = 4

    def encode(self, video, **kwargs):
        if isinstance(video, list):
            return [item.new_zeros(4, 1, item.shape[-2] // 2, item.shape[-1] // 2) for item in video]
        b, _, t, h, w = video.shape
        return video.new_zeros(b, 4, (t - 1) // 4 + 1, h // 2, w // 2)


def tiny_config(**overrides) -> FastWAMConfig:
    common = {
        "hidden_dim": 12,
        "ffn_dim": 16,
        "freq_dim": 8,
        "text_dim": 8,
        "num_heads": 2,
        "attn_head_dim": 6,
        "num_layers": 3,
        "eps": 1e-6,
    }
    fields = {
        "action_dim": 3,
        "proprio_dim": 2,
        "action_horizon": 8,
        "n_action_steps": 8,
        "num_video_frames": 5,
        "action_video_freq_ratio": 1,
        "image_size": (16, 32),
        "context_len": 4,
        "base_model_id": None,
        "load_text_encoder": False,
        "device": "cpu",
        "fp32_attention": False,
        "num_inference_steps": 2,
        "video_dit_config": dict(
            common,
            patch_size=[1, 2, 2],
            in_dim=4,
            out_dim=4,
            seperated_timestep=True,
            action_conditioned=False,
            video_attention_mask_mode="first_frame_causal",
        ),
        "action_dit_config": dict(common, action_dim=3),
    }
    fields.update(overrides)
    fields["action_dit_config"]["action_dim"] = fields["action_dim"]
    return FastWAMConfig(**fields)


@pytest.fixture
def offline_components(monkeypatch):
    monkeypatch.setattr(modeling_fastwam, "load_pretrained_wan_vae", lambda **kw: ShapeVAE())
    monkeypatch.setattr(modeling_fastwam, "build_wan_tokenizer", lambda **kw: None)


def training_batch():
    return {
        "observation.images.image": torch.rand(1, 3, 16, 32),
        "observation.state": torch.rand(1, 2),
        "context": torch.randn(1, 4, 8),
        "context_mask": torch.ones(1, 4, dtype=torch.bool),
        "action": torch.randn(1, 32, 3),
    }


@pytest.mark.parametrize("flags", [{}, DOT_FLAGS], ids=["default", "fasterwam"])
@pytest.mark.parametrize(
    "device,dtype",
    [
        ("cpu", torch.float32),
        ("cpu", torch.bfloat16),
        pytest.param(
            "cuda",
            torch.bfloat16,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA dtype coverage requires CUDA"
            ),
        ),
    ],
)
def test_public_dtype_train_infer_and_checkpoint(offline_components, tmp_path, flags, device, dtype):
    torch.manual_seed(1234)
    policy = FastWAMPolicy(tiny_config(dtype=dtype, device=device, **flags))
    batch = training_batch()
    with torch.autocast(device, dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        loss, _ = policy(batch)
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    assert any(grad.abs().sum() > 0 for grad in grads)
    assert {p.dtype for p in policy.model.mot.parameters()} == {dtype}
    if flags:
        assert policy.model.mot.lite_kv_fusion.logits.grad.abs().sum() > 0
        for projection in ("k", "v"):
            weight = getattr(policy.model.mot.layers[0].blocks["video"].self_attn, projection).weight
            assert weight.grad.abs().sum() > 0
        assert policy.model.action_expert.text_embedding is None
        assert not any(
            "cross_attn" in key for key in policy.model.mot.layers[0].blocks["action"].state_dict()
        )
    else:
        assert policy.model.mot.lite_kv_fusion is None
        assert policy.model.action_expert.text_embedding is not None
    noise = torch.randn(1, 8, 3)
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        policy.eval()
        action = policy.predict_action_chunk(batch, noise=noise)
        policy.save_pretrained(tmp_path)
        loaded = FastWAMPolicy.from_pretrained(tmp_path, strict=True)
        reloaded_action = loaded.predict_action_chunk(batch, noise=noise)
    assert action.shape == (1, 8, 3)
    assert torch.isfinite(action).all()
    torch.testing.assert_close(reloaded_action, action, rtol=0, atol=0)
    assert loaded.config.dtype == dtype


def test_cli_backbone_conversion_loads_weights_and_trains(offline_components, tmp_path, monkeypatch):
    torch.manual_seed(7)
    original = FastWAMPolicy(tiny_config(dtype=torch.float32))
    original.save_pretrained(tmp_path / "backbone")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lerobot-train",
            f"--policy.path={tmp_path / 'backbone'}",
            "--policy.action_expert_layers=1",
            "--policy.lite_kv_fusion=true",
            "--policy.retracted_1d_rope=true",
            "--policy.world_model_only_conditioning=true",
            "--policy.push_to_hub=false",
            "--dataset.repo_id=local/fixture",
        ],
    )

    @parser.wrap()
    def parse_config(cfg: TrainPipelineConfig) -> TrainPipelineConfig:
        cfg.validate()
        return cfg

    cfg = parse_config()
    with pytest.raises(ValueError, match="checkpoint.*architecture"):
        FastWAMPolicy.from_pretrained(tmp_path / "backbone", config=cfg.policy, strict=True)
    changed_backbone = deepcopy(cfg.policy)
    changed_backbone.video_dit_config["num_layers"] = 2
    with pytest.raises(ValueError, match="checkpoint.*architecture"):
        FastWAMPolicy.from_pretrained(tmp_path / "backbone", config=changed_backbone)
    meta = SimpleNamespace(
        features={
            "observation.images.image": {
                "dtype": "image",
                "shape": (3, 16, 32),
                "names": ["channels", "height", "width"],
            },
            "observation.state": {"dtype": "float32", "shape": (2,), "names": ["a", "b"]},
            "action": {"dtype": "float32", "shape": (3,), "names": ["a", "b", "c"]},
        },
        stats={},
    )
    policy = make_policy(cfg.policy, ds_meta=meta)
    original_state = original.state_dict()
    for key, value in policy.state_dict().items():
        if key != "model.mot.lite_kv_fusion.logits":
            torch.testing.assert_close(value, original_state[key], rtol=0, atol=0)
    fusion = policy.model.mot.lite_kv_fusion
    assert torch.count_nonzero(fusion.logits) == 0
    policy.train()
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=0.01)
    loss, _ = policy(training_batch())
    loss.backward()
    assert torch.isfinite(loss)
    assert fusion.logits.grad is not None and fusion.logits.grad.abs().sum() > 0
    optimizer.step()
    assert torch.count_nonzero(fusion.logits) > 0
    policy.save_pretrained(tmp_path / "converted")
    loaded = FastWAMPolicy.from_pretrained(tmp_path / "converted", strict=True)
    torch.testing.assert_close(loaded.state_dict(), policy.state_dict(), rtol=0, atol=0)
    incompatible = deepcopy(policy.config)
    incompatible.retracted_1d_rope = False
    with pytest.raises(ValueError, match="checkpoint.*architecture"):
        FastWAMPolicy.from_pretrained(tmp_path / "converted", config=incompatible)
    # An embodiment change must still keep the compatible backbone weights.
    embodiment = tiny_config(dtype=policy.config.dtype, action_dim=2, proprio_dim=3, **DOT_FLAGS)
    adapted = FastWAMPolicy.from_pretrained(tmp_path / "converted", config=embodiment)
    assert adapted.config.action_dim == 2 and adapted.config.proprio_dim == 3
    torch.testing.assert_close(
        adapted.model.video_expert.time_embedding[0].weight,
        policy.model.video_expert.time_embedding[0].weight,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_fusion_uses_float32_softmax_with_policy_dtype_storage(dtype):
    fusion = LiteKVFusion(1, 2, 3).to(dtype=dtype)
    with torch.no_grad():
        fusion.logits.copy_(torch.tensor([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]))
    expected = torch.softmax(fusion.logits[0].float(), dim=-1)
    assert fusion.logits.dtype == dtype
    assert fusion.mixing_weights(0).dtype == torch.float32
    torch.testing.assert_close(fusion.mixing_weights(0), expected, rtol=0, atol=0)
    cache = [
        {key: torch.randn(1, 2, 12, dtype=dtype, requires_grad=True) for key in ("k", "v")} for _ in range(3)
    ]
    k, v = fusion(cache, 0)
    assert k.dtype == v.dtype == dtype
    (k.float().square().mean() + v.float().square().mean()).backward()
    assert torch.isfinite(fusion.logits.grad).all() and fusion.logits.grad.abs().sum() > 0
    assert all(torch.isfinite(entry[key].grad).all() for entry in cache for key in ("k", "v"))
