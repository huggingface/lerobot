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

from copy import deepcopy

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")
pytest.importorskip("diffusers")

from lerobot.policies.fastwam import modeling_fastwam
from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy

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


@pytest.mark.parametrize("flags", [{}, DOT_FLAGS], ids=["default", "fasterwam"])
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        pytest.param(
            torch.bfloat16,
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="bfloat16 requires CUDA autocast"),
        ),
    ],
)
def test_public_dtype_train_infer_and_checkpoint(offline_components, tmp_path, flags, dtype):
    torch.manual_seed(1234)
    policy = FastWAMPolicy(
        tiny_config(dtype=dtype, device="cuda" if dtype == torch.bfloat16 else "cpu", **flags)
    )
    batch = {
        "observation.images.image": torch.rand(1, 3, 16, 32),
        "observation.state": torch.rand(1, 2),
        "context": torch.randn(1, 4, 8),
        "context_mask": torch.ones(1, 4, dtype=torch.bool),
        "action": torch.randn(1, 32, 3),
    }
    loss, _ = policy(batch)
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    assert any(grad.abs().sum() > 0 for grad in grads)
    assert policy.model.video_expert.time_embedding[0].weight.dtype == dtype
    noise = torch.randn(1, 8, 3)
    with torch.no_grad():
        policy.eval()
        action = policy.predict_action_chunk(batch, noise=noise)
        policy.save_pretrained(tmp_path)
        loaded = FastWAMPolicy.from_pretrained(tmp_path, strict=True)
        reloaded_action = loaded.predict_action_chunk(batch, noise=noise)
    assert action.shape == (1, 8, 3)
    assert torch.isfinite(action).all()
    torch.testing.assert_close(reloaded_action, action, rtol=0, atol=0)
    assert loaded.config.dtype == dtype


@pytest.mark.parametrize(
    "override", ["default", "retracted_1d_rope", "world_model_only_conditioning", "depth"]
)
def test_checkpoint_rejects_incompatible_explicit_config(offline_components, tmp_path, override):
    policy = FastWAMPolicy(tiny_config(**DOT_FLAGS))
    policy.save_pretrained(tmp_path)
    config = deepcopy(policy.config)
    if override == "default":
        config = tiny_config()
    elif override == "depth":
        config.action_dit_config["num_layers"] = 2
        config.action_expert_layers = 2
    else:
        setattr(config, override, False)
    with pytest.raises(ValueError, match="checkpoint.*architecture"):
        FastWAMPolicy.from_pretrained(tmp_path, config=config)


def test_checkpoint_preserves_explicit_embodiment_overrides(offline_components, tmp_path):
    policy = FastWAMPolicy(tiny_config(**DOT_FLAGS))
    with torch.no_grad():
        policy.model.video_expert.time_embedding[0].weight.fill_(0.125)
    policy.save_pretrained(tmp_path)
    config = tiny_config(action_dim=2, proprio_dim=3, **DOT_FLAGS)
    # A resolved depth is equivalent to the convenience override.
    config.action_expert_layers = None
    loaded = FastWAMPolicy.from_pretrained(tmp_path, config=config)
    assert loaded.config.action_dim == 2
    assert loaded.config.proprio_dim == 3
    torch.testing.assert_close(
        loaded.model.video_expert.time_embedding[0].weight, policy.model.video_expert.time_embedding[0].weight
    )
    assert loaded.model.action_expert.action_dim == 2


@pytest.mark.parametrize("flags", [{}, DOT_FLAGS], ids=["default", "fasterwam"])
def test_weights_only_checkpoint_with_explicit_config(offline_components, tmp_path, caplog, flags):
    policy = FastWAMPolicy(tiny_config(**flags))
    policy.save_pretrained(tmp_path)
    (tmp_path / "config.json").unlink()
    config = deepcopy(policy.config)

    loaded = FastWAMPolicy.from_pretrained(tmp_path, config=config, strict=True)

    assert loaded.config is config
    torch.testing.assert_close(loaded.state_dict(), policy.state_dict(), rtol=0, atol=0)
    assert "skipping architecture validation" in caplog.text
