# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real reduced-size video/action transformers; synthetic VAE, no downloaded weights."""

import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from lerobot.policies.longwam._core.action_dit import ActionDiT
from lerobot.policies.longwam._core.long_wam import LongWAM
from lerobot.policies.longwam._core.mot import MoT
from lerobot.policies.longwam._core.wan_video_dit import WanVideoDiT
from lerobot.policies.longwam._core.wan_video_vae import Resample
from lerobot.policies.longwam.modeling_longwam import LongWAMPolicy
from tests.policies.longwam.test_longwam_policy import config, observation, training_batch


class TinyVAE(nn.Module):
    temporal_downsample_factor = 4
    upsampling_factor = 8
    z_dim = 4

    def encode(self, video, **kwargs):
        sampled = video[:, :, ::4, ::8, ::8]
        return torch.cat([sampled, sampled.mean(1, keepdim=True)], dim=1)


def test_vae_cache_index_default_is_not_shared_between_calls():
    layer = Resample(4, mode="downsample3d")
    video = torch.zeros(1, 4, 1, 4, 4)
    for _ in range(2):
        output, cache, index = layer(video, feat_cache=[None])
        assert output.shape == (1, 4, 1, 2, 2)
        assert len(cache) == 1 and index == [1]


def tiny_core(cfg, **kwargs):
    common = {
        "hidden_dim": 32,
        "ffn_dim": 64,
        "text_dim": 16,
        "freq_dim": 16,
        "eps": 1e-6,
        "num_heads": 2,
        "attn_head_dim": 24,
        "num_layers": 1,
        "use_gradient_checkpointing": False,
    }
    video = WanVideoDiT(
        **common,
        in_dim=4,
        out_dim=4,
        patch_size=(1, 2, 2),
        has_image_input=False,
        seperated_timestep=True,
        fuse_vae_embedding_in_latents=True,
        video_attention_mask_mode="per_frame_causal",
    )
    action = ActionDiT(**common, action_dim=cfg.action_dim)
    return LongWAM(
        video_expert=video,
        action_expert=action,
        mot=MoT({"video": video, "action": action}, mot_checkpoint_mixed_attn=False),
        vae=TinyVAE(),
        text_dim=16,
        proprio_dim=cfg.proprio_dim,
        longwam_past_obs_size=cfg.past_obs_steps,
        longwam_num_imagine_frames=cfg.future_video_steps // 16,
        longwam_joint_denoise=cfg.denoising == "cod",
        longwam_imagine_sigma=cfg.imagine_sigma,
        longwam_imagine_infer_steps=1,
    )


@pytest.mark.parametrize("mode", ["idm", "cod"])
def test_real_transformer_loss_backward_save_load_and_inference(monkeypatch, tmp_path, mode):
    monkeypatch.setattr(LongWAMPolicy, "_build_core_model", staticmethod(tiny_core))
    cfg = config(denoising=mode, imagine_sigma=0 if mode == "cod" else 0.9, num_inference_steps=2)
    torch.manual_seed(7)
    policy = LongWAMPolicy(cfg)
    batch = training_batch(cfg)
    batch.update(context=torch.randn(2, 4, 16), context_mask=torch.ones(2, 4, dtype=torch.bool))
    optimizer = cfg.get_optimizer_preset().build(policy.get_optim_params())
    before = {k: p.detach().clone() for k, p in policy.model.named_parameters()}
    loss, metrics = policy(batch)
    assert torch.isfinite(loss)
    assert "loss_video" in metrics and "loss_action" in metrics
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in policy.parameters())
    optimizer.step()
    assert any(not torch.equal(before[k], p) for k, p in policy.model.named_parameters())
    policy.save_pretrained(tmp_path)
    restored = LongWAMPolicy.from_pretrained(tmp_path)
    data = observation(cfg, 0)
    data.update(context=batch["context"][:1], context_mask=batch["context_mask"][:1])
    try:
        left, right = policy.eval().select_action(data), restored.select_action(data)
        assert left.shape == (1, 3) and torch.isfinite(left).all()
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    finally:
        policy.close()
        restored.close()
