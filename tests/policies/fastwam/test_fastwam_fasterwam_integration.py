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

"""CPU integration tests for the Faster-WAM (arXiv 2608.02365) DoT variant.

Covers: one training step through `training_loss` on the DoT path (gradients must reach
the video backbone AND the fusion logits), cached inference (`infer_action`) with the
prepared fused context, disabled-variant equivalence (the split DoT flow must match the
joint MoT path when Lite KV-Fusion is off), and the world-model-only context guards.
All models are tiny; nothing downloads and nothing touches a GPU.
"""

import pytest
import torch
from torch import nn

pytest.importorskip("transformers", reason="fastwam requires the `fastwam` extra (transformers)")
pytest.importorskip("diffusers", reason="fastwam requires the `fastwam` extra (diffusers)")

from lerobot.policies.fastwam.wan import ActionDiT, FastWAM, LiteKVFusion, MoT, WanVideoDiT

HEADS = 2
HEAD_DIM = 6
HIDDEN = HEADS * HEAD_DIM  # WanVideoDiT requires attn_head_dim == hidden_dim // num_heads
Z_DIM = 4
TEXT_DIM = 8
ACTION_DIM = 3


def _video_dit_config(num_layers: int) -> dict:
    return {
        "patch_size": [1, 2, 2],
        "in_dim": Z_DIM,
        "hidden_dim": HIDDEN,
        "ffn_dim": 16,
        "freq_dim": 8,
        "text_dim": TEXT_DIM,
        "out_dim": Z_DIM,
        "num_heads": HEADS,
        "attn_head_dim": HEAD_DIM,
        "num_layers": num_layers,
        "eps": 1.0e-6,
        "seperated_timestep": True,
        "use_gradient_checkpointing": False,
        "video_attention_mask_mode": "first_frame_causal",
        "action_conditioned": False,
        "action_dim": ACTION_DIM,
        "action_group_causal_mask_mode": "group_diagonal",
        "fp32_attention": False,
    }


def _action_dit_config(num_layers: int, use_text_cross_attention: bool = True) -> dict:
    return {
        "action_dim": ACTION_DIM,
        "hidden_dim": HIDDEN,
        "ffn_dim": 16,
        "num_heads": HEADS,
        "attn_head_dim": HEAD_DIM,
        "num_layers": num_layers,
        "text_dim": TEXT_DIM,
        "freq_dim": 8,
        "eps": 1.0e-6,
        "use_gradient_checkpointing": False,
        "fp32_attention": False,
        "use_text_cross_attention": use_text_cross_attention,
    }


class FakeWanVAE(nn.Module):
    """Shape-only VAE stand-in. Video [B,3,T,H,W] -> latents [B,Z,(T-1)//4+1,H/2,W/2];
    a single image (passed as a one-element list) -> [Z,1,H/2,W/2]."""

    z_dim = Z_DIM
    upsampling_factor = 2
    temporal_downsample_factor = 4

    def encode(self, video, device=None, tiled=False, tile_size=None, tile_stride=None):
        if isinstance(video, list):
            image = video[0]  # [C, 1, H, W]
            _, _, h, w = image.shape
            # The real VAE returns per-item [Z, 1, H', W']; `_encode_input_image_latents_tensor`
            # unsqueezes it back to [1, Z, 1, H', W'].
            return [torch.zeros(Z_DIM, 1, h // 2, w // 2)]
        b, _, t, h, w = video.shape
        return torch.zeros(b, Z_DIM, (t - 1) // 4 + 1, h // 2, w // 2)

    def decode(self, latents, device=None, tiled=False, tile_size=None, tile_stride=None):
        b, _, t, h, w = latents.shape
        return torch.zeros(b, 3, (t - 1) * 4 + 1, h * 2, w * 2)

    def requires_grad_(self, flag: bool):
        return self


def _build_fastwam(
    num_video_layers: int,
    num_action_layers: int,
    lite_kv_fusion: bool,
    world_model_only: bool,
) -> FastWAM:
    video_expert = WanVideoDiT(**_video_dit_config(num_video_layers))
    action_expert = ActionDiT(
        **_action_dit_config(num_action_layers, use_text_cross_attention=not world_model_only)
    )
    fusion = (
        LiteKVFusion(
            num_action_layers=num_action_layers,
            num_heads=HEADS,
            num_video_layers=num_video_layers,
        )
        if lite_kv_fusion
        else None
    )
    mot = MoT(
        mixtures={"video": video_expert, "action": action_expert},
        mot_checkpoint_mixed_attn=False,
        lite_kv_fusion=fusion,
        retracted_1d_rope=lite_kv_fusion,  # retraction requires fusion in this task's design
    )
    return FastWAM(
        video_expert=video_expert,
        action_expert=action_expert,
        mot=mot,
        vae=FakeWanVAE(),
        text_encoder=None,
        tokenizer=None,
        text_dim=TEXT_DIM,
        proprio_dim=None,
        device="cpu",
        torch_dtype=torch.float32,
        world_model_only_conditioning=world_model_only,
    )


def _training_sample() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "video": torch.rand(1, 3, 5, 16, 32),  # T%4==1; fake VAE -> latents [1,Z,2,8,16]
        "action": torch.randn(1, 32, ACTION_DIM),  # 32 % (5-1) == 0
        "context": torch.randn(1, 4, TEXT_DIM),
        "context_mask": torch.ones(1, 4, dtype=torch.bool),
    }


def test_dot_training_step_reaches_video_backbone_and_fusion_logits():
    torch.manual_seed(0)
    model = _build_fastwam(
        num_video_layers=3, num_action_layers=1, lite_kv_fusion=True, world_model_only=True
    )
    model.train()
    loss, metrics = model.training_loss(_training_sample())
    assert torch.isfinite(loss)
    assert set(metrics) >= {"loss_video", "loss_action"}
    loss.backward()

    fusion = model.mot.lite_kv_fusion
    assert fusion.logits.grad is not None
    assert fusion.logits.grad.abs().sum() > 0
    # Action-loss gradients must reach the video backbone through the fused cache.
    video_param = model.video_expert.blocks[0].self_attn.q.weight
    assert video_param.grad is not None
    assert video_param.grad.abs().sum() > 0
    # The world-model-only action expert has no cross-attention modules at all.
    assert not any("cross_attn" in name for name in model.action_expert.state_dict())


def test_dot_cached_inference_smoke():
    torch.manual_seed(1)
    model = _build_fastwam(
        num_video_layers=3, num_action_layers=1, lite_kv_fusion=True, world_model_only=True
    )
    model.eval()
    image = torch.rand(1, 3, 16, 32)
    out = model.infer_action(
        prompt=None,
        input_image=image,
        action_horizon=8,
        context=torch.randn(1, 4, TEXT_DIM),
        context_mask=torch.ones(1, 4, dtype=torch.bool),
        num_inference_steps=2,
        seed=0,
        rand_device="cpu",
    )
    action = out["action"]
    # `infer_action` strips the singleton batch dim.
    assert action.shape == (8, ACTION_DIM)
    assert torch.isfinite(action).all()


def test_split_dot_flow_matches_joint_flow_when_fusion_disabled():
    """Disabled-variant equivalence: with Lite KV-Fusion off and equal expert depths, the
    split flow (video prefill + identity prepare + action_cached) must reproduce the
    joint MoT forward for both experts on identical inputs and masks."""
    torch.manual_seed(2)
    expert_config = _action_dit_config(3)
    expert_config["action_dim"] = 8  # token width for this expert-level equivalence test
    video_expert = ActionDiT(**expert_config)
    action_expert = ActionDiT(**expert_config)
    mot = MoT(
        mixtures={"video": video_expert, "action": action_expert},
        mot_checkpoint_mixed_attn=False,
    ).eval()

    batch, seq_v, seq_a = 1, 6, 2
    # MoT operates on post-`pre_dit` tokens, i.e. already at the experts' hidden width.
    video_tokens = torch.randn(batch, seq_v, HIDDEN)
    action_tokens = torch.randn(batch, seq_a, HIDDEN)
    video_freqs = video_expert.freqs[:seq_v].view(seq_v, 1, -1)
    action_freqs = action_expert.freqs[:seq_a].view(seq_a, 1, -1)
    # AdaLN modulation lives at the expert's hidden width.
    video_t_mod = torch.zeros(batch, 6, HIDDEN)
    action_t_mod = torch.zeros(batch, 6, HIDDEN)
    # Cross-attention context is post-`text_embedding` (hidden width), as `_run_training_mot`
    # passes `pre_dit` outputs.
    video_ctx = {"context": torch.randn(batch, 2, HIDDEN), "mask": torch.ones(batch, 2, dtype=torch.bool)}
    action_ctx = {"context": torch.randn(batch, 2, HIDDEN), "mask": torch.ones(batch, 2, dtype=torch.bool)}
    video_vv_mask = torch.ones(seq_v, seq_v, dtype=torch.bool)
    # The real Fast-WAM/Faster-WAM joint mask: video rows never attend to action rows,
    # action rows attend to action rows plus the first-frame video tokens only. The
    # equivalence only holds under this masking (it is what `_build_mot_attention_mask`
    # builds).
    joint_mask = torch.zeros(seq_v + seq_a, seq_v + seq_a, dtype=torch.bool)
    joint_mask[:seq_v, :seq_v] = True
    joint_mask[seq_v:, seq_v:] = True
    joint_mask[seq_v:, :2] = True  # first-frame video tokens

    with torch.no_grad():
        tokens_joint = mot(
            embeds_all={"video": video_tokens, "action": action_tokens},
            attention_mask=joint_mask,
            freqs_all={"video": video_freqs, "action": action_freqs},
            context_all={"video": video_ctx, "action": action_ctx},
            t_mod_all={"video": video_t_mod, "action": action_t_mod},
        )
        video_out, cache = mot.run_video_prefill(
            video_tokens=video_tokens,
            video_freqs=video_freqs,
            video_t_mod=video_t_mod,
            video_context_payload=video_ctx,
            video_attention_mask=video_vv_mask,
        )
        action_kv = mot.prepare_action_kv_context(cache)
        action_out = mot.forward_action_with_video_cache(
            action_tokens=action_tokens,
            action_freqs=action_freqs,
            action_t_mod=action_t_mod,
            action_context_payload=action_ctx,
            video_kv_cache=action_kv,
            attention_mask=joint_mask,
            video_seq_len=seq_v,
        )

    torch.testing.assert_close(video_out, tokens_joint["video"], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(action_out, tokens_joint["action"], rtol=1e-5, atol=1e-5)


def test_world_model_only_context_guards():
    expert = ActionDiT(**_action_dit_config(1, use_text_cross_attention=False))
    with pytest.raises(ValueError, match="without text cross-attention"):
        expert.pre_dit(
            action_tokens=torch.randn(1, 2, ACTION_DIM),
            timestep=torch.zeros(1),
            context=torch.randn(1, 2, TEXT_DIM),
        )
    tokens = expert.pre_dit(
        action_tokens=torch.randn(1, 2, ACTION_DIM), timestep=torch.zeros(1), context=None
    )
    assert tokens["context"] is None and tokens["context_mask"] is None

    default_expert = ActionDiT(**_action_dit_config(1, use_text_cross_attention=True))
    with pytest.raises(ValueError, match="`context` is required"):
        default_expert.pre_dit(
            action_tokens=torch.randn(1, 2, ACTION_DIM), timestep=torch.zeros(1), context=None
        )
