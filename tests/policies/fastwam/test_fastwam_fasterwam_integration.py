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

"""Default joint/cache equivalence and action-context validation."""

import pytest
import torch

pytest.importorskip("transformers", reason="fastwam requires the `fastwam` extra (transformers)")
pytest.importorskip("diffusers", reason="fastwam requires the `fastwam` extra (diffusers)")

from lerobot.policies.fastwam.wan import ActionDiT, MoT

HEADS = 2
HEAD_DIM = 6
HIDDEN = HEADS * HEAD_DIM  # WanVideoDiT requires attn_head_dim == hidden_dim // num_heads
TEXT_DIM = 8
ACTION_DIM = 3


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


def test_missing_text_embedding_raises_explicit_error():
    expert = ActionDiT(**_action_dit_config(1))
    expert.text_embedding = None
    with pytest.raises(ValueError, match="text embedding module"):
        expert.pre_dit(torch.randn(1, 2, ACTION_DIM), torch.zeros(1), torch.randn(1, 2, TEXT_DIM))
