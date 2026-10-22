# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 The FastWAM Authors
# SPDX-License-Identifier: MIT AND Apache-2.0
"""Long-WAM action expert and mixed attention, adapted from NVlabs/LongLive (1b9e348).

Originally derived from FastWAM. Upstream license notice for the adapted code:

MIT License

Copyright (c) 2026 The FastWAM Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import TYPE_CHECKING, Self

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.checkpoint import checkpoint

from lerobot.utils.import_utils import _diffusers_available

from .configuration_longwam import LongWAMConfig

if TYPE_CHECKING or _diffusers_available:
    from diffusers import WanTransformer3DModel
    from diffusers.models.embeddings import apply_rotary_emb, get_1d_rotary_pos_embed
    from diffusers.models.transformers.transformer_wan import (
        WanAttention,
        WanAttnProcessor,
        WanTimeTextImageEmbedding,
        WanTransformerBlock,
    )

KVCache = list[tuple[Tensor, Tensor]]
Rotary = tuple[Tensor, Tensor]


class ActionExpert(nn.Module):
    def __init__(self, config: LongWAMConfig) -> None:
        super().__init__()
        vc, dim = config.video_config, config.action_hidden_dim
        heads, head_dim = vc["num_attention_heads"], vc["attention_head_dim"]
        self.action_encoder = nn.Linear(config.action_dim, dim)
        self.condition_embedder = WanTimeTextImageEmbedding(dim, vc["freq_dim"], 6 * dim, vc["text_dim"])
        self.blocks = nn.ModuleList()
        for _ in range(vc["num_layers"]):
            block = WanTransformerBlock(
                dim, config.action_ffn_dim, heads, cross_attn_norm=True, eps=vc.get("eps", 1e-6)
            )
            # The action residual width is smaller, but video/action QKV widths must match.
            for name, cross in (("attn1", False), ("attn2", True)):
                setattr(
                    block,
                    name,
                    WanAttention(
                        dim,
                        heads,
                        head_dim,
                        eps=vc.get("eps", 1e-6),
                        is_cross_attention=cross,
                        processor=WanAttnProcessor(),
                    ),
                )
            self.blocks.append(block)
        self.head = nn.Linear(dim, config.action_dim)
        cos, sin = get_1d_rotary_pos_embed(
            head_dim,
            max(config.action_horizon, config.prediction_steps),
            use_real=True,
            repeat_interleave_real=True,
            freqs_dtype=torch.float32 if torch.backends.mps.is_available() else torch.float64,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)


def _attention_inputs(block: nn.Module, x: Tensor, modulation: Tensor, rope: Rotary):
    # Preserve the released Long-WAM checkpoint's modulation/residual precision.
    shift, scale, gate, ff_shift, ff_scale, ff_gate = (
        block.scale_shift_table.unsqueeze(1).to(modulation) + modulation
    ).unbind(2)
    h = block.norm1(x) * (1 + scale) + shift
    attn = block.attn1
    q = attn.norm_q(attn.to_q(h)).unflatten(-1, (attn.heads, -1)).transpose(1, 2)
    k = attn.norm_k(attn.to_k(h)).unflatten(-1, (attn.heads, -1)).transpose(1, 2)
    v = attn.to_v(h).unflatten(-1, (attn.heads, -1)).transpose(1, 2)
    return apply_rotary_emb(q, rope), apply_rotary_emb(k, rope), v, (gate, ff_shift, ff_scale, ff_gate)


def _finish_block(block: nn.Module, x: Tensor, attention: Tensor, gates, context: Tensor) -> Tensor:
    gate, shift, scale, ff_gate = gates
    attention = attention.transpose(1, 2).flatten(2)
    attention = block.attn1.to_out[1](block.attn1.to_out[0](attention))
    x = x + attention * gate
    x = x + block.attn2(block.norm2(x), context)
    h = block.norm3(x) * (1 + scale) + shift
    return x + block.ffn(h) * ff_gate


def _mixed_block(video_block, action_block, video, action, video_condition, action_condition, mask, cache):
    """One shared attention operation; IDM can reuse detached video keys and values."""
    video_qkv = action_qkv = None
    if video is not None:
        video_qkv = _attention_inputs(video_block, video, video_condition[0], video_condition[2])
    if action is not None:
        action_qkv = _attention_inputs(action_block, action, action_condition[0], action_condition[2])
    qkv = [values[:3] for values in (video_qkv, action_qkv) if values is not None]
    q, k, v = (torch.cat(values, dim=2) for values in zip(*qkv, strict=True))
    if cache is not None:
        k, v = torch.cat((cache[0], k), dim=2), torch.cat((cache[1], v), dim=2)
    attention = functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    video_len = 0 if video is None else video.shape[1]
    if video_qkv is not None:
        video = _finish_block(
            video_block, video, attention[:, :, :video_len], video_qkv[3], video_condition[1]
        )
        cache = video_qkv[1:3]
    if action_qkv is not None:
        action = _finish_block(
            action_block, action, attention[:, :, video_len:], action_qkv[3], action_condition[1]
        )
    return video, action, cache


class LongWAMDiT(nn.Module):
    """Wan video backbone plus Long-WAM's action expert and causal cross-expert mask."""

    def __init__(self, config: LongWAMConfig) -> None:
        super().__init__()
        self.config = config
        self.video = WanTransformer3DModel.from_config(config.video_config)
        self.action = ActionExpert(config)

    def _apply(self, fn, recurse: bool = True) -> Self:
        # Moving weights to BF16 must not round the rotary frequencies.
        rotations = [
            (self.video.rope, "freqs_cos"),
            (self.video.rope, "freqs_sin"),
            (self.action, "rope_cos"),
            (self.action, "rope_sin"),
        ]
        saved = [getattr(module, name) for module, name in rotations]
        super()._apply(fn, recurse)
        device = self.video.patch_embedding.weight.device
        for (module, name), value in zip(rotations, saved, strict=True):
            setattr(module, name, value.to(device=device))
        return self

    def forward(
        self,
        latents: Tensor | None,
        actions: Tensor | None,
        video_time: Tensor | None,
        action_time: Tensor | None,
        context: Tensor,
        *,
        video_cache: KVCache | None = None,
        return_cache: bool = False,
    ) -> tuple[Tensor | None, Tensor | None, KVCache]:
        video = action = video_condition = action_condition = mask = None
        cfg = self.config
        if latents is not None:
            if video_time is None:
                raise ValueError("Video latents require a video timestep")
            b, _, frames, height, width = latents.shape
            pt, ph, pw = self.video.config.patch_size
            grid = (frames // pt, height // ph, width // pw)
            tokens_per_frame = grid[1] * grid[2]
            frame_ids = torch.arange(frames, device=latents.device).repeat_interleave(tokens_per_frame)
            times = video_time[:, None] * (frame_ids >= cfg.past_obs_steps // 16 + 1)
            embed, modulation, text, _ = self.video.condition_embedder(
                times.flatten(), context, timestep_seq_len=times.shape[1]
            )
            video_condition = (
                modulation.unflatten(-1, (6, -1)),
                text,
                tuple(freq.squeeze(0).squeeze(1) for freq in self.video.rope(latents)),
            )
            video = self.video.patch_embedding(latents).flatten(2).transpose(1, 2)
            video_len = video.shape[1]
            action_len = 0 if actions is None else actions.shape[1]
            mask = torch.zeros(
                video_len + action_len, video_len + action_len, dtype=torch.bool, device=video.device
            )
            mask[:video_len, :video_len] = frame_ids[:, None] >= frame_ids[None, :]
            mask[video_len:, :] = True
            if actions is not None and cfg.denoising == "cod":
                mask[(cfg.past_obs_steps // 16 + 1) * tokens_per_frame : video_len, :] = True
        if actions is not None:
            _, modulation, text, _ = self.action.condition_embedder(action_time, context)
            action_condition = (
                modulation.unflatten(-1, (6, -1)).unsqueeze(1),
                text,
                (self.action.rope_cos[: actions.shape[1]], self.action.rope_sin[: actions.shape[1]]),
            )
            action = self.action.action_encoder(actions)
        caches = []
        for i, (vb, ab) in enumerate(zip(self.video.blocks, self.action.blocks, strict=True)):
            args = (
                vb,
                ab,
                video,
                action,
                video_condition,
                action_condition,
                mask,
                None if video_cache is None else video_cache[i],
            )
            if self.training and cfg.use_gradient_checkpointing and torch.is_grad_enabled():
                video, action, kv = checkpoint(_mixed_block, *args, use_reentrant=False)
            else:
                video, action, kv = _mixed_block(*args)
            if return_cache:
                caches.append(kv)
        if video is not None:
            shift, scale = (self.video.scale_shift_table.unsqueeze(0) + embed.unsqueeze(2)).unbind(2)
            video = self.video.proj_out(self.video.norm_out(video) * (1 + scale) + shift)
            video = video.reshape(b, *grid, pt, ph, pw, -1).permute(0, 7, 1, 4, 2, 5, 3, 6)
            video = video.flatten(6, 7).flatten(4, 5).flatten(2, 3)
        if action is not None:
            action = self.action.head(action)
        return video, action, caches
