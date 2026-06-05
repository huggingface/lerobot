# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
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

"""Flex-attention backend for the LingBot-VA Wan transformer (training only).

This module is imported lazily and ONLY when ``attn_mode='flex'`` is requested. It builds
the block-causal / window / noise-vs-clean attention masks used during the dual-stream
flow-matching training described in the LingBot-VA paper. Inference uses the ``torch``
SDPA backend (see :mod:`wan_attention`) which does not need flex-attention.

``torch.nn.attention.flex_attention`` requires a recent PyTorch build with the relevant
inductor support; importing this module on an unsupported build raises ``ImportError``.
"""

from collections.abc import Callable
from functools import partial
from typing import ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.attention.flex_attention import (
    BlockMask,
    and_masks,
    create_block_mask,
    flex_attention,
    or_masks,
)


class FlexAttnFunc(nn.Module):
    flex_attn: ClassVar[Callable] = torch.compile(flex_attention, dynamic=True)
    compiled_create_block_mask: ClassVar[Callable] = torch.compile(create_block_mask)
    attention_mask: ClassVar[BlockMask] = None
    cross_attention_mask: ClassVar[BlockMask] = None

    def __init__(self, is_cross=False) -> None:
        super().__init__()
        self.is_cross = is_cross

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dtype=torch.bfloat16,
    ) -> torch.Tensor:
        q_varlen = rearrange(query[0], "s n d -> 1 n s d")
        k_varlen = rearrange(key[0], "s n d -> 1 n s d")
        v_varlen = rearrange(value[0], "s n d -> 1 n s d")

        half_dtypes = (torch.float16, torch.bfloat16)
        assert dtype in half_dtypes

        def half(x):
            return x if x.dtype in half_dtypes else x.to(dtype)

        q_varlen = half(q_varlen)
        k_varlen = half(k_varlen)
        v_varlen = half(v_varlen)
        q_varlen = q_varlen.to(v_varlen.dtype)
        k_varlen = k_varlen.to(v_varlen.dtype)

        block_mask = FlexAttnFunc.cross_attention_mask if self.is_cross else FlexAttnFunc.attention_mask

        x_out = FlexAttnFunc.flex_attn(
            q_varlen,
            k_varlen,
            v_varlen,
            block_mask=block_mask,
            kernel_options={
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_M1": 32,
                "BLOCK_N1": 64,
                "BLOCK_M2": 64,
                "BLOCK_N2": 32,
            },
        )

        x_out = rearrange(x_out, "b n s d -> b s n d")
        return x_out

    @staticmethod
    @torch.no_grad()
    def init_mask(
        latent_shape,
        action_shape,
        padded_length,
        chunk_size,
        window_size,
        patch_size,
        device,
    ):
        torch._inductor.config.realize_opcount_threshold = 100
        B, _, L_F, L_H, L_W = latent_shape
        _, _, A_F, A_H, A_W = action_shape

        latent_seq_id = (
            torch.arange(B)[:, None, None, None]
            .expand(-1, L_F // patch_size[0], L_H // patch_size[1], L_W // patch_size[2])
            .flatten()
        )
        action_seq_id = torch.arange(B)[:, None, None, None].expand(-1, A_F, A_H, A_W).flatten()
        seq_ids = torch.cat([latent_seq_id] * 2 + [action_seq_id] * 2)

        latent_frame_id = (
            torch.arange(L_F)[None, :, None, None]
            .expand(B, -1, L_H // patch_size[1], L_W // patch_size[2])[None]
            .flatten()
        )
        action_frame_id = torch.arange(A_F)[None, :, None, None].expand(B, -1, A_H, A_W)[None].flatten()
        frame_ids = torch.cat(
            [latent_frame_id // chunk_size * 2] * 2 + [action_frame_id // chunk_size * 2 + 1] * 2
        )

        noise_ids = torch.cat(
            [
                torch.zeros_like(latent_frame_id),
                torch.ones_like(latent_frame_id),
                torch.zeros_like(action_frame_id),
                torch.ones_like(action_frame_id),
            ]
        )

        seq_ids = F.pad(seq_ids, (0, padded_length), value=-1)
        frame_ids = F.pad(frame_ids, (0, padded_length), value=-1)
        noise_ids = F.pad(noise_ids, (0, padded_length), value=-1)

        mask_mod = FlexAttnFunc._get_mask_mod(
            seq_ids.long().to(device), frame_ids.long().to(device), noise_ids.long().to(device), window_size
        )
        block_mask = FlexAttnFunc.compiled_create_block_mask(
            mask_mod, 1, 1, len(seq_ids), len(seq_ids), device=device, _compile=True
        )
        FlexAttnFunc.attention_mask = block_mask

        text_seq_ids = torch.arange(B)[:, None].expand(-1, 512).flatten()
        mask_mod_cross = FlexAttnFunc._get_cross_mask_mod(
            seq_ids.long().to(device), text_seq_ids.long().to(device)
        )
        block_mask_cross = FlexAttnFunc.compiled_create_block_mask(
            mask_mod_cross, 1, 1, len(seq_ids), len(text_seq_ids), device=device, _compile=True
        )
        FlexAttnFunc.cross_attention_mask = block_mask_cross

    @staticmethod
    @torch.no_grad()
    def _get_cross_mask_mod(seq_ids, text_seq_ids):
        def seq_mask(b, h, q_idx, kv_idx):
            return (
                (seq_ids[q_idx] == text_seq_ids[kv_idx]) & (seq_ids[q_idx] >= 0) & (text_seq_ids[kv_idx] >= 0)
            )

        return seq_mask

    @staticmethod
    @torch.no_grad()
    def _get_mask_mod(seq_ids, frame_ids, noise_ids, window_size):
        def seq_mask(b, h, q_idx, kv_idx):
            return (seq_ids[q_idx] == seq_ids[kv_idx]) & (seq_ids[q_idx] >= 0) & (seq_ids[kv_idx] >= 0)

        def block_causal_mask(b, h, q_idx, kv_idx):
            return frame_ids[kv_idx] <= frame_ids[q_idx]

        def block_causal_mask_exclude_self(b, h, q_idx, kv_idx):
            return frame_ids[kv_idx] < frame_ids[q_idx]

        def block_self_mask(b, h, q_idx, kv_idx):
            return frame_ids[kv_idx] == frame_ids[q_idx]

        def clean2clean_mask(b, h, q_idx, kv_idx):
            return (noise_ids[q_idx] == 1) & (noise_ids[kv_idx] == 1)

        def noise2clean_mask(b, h, q_idx, kv_idx):
            return (noise_ids[q_idx] == 0) & (noise_ids[kv_idx] == 1)

        def noise2noise_mask(b, h, q_idx, kv_idx):
            return (noise_ids[q_idx] == 0) & (noise_ids[kv_idx] == 0)

        def block_window_mask(b, h, q_idx, kv_idx, window_size: int):
            return (frame_ids[q_idx] - frame_ids[kv_idx]).abs() <= window_size

        mask_list = []
        mask_list.append(and_masks(clean2clean_mask, block_causal_mask))
        mask_list.append(and_masks(noise2clean_mask, block_causal_mask_exclude_self))
        mask_list.append(and_masks(noise2noise_mask, block_self_mask))
        mask = or_masks(*mask_list)
        mask = and_masks(mask, seq_mask)
        mask = and_masks(mask, partial(block_window_mask, window_size=window_size))
        return mask


__all__ = ["FlexAttnFunc"]
