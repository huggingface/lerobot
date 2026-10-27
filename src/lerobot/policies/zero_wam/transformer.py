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

"""Zero-WAM mixture-of-transformers backbone (``WanICLTransformer3DModel``).

Vendored from upstream ``wan_va/modules/icl_model.py`` so the released checkpoints load
unchanged. Every block has a video expert and an action expert that attend jointly; human-video
(ICL) tokens run through the video expert as persistent clean context. ICL tokens are visible to
target video queries only, never to action queries.

Changes against upstream: the frame/height/width RoPE and timestep/text embedders are imported from
the LingBot-VA port (functionally identical), ``torch.compile`` of flex attention is deferred to
first use, and a dense-mask SDPA backend (``set_attention_backend("sdpa")``) is available for CPU
and for environments without a working flex-attention compiler.
"""

import math
from copy import deepcopy
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torch.utils.checkpoint
from einops import rearrange
from torch.nn.attention.flex_attention import (
    and_masks,
    create_block_mask,
    create_mask,
    flex_attention,
    or_masks,
)

from lerobot.policies.lingbot_va.utils import WanRotaryPosEmbed, WanTimeTextImageEmbedding
from lerobot.utils.import_utils import _diffusers_available

if TYPE_CHECKING or _diffusers_available:
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from diffusers.models.attention import FeedForward
    from diffusers.models.modeling_utils import ModelMixin
    from diffusers.models.normalization import FP32LayerNorm
else:
    FeedForward = FP32LayerNorm = None

    class ModelMixin:
        pass

    class ConfigMixin:
        pass

    def register_to_config(func):
        return func


ICL_CACHE_TYPE = 2
PREDICTION_CACHE_TYPE = 1
OBSERVATION_CACHE_TYPE = 0

_ATTENTION_BACKENDS = ("flex", "sdpa")
_compiled: dict = {}


def _compiled_flex_attention(*args, **kwargs):
    if "attn" not in _compiled:
        _compiled["attn"] = torch.compile(flex_attention, dynamic=True)
    return _compiled["attn"](*args, **kwargs)


def _compiled_create_block_mask(*args, **kwargs):
    if "mask" not in _compiled:
        _compiled["mask"] = torch.compile(create_block_mask)
    return _compiled["mask"](*args, **kwargs)


class _DenseMask:
    """Boolean ``[1, 1, Q, KV]`` mask for the SDPA backend.

    Query rows without any visible key (alignment padding) are given key 0 so SDPA never produces
    NaN; those rows are discarded by the caller, like flex attention's zero rows.
    """

    def __init__(self, mask_mod, q_len: int, kv_len: int, device: torch.device):
        mask = create_mask(mask_mod, 1, 1, q_len, kv_len, device=device)
        mask[..., 0] |= ~mask.any(dim=-1)
        self.mask = mask


def _apply_rotary_emb(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    x_complex = torch.view_as_complex(x.to(torch.float64).reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(x_complex * freqs).flatten(3).to(x.dtype)


class ICLAttentionBackend:
    """Holds the per-forward masks shared by all Transformer blocks."""

    self_mask = None
    cross_mask = None
    backend = "flex"

    @classmethod
    def set_backend(cls, backend: str) -> None:
        if backend not in _ATTENTION_BACKENDS:
            raise ValueError(f"attention backend must be one of {_ATTENTION_BACKENDS}, got {backend!r}")
        cls.backend = backend

    @classmethod
    def _make_mask(cls, mask_mod, q_len: int, kv_len: int, device, compile_mask: bool):
        if cls.backend == "sdpa":
            return _DenseMask(mask_mod, q_len, kv_len, device)
        mask_builder = _compiled_create_block_mask if compile_mask else create_block_mask
        return mask_builder(mask_mod, 1, 1, q_len, kv_len, device=device, _compile=compile_mask)

    @classmethod
    def build_self_mask(
        cls,
        query_type_ids: torch.Tensor,
        key_type_ids: torch.Tensor,
        query_seq_ids: torch.Tensor,
        key_seq_ids: torch.Tensor,
        query_frame_ids: torch.Tensor,
        key_frame_ids: torch.Tensor,
        window_size: int,
        device: torch.device,
        compile_mask: bool = True,
    ):
        def window_mask(b, h, q_idx, kv_idx):
            return (window_size == -1) | (
                (query_frame_ids[q_idx] - key_frame_ids[kv_idx]).abs() <= window_size
            )

        def valid_sequence_mask(b, h, q_idx, kv_idx):
            return (key_type_ids[kv_idx] != -1) & (query_seq_ids[q_idx] == key_seq_ids[kv_idx])

        def non_icl_key_mask(b, h, q_idx, kv_idx):
            return key_type_ids[kv_idx] != ICL_CACHE_TYPE

        def video_to_icl_mask(b, h, q_idx, kv_idx):
            return (
                (query_type_ids[q_idx] == 0)
                & (key_type_ids[kv_idx] == ICL_CACHE_TYPE)
                & (query_seq_ids[q_idx] == key_seq_ids[kv_idx])
            )

        mask_mod = or_masks(
            and_masks(valid_sequence_mask, window_mask, non_icl_key_mask),
            video_to_icl_mask,
        )

        cls.self_mask = cls._make_mask(mask_mod, len(query_type_ids), len(key_type_ids), device, compile_mask)
        return cls.self_mask

    @classmethod
    def build_training_self_mask(
        cls,
        seq_ids: torch.Tensor,
        frame_ids: torch.Tensor,
        noise_ids: torch.Tensor,
        type_ids: torch.Tensor,
        icl_ids: torch.Tensor,
        window_size: int,
        device: torch.device,
        compile_mask: bool = True,
    ):
        """Build the temporal-forcing mask for one Robotwin ICL sample."""

        def valid_pair(b, h, q_idx, kv_idx):
            return (seq_ids[q_idx] >= 0) & (seq_ids[q_idx] == seq_ids[kv_idx])

        def in_window(b, h, q_idx, kv_idx):
            return (window_size == -1) | ((frame_ids[q_idx] - frame_ids[kv_idx]).abs() <= window_size)

        def base_tf_pair(b, h, q_idx, kv_idx):
            clean_to_clean = (
                (noise_ids[q_idx] == 1) & (noise_ids[kv_idx] == 1) & (frame_ids[kv_idx] <= frame_ids[q_idx])
            )
            noisy_to_clean = (
                (noise_ids[q_idx] == 0) & (noise_ids[kv_idx] == 1) & (frame_ids[kv_idx] < frame_ids[q_idx])
            )
            noisy_to_noisy = (
                (noise_ids[q_idx] == 0) & (noise_ids[kv_idx] == 0) & (frame_ids[kv_idx] == frame_ids[q_idx])
            )
            return (
                (icl_ids[q_idx] == 0)
                & (icl_ids[kv_idx] == 0)
                & (clean_to_clean | noisy_to_clean | noisy_to_noisy)
            )

        def target_video_to_icl(b, h, q_idx, kv_idx):
            return (
                (seq_ids[q_idx] >= 0)
                & (seq_ids[q_idx] == seq_ids[kv_idx])
                & (icl_ids[q_idx] == 0)
                & (icl_ids[kv_idx] == 1)
                & (type_ids[q_idx] == 0)
            )

        def icl_to_icl(b, h, q_idx, kv_idx):
            return (
                (seq_ids[q_idx] >= 0)
                & (seq_ids[q_idx] == seq_ids[kv_idx])
                & (icl_ids[q_idx] == 1)
                & (icl_ids[kv_idx] == 1)
            )

        def mask_mod(b, h, q_idx, kv_idx):
            base = (
                valid_pair(b, h, q_idx, kv_idx)
                & in_window(b, h, q_idx, kv_idx)
                & base_tf_pair(b, h, q_idx, kv_idx)
            )
            return base | target_video_to_icl(b, h, q_idx, kv_idx) | icl_to_icl(b, h, q_idx, kv_idx)

        cls.self_mask = cls._make_mask(mask_mod, len(seq_ids), len(seq_ids), device, compile_mask)
        return cls.self_mask

    @classmethod
    def build_cross_mask(
        cls,
        query_seq_ids: torch.Tensor,
        encoder_seq_ids: torch.Tensor,
        device: torch.device,
        compile_mask: bool = True,
    ):
        def sequence_mask(b, h, q_idx, kv_idx):
            return query_seq_ids[q_idx] == encoder_seq_ids[kv_idx]

        cls.cross_mask = cls._make_mask(
            sequence_mask, len(query_seq_ids), len(encoder_seq_ids), device, compile_mask
        )
        return cls.cross_mask

    @classmethod
    def apply(cls, query, key, value, *, cross_attention: bool, block_mask=None):
        if block_mask is None:
            block_mask = cls.cross_mask if cross_attention else cls.self_mask
        if block_mask is None:
            raise RuntimeError("Attention mask was not initialized")
        output_dtype = value.dtype
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        if isinstance(block_mask, _DenseMask):
            query = query.to(value.dtype)
            key = key.to(value.dtype)
            return F.scaled_dot_product_attention(query, key, value, attn_mask=block_mask.mask).transpose(
                1, 2
            )
        half_dtypes = (torch.float16, torch.bfloat16)
        if query.dtype not in half_dtypes:
            query = query.to(torch.bfloat16)
        if key.dtype not in half_dtypes:
            key = key.to(torch.bfloat16)
        if value.dtype not in half_dtypes:
            value = value.to(torch.bfloat16)
        query = query.to(value.dtype)
        key = key.to(value.dtype)

        return (
            _compiled_flex_attention(
                query,
                key,
                value,
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
            .transpose(1, 2)
            .to(output_dtype)
        )


class WanICLAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        eps: float,
        dropout: float = 0.0,
        cross_attention_dim_head: int | None = None,
        action_inner_dim: int = -1,
        attn_moe: bool = True,
    ):
        super().__init__()
        self.inner_dim = heads * dim_head
        self.heads = heads
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = (
            self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads
        )
        action_dim = action_inner_dim if action_inner_dim > 0 else dim

        self.to_q = nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, dim, bias=True), nn.Dropout(dropout)])
        self.norm_q = nn.RMSNorm(self.inner_dim, eps=eps, elementwise_affine=True)
        self.norm_k = nn.RMSNorm(self.inner_dim, eps=eps, elementwise_affine=True)

        if not attn_moe:
            raise ValueError("The released ICL checkpoints require attn_moe=True")
        self.action_to_q = nn.Linear(action_dim, self.inner_dim, bias=True)
        self.action_to_out = nn.ModuleList(
            [nn.Linear(self.inner_dim, action_dim, bias=True), nn.Dropout(dropout)]
        )
        self.action_norm_q = deepcopy(self.norm_q)
        if cross_attention_dim_head is None:
            self.action_to_k = nn.Linear(action_dim, self.kv_inner_dim, bias=True)
            self.action_to_v = nn.Linear(action_dim, self.kv_inner_dim, bias=True)
            self.action_norm_k = deepcopy(self.norm_k)
        else:
            # Keep these aliases: they are present in the released state dict.
            self.action_to_k = self.to_k
            self.action_to_v = self.to_v
            self.action_norm_k = self.norm_k

        self.attn_caches: dict[str, dict[str, torch.Tensor]] = {}
        self.self_block_mask = None
        self.cross_block_mask = None

    def set_block_masks(self, self_mask=None, cross_mask=None) -> None:
        self.self_block_mask = self_mask
        self.cross_block_mask = cross_mask

    def clear_cache(self, cache_name: str) -> None:
        self.attn_caches.pop(cache_name, None)

    def prune_cache(self, cache_name: str, keep: torch.Tensor) -> None:
        cache = self.attn_caches.get(cache_name)
        if cache is None:
            return
        cache["k"] = cache["k"][:, keep]
        cache["v"] = cache["v"][:, keep]

    def _project(
        self,
        hs_latent: torch.Tensor | None,
        hs_action: torch.Tensor | None,
        hs_pad: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
    ):
        latent_length = 0 if hs_latent is None else hs_latent.shape[1]
        action_length = 0 if hs_action is None else hs_action.shape[1]

        if encoder_hidden_states is not None:
            query_parts = []
            if hs_latent is not None:
                query_parts.append(self.norm_q(self.to_q(hs_latent)).unflatten(2, (self.heads, -1)))
            if hs_action is not None:
                query_parts.append(
                    self.action_norm_q(self.action_to_q(hs_action)).unflatten(2, (self.heads, -1))
                )
            if not query_parts:
                raise ValueError("Cross attention requires video or action queries")
            reference = query_parts[0]
            query_parts.append(
                reference.new_zeros(reference.shape[0], hs_pad.shape[1], self.heads, reference.shape[-1])
            )
            query = torch.cat(query_parts, dim=1)
            key = self.norm_k(self.to_k(encoder_hidden_states)).unflatten(2, (self.heads, -1))
            value = self.to_v(encoder_hidden_states).unflatten(2, (self.heads, -1))
            return query, key, value, latent_length, action_length

        projected = []
        if hs_latent is not None:
            projected.append(
                (
                    self.norm_q(self.to_q(hs_latent)).unflatten(2, (self.heads, -1)),
                    self.norm_k(self.to_k(hs_latent)).unflatten(2, (self.heads, -1)),
                    self.to_v(hs_latent).unflatten(2, (self.heads, -1)),
                )
            )
        if hs_action is not None:
            projected.append(
                (
                    self.action_norm_q(self.action_to_q(hs_action)).unflatten(2, (self.heads, -1)),
                    self.action_norm_k(self.action_to_k(hs_action)).unflatten(2, (self.heads, -1)),
                    self.action_to_v(hs_action).unflatten(2, (self.heads, -1)),
                )
            )
        if not projected:
            raise ValueError("Self attention requires video or action tokens")
        reference = projected[0][0]
        padding = reference.new_zeros(reference.shape[0], hs_pad.shape[1], self.heads, reference.shape[-1])
        query = torch.cat([part[0] for part in projected] + [padding], dim=1)
        key = torch.cat([part[1] for part in projected] + [padding], dim=1)
        value = torch.cat([part[2] for part in projected] + [padding], dim=1)
        return query, key, value, latent_length, action_length

    def forward(
        self,
        hs_latent: torch.Tensor | None,
        hs_action: torch.Tensor | None,
        hs_pad: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        rotary_emb: torch.Tensor | None,
        update_cache: int,
        cache_name: str,
    ):
        query, key, value, latent_length, action_length = self._project(
            hs_latent, hs_action, hs_pad, encoder_hidden_states
        )
        is_cross_attention = encoder_hidden_states is not None
        if rotary_emb is not None:
            query = _apply_rotary_emb(query, rotary_emb)
            key = _apply_rotary_emb(key, rotary_emb)

        if not is_cross_attention:
            cache = self.attn_caches.get(cache_name)
            if cache is not None:
                key = torch.cat([cache["k"], key], dim=1)
                value = torch.cat([cache["v"], value], dim=1)

        hidden_states = ICLAttentionBackend.apply(
            query,
            key,
            value,
            cross_attention=is_cross_attention,
            block_mask=(self.cross_block_mask if is_cross_attention else self.self_block_mask),
        ).flatten(2, 3)

        if not is_cross_attention and update_cache:
            self.attn_caches[cache_name] = {"k": key, "v": value}

        out_latent = None
        out_action = None
        if latent_length:
            out_latent = self.to_out[1](self.to_out[0](hidden_states[:, :latent_length]))
        if action_length:
            action_start = latent_length
            out_action = self.action_to_out[1](
                self.action_to_out[0](hidden_states[:, action_start : action_start + action_length])
            )
        return out_latent, out_action


class WanICLTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        action_ffn_dim: int,
        num_heads: int,
        cross_attn_norm: bool,
        eps: float,
        action_inner_dim: int,
        attn_moe: bool = True,
        fully_share: bool = False,
    ):
        super().__init__()
        if fully_share:
            raise ValueError("The released ICL checkpoints require fully_share=False")
        action_dim = action_inner_dim if action_inner_dim > 0 else dim

        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.action_norm1 = FP32LayerNorm(action_dim, eps, elementwise_affine=False)
        self.attn1 = WanICLAttention(
            dim,
            num_heads,
            dim // num_heads,
            eps,
            action_inner_dim=action_dim,
            attn_moe=attn_moe,
        )
        self.attn2 = WanICLAttention(
            dim,
            num_heads,
            dim // num_heads,
            eps,
            cross_attention_dim_head=dim // num_heads,
            action_inner_dim=action_dim,
            attn_moe=attn_moe,
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.action_norm2 = (
            FP32LayerNorm(action_dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        )
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.action_ffn = FeedForward(
            action_dim,
            inner_dim=action_ffn_dim,
            activation_fn="gelu-approximate",
        )
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.action_norm3 = FP32LayerNorm(action_dim, eps, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.scale_shift_table_action = nn.Parameter(torch.randn(1, 6, action_dim) / action_dim**0.5)

    @staticmethod
    def _modulation(table, temb):
        values = table[None] + temb.float()
        return values.unbind(dim=2)

    def forward(
        self,
        hs_latent: torch.Tensor | None,
        hs_action: torch.Tensor | None,
        hs_pad: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb_latent: torch.Tensor | None,
        temb_action: torch.Tensor | None,
        rotary_emb: torch.Tensor,
        update_cache: int = 0,
        cache_name: str = "pos",
    ):
        latent_mod: tuple[torch.Tensor, ...] = (
            self._modulation(self.scale_shift_table, temb_latent) if hs_latent is not None else ()
        )
        action_mod: tuple[torch.Tensor, ...] = (
            self._modulation(self.scale_shift_table_action, temb_action) if hs_action is not None else ()
        )

        norm_latent = None
        if hs_latent is not None:
            shift, scale = latent_mod[0], latent_mod[1]
            norm_latent = (self.norm1(hs_latent.float()) * (1 + scale) + shift).to(hs_latent.dtype)
        norm_action = None
        if hs_action is not None:
            shift, scale = action_mod[0], action_mod[1]
            norm_action = (self.action_norm1(hs_action.float()) * (1 + scale) + shift).to(hs_action.dtype)

        attn_latent, attn_action = self.attn1(
            norm_latent,
            norm_action,
            hs_pad,
            None,
            rotary_emb,
            update_cache,
            cache_name,
        )
        if hs_latent is not None:
            hs_latent = (hs_latent.float() + attn_latent.float() * latent_mod[2]).to(hs_latent.dtype)
        if hs_action is not None:
            hs_action = (hs_action.float() + attn_action.float() * action_mod[2]).to(hs_action.dtype)

        cross_latent, cross_action = self.attn2(
            None if hs_latent is None else self.norm2(hs_latent.float()).to(hs_latent.dtype),
            None if hs_action is None else self.action_norm2(hs_action.float()).to(hs_action.dtype),
            hs_pad,
            encoder_hidden_states,
            None,
            0,
            cache_name,
        )
        if hs_latent is not None:
            hs_latent = hs_latent + cross_latent
            ff_input = (self.norm3(hs_latent.float()) * (1 + latent_mod[4]) + latent_mod[3]).to(
                hs_latent.dtype
            )
            hs_latent = (hs_latent.float() + self.ffn(ff_input).float() * latent_mod[5]).to(hs_latent.dtype)
        if hs_action is not None:
            hs_action = hs_action + cross_action
            ff_input = (self.action_norm3(hs_action.float()) * (1 + action_mod[4]) + action_mod[3]).to(
                hs_action.dtype
            )
            hs_action = (hs_action.float() + self.action_ffn(ff_input).float() * action_mod[5]).to(
                hs_action.dtype
            )
        return hs_latent, hs_action


class WanICLTransformer3DModel(ModelMixin, ConfigMixin):
    """Inference model for the released Robotwin ICL checkpoints."""

    _supports_gradient_checkpointing = False
    _no_split_modules = ["WanICLTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size=(1, 2, 2),
        num_attention_heads=24,
        attention_head_dim=128,
        in_channels=48,
        out_channels=48,
        action_dim=30,
        text_dim=4096,
        freq_dim=256,
        ffn_dim=14336,
        num_layers=30,
        cross_attn_norm=True,
        eps=1e-6,
        rope_max_seq_len=1024,
        pos_embed_seq_len=None,
        action_inner_dim=3072,
        action_ffn_dim=14336,
        action_embedder_bias=True,
        attn_moe=True,
        fully_share=False,
        attn_window=64,
        enable_mcp=True,
        num_mcp_modules=4,
        mcp_blocks_per_group=1,
        mcp_hidden_collect_layers=(3, 11, 19, 29),
    ):
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.action_inner_dim = action_inner_dim
        self.attn_window = attn_window

        self.rope = WanRotaryPosEmbed(attention_head_dim, self.patch_size, rope_max_seq_len)
        patch_channels = in_channels * math.prod(self.patch_size)
        self.patch_embedding_mlp = nn.Linear(patch_channels, self.inner_dim)
        # Kept for exact compatibility with the training checkpoint.
        self.patch_embedding = nn.Conv3d(
            in_channels,
            self.inner_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.action_embedder = nn.Linear(action_dim, action_inner_dim, bias=action_embedder_bias)
        self.condition_embedder = WanTimeTextImageEmbedding(
            self.inner_dim,
            freq_dim,
            self.inner_dim * 6,
            text_dim,
            pos_embed_seq_len,
        )
        self.condition_embedder_action = deepcopy(self.condition_embedder)

        block_kwargs = {
            "dim": self.inner_dim,
            "ffn_dim": ffn_dim,
            "action_ffn_dim": action_ffn_dim,
            "num_heads": num_attention_heads,
            "cross_attn_norm": cross_attn_norm,
            "eps": eps,
            "action_inner_dim": action_inner_dim,
            "attn_moe": attn_moe,
            "fully_share": fully_share,
        }
        self.blocks = nn.ModuleList([WanICLTransformerBlock(**block_kwargs) for _ in range(num_layers)])

        self.enable_mcp = enable_mcp
        self.mcp_hidden_collect_layers = list(mcp_hidden_collect_layers)
        if enable_mcp:
            self.mcp_mlp_hidden = nn.Sequential(
                nn.Linear(
                    self.inner_dim * len(self.mcp_hidden_collect_layers),
                    self.inner_dim,
                    bias=True,
                ),
                nn.SiLU(),
                nn.Linear(self.inner_dim, self.inner_dim, bias=True),
            )
            self.mcp_projections = nn.ModuleList(
                [nn.Linear(self.inner_dim * 2, self.inner_dim, bias=True) for _ in range(num_mcp_modules)]
            )
            self.mcp_blocks = nn.ModuleList(
                [
                    nn.ModuleList(
                        [WanICLTransformerBlock(**block_kwargs) for _ in range(mcp_blocks_per_group)]
                    )
                    for _ in range(num_mcp_modules)
                ]
            )

        self.norm_out = FP32LayerNorm(self.inner_dim, eps, elementwise_affine=False)
        self.action_norm_out = FP32LayerNorm(action_inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(self.inner_dim, out_channels * math.prod(self.patch_size))
        self.action_proj_out = nn.Linear(action_inner_dim, action_dim)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, self.inner_dim) / self.inner_dim**0.5)
        self.scale_shift_table_action = nn.Parameter(
            torch.randn(1, 2, action_inner_dim) / action_inner_dim**0.5
        )
        self.gradient_checkpointing = False
        self.clear_cache()

    def tie_cross_attention_aliases(self) -> None:
        """Re-point each cross-attention's action K/V to the shared video K/V (after ``assign`` loads)."""
        for block in [*self.blocks, *(b for group in getattr(self, "mcp_blocks", []) for b in group)]:
            block.attn2.action_to_k = block.attn2.to_k
            block.attn2.action_to_v = block.attn2.to_v
            block.attn2.action_norm_k = block.attn2.norm_k

    def _run_block(self, block, *args):
        if not (self.gradient_checkpointing and torch.is_grad_enabled()):
            return block(*args)
        # Masks live on the attention modules and are replaced by every later forward; the backward
        # recompute must see this forward's masks, not the ones of the last sample in the batch.
        self_mask, cross_mask = block.attn1.self_block_mask, block.attn2.cross_block_mask

        def run(*inputs):
            block.attn1.set_block_masks(self_mask=self_mask)
            block.attn2.set_block_masks(cross_mask=cross_mask)
            return block(*inputs)

        return torch.utils.checkpoint.checkpoint(run, *args, use_reentrant=False)

    def disable_mcp_modules(self) -> None:
        self.enable_mcp = False

    def clear_cache(self, cache_name: str = "pos") -> None:
        self.type_ids_cache = None
        self.seq_ids_cache = None
        self.frame_ids_cache = None
        self.cache_type_ids_cache = None
        if hasattr(self, "blocks"):
            for block in self.blocks:
                block.attn1.clear_cache(cache_name)

    def _prune_cache(self, keep: torch.Tensor, cache_name: str = "pos") -> None:
        if self.type_ids_cache is None:
            return
        self.type_ids_cache = self.type_ids_cache[keep]
        self.seq_ids_cache = self.seq_ids_cache[keep]
        self.frame_ids_cache = self.frame_ids_cache[keep]
        self.cache_type_ids_cache = self.cache_type_ids_cache[keep]
        for block in self.blocks:
            block.attn1.prune_cache(cache_name, keep)

    def clear_prediction_cache(self, attn_window: int | None = None, cache_name: str = "pos") -> None:
        """Drop generated type-1 entries while retaining ICL for the episode."""
        if self.cache_type_ids_cache is None:
            return
        keep_icl = self.cache_type_ids_cache == ICL_CACHE_TYPE
        keep_obs = self.cache_type_ids_cache == OBSERVATION_CACHE_TYPE
        window = self.attn_window if attn_window is None else attn_window
        if window > 0 and keep_obs.any():
            next_frame_id = self.frame_ids_cache[keep_obs].max() + 1
            keep_obs = keep_obs & ((self.frame_ids_cache - next_frame_id).abs() <= window)
        self._prune_cache(keep_icl | keep_obs, cache_name)

    def cache_counts(self) -> dict[int, int]:
        if self.cache_type_ids_cache is None:
            return {}
        valid_tokens = self.frame_ids_cache != -1
        return {
            cache_type: int(((self.cache_type_ids_cache == cache_type) & valid_tokens).sum())
            for cache_type in (
                OBSERVATION_CACHE_TYPE,
                PREDICTION_CACHE_TYPE,
                ICL_CACHE_TYPE,
            )
        }

    def _embed_stream(self, stream, mode):
        latents = stream["noisy_latents"]
        if mode == "video":
            hidden_states = rearrange(
                latents,
                "b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)",
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                p3=self.patch_size[2],
            )
            hidden_states = self.patch_embedding_mlp(hidden_states)
            repeats = (latents.shape[-2] // self.patch_size[1]) * (latents.shape[-1] // self.patch_size[2])
            condition_embedder = self.condition_embedder
        else:
            hidden_states = rearrange(latents, "b c f h w -> b (f h w) c")
            hidden_states = self.action_embedder(hidden_states)
            repeats = latents.shape[-2] * latents.shape[-1]
            condition_embedder = self.condition_embedder_action
        timesteps = stream["timesteps"]
        if timesteps.ndim == 1:
            timesteps = timesteps[None]
        token_timesteps = torch.repeat_interleave(timesteps, repeats, dim=1)
        temb, timestep_proj = condition_embedder(token_timesteps, dtype=hidden_states.dtype)
        return hidden_states, temb, timestep_proj.unflatten(2, (6, -1))

    def _append_metadata(
        self,
        query_types,
        query_seq_ids,
        query_frame_ids,
        cache_type_ids,
    ) -> None:
        stored_types = query_types.clone()
        stored_types[cache_type_ids == ICL_CACHE_TYPE] = ICL_CACHE_TYPE
        for name, value in (
            ("type_ids_cache", stored_types),
            ("seq_ids_cache", query_seq_ids),
            ("frame_ids_cache", query_frame_ids),
            ("cache_type_ids_cache", cache_type_ids),
        ):
            previous = getattr(self, name)
            setattr(self, name, value if previous is None else torch.cat([previous, value]))

    @staticmethod
    def _set_masks(blocks, self_mask, cross_mask) -> None:
        for block in blocks:
            block.attn1.set_block_masks(self_mask=self_mask)
            block.attn2.set_block_masks(cross_mask=cross_mask)

    def _training_embed(self, latents, timesteps, mode):
        model_dtype = self.patch_embedding_mlp.weight.dtype
        stream = {
            "noisy_latents": latents.to(dtype=model_dtype),
            "timesteps": timesteps,
        }
        return self._embed_stream(stream, mode)

    @staticmethod
    def _pad_metadata(value, padded_length):
        return F.pad(value, (0, padded_length), value=-1)

    def _build_training_masks(
        self,
        seq_ids,
        frame_ids,
        noise_ids,
        type_ids,
        icl_ids,
        cross_seq_ids,
        encoder_seq_ids,
        window_size,
        blocks,
    ) -> None:
        self_mask = ICLAttentionBackend.build_training_self_mask(
            seq_ids=seq_ids,
            frame_ids=frame_ids,
            noise_ids=noise_ids,
            type_ids=type_ids,
            icl_ids=icl_ids,
            window_size=window_size,
            device=seq_ids.device,
        )
        cross_mask = ICLAttentionBackend.build_cross_mask(
            query_seq_ids=cross_seq_ids,
            encoder_seq_ids=encoder_seq_ids,
            device=seq_ids.device,
        )
        self._set_masks(blocks, self_mask, cross_mask)

    def _forward_training_mcp(
        self,
        input_dict,
        collected_hidden_states,
        hs_action,
        action_timestep_proj,
        text_hidden_states,
        encoder_seq_ids,
        target_grid,
        action_grid,
        target_frame_ids,
        action_frame_ids,
        target_clean_timestep_proj,
        target_length,
        action_length,
    ):
        mcp_streams = input_dict.get("mcp_latent_dicts", [])
        if not self.enable_mcp or not mcp_streams:
            return []
        if len(collected_hidden_states) != len(self.mcp_hidden_collect_layers):
            raise RuntimeError("MCP did not collect every configured backbone hidden state")

        fused = self.mcp_mlp_hidden(torch.cat(collected_hidden_states, dim=-1))
        main_noisy = fused[:, :target_length]
        main_clean = fused[:, target_length : target_length * 2]
        device = main_noisy.device
        outputs = []

        for module_index, (group, stream) in enumerate(zip(self.mcp_blocks, mcp_streams, strict=True)):
            future_hs, future_temb, future_timestep_proj = self._training_embed(
                stream["noisy_latents"], stream["timesteps"], "video"
            )
            if future_hs.shape[1] != target_length:
                raise ValueError("MCP and target video token lengths must match")
            mcp_hs = self.mcp_projections[module_index](torch.cat([main_noisy, future_hs], dim=-1))
            mcp_hs = torch.cat([mcp_hs, main_clean], dim=1)
            mcp_action = hs_action

            future_grid = stream["grid_id"][0]
            full_grid = torch.cat([future_grid, target_grid, action_grid, action_grid], dim=1)
            total_length = target_length * 2 + action_length * 2
            padded_length = 128 - total_length % 128
            hs_pad = mcp_hs.new_zeros(mcp_hs.shape[0], padded_length, self.inner_dim)
            rotary_emb = self.rope(full_grid[None])[:, :, None]
            rotary_emb = F.pad(rotary_emb, (0, 0, 0, 0, 0, padded_length))

            seq_ids = torch.zeros(total_length, device=device, dtype=torch.int)
            frame_ids = torch.cat(
                [
                    target_frame_ids,
                    target_frame_ids,
                    action_frame_ids,
                    action_frame_ids,
                ]
            )
            noise_ids = torch.cat(
                [
                    torch.zeros(target_length, device=device, dtype=torch.int),
                    torch.ones(target_length, device=device, dtype=torch.int),
                    torch.zeros(action_length, device=device, dtype=torch.int),
                    torch.ones(action_length, device=device, dtype=torch.int),
                ]
            )
            type_ids = torch.cat(
                [
                    torch.zeros(target_length * 2, device=device, dtype=torch.int),
                    torch.ones(action_length * 2, device=device, dtype=torch.int),
                ]
            )
            icl_ids = torch.zeros_like(type_ids)
            cross_seq_ids = torch.zeros_like(seq_ids)
            seq_ids = self._pad_metadata(seq_ids, padded_length)
            frame_ids = self._pad_metadata(frame_ids, padded_length)
            noise_ids = self._pad_metadata(noise_ids, padded_length)
            type_ids = self._pad_metadata(type_ids, padded_length)
            icl_ids = self._pad_metadata(icl_ids, padded_length)
            cross_seq_ids = self._pad_metadata(cross_seq_ids, padded_length)
            self._build_training_masks(
                seq_ids,
                frame_ids,
                noise_ids,
                type_ids,
                icl_ids,
                cross_seq_ids,
                encoder_seq_ids,
                input_dict["window_size"],
                group,
            )

            mcp_timestep_proj = torch.cat([future_timestep_proj, target_clean_timestep_proj], dim=1)
            for block in group:
                mcp_hs, mcp_action = self._run_block(
                    block,
                    mcp_hs,
                    mcp_action,
                    hs_pad,
                    text_hidden_states,
                    mcp_timestep_proj,
                    action_timestep_proj,
                    rotary_emb,
                )

            mcp_output = mcp_hs[:, :target_length]
            shift, scale = (self.scale_shift_table[None] + future_temb[:, :, None]).unbind(dim=2)
            mcp_output = (self.norm_out(mcp_output.float()) * (1 + scale) + shift).to(mcp_output.dtype)
            mcp_output = self.proj_out(mcp_output)
            outputs.append(
                rearrange(
                    mcp_output,
                    "1 l (n c) -> 1 (l n) c",
                    n=math.prod(self.patch_size),
                )
            )
        return outputs

    def forward_train(self, input_dict):
        latent = input_dict["latent_dict"]
        action = input_dict["action_dict"]
        if latent["noisy_latents"].shape[0] != 1:
            raise ValueError("Open-source ICL training uses one sample per rank; set batch_size=1")

        target_noisy_hs, target_temb, target_noisy_timestep_proj = self._training_embed(
            latent["noisy_latents"], latent["timesteps"], "video"
        )
        target_clean_hs, _, target_clean_timestep_proj = self._training_embed(
            latent["latent"], latent["cond_timesteps"], "video"
        )
        action_noisy_hs, action_temb, action_noisy_timestep_proj = self._training_embed(
            action["noisy_latents"], action["timesteps"], "action"
        )
        action_clean_hs, _, action_clean_timestep_proj = self._training_embed(
            action["latent"], action["cond_timesteps"], "action"
        )

        target_length = target_noisy_hs.shape[1]
        action_length = action_noisy_hs.shape[1]
        latent_parts = [target_noisy_hs, target_clean_hs]
        latent_timestep_parts = [
            target_noisy_timestep_proj,
            target_clean_timestep_proj,
        ]
        grid_parts = [latent["grid_id"][0], latent["grid_id"][0]]

        icl = input_dict.get("icl_latent_dict")
        icl_length = 0
        if icl is not None:
            icl_hs, _, icl_timestep_proj = self._training_embed(icl["latent"], icl["timesteps"], "video")
            icl_length = icl_hs.shape[1]
            latent_parts.append(icl_hs)
            latent_timestep_parts.append(icl_timestep_proj)
            grid_parts.append(icl["grid_id"][0])

        hs_latent = torch.cat(latent_parts, dim=1)
        hs_action = torch.cat([action_noisy_hs, action_clean_hs], dim=1)
        latent_timestep_proj = torch.cat(latent_timestep_parts, dim=1)
        action_timestep_proj = torch.cat([action_noisy_timestep_proj, action_clean_timestep_proj], dim=1)
        target_grid = latent["grid_id"][0]
        action_grid = action["grid_id"][0]
        full_grid = torch.cat(grid_parts + [action_grid, action_grid], dim=1)

        total_length = hs_latent.shape[1] + hs_action.shape[1]
        padded_length = 128 - total_length % 128
        hs_pad = hs_latent.new_zeros(hs_latent.shape[0], padded_length, self.inner_dim)
        rotary_emb = self.rope(full_grid[None])[:, :, None]
        rotary_emb = F.pad(rotary_emb, (0, 0, 0, 0, 0, padded_length))

        device = hs_latent.device
        chunk_size = int(input_dict["chunk_size"])
        max_chunk_size = int(input_dict["max_frame_chunk_size"])
        target_frame_ids = (target_grid[0] // chunk_size * 2).to(torch.int)
        action_frame_ids = (action_grid[0] // chunk_size * 2 + 1).to(torch.int)
        if icl is not None:
            icl_frame_ids = (icl["grid_id"][0][0] // max_chunk_size).to(torch.int)
        else:
            icl_frame_ids = torch.empty(0, device=device, dtype=torch.int)

        seq_ids = torch.zeros(total_length, device=device, dtype=torch.int)
        frame_ids = torch.cat(
            [
                target_frame_ids,
                target_frame_ids,
                icl_frame_ids,
                action_frame_ids,
                action_frame_ids,
            ]
        )
        noise_ids = torch.cat(
            [
                torch.zeros(target_length, device=device, dtype=torch.int),
                torch.ones(target_length, device=device, dtype=torch.int),
                torch.ones(icl_length, device=device, dtype=torch.int),
                torch.zeros(action_length, device=device, dtype=torch.int),
                torch.ones(action_length, device=device, dtype=torch.int),
            ]
        )
        type_ids = torch.cat(
            [
                torch.zeros(
                    target_length * 2 + icl_length,
                    device=device,
                    dtype=torch.int,
                ),
                torch.ones(action_length * 2, device=device, dtype=torch.int),
            ]
        )
        icl_ids = torch.cat(
            [
                torch.zeros(target_length * 2, device=device, dtype=torch.int),
                torch.ones(icl_length, device=device, dtype=torch.int),
                torch.zeros(action_length * 2, device=device, dtype=torch.int),
            ]
        )
        cross_seq_ids = torch.cat(
            [
                torch.zeros(target_length * 2, device=device, dtype=torch.int),
                torch.ones(icl_length, device=device, dtype=torch.int),
                torch.zeros(action_length * 2, device=device, dtype=torch.int),
            ]
        )
        seq_ids = self._pad_metadata(seq_ids, padded_length)
        frame_ids = self._pad_metadata(frame_ids, padded_length)
        noise_ids = self._pad_metadata(noise_ids, padded_length)
        type_ids = self._pad_metadata(type_ids, padded_length)
        icl_ids = self._pad_metadata(icl_ids, padded_length)
        cross_seq_ids = self._pad_metadata(cross_seq_ids, padded_length)

        text_emb = input_dict["text_emb"].to(
            device=device, dtype=self.condition_embedder.text_embedder.linear_1.weight.dtype
        )
        text_hidden_states = self.condition_embedder.text_embedder(text_emb)
        encoder_seq_ids = input_dict["encoder_seq_ids"].to(device=device, dtype=torch.int)
        self._build_training_masks(
            seq_ids,
            frame_ids,
            noise_ids,
            type_ids,
            icl_ids,
            cross_seq_ids,
            encoder_seq_ids,
            input_dict["window_size"],
            self.blocks,
        )

        collected = []
        for layer_index, block in enumerate(self.blocks):
            hs_latent, hs_action = self._run_block(
                block,
                hs_latent,
                hs_action,
                hs_pad,
                text_hidden_states,
                latent_timestep_proj,
                action_timestep_proj,
                rotary_emb,
            )
            if self.enable_mcp and layer_index in self.mcp_hidden_collect_layers:
                collected.append(hs_latent[:, : target_length * 2])

        latent_output = hs_latent[:, :target_length]
        action_output = hs_action[:, :action_length]
        shift, scale = (self.scale_shift_table[None] + target_temb[:, :, None]).unbind(dim=2)
        latent_output = (self.norm_out(latent_output.float()) * (1 + scale) + shift).to(latent_output.dtype)
        latent_output = self.proj_out(latent_output)
        latent_output = rearrange(
            latent_output,
            "1 l (n c) -> 1 (l n) c",
            n=math.prod(self.patch_size),
        )

        action_shift, action_scale = (self.scale_shift_table_action[None] + action_temb[:, :, None]).unbind(
            dim=2
        )
        action_output = (self.action_norm_out(action_output.float()) * (1 + action_scale) + action_shift).to(
            action_output.dtype
        )
        action_output = self.action_proj_out(action_output)

        mcp_outputs = self._forward_training_mcp(
            input_dict=input_dict,
            collected_hidden_states=collected,
            hs_action=hs_action,
            action_timestep_proj=action_timestep_proj,
            text_hidden_states=text_hidden_states,
            encoder_seq_ids=encoder_seq_ids,
            target_grid=target_grid,
            action_grid=action_grid,
            target_frame_ids=target_frame_ids,
            action_frame_ids=action_frame_ids,
            target_clean_timestep_proj=target_clean_timestep_proj,
            target_length=target_length,
            action_length=action_length,
        )
        if self.enable_mcp:
            return latent_output, action_output, mcp_outputs
        return latent_output, action_output

    def _forward_stream(
        self,
        input_dict,
        mode: str,
        update_cache: int,
        cache_name: str,
        clean_window_cache: bool,
    ):
        stream_key = "latent_res_lst" if mode == "video" else "action_res_lst"
        grid_key = "latent_grid_id" if mode == "video" else "action_grid_id"
        stream = input_dict[stream_key]
        hidden_states, temb, timestep_proj = self._embed_stream(stream, mode)
        if hidden_states.shape[0] != 1:
            raise ValueError("ICL inference expects a single sequence")

        grid_id = input_dict.get(grid_key, stream.get("grid_id"))
        if grid_id is None:
            raise KeyError(f"Missing {grid_key}")
        if grid_id.ndim == 3:
            grid_id = grid_id[0]
        rotary_emb = self.rope(grid_id[None])[:, :, None]

        token_count = hidden_states.shape[1]
        device = hidden_states.device
        query_type = torch.full([token_count], 0 if mode == "video" else 1, device=device, dtype=torch.int)
        query_seq = input_dict.get(
            "current_seq_ids", torch.zeros(token_count, device=device, dtype=torch.int)
        ).to(device=device, dtype=torch.int)
        query_frame = input_dict.get("current_frame_ids", grid_id[0].to(device=device, dtype=torch.int)).to(
            device=device, dtype=torch.int
        )
        cache_types = stream.get(
            "cache_type_ids",
            torch.full(
                [token_count],
                int(input_dict.get("cache_type_id", PREDICTION_CACHE_TYPE)),
                device=device,
                dtype=torch.int,
            ),
        ).to(device=device, dtype=torch.int)
        for name, value in (
            ("current_seq_ids", query_seq),
            ("current_frame_ids", query_frame),
            ("cache_type_ids", cache_types),
        ):
            if value.numel() != token_count:
                raise ValueError(f"{name} has {value.numel()} entries for {token_count} tokens")

        padded_length = 128 - token_count % 128
        hs_pad = hidden_states.new_zeros(1, padded_length, self.inner_dim)
        rotary_emb = F.pad(rotary_emb, (0, 0, 0, 0, 0, padded_length))
        query_type = F.pad(query_type, (0, padded_length), value=-1)
        query_seq = F.pad(query_seq, (0, padded_length), value=-1)
        query_frame = F.pad(query_frame, (0, padded_length), value=-1)
        # VA keeps alignment padding in the KV cache for persistent ICL and
        # observation chunks.  Preserve the owning cache type so pruning drops
        # prediction padding but retains the same KV layout as deployment.  The
        # modality type remains -1, so these padding keys are still masked out.
        padding_cache_type = int(cache_types[0].item())
        cache_types = F.pad(cache_types, (0, padded_length), value=padding_cache_type)

        key_type = query_type if self.type_ids_cache is None else torch.cat([self.type_ids_cache, query_type])
        key_seq = query_seq if self.seq_ids_cache is None else torch.cat([self.seq_ids_cache, query_seq])
        key_frame = (
            query_frame if self.frame_ids_cache is None else torch.cat([self.frame_ids_cache, query_frame])
        )
        self_mask = ICLAttentionBackend.build_self_mask(
            query_type,
            key_type,
            query_seq,
            key_seq,
            query_frame,
            key_frame,
            self.attn_window,
            device,
            compile_mask=mode != "action",
        )

        text_emb = input_dict["text_emb"]
        text_hidden_states = self.condition_embedder.text_embedder(text_emb)
        encoder_seq = input_dict.get(
            "encoder_seq_ids",
            torch.zeros(text_hidden_states.shape[1], device=device, dtype=torch.int),
        ).to(device=device, dtype=torch.int)
        cross_mask = ICLAttentionBackend.build_cross_mask(
            query_seq,
            encoder_seq,
            device,
            compile_mask=mode != "action",
        )
        self._set_masks(self.blocks, self_mask, cross_mask)

        hs_latent = hidden_states if mode == "video" else None
        hs_action = hidden_states if mode == "action" else None
        for block in self.blocks:
            hs_latent, hs_action = block(
                hs_latent,
                hs_action,
                hs_pad,
                text_hidden_states,
                timestep_proj if mode == "video" else None,
                timestep_proj if mode == "action" else None,
                rotary_emb,
                update_cache=update_cache,
                cache_name=cache_name,
            )

        output_states = hs_latent if mode == "video" else hs_action
        assert output_states is not None
        output_table = self.scale_shift_table if mode == "video" else self.scale_shift_table_action
        output_norm = self.norm_out if mode == "video" else self.action_norm_out
        shift, scale = (output_table[None] + temb[:, :, None]).unbind(dim=2)
        output_states = (output_norm(output_states.float()) * (1 + scale) + shift).to(output_states.dtype)
        if mode == "video":
            output = self.proj_out(output_states)
            output = rearrange(
                output,
                "1 l (n c) -> 1 (l n) c",
                n=math.prod(self.patch_size),
            )
        else:
            output = self.action_proj_out(output_states)

        if update_cache:
            self._append_metadata(query_type, query_seq, query_frame, cache_types)
            if clean_window_cache:
                self.clear_prediction_cache(cache_name=cache_name)
        return output

    def forward(
        self,
        input_dict,
        update_cache=0,
        cache_name="pos",
        mode="forward_latent_only",
        clean_window_cache=False,
        return_dict=False,
        train_mode=False,
        **kwargs,
    ):
        del kwargs
        if return_dict:
            raise ValueError("WanICLTransformer3DModel returns tuples or tensors")
        if train_mode:
            return self.forward_train(input_dict)
        if mode in ("forward_latent_only", "forward_latent_only_flash"):
            return self._forward_stream(
                input_dict,
                "video",
                update_cache,
                cache_name,
                clean_window_cache,
            )
        if mode in ("forward_action_only", "forward_action_only_flash"):
            return self._forward_stream(
                input_dict,
                "action",
                update_cache,
                cache_name,
                clean_window_cache,
            )
        raise ValueError(f"Unsupported ICL inference mode: {mode}")


__all__ = [
    "ICL_CACHE_TYPE",
    "OBSERVATION_CACHE_TYPE",
    "PREDICTION_CACHE_TYPE",
    "WanICLTransformer3DModel",
]
