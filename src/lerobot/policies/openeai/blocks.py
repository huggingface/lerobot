# Copyright 2026 The OpenEAI team and The HuggingFace Inc. team. All rights reserved.
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

"""DiT blocks for OpenEAI VLA.

Adapted from OpenEAI-VLA/openeai/models/openeai/blocks.py
"""

import math
from typing import Final

import numpy as np
import torch
from timm.models.vision_transformer import Attention, Mlp, RmsNorm, use_fused_attn
from torch import nn


class CrossAttention(nn.Module):
    """A cross-attention layer with flash attention."""

    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0,
        proj_drop: float = 0,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn()

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, c: torch.Tensor, mask=None) -> torch.Tensor:
        batch_size, seq_len_x, dim_x = x.shape
        _, seq_len_c, _ = c.shape
        q = self.q(x).reshape(batch_size, seq_len_x, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = (
            self.kv(c).reshape(batch_size, seq_len_c, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.to(torch.bool)
            mask = mask.reshape(batch_size, 1, 1, seq_len_c).expand(-1, -1, seq_len_x, -1)

        if self.fused_attn:
            x = nn.functional.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                attn_mask=mask,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                attn = attn.masked_fill_(mask.logical_not(), float("-inf"))
            attn = attn.softmax(dim=-1)
            if self.attn_drop.p > 0:
                attn = self.attn_drop(attn)
            x = attn @ v

        x = x.permute(0, 2, 1, 3).reshape(batch_size, seq_len_x, dim_x)
        x = self.proj(x)
        if self.proj_drop.p > 0:
            x = self.proj_drop(x)
        return x


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = RmsNorm(hidden_dim, eps=1e-6)
        self.attn = Attention(
            dim=hidden_dim,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            norm_layer=RmsNorm,
            attn_drop=dropout,
            proj_drop=dropout,
        )
        self.norm2 = RmsNorm(hidden_dim, eps=1e-6)
        self.cross_attn = CrossAttention(
            dim=hidden_dim,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            norm_layer=RmsNorm,
            attn_drop=dropout,
            proj_drop=dropout,
        )
        self.ffn = Mlp(
            in_features=hidden_dim,
            hidden_features=int(hidden_dim * ff_ratio),
            out_features=hidden_dim,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=dropout,
        )
        self.norm3 = RmsNorm(hidden_dim, eps=1e-6)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, attn_mask=None, cond_mask=None):
        # Self-attention
        residual = x
        x = self.norm1(x)
        x = self.attn(x, attn_mask)
        x = residual + x

        # Cross-attention
        residual = x
        x = self.norm2(x)
        x = self.cross_attn(x, cond, cond_mask)
        x = residual + x

        # FFN
        residual = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = residual + x
        return x


def create_sinusoidal_pos_embedding(
    time: torch.Tensor, dimension: int, min_period: float = 4e-3, max_period: float = 4.0
) -> torch.Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=time.dtype, device=time.device)
    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos) -> np.ndarray:
    """Sin-cos positional embedding from grid coordinates.

    Args:
        embed_dim: embedding dimension (must be even).
        pos: positions, accepts numpy array, torch.Tensor, list, or scalar.

    Returns:
        Numpy array of shape (len(pos), embed_dim).
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    if isinstance(pos, torch.Tensor):
        pos = pos.detach().cpu().numpy()
    if not isinstance(pos, np.ndarray):
        pos = np.array(pos, dtype=np.float64)
    pos = pos.reshape(-1).astype(np.float64)
    out = np.einsum("m,d->md", pos, omega)
    emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)
    return emb


def make_timm_attn_mask(pad_mask: torch.Tensor) -> torch.Tensor:
    """Convert 1D pad mask to 2D attention mask for timm FlashAttention."""
    pad_2d_mask = pad_mask[:, None, :] * pad_mask[:, :, None]
    pad_2d_mask = torch.where(pad_2d_mask > 0, 0.0, float("-inf"))
    return pad_2d_mask.unsqueeze(1)
