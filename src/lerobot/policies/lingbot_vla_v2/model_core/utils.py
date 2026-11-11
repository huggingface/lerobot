# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
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

import math

import einops
import torch
import torch.nn as nn
from torch import Tensor


def create_sinusoidal_pos_embedding(
    time: torch.tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device="cpu",
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float32, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def sample_beta(alpha, beta, bsize, device):
    gamma1 = torch.rand((bsize,), device=device).pow(1 / alpha)
    gamma2 = torch.rand((bsize,), device=device).pow(1 / beta)
    return gamma1 / (gamma1 + gamma2)


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def our_eager_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
):
    """
    Performs eager attention, optimized with torch.einsum.

    Args:
        query_states: Query tensor of shape [batch_size, seq_len, num_attention_heads, head_dim].
        key_states: Key tensor of shape [batch_size, seq_len, num_key_value_heads, head_dim].
        value_states: Value tensor of shape [batch_size, seq_len, num_key_value_heads, head_dim].
        attention_mask: Bool attention mask (True = attend) of shape [batch_size, seq_len, seq_len]
            or [batch_size, 1, seq_len, seq_len]. None applies no mask.

    Returns:
        Output tensor of shape [batch_size, seq_len, num_attention_heads * head_dim].
    """
    bsize, seq_len, num_att_heads, head_dim = query_states.shape
    num_key_value_heads = key_states.shape[2]
    num_key_value_groups = num_att_heads // num_key_value_heads

    key_states = einops.repeat(key_states, "b l h d -> b l (h g) d", g=num_key_value_groups)
    value_states = einops.repeat(value_states, "b l h d -> b l (h g) d", g=num_key_value_groups)

    query_states_permuted = torch.einsum("blhd->bhld", query_states)
    key_states_permuted = torch.einsum("blhd->bhld", key_states)

    att_weights = torch.einsum("bhqd,bhkd->bhqk", query_states_permuted, key_states_permuted)
    att_weights *= head_dim**-0.5

    big_neg = -2.3819763e38
    if attention_mask is not None:
        if attention_mask.dim() == 3:  # [B, L, L] -> [B, 1, L, L] to broadcast over heads
            attention_mask = attention_mask[:, None, :, :]
        att_weights = torch.where(attention_mask, att_weights, big_neg)

    probs = nn.functional.softmax(att_weights, dim=-1)
    probs = probs.to(dtype=value_states.dtype)

    value_states_permuted = torch.einsum("blhd->bhld", value_states)  # [B, H, L_v, D]
    att_output = torch.einsum("bhqk,bhkv->bhqv", probs, value_states_permuted)  # [B, H, L_q, D]
    att_output = torch.einsum("bhld->blhd", att_output)  # [B, L, H, D]
    att_output = att_output.reshape(bsize, seq_len, num_att_heads * head_dim)

    return att_output


def our_sdpa_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
):
    """SDPA attention with the SAME (b, l, h, d) in / (b, l, h*d) out contract as
    ``our_eager_attention_forward``.

    Uses ``torch.nn.functional.scaled_dot_product_attention`` — the same softmax attention
    (fidelity-preserving: identical math up to floating-point reassociation, no approximation),
    but it fuses the softmax and never materializes the ``[b, h, q, k]`` score matrix, so it is
    O(seq) in memory instead of O(seq^2) like the eager path. Torch-native (auto-selects the
    flash / memory-efficient / math backend), no compiled dependency. Grouped-query attention
    is handled by ``enable_gqa`` (num_kv_heads < num_att_heads). The default SDPA scale is
    ``1/sqrt(head_dim)``, matching the eager path.

    Args:
        query_states: ``[batch, seq, num_att_heads, head_dim]``.
        key_states / value_states: ``[batch, seq, num_kv_heads, head_dim]``.
        attention_mask: bool tensor, ``True`` = attend; ``[batch, seq, seq]`` or ``[batch, 1, seq, seq]``.
    """
    bsize, seq_len, num_att_heads, head_dim = query_states.shape
    num_kv_heads = key_states.shape[2]

    # (b, l, h, d) -> (b, h, l, d)
    q = query_states.transpose(1, 2)
    k = key_states.transpose(1, 2)
    v = value_states.transpose(1, 2)

    mask = attention_mask
    if mask is not None:
        if mask.dim() == 3:  # (b, q, k) -> (b, 1, q, k) to broadcast over heads
            mask = mask.unsqueeze(1)
        if mask.dtype != torch.bool:
            mask = mask.bool()

    att_output = nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,  # bool: True keeps, False masks (matches the eager where-mask)
        enable_gqa=num_kv_heads != num_att_heads,
    )

    # (b, h, l, d) -> (b, l, h*d)
    att_output = att_output.transpose(1, 2).reshape(bsize, seq_len, num_att_heads * head_dim)
    return att_output
