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

import torch
import torch.nn.functional as F  # noqa: N812
from packaging.version import Version

if Version(torch.__version__) > Version("2.5.0"):
    # Ffex attention is only available from torch 2.5 onwards
    from torch.nn.attention.flex_attention import (
        _mask_mod_signature,
        _round_up_to_multiple,
        create_block_mask,
        create_mask,
        flex_attention,
    )

# more aggressive compile setting
_flex_attention = torch.compile(flex_attention, mode="max-autotune-no-cudagraphs")


def flex_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    scaling=None,
):
    """
    This is defined out of classes to make compile happy.
    """
    batch_size, seq_len, num_att_heads, head_dim = query_states.shape
    original_dtype = query_states.dtype
    # num_key_value_heads = key_states.shape[2]
    # num_key_value_groups = num_att_heads // num_key_value_heads

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    query_states = query_states.to(torch.float32)
    key_states = key_states.to(torch.float32)
    value_states = value_states.to(torch.float32)

    causal_mask = attention_mask
    if causal_mask is not None:
        causal_mask = causal_mask[:, None, :, : key_states.shape[2]]

        if causal_mask.shape[1] == 1 and query_states.shape[1] > 1:
            causal_mask = causal_mask.expand(-1, query_states.shape[1], -1, -1)

    def precomputed_mask_factory(precomputed_mask: torch.Tensor) -> _mask_mod_signature:
        def mask_mod(b, h, q_idx, kv_idx):
            # Danger zone: if b,h,q_idx,kv_idx exceed the shape, device-side assert occurs.
            return precomputed_mask[b][h][q_idx][kv_idx]

        return mask_mod

    b_mask, h_mask, q_len, kv_len = causal_mask.shape  # The shape of your mask

    block_size = 128  # some gpus do not support 64
    q_len_rounded = _round_up_to_multiple(q_len, block_size)
    kv_len_rounded = _round_up_to_multiple(kv_len, block_size)

    # *CRITICAL* we do need to expand here, else we get a CUDA index error

    pad_q = q_len_rounded - q_len
    pad_k = kv_len_rounded - kv_len

    if pad_q > 0:
        query_states = F.pad(query_states, (0, 0, 0, pad_q), value=0.0)  # [B, H, q_len_rounded, D]
    if pad_k > 0:
        key_states = F.pad(key_states, (0, 0, 0, pad_k), value=0.0)
        value_states = F.pad(value_states, (0, 0, 0, pad_k), value=0.0)
    padded_causal_mask = F.pad(causal_mask, (0, pad_k, 0, pad_q), value=0.0)
    mask_mod_fn_orig = precomputed_mask_factory(padded_causal_mask)

    mask_4d = create_mask(
        mod_fn=mask_mod_fn_orig,
        B=b_mask,
        H=h_mask,
        Q_LEN=q_len_rounded,
        KV_LEN=kv_len_rounded,
        device=causal_mask.device,
    )

    mask_mod_fn_padded = precomputed_mask_factory(mask_4d)
    block_mask = create_block_mask(
        mask_mod=mask_mod_fn_padded,
        B=b_mask,
        H=h_mask,
        Q_LEN=q_len_rounded,
        KV_LEN=kv_len_rounded,
        BLOCK_SIZE=block_size,
        device=causal_mask.device,
        _compile=False,
    )

    #  mask is applied inside the kernel, ideally more efficiently than score_mod.
    attn_output = _flex_attention(
        query_states,
        key_states,
        value_states,
        block_mask=block_mask,
        kernel_options={
            "BLOCK_M": 32,
            "BLOCK_N": 32,
            "BLOCK_M1": 32,
            "BLOCK_N1": 32,
            "BLOCK_M2": 32,
            "BLOCK_N2": 32,
            "num_stages": 2,
            "num_warps": 4,
        },
        enable_gqa=True,  # because we shaped query/key states for GQA
        scale=head_dim**-0.5
        if scaling is None
        else scaling,  # on certain device, the head_dim ** 0.5 break compile !
    )
    attn_output = attn_output[:, :, :seq_len, :].to(dtype=original_dtype)
    attn_output = attn_output.transpose(1, 2).contiguous()  # [B, Q_LEN, H, head_dim]
    attn_output = attn_output.reshape(
        batch_size,
        -1,
        attn_output.shape[2] * attn_output.shape[3],  # merges [H, head_dim]
    )
    return attn_output
