# Adapted from NVlabs/LongLive at 1b9e348897cafa206fc6dd6feb80b59e2a8b1aa9.
# Changes: LeRobot package imports and reference-only runtime integration.
# Long-WAM file attribution; existing upstream notices are retained below.
# SPDX-FileCopyrightText: Copyright (c) 2026 The FastWAM Authors
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: The DiffSynth-Studio contributors
# SPDX-FileCopyrightText: Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# SPDX-License-Identifier: MIT AND Apache-2.0
# Provenance: Modified from the source snapshot listed below.
# Source: https://github.com/yuantianyuan01/FastWAM @ 7faa71108368fbb3b6885649f112af607427a2d4 :: src/fastwam/models/wan22/wan_video_dit.py
# Changes: Long-WAM integration, portable imports and/or robot training/evaluation adaptations.
# License texts and provenance: see the Long-WAM _core/NOTICE.md.
# Source: https://github.com/modelscope/DiffSynth-Studio @ 974cfa37f27ac55eba3b6d10efa21f876900572d :: diffsynth/models/wan_video_dit.py
# Source: https://github.com/Wan-Video/Wan2.2 :: wan/modules/model.py
# Changes: Nested DiffSynth-Studio/Wan implementations incorporated through FastWAM retain their Apache-2.0 terms.
# Source: https://github.com/Aaronhuang-778/Long-WAM @ fbe5e556b7897c16aac1ce8cb4706764e00c74e1 :: src/longwam/models/wan22/wan_video_dit.py
# Changes: Integrated shared runtime and accelerated inference while retaining release contracts.
# End Long-WAM attribution.

import math
from logging import getLogger
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from .helpers.gradient import gradient_checkpoint_forward

logger = getLogger(__name__)


_FLEX_GROUP_CAUSAL_OPTIONS = {
    # Thor CC 11.0 real-shape sweep: the smaller query tile and explicit
    # resource budget reduce the 255-register/180-KiB pressure of the default
    # 128x128, 8-warp lowering while preserving the exact mask semantics.
    "BLOCK_M": 64,
    "BLOCK_N": 128,
    "num_warps": 4,
    "num_stages": 2,
    "PRESCALE_QK": False,
    "ROWS_GUARANTEED_SAFE": True,
    "BLOCKS_ARE_CONTIGUOUS": True,
}


def build_flex_group_causal_mask(
    *,
    device: torch.device | str,
    batch_size: int,
    num_heads: int,
    sequence_length: int,
    tokens_per_frame: int,
    query_length: int | None = None,
    query_start: int = 0,
) -> BlockMask:
    """Build the exact Primary per-frame-causal sparse mask outside inference."""
    if sequence_length <= 0 or tokens_per_frame <= 0:
        raise ValueError("sequence length and tokens per frame must be positive")
    if sequence_length % tokens_per_frame:
        raise ValueError("sequence length must be divisible by tokens per frame")

    query_length = sequence_length if query_length is None else int(query_length)
    query_start = int(query_start)
    if query_length <= 0 or query_start < 0:
        raise ValueError("query length must be positive and query start non-negative")
    if query_start + query_length > sequence_length:
        raise ValueError("query range exceeds sequence length")

    def group_causal(_batch, _head, query, key):
        absolute_query = query + query_start
        return (key // tokens_per_frame) <= (absolute_query // tokens_per_frame)

    return create_block_mask(
        group_causal,
        B=batch_size,
        H=num_heads,
        Q_LEN=query_length,
        KV_LEN=sequence_length,
        device=str(device),
        BLOCK_SIZE=128,
        _compile=True,
    )


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    ctx_mask: torch.Tensor | None = None,
    compatibility_mode: bool = True,
    sdpa_backend: str | None = None,
    sdpa_block_mask: BlockMask | None = None,
):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        if sdpa_backend is None:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=ctx_mask)
        elif sdpa_backend == "cudnn":
            from lerobot.policies.longwam._core.projections import cudnn_attention

            x = cudnn_attention(q, k, v, ctx_mask)
        elif sdpa_backend == "flex-group-causal":
            if sdpa_block_mask is None:
                raise RuntimeError("flex-group-causal SDPA requires a prebuilt BlockMask")
            if k.shape != v.shape or q.shape[0] != k.shape[0]:
                raise RuntimeError("flex-group-causal SDPA requires compatible Q/K/V shapes")
            qk_shape = (int(q.shape[-2]), int(k.shape[-2]))
            if q.shape[-1] != 128 or k.shape[-1] != 128 or q.shape[1] != 24:
                raise RuntimeError("flex-group-causal SDPA is validated only for Primary 24x128 heads")
            if qk_shape not in {(588, 588), (392, 392), (196, 588)}:
                raise RuntimeError(f"unvalidated Primary FlexAttention Q/KV shape {qk_shape}")
            if ctx_mask is None or tuple(ctx_mask.shape) != qk_shape:
                raise RuntimeError("flex-group-causal SDPA mask must match the Q/KV shape")
            x = flex_attention(
                q,
                k,
                v,
                block_mask=sdpa_block_mask,
                kernel_options=_FLEX_GROUP_CAUSAL_OPTIONS,
            )
        else:
            raise ValueError(f"Unsupported SDPA backend: {sdpa_backend!r}")
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
        return x
    else:
        raise NotImplementedError(
            "Only compatibility mode is implemented for flash attention. Please set compatibility_mode=True."
        )


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(10000, -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2)),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # Store the canonical complex128 values as an explicit FP64 real/imag pair.
    # Keeping complex tensors out of the runtime graph lets TorchInductor lower
    # the complete RoPE arithmetic instead of inserting complex-op fallbacks.
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    return torch.view_as_real(torch.polar(torch.ones_like(freqs), freqs)).contiguous()


def rope_apply(
    x,
    freqs,
    num_heads,
    compute_dtype: torch.dtype | None = None,
    direct_output: bool = False,
):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    # Backward-compatible support for callers that still provide a complex
    # table. Long-WAM's own precomputed and resident tables use FP64 pairs, so
    # this branch is absent from its compiled CUDA graphs.
    if freqs.is_complex():
        freqs = torch.view_as_real(freqs)
    if compute_dtype is None:
        compute_dtype = torch.float32 if freqs.device.type == "npu" else torch.float64
    elif compute_dtype not in (torch.float32, torch.float64):
        raise ValueError(f"Unsupported RoPE compute dtype: {compute_dtype}")
    if direct_output:
        if compute_dtype != torch.float32:
            raise ValueError("Direct RoPE output requires float32 arithmetic.")
        x_compute = x.to(compute_dtype)
        head_dim = x_compute.shape[-1]
        pair_index = torch.arange(head_dim, device=x.device) ^ 1
        paired = x_compute[..., pair_index]
        freqs = freqs.to(compute_dtype)
        f_real = freqs[..., 0].repeat_interleave(2, dim=-1)
        f_imag = freqs[..., 1].repeat_interleave(2, dim=-1)
        real_term = x_compute * f_real
        imag_term = paired * f_imag
        even = (torch.arange(head_dim, device=x.device) & 1) == 0
        x_out = torch.where(
            even,
            real_term - imag_term,
            real_term + imag_term,
        )
        return x_out.flatten(2).to(x.dtype)
    x_pair = x.reshape(*x.shape[:-1], -1, 2).to(compute_dtype)
    freqs = freqs.to(compute_dtype)
    x_real, x_imag = x_pair[..., 0], x_pair[..., 1]
    f_real, f_imag = freqs[..., 0], freqs[..., 1]
    x_out = torch.stack(
        (
            x_real * f_real - x_imag * f_imag,
            x_real * f_imag + x_imag * f_real,
        ),
        dim=-1,
    ).flatten(2)
    return x_out.to(x.dtype)


def create_group_causal_attn_mask(
    num_temporal_groups: int, num_query_per_group: int, num_key_per_group: int, mode: str = "causal"
) -> torch.Tensor:
    """
    Creates a group-based attention mask for scaled dot-product attention with two modes:
    'causal' and 'group_diagonal'.

    Parameters:
    - num_temporal_groups (int): The number of temporal groups (e.g., frames in a video sequence).
    - num_query_per_group (int): The number of query tokens per temporal group. (e.g., latent tokens in a frame, H x W).
    - num_key_per_group (int): The number of key tokens per temporal group. (e.g., action tokens per frame).
    - mode (str): The mode of the attention mask. Options are:
        - 'causal': Query tokens can attend to key tokens from the same or previous temporal groups.
        - 'group_diagonal': Query tokens can attend only to key tokens from the same temporal group.

    Returns:
    - attn_mask (torch.Tensor): A boolean tensor of shape (L, S), where:
        - L = num_temporal_groups * num_query_per_group (total number of query tokens)
        - S = num_temporal_groups * num_key_per_group (total number of key tokens)
      The mask indicates where attention is allowed (True) and disallowed (False).

    Example:
    Input:
        num_temporal_groups = 3
        num_query_per_group = 4
        num_key_per_group = 2
    Output:
        Causal Mask Shape: torch.Size([12, 6])
        Group Diagonal Mask Shape: torch.Size([12, 6])
        if mode='causal':
        tensor([[ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True,  True,  True,  True,  True],
                [ True,  True,  True,  True,  True,  True],
                [ True,  True,  True,  True,  True,  True],
                [ True,  True,  True,  True,  True,  True]])

        if mode='group_diagonal':
        tensor([[ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True, False, False, False, False],
                [False, False,  True,  True, False, False],
                [False, False,  True,  True, False, False],
                [False, False,  True,  True, False, False],
                [False, False,  True,  True, False, False],
                [False, False, False, False,  True,  True],
                [False, False, False, False,  True,  True],
                [False, False, False, False,  True,  True],
                [False, False, False, False,  True,  True]])

    """
    assert mode in ["causal", "group_diagonal"], f"Mode {mode} must be 'causal' or 'group_diagonal'"

    # Total number of query and key tokens
    total_num_query_tokens = num_temporal_groups * num_query_per_group  # Total number of query tokens (L)
    total_num_key_tokens = num_temporal_groups * num_key_per_group  # Total number of key tokens (S)

    # Generate time indices for query and key tokens (shape: [L] and [S])
    query_time_indices = torch.arange(num_temporal_groups).repeat_interleave(
        num_query_per_group
    )  # Shape: [L]
    key_time_indices = torch.arange(num_temporal_groups).repeat_interleave(num_key_per_group)  # Shape: [S]

    # Expand dimensions to compute outer comparison
    query_time_indices = query_time_indices.unsqueeze(1)  # Shape: [L, 1]
    key_time_indices = key_time_indices.unsqueeze(0)  # Shape: [1, S]

    if mode == "causal":
        # Causal Mode: Query can attend to keys where key_time <= query_time
        attn_mask = query_time_indices >= key_time_indices  # Shape: [L, S]
    elif mode == "group_diagonal":
        # Group Diagonal Mode: Query can attend only to keys where key_time == query_time
        attn_mask = query_time_indices == key_time_indices  # Shape: [L, S]

    assert attn_mask.shape == (total_num_query_tokens, total_num_key_tokens), "Attention mask shape mismatch"
    return attn_mask


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q, k, v, ctx_mask=None):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=ctx_mask)
        return x


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = self.num_heads * self.attn_head_dim
        self.rope_compute_dtype = None
        self.rope_direct_output = False
        self.sdpa_backend = None
        self.sdpa_block_mask = None

        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)

        # self.attn = AttentionModule(self.num_heads)

    def project_qkv(self, x: torch.Tensor):
        if getattr(self, "_edge_batched_qkv_gemm", False):
            return self._project_batched_qkv(x)
        if not getattr(self, "_edge_direct_shared_qkv_quantization", False):
            return self.q(x), self.k(x), self.v(x)

        if x.numel() == 0:
            return tuple(
                x.new_empty(*x.shape[:-1], projection.out_features) for projection in (self.q, self.k, self.v)
            )

        from fouroversix.quantize import quantize_to_fp4

        quantized_x = quantize_to_fp4(
            x.reshape(-1, x.shape[-1]),
            self.q.config.get_activation_config(),
        )
        return tuple(
            self._project_shared_quantized_linear(quantized_x, x.shape, projection)
            for projection in (self.q, self.k, self.v)
        )

    @staticmethod
    def _project_shared_quantized_linear(
        quantized_input,
        input_shape: torch.Size,
        projection: nn.Module,
    ) -> torch.Tensor:
        """Project one shared activation with direct-forward bias semantics.

        The direct video self-attention path adds bias after the BF16 GEMM.
        Keep that rounding order even when the expert-wide fused-bias epilogue
        is enabled: the latter is consumed by MoT prefill helpers, while the
        quantized replacement modules in this direct path retain their own
        forward contract.
        """
        config = projection.config
        weight = projection.quantized_weight()
        from fouroversix.matmul import fp4_matmul

        out = fp4_matmul(
            quantized_input,
            weight,
            backend=config.matmul_backend,
            out_dtype=config.output_dtype,
        ).reshape(*input_shape[:-1], weight.original_shape[0])
        if projection.bias is not None:
            out = out + projection.bias
        return out

    def _project_batched_qkv(self, x: torch.Tensor):
        from fouroversix.matmul.cutlass.ops import (
            qkv_gemm_nvfp4nvfp4_accum_fp32_out_bf16_tnt,
        )
        from fouroversix.quantize import quantize_to_fp4

        quantized_x = quantize_to_fp4(
            x.reshape(-1, x.shape[-1]),
            self.q.config.get_activation_config(),
        )
        weight = self.q.quantized_weight()
        denominator = (
            quantized_x.scale_rule.max_allowed_e2m1_value()
            * quantized_x.scale_rule.max_allowed_e4m3_value()
            * weight.scale_rule.max_allowed_e2m1_value()
            * weight.scale_rule.max_allowed_e4m3_value()
        )
        alpha = (quantized_x.amax * self._edge_batched_qkv_weight_amax / denominator).to(torch.float32)
        projected = qkv_gemm_nvfp4nvfp4_accum_fp32_out_bf16_tnt(
            quantized_x.values,
            self._edge_batched_qkv_values,
            quantized_x.scale_factors,
            self._edge_batched_qkv_scales,
            alpha,
            quantized_x.original_shape[0],
        )
        bias_shape = [3] + [1] * (x.ndim - 1) + [self.attn_hidden_dim]
        projected = projected.reshape(
            3, *x.shape[:-1], self.attn_hidden_dim
        ) + self._edge_batched_qkv_output_bias.reshape(bias_shape)
        return projected.unbind(0)

    def forward(self, x, freqs, self_attn_mask: torch.Tensor | None = None):
        q, k, v = self.project_qkv(x)
        q = self.norm_q(q)
        k = self.norm_k(k)
        q = rope_apply(
            q,
            freqs,
            self.num_heads,
            self.rope_compute_dtype,
            self.rope_direct_output,
        )
        k = rope_apply(
            k,
            freqs,
            self.num_heads,
            self.rope_compute_dtype,
            self.rope_direct_output,
        )
        x = flash_attention(
            q=q,
            k=k,
            v=v,
            num_heads=self.num_heads,
            ctx_mask=self_attn_mask,
            sdpa_backend=self.sdpa_backend,
            sdpa_block_mask=self.sdpa_block_mask,
        )
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        attn_head_dim: int,
        num_heads: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.attn_hidden_dim = self.num_heads * self.attn_head_dim

        self.q = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.k = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.v = nn.Linear(hidden_dim, self.attn_hidden_dim)
        self.o = nn.Linear(self.attn_hidden_dim, hidden_dim)
        self.norm_q = RMSNorm(self.attn_hidden_dim, eps=eps)
        self.norm_k = RMSNorm(self.attn_hidden_dim, eps=eps)

        # self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, ctx_mask: torch.Tensor | None = None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        return self.forward_projected(q, k, v, ctx_mask=ctx_mask)

    def forward_projected(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=ctx_mask)
        return self.o(x)

    def forward_with_cached_kv(
        self,
        x: torch.Tensor,
        ctx_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        k = getattr(self, "_edge_cached_k", None)
        v = getattr(self, "_edge_cached_v", None)
        if k is None or v is None:
            raise RuntimeError("Cross-attention K/V cache has not been installed")
        q = self.norm_q(self.q(x))
        return self.forward_projected(q, k, v, ctx_mask=ctx_mask)


class GateModule(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual


class DiTBlock(nn.Module):
    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_head_dim = attn_head_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.cross_attn = CrossAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, hidden_dim)
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)
        self.gate = GateModule()

    def forward(
        self,
        x,
        context,
        t_mod,
        freqs,
        context_mask=None,
        self_attn_mask: torch.Tensor | None = None,
        use_cached_cross_kv: bool = False,
    ):
        if context_mask is not None and context_mask.dim() == 3:
            context_mask = context_mask.unsqueeze(1)  # (B, 1, seq_len, context_len), 1 for heads
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            # means t_mod has separate modulation for each token, otherwise same modulation for all tokens in the block
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs, self_attn_mask=self_attn_mask))
        normalized = self.norm3(x)
        if use_cached_cross_kv:
            cross = self.cross_attn.forward_with_cached_kv(
                normalized,
                ctx_mask=context_mask,
            )
        else:
            cross = self.cross_attn(normalized, context, ctx_mask=context_mask)
        x = x + cross
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (
                self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)
            ).chunk(2, dim=2)
            x = self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(
                2, dim=1
            )
            x = self.head(self.norm(x) * (1 + scale) + shift)
        return x


class WanVideoDiT(torch.nn.Module):
    VIDEO_ROPE_MODES = frozenset({"cpu_per_step", "gpu_resident"})

    def __init__(
        self,
        hidden_dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: tuple[int, int, int],
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = False,
        require_clip_embedding: bool = False,
        fuse_vae_embedding_in_latents: bool = True,
        action_conditioned: bool = False,
        action_dim: int = 7,
        action_group_causal_mask_mode="causal",
        video_attention_mask_mode: str = "bidirectional",
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.video_attention_mask_mode = str(video_attention_mask_mode)

        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}")

        self.action_conditioned = action_conditioned
        self.action_dim = action_dim
        assert not has_image_input
        assert not require_clip_embedding
        assert not require_vae_embedding and fuse_vae_embedding_in_latents, (
            "Only support fusing vae embedding in latents"
        )

        self.patch_embedding = nn.Conv3d(in_dim, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim), nn.GELU(approximate="tanh"), nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, attn_head_dim, num_heads, ffn_dim, eps) for _ in range(num_layers)]
        )
        self.head = Head(hidden_dim, out_dim, patch_size, eps)
        self.freqs = precompute_freqs_cis_3d(attn_head_dim)
        self._video_rope_grid_cache: torch.Tensor | None
        self.register_buffer("_video_rope_grid_cache", None, persistent=False)
        self._video_rope_mode = "cpu_per_step"
        self._video_rope_cache_generation = 0
        self._video_rope_cache_build_count = 0
        self._video_rope_cache_dtype = torch.float64
        self._video_rope_grid_size: tuple[int, int, int] | None = None
        self._video_rope_source_identity: tuple[tuple[int, int], ...] | None = None
        self._video_rope_cache_identity: tuple[int, int, int] | None = None
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, hidden_dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        self.control_adapter = None

        if self.action_conditioned:
            self.action_embedding = nn.Linear(action_dim, hidden_dim)
            self.action_group_causal_mask_mode = action_group_causal_mask_mode

        self.use_gradient_checkpointing = use_gradient_checkpointing
        if self.use_gradient_checkpointing:
            logger.info(
                "Using gradient checkpointing for DiT blocks. This will save memory but use more computation."
            )

    @property
    def video_rope_mode(self) -> str:
        return self._video_rope_mode

    @property
    def video_rope_cache(self) -> torch.Tensor | None:
        return self._video_rope_grid_cache

    def configure_video_rope_mode(self, mode: str) -> None:
        normalized = str(mode)
        if normalized not in self.VIDEO_ROPE_MODES:
            raise ValueError(
                f"unsupported Video RoPE mode {normalized!r}; expected one of {sorted(self.VIDEO_ROPE_MODES)}"
            )
        if normalized == self._video_rope_mode:
            return
        if self._video_rope_grid_cache is not None:
            raise RuntimeError("Video RoPE mode cannot change while a resident cache exists.")
        self._video_rope_mode = normalized
        self._video_rope_cache_generation += 1

    def invalidate_video_rope_cache(self) -> None:
        had_cache = self._video_rope_grid_cache is not None
        self._video_rope_grid_cache = None
        self._video_rope_grid_size = None
        self._video_rope_source_identity = None
        self._video_rope_cache_identity = None
        if had_cache:
            self._video_rope_cache_generation += 1

    def _apply(self, fn, recurse: bool = True):
        if hasattr(self, "_video_rope_grid_cache"):
            self.invalidate_video_rope_cache()
        return super()._apply(fn, recurse=recurse)

    def _video_rope_source_descriptor(self) -> tuple[tuple[int, int], ...]:
        if not isinstance(self.freqs, tuple) or len(self.freqs) != 3:
            raise RuntimeError("Video RoPE source must remain a three-tensor tuple.")
        if not all(torch.is_tensor(value) for value in self.freqs):
            raise RuntimeError("Video RoPE source entries must remain tensors.")
        return tuple((id(value), int(value._version)) for value in self.freqs)

    def _assemble_video_rope(
        self,
        *,
        grid_size: tuple[int, int, int],
    ) -> torch.Tensor:
        f, h, w = (int(value) for value in grid_size)
        if min(f, h, w) <= 0:
            raise ValueError(f"Video RoPE grid must be positive, got {grid_size}.")
        if (
            f > int(self.freqs[0].shape[0])
            or h > int(self.freqs[1].shape[0])
            or w > int(self.freqs[2].shape[0])
        ):
            raise ValueError(f"Video RoPE grid {grid_size} exceeds source capacity.")
        return torch.cat(
            [
                self.freqs[0][:f].view(f, 1, 1, -1, 2).expand(f, h, w, -1, 2),
                self.freqs[1][:h].view(1, h, 1, -1, 2).expand(f, h, w, -1, 2),
                self.freqs[2][:w].view(1, 1, w, -1, 2).expand(f, h, w, -1, 2),
            ],
            dim=-2,
        ).reshape(f * h * w, 1, -1, 2)

    def prepare_video_rope_cache(
        self,
        *,
        device: torch.device | str,
        grid_size: tuple[int, int, int],
        storage_dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        if self._video_rope_mode != "gpu_resident":
            raise RuntimeError("prepare_video_rope_cache requires video_rope_mode='gpu_resident'.")
        if self._video_rope_grid_cache is not None or self._video_rope_cache_build_count != 0:
            raise RuntimeError("Video RoPE cache may be built only once per model setup.")
        target = torch.device(device)
        if target.type != "cuda":
            raise ValueError(f"Video RoPE residency requires CUDA, got {target}.")
        if storage_dtype not in (torch.float32, torch.float64):
            raise ValueError(f"Video RoPE resident storage must be float32 or float64, got {storage_dtype}.")
        f, h, w = (int(value) for value in grid_size)
        normalized_grid = (f, h, w)
        source = self._assemble_video_rope(grid_size=normalized_grid)
        cache = source.to(
            device=target,
            dtype=storage_dtype,
            non_blocking=True,
        ).detach()
        if (
            tuple(cache.shape) != tuple(source.shape)
            or tuple(cache.stride()) != tuple(source.stride())
            or cache.dtype != storage_dtype
            or cache.device != target
            or cache.requires_grad
        ):
            raise RuntimeError("Video RoPE resident cache does not preserve the eager contract.")
        self._video_rope_grid_cache = cache
        self._video_rope_cache_dtype = storage_dtype
        self._video_rope_grid_size = normalized_grid
        self._video_rope_source_identity = self._video_rope_source_descriptor()
        self._video_rope_cache_identity = (
            id(cache),
            int(cache.data_ptr()),
            int(cache._version),
        )
        self._video_rope_cache_generation += 1
        self._video_rope_cache_build_count += 1
        return cache

    def select_video_rope_freqs(
        self,
        *,
        grid_size: tuple[int, int, int],
        device: torch.device,
    ) -> torch.Tensor:
        f, h, w = (int(value) for value in grid_size)
        normalized_grid = (f, h, w)
        if self._video_rope_mode == "cpu_per_step":
            return self._assemble_video_rope(
                grid_size=normalized_grid,
            ).to(device)
        if self._video_rope_mode != "gpu_resident":
            raise RuntimeError(f"invalid Video RoPE mode: {self._video_rope_mode!r}")
        cache = self._video_rope_grid_cache
        if cache is None:
            raise RuntimeError("Prepare the resident Video RoPE cache before inference.")
        if hasattr(torch, "compiler") and torch.compiler.is_compiling():
            # Address/version checks belong to the outer setup and evidence
            # path; tracing them would turn tensor version values into
            # data-dependent symbolic guards.
            token_count = math.prod(normalized_grid)
            return cache[:token_count]
        cached_grid = self._video_rope_grid_size
        prefix_compatible = bool(
            cached_grid is not None
            and normalized_grid[0] <= cached_grid[0]
            and normalized_grid[1:] == cached_grid[1:]
        )
        if (
            cache is None
            or not prefix_compatible
            or self._video_rope_source_identity != self._video_rope_source_descriptor()
            or self._video_rope_cache_identity != (id(cache), int(cache.data_ptr()), int(cache._version))
            or cache.device != device
            or cache.dtype != self._video_rope_cache_dtype
            or cache.requires_grad
        ):
            raise RuntimeError("Video RoPE resident cache identity changed.")
        token_count = math.prod(normalized_grid)
        return cache[:token_count]

    def video_rope_descriptor(self) -> dict[str, Any]:
        cache = self._video_rope_grid_cache
        return {
            "mode": self._video_rope_mode,
            "cache_generation": int(self._video_rope_cache_generation),
            "cache_build_count": int(self._video_rope_cache_build_count),
            "storage_dtype": str(self._video_rope_cache_dtype),
            "grid_size": (
                None
                if self._video_rope_grid_size is None
                else [int(item) for item in self._video_rope_grid_size]
            ),
            "cache": (
                None
                if cache is None
                else {
                    "data_ptr": int(cache.data_ptr()),
                    "shape": [int(item) for item in cache.shape],
                    "stride": [int(item) for item in cache.stride()],
                    "dtype": str(cache.dtype),
                    "device": str(cache.device),
                    "payload_bytes": int(cache.numel() * cache.element_size()),
                }
            ),
        }

    def patchify(self, x: torch.Tensor, control_camera_latents_input: torch.Tensor | None = None):
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera, strict=False)]
            x = x[0].unsqueeze(0)
        return x

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )

    def _validate_forward_inputs(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None,
        action: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"`latents` must be 5D [B, C, T, H, W], got shape {tuple(x.shape)}")
        num_latent_frames = x.shape[2]
        if context.ndim != 3:
            raise ValueError(f"`context` must be 3D [B, L, D], got shape {tuple(context.shape)}")
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B] or [1], got shape {tuple(timestep.shape)}")
        if self.action_conditioned:
            allow_text_only_single_frame = num_latent_frames == 1 and action is None
            if not allow_text_only_single_frame:
                assert action is not None, "Action input is required for action-conditioned model."
                if action.ndim != 3:
                    raise ValueError(
                        f"`action` must be 3D [B, action_horizon, action_dim], got shape {tuple(action.shape)}"
                    )
                if action.shape[2] != self.action_dim:
                    raise ValueError(
                        f"`action` last dimension must be {self.action_dim}, got {action.shape[2]}"
                    )
                if num_latent_frames <= 1:
                    raise ValueError(
                        f"video length must be > 1 for action-conditioned model, got {num_latent_frames}"
                    )
                if action.shape[1] % (num_latent_frames - 1) != 0:
                    raise ValueError(
                        f"action horizon must be divisible by (num_latent_frames - 1), got action_horizon={action.shape[1]}"
                    )
        if context_mask is None:
            context_mask = torch.ones(
                (context.shape[0], context.shape[1]), dtype=torch.bool, device=context.device
            )
        else:
            if context_mask.ndim != 2:
                raise ValueError(f"`context_mask` must be 2D [B, L], got shape {tuple(context_mask.shape)}")
            if context_mask.shape[0] != context.shape[0] or context_mask.shape[1] != context.shape[1]:
                raise ValueError(
                    f"`context_mask` shape must match `context` shape [B, L], got {tuple(context_mask.shape)} vs {tuple(context.shape)}"
                )

        batch_size = x.shape[0]
        if batch_size != context.shape[0]:
            if not self.training and batch_size == 1:
                x = x.expand(context.shape[0], -1, -1, -1, -1)
                batch_size = context.shape[0]
            else:
                raise ValueError(
                    f"Batch mismatch between latents and context: {batch_size} vs {context.shape[0]}."
                )

        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(
                f"`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}"
            )
        if timestep.shape[0] == 1 and batch_size > 1:
            assert not self.training, "During training, timestep length must match batch_size."
            timestep = timestep.expand(batch_size)
        return x, timestep, context_mask

    def build_video_to_video_mask(
        self,
        video_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        if video_seq_len <= 0:
            raise ValueError(f"`video_seq_len` must be positive, got {video_seq_len}")
        if video_tokens_per_frame <= 0:
            raise ValueError(f"`video_tokens_per_frame` must be positive, got {video_tokens_per_frame}")

        if self.video_attention_mask_mode == "bidirectional":
            return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)

        if self.video_attention_mask_mode == "per_frame_causal":
            if video_seq_len % video_tokens_per_frame != 0:
                raise ValueError(
                    "`video_seq_len` must be divisible by `video_tokens_per_frame` in `per_frame_causal` mode, "
                    f"got {video_seq_len} and {video_tokens_per_frame}"
                )
            num_video_frames = video_seq_len // video_tokens_per_frame
            frame_causal = torch.tril(
                torch.ones((num_video_frames, num_video_frames), dtype=torch.bool, device=device)
            )
            return frame_causal.repeat_interleave(video_tokens_per_frame, dim=0).repeat_interleave(
                video_tokens_per_frame, dim=1
            )

        if self.video_attention_mask_mode == "first_frame_causal":
            video_mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
            first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
            video_mask[:first_frame_tokens, first_frame_tokens:] = False
            return video_mask

        raise ValueError(f"Unsupported video attention mask mode: {self.video_attention_mask_mode}")

    def pre_dit(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        fuse_vae_embedding_in_latents: bool = False,
        control_camera_latents_input: torch.Tensor | None = None,
        num_clean_frames: int = 1,
        context_is_projected: bool = False,
    ) -> dict[str, Any]:
        x, timestep, context_mask = self._validate_forward_inputs(
            x=x,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=action,
        )

        batch_size = x.shape[0]
        patch_h = int(self.patch_size[1])
        patch_w = int(self.patch_size[2])
        if x.shape[3] % patch_h != 0 or x.shape[4] % patch_w != 0:
            raise ValueError(
                "Latent spatial shape must be divisible by DiT patch size, "
                f"got HxW=({x.shape[3]}, {x.shape[4]}), patch=({patch_h}, {patch_w})"
            )
        tokens_per_frame = (x.shape[3] // patch_h) * (x.shape[4] // patch_w)

        if self.seperated_timestep and fuse_vae_embedding_in_latents:
            if not hasattr(self, "patch_size") or len(self.patch_size) < 3:
                raise ValueError(f"Invalid dit.patch_size: {getattr(self, 'patch_size', None)}")

            token_timesteps = torch.ones(
                (batch_size, x.shape[2], tokens_per_frame),
                dtype=timestep.dtype,
                device=timestep.device,
            ) * timestep.view(batch_size, 1, 1)
            # v0b: the first `num_clean_frames` latent frames are clean (t=0); the rest carry the
            # sampled noise timestep. num_clean_frames=1 reproduces v0a exactly.
            token_timesteps[:, 0 : max(1, int(num_clean_frames)), :] = 0
            token_timesteps = token_timesteps.reshape(batch_size, -1)
            token_t_emb = sinusoidal_embedding_1d(self.freq_dim, token_timesteps.reshape(-1))
            t = self.time_embedding(token_t_emb).reshape(batch_size, -1, self.hidden_dim)
            t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))
        else:
            raise NotImplementedError(
                "Only support seperated_timestep with fuse_vae_embedding_in_latents for now."
            )
            t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
            t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))
        x = self.patchify(x, control_camera_latents_input=control_camera_latents_input)
        f, h, w = x.shape[2:]

        if not context_is_projected:
            context = self.text_embedding(context)  # (B, L, dim)
        context_len = context.shape[1]
        if self.action_conditioned and action is not None:
            action_len = action.shape[1]
            action_emb = self.action_embedding(action)  # (B, action_len, dim)
            action_pos_embed = sinusoidal_embedding_1d(
                self.hidden_dim, torch.arange(action_len, device=action_emb.device)
            )  # (action_len, dim)
            action_emb = action_emb + action_pos_embed.unsqueeze(0)  # (B, action_len, dim)
            context = torch.cat([context, action_emb], dim=1)  # (B, context_len + action_len, dim)

            # new mask
            num_temporal_groups = f - 1  # first latent frame do not attend to actions
            if num_temporal_groups <= 0:
                raise ValueError(
                    "Action-conditioned context mask requires at least 2 latent frames when `action` is provided."
                )
            assert action_emb.shape[1] % num_temporal_groups == 0, (
                f"Action embedding length {action_emb.shape[1]} must be divisible by number of temporal groups {num_temporal_groups}"
            )
            # Each latent frame (from the 2nd one) attends to the corresponding group of action tokens
            action_group_mask = create_group_causal_attn_mask(
                num_temporal_groups=num_temporal_groups,
                num_query_per_group=tokens_per_frame,
                num_key_per_group=action_len // num_temporal_groups,
                mode=self.action_group_causal_mask_mode,
            ).to(context.device)  # ((f-1)*tokens_per_frame, action_len)

            seq_len = f * h * w  # query length
            final_context_mask = torch.zeros(
                (batch_size, seq_len, context.shape[1]), dtype=torch.bool, device=context.device
            )  # (B, seq_len, L + action_len)
            # all latent frames attend to text tokens
            final_context_mask[:, :, :context_len] = context_mask.unsqueeze(1).expand(
                -1, seq_len, -1
            )  # (B, seq_len, L)
            # latent frames from the 2nd one attend to action tokens
            final_context_mask[:, tokens_per_frame:, context_len:] = action_group_mask.unsqueeze(0).expand(
                batch_size, -1, -1
            )  # (B, seq_len, action_len)
            context_mask = final_context_mask
        elif self.action_conditioned and action is None:
            if f != 1:
                raise ValueError(
                    "Action-conditioned model requires `action` unless running single-frame text-only mode with num_latent_frames=1."
                )
            context_mask = context_mask.unsqueeze(1).expand(-1, f * h * w, -1)  # (B, seq_len, L)
        else:
            context_mask = context_mask.unsqueeze(1).expand(-1, f * h * w, -1)  # (B, seq_len, L)

        x_tokens = rearrange(x, "b c f h w -> b (f h w) c").contiguous()

        freqs = self.select_video_rope_freqs(
            grid_size=(int(f), int(h), int(w)),
            device=x_tokens.device,
        )

        return {
            "tokens": x_tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context,
            "context_mask": context_mask,
            "meta": {
                "grid_size": (f, h, w),
                "tokens_per_frame": tokens_per_frame,
                "batch_size": batch_size,
            },
        }

    def post_dit(self, x_tokens: torch.Tensor, pre_state: dict[str, Any]) -> torch.Tensor:
        f, h, w = pre_state["meta"]["grid_size"]
        x = self.head(x_tokens, pre_state["t"])
        x = self.unpatchify(x, (f, h, w))
        return x

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
        fuse_vae_embedding_in_latents: bool = False,
    ):
        pre_state = self.pre_dit(
            x=x,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        x_tokens = pre_state["tokens"]
        context_emb = pre_state["context"]
        t_mod = pre_state["t_mod"]
        freqs = pre_state["freqs"]
        context_attn_mask = pre_state["context_mask"]
        self_attn_mask = (
            self.build_video_to_video_mask(
                video_seq_len=x_tokens.shape[1],
                video_tokens_per_frame=int(pre_state["meta"]["tokens_per_frame"]),
                device=x_tokens.device,
            )
            if self.video_attention_mask_mode != "bidirectional"
            else None
        )  # special rule for faster speed

        for block in self.blocks:
            if self.use_gradient_checkpointing:
                x_tokens = gradient_checkpoint_forward(
                    block,
                    self.use_gradient_checkpointing,
                    x_tokens,
                    context_emb,
                    t_mod,
                    freqs,
                    context_mask=context_attn_mask,
                    self_attn_mask=self_attn_mask,
                )
            else:
                x_tokens = block(
                    x_tokens,
                    context_emb,
                    t_mod,
                    freqs,
                    context_mask=context_attn_mask,
                    self_attn_mask=self_attn_mask,
                )

        return self.post_dit(x_tokens, pre_state)
