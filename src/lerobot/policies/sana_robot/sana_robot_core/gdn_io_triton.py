# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://github.com/NVlabs/Sana/tree/robot) at commit 1e7d6cc51,
# original path sana_robot/kernels/training/gdn_io.py. Import paths are rewritten to the LeRobot package.
# Keep this file in sync with the upstream file.

from __future__ import annotations

import torch
import triton
import triton.language as tl

_ROWS = 64


@triton.jit
def _qk_norm_rope_fwd(
    X,
    W,
    COS,
    SIN,
    OUT,
    RSTD,
    row_stride,
    N,
    H,
    eps,
    C: tl.constexpr,
    HALF: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    """One token row r = b * N + n: RMSNorm of X[r, :C] (pairs j = (2j, 2j + 1)), weight W, then the rotation of every pair by (COS, SIN)[r, j % HALF] into OUT[b, j // HALF, n, 2 (j % HALF) + {0, 1}]."""
    r = tl.program_id(0).to(tl.int64)
    b = r // N
    n = r % N
    j = tl.arange(0, BLOCK_P)
    mask = j < C // 2
    xe = tl.load(X + r * row_stride + 2 * j, mask=mask, other=0.0).to(tl.float32)
    xo = tl.load(X + r * row_stride + 2 * j + 1, mask=mask, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(xe * xe + xo * xo, axis=0) / C + eps)
    we = tl.load(W + 2 * j, mask=mask, other=0.0).to(tl.float32)
    wo = tl.load(W + 2 * j + 1, mask=mask, other=0.0).to(tl.float32)
    ne = xe * rstd * we
    no = xo * rstd * wo
    i = j % HALF
    h = j // HALF
    c = tl.load(COS + r * HALF + i, mask=mask, other=0.0)
    s = tl.load(SIN + r * HALF + i, mask=mask, other=0.0)
    out = ((b * H + h) * N + n) * D + 2 * i
    tl.store(OUT + out, (ne * c - no * s).to(OUT.dtype.element_ty), mask=mask)
    tl.store(OUT + out + 1, (ne * s + no * c).to(OUT.dtype.element_ty), mask=mask)
    tl.store(RSTD + r, rstd)


@triton.jit
def _qk_norm_rope_bwd(
    X,
    W,
    COS,
    SIN,
    DOUT,
    RSTD,
    DX,
    DW,
    row_stride,
    T,
    N,
    H,
    C: tl.constexpr,
    HALF: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,
    ROWS: tl.constexpr,
):
    """ROWS token rows per program: the inverse rotation of DOUT, the RMSNorm backward into DX [T, C] and the weight gradient partial DW[program, :C]."""
    pid = tl.program_id(0)
    j = tl.arange(0, BLOCK_P)
    mask = j < C // 2
    we = tl.load(W + 2 * j, mask=mask, other=0.0).to(tl.float32)
    wo = tl.load(W + 2 * j + 1, mask=mask, other=0.0).to(tl.float32)
    i = j % HALF
    h = j // HALF
    dwe = tl.zeros([BLOCK_P], dtype=tl.float32)
    dwo = tl.zeros([BLOCK_P], dtype=tl.float32)
    for k in range(ROWS):
        r = (pid * ROWS + k).to(tl.int64)
        live = mask & (r < T)
        b = r // N
        n = r % N
        xe = tl.load(X + r * row_stride + 2 * j, mask=live, other=0.0).to(tl.float32)
        xo = tl.load(X + r * row_stride + 2 * j + 1, mask=live, other=0.0).to(tl.float32)
        rstd = tl.load(RSTD + r, mask=r < T, other=0.0)
        c = tl.load(COS + r * HALF + i, mask=live, other=0.0)
        s = tl.load(SIN + r * HALF + i, mask=live, other=0.0)
        out = ((b * H + h) * N + n) * D + 2 * i
        dre = tl.load(DOUT + out, mask=live, other=0.0).to(tl.float32)
        dro = tl.load(DOUT + out + 1, mask=live, other=0.0).to(tl.float32)
        dne = dre * c + dro * s
        dno = dro * c - dre * s
        dwe += dne * xe * rstd
        dwo += dno * xo * rstd
        ge = we * dne
        go = wo * dno
        coef = rstd * rstd * rstd / C * tl.sum(ge * xe + go * xo, axis=0)
        tl.store(DX + r * C + 2 * j, (rstd * ge - coef * xe).to(DX.dtype.element_ty), mask=live)
        tl.store(DX + r * C + 2 * j + 1, (rstd * go - coef * xo).to(DX.dtype.element_ty), mask=live)
    tl.store(DW + pid * C + 2 * j, dwe, mask=mask)
    tl.store(DW + pid * C + 2 * j + 1, dwo, mask=mask)


def rope_cos_sin(freqs: torch.Tensor, batch: int, tokens: int):
    """(cos, sin) float32 [B, N, D // 2] of a complex rotary table [B or 1, 1, N, D // 2] (the layer's ``rotary_emb``)."""
    table = freqs.reshape(freqs.shape[0], tokens, freqs.shape[-1])
    if table.is_complex():
        table = table.to(torch.complex64)
        (cos, sin) = (table.real, table.imag)
    if cos.shape[0] != batch:
        (cos, sin) = (cos.expand(batch, -1, -1), sin.expand(batch, -1, -1))
    return (cos.contiguous(), sin.contiguous())


class _QKNormRope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, cos, sin, eps, heads):
        (bsz, tokens, channels) = x.shape
        half = cos.shape[-1]
        dim = 2 * half
        out = torch.empty((bsz, heads, tokens, dim), device=x.device, dtype=x.dtype)
        rstd = torch.empty(bsz * tokens, device=x.device, dtype=torch.float32)
        block = triton.next_power_of_2(channels // 2)
        _qk_norm_rope_fwd[bsz * tokens,](
            x,
            weight,
            cos,
            sin,
            out,
            rstd,
            x.stride(1),
            tokens,
            heads,
            float(eps),
            C=channels,
            HALF=half,
            D=dim,
            BLOCK_P=block,
            num_warps=8,
        )
        ctx.save_for_backward(x, weight, cos, sin, rstd)
        ctx.heads = heads
        return out

    @staticmethod
    def backward(ctx, grad):
        (x, weight, cos, sin, rstd) = ctx.saved_tensors
        (bsz, tokens, channels) = x.shape
        rows = bsz * tokens
        half = cos.shape[-1]
        grad = grad.contiguous()
        dx = torch.empty((bsz, tokens, channels), device=x.device, dtype=x.dtype)
        programs = triton.cdiv(rows, _ROWS)
        dw = torch.empty((programs, channels), device=x.device, dtype=torch.float32)
        _qk_norm_rope_bwd[programs,](
            x,
            weight,
            cos,
            sin,
            grad,
            rstd,
            dx,
            dw,
            x.stride(1),
            rows,
            tokens,
            ctx.heads,
            C=channels,
            HALF=half,
            D=2 * half,
            BLOCK_P=triton.next_power_of_2(channels // 2),
            ROWS=_ROWS,
            num_warps=8,
        )
        return (dx, dw.sum(0).to(weight.dtype), None, None, None, None)


def gdn_qk_norm_rope(q_raw, k_raw, q_weight, k_weight, freqs, eps: float, heads: int):
    """The rotated numerator features (q, k) [B, H, N, D] of the q / k slices [B, N, C] of the packed qkv projection: RMSNorm over C with q_weight / k_weight and eps, the head split, then the RoPE of the complex table freqs (adjacent channel pairs), float32 inside."""
    (bsz, tokens, _channels) = q_raw.shape
    (cos, sin) = rope_cos_sin(freqs, bsz, tokens)
    return (
        _QKNormRope.apply(q_raw, q_weight, cos, sin, eps, heads),
        _QKNormRope.apply(k_raw, k_weight, cos, sin, eps, heads),
    )


@triton.jit
def _out_norm_gate_fwd(
    O,
    G,
    W,
    Y,
    RSTD,
    so_b,
    so_h,
    so_n,
    N,
    eps,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    SILU: tl.constexpr,
):
    """One token row r = b * N + n: RMSNorm of every head's D channels of O[b, :, n, :] with W, times the gate act(G[r]) into Y[r] ([B * N, H * D])."""
    r = tl.program_id(0).to(tl.int64)
    b = r // N
    n = r % N
    hh = tl.arange(0, BLOCK_H)[:, None]
    dd = tl.arange(0, D)[None, :]
    mask = hh < H
    o = tl.load(O + b * so_b + hh * so_h + n * so_n + dd, mask=mask, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(o * o, axis=1) / D + eps)
    w = tl.load(W + dd).to(tl.float32)
    g = tl.load(G + r * (H * D) + hh * D + dd, mask=mask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(g)
    act = g * sig if SILU else sig
    tl.store(Y + r * (H * D) + hh * D + dd, (o * rstd[:, None] * w * act).to(Y.dtype.element_ty), mask=mask)
    tl.store(RSTD + r * H + tl.arange(0, BLOCK_H), rstd, mask=tl.arange(0, BLOCK_H) < H)


@triton.jit
def _out_norm_gate_bwd(
    O,
    G,
    W,
    DY,
    RSTD,
    DO,
    DG,
    DW,
    so_b,
    so_h,
    so_n,
    T,
    N,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    SILU: tl.constexpr,
    ROWS: tl.constexpr,
):
    """ROWS token rows per program: DO (O's layout), DG [T, H * D] and the o_norm weight gradient partial DW[program]."""
    pid = tl.program_id(0)
    hh = tl.arange(0, BLOCK_H)[:, None]
    dd = tl.arange(0, D)[None, :]
    w = tl.load(W + dd).to(tl.float32)
    dw = tl.zeros([BLOCK_H, D], dtype=tl.float32)
    for k in range(ROWS):
        r = (pid * ROWS + k).to(tl.int64)
        live = (hh < H) & (r < T)
        b = r // N
        n = r % N
        o_off = b * so_b + hh * so_h + n * so_n + dd
        o = tl.load(O + o_off, mask=live, other=0.0).to(tl.float32)
        g = tl.load(G + r * (H * D) + hh * D + dd, mask=live, other=0.0).to(tl.float32)
        dy = tl.load(DY + r * (H * D) + hh * D + dd, mask=live, other=0.0).to(tl.float32)
        rstd = tl.load(
            RSTD + r * H + tl.arange(0, BLOCK_H), mask=(tl.arange(0, BLOCK_H) < H) & (r < T), other=0.0
        )[:, None]
        sig = tl.sigmoid(g)
        if SILU:
            act = g * sig
            dact = sig + g * sig * (1.0 - sig)
        else:
            act = sig
            dact = sig * (1.0 - sig)
        normed = o * rstd
        dnormed = dy * act * w
        dw += dy * act * normed
        tl.store(DG + r * (H * D) + hh * D + dd, (dy * normed * w * dact).to(DG.dtype.element_ty), mask=live)
        coef = rstd * rstd * rstd / D * tl.sum(dnormed * o, axis=1)[:, None]
        tl.store(DO + o_off, (rstd * dnormed - coef * o).to(DO.dtype.element_ty), mask=live)
    tl.store(DW + pid * D + tl.arange(0, D), tl.sum(dw, axis=0))


class _OutNormGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, o, gate, weight, eps, silu):
        (bsz, heads, tokens, dim) = o.shape
        y = torch.empty((bsz, tokens, heads * dim), device=o.device, dtype=o.dtype)
        rstd = torch.empty((bsz * tokens, heads), device=o.device, dtype=torch.float32)
        block_h = triton.next_power_of_2(heads)
        _out_norm_gate_fwd[bsz * tokens,](
            o,
            gate,
            weight,
            y,
            rstd,
            o.stride(0),
            o.stride(1),
            o.stride(2),
            tokens,
            float(eps),
            H=heads,
            D=dim,
            BLOCK_H=block_h,
            SILU=bool(silu),
            num_warps=8,
        )
        ctx.save_for_backward(o, gate, weight, rstd)
        ctx.silu = bool(silu)
        return y

    @staticmethod
    def backward(ctx, grad):
        (o, gate, weight, rstd) = ctx.saved_tensors
        (bsz, heads, tokens, dim) = o.shape
        rows = bsz * tokens
        grad = grad.contiguous()
        do = torch.empty_like(o)
        dg = torch.empty_like(gate)
        programs = triton.cdiv(rows, _ROWS)
        dw = torch.empty((programs, dim), device=o.device, dtype=torch.float32)
        _out_norm_gate_bwd[programs,](
            o,
            gate,
            weight,
            grad,
            rstd,
            do,
            dg,
            dw,
            o.stride(0),
            o.stride(1),
            o.stride(2),
            rows,
            tokens,
            H=heads,
            D=dim,
            BLOCK_H=triton.next_power_of_2(heads),
            SILU=ctx.silu,
            ROWS=_ROWS,
            num_warps=8,
        )
        return (do, dg, dw.sum(0).to(weight.dtype), None, None)


def gdn_output_norm_gate(o, gate, weight, eps: float, act: str):
    """o_norm and the output gate of the GDN read o [B, H, N, D]: RMSNorm over every head's D channels with weight and eps, times act(gate) for the gate projection gate [B, N, H * D] (act "sigmoid" or "silu"); returns [B, N, H * D] in o's dtype, float32 inside."""
    if not (o.is_contiguous() or o.transpose(1, 2).is_contiguous()):
        o = o.contiguous()
    return _OutNormGate.apply(o, gate.contiguous(), weight, eps, act == "silu")
