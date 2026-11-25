# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://github.com/NVlabs/Sana/tree/robot) at commit 1e7d6cc51,
# original path sana_robot/kernels/training/gdn_teacher_forcing.py. Import paths are rewritten to the LeRobot package.
# Keep this file in sync with the upstream file.

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from lerobot.policies.sana_robot.sana_robot_core.teacher_forcing import (
    teacher_forcing_observation_index,
    teacher_forcing_prediction_index,
)

HEAD_DIM = 128
FWD = {"BLOCK_V": 64, "BLOCK_T": 128, "num_warps": 8, "num_stages": 1}
BWD_STATE = {"BLOCK_V": 64, "BLOCK_T": 32, "num_warps": 4, "num_stages": 2}
BWD_TOKEN = {"BLOCK_V": 64, "BLOCK_T": 64, "num_warps": 8, "num_stages": 2}
BWD_Z = {"BLOCK_T": 128, "num_warps": 4}
(ZERO, PERSISTENT, BRANCH) = (0, 1, 2)


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    QR,
    KR,
    BETA,
    DECAY,
    MODE,
    RLO,
    RHI,
    OLO,
    OHI,
    RIDX,
    OIDX,
    O,
    DEN,
    RBUF,
    RZBUF,
    SBUF,
    ZBUF,
    sq_b,
    sq_h,
    sq_n,
    sk_b,
    sk_h,
    sk_n,
    sv_b,
    sv_h,
    sv_n,
    sqr_b,
    sqr_h,
    sqr_n,
    skr_b,
    skr_h,
    skr_n,
    so_b,
    so_h,
    so_n,
    sbeta_b,
    sbeta_h,
    H,
    N,
    M,
    eps,
    HEAD_DIM: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_T: tl.constexpr,
    NORMALIZER: tl.constexpr,
):
    bh = tl.program_id(0)
    vt = tl.program_id(1)
    b_idx = (bh // H).to(tl.int64)
    h_idx = (bh % H).to(tl.int64)
    q_base = b_idx * sq_b + h_idx * sq_h
    k_base = b_idx * sk_b + h_idx * sk_h
    v_base = b_idx * sv_b + h_idx * sv_h
    qr_base = b_idx * sqr_b + h_idx * sqr_h
    kr_base = b_idx * skr_b + h_idx * skr_h
    o_base = b_idx * so_b + h_idx * so_h
    beta_base = b_idx * sbeta_b + h_idx * sbeta_h
    row = bh.to(tl.int64) * N
    offs_d = tl.arange(0, HEAD_DIM)
    offs_v = vt * BLOCK_V + tl.arange(0, BLOCK_V)
    offs_t = tl.arange(0, BLOCK_T)
    tile = offs_d[:, None] * HEAD_DIM + offs_v[None, :]
    low = K.dtype.element_ty
    P = tl.zeros([HEAD_DIM, BLOCK_V], dtype=tl.float32)
    zP = tl.zeros([HEAD_DIM], dtype=tl.float32)
    Bst = tl.zeros([HEAD_DIM, BLOCK_V], dtype=tl.float32)
    zB = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for c in range(0, M):
        mode = tl.load(MODE + c)
        src = mode % 3
        dst = mode // 3
        rlo = tl.load(RLO + c)
        rhi = tl.load(RHI + c)
        olo = tl.load(OLO + c)
        ohi = tl.load(OHI + c)
        scale = 1.0 / tl.sqrt((HEAD_DIM * (rhi - rlo)).to(tl.float32))
        g = tl.load(DECAY + bh * M + c)
        S = tl.where(src == 1, P, tl.where(src == 2, Bst, 0.0))
        z = tl.where(src == 1, zP, tl.where(src == 2, zB, 0.0))
        blk = bh.to(tl.int64) * M + c
        state_off = blk * HEAD_DIM * HEAD_DIM
        tl.store(SBUF + state_off + tile, S)
        if NORMALIZER:
            if vt == 0:
                tl.store(ZBUF + blk * HEAD_DIM + offs_d, z)
        St = S * g
        zt = z * g
        R = St
        r = zt
        for p0 in range(rlo, rhi, BLOCK_T):
            pos = p0 + offs_t
            mask = pos < rhi
            tok = tl.load(RIDX + pos, mask=mask, other=0).to(tl.int64)
            k = tl.load(K + k_base + tok[:, None] * sk_n + offs_d[None, :], mask=mask[:, None], other=0.0)
            v = tl.load(V + v_base + tok[:, None] * sv_n + offs_v[None, :], mask=mask[:, None], other=0.0)
            b = tl.load(BETA + beta_base + tok, mask=mask, other=0.0).to(tl.float32)
            R += tl.dot(tl.trans((k.to(tl.float32) * (scale * b)[:, None]).to(low)), v)
            if NORMALIZER:
                kd = tl.load(
                    KR + kr_base + tok[:, None] * skr_n + offs_d[None, :], mask=mask[:, None], other=0.0
                ).to(tl.float32)
                r += tl.sum(kd * (scale * b)[:, None], axis=0)
        tl.store(RBUF + state_off + tile, R)
        if NORMALIZER:
            if vt == 0:
                tl.store(RZBUF + blk * HEAD_DIM + offs_d, r)
        R_low = R.to(low)
        for p0 in range(olo, ohi, BLOCK_T):
            pos = p0 + offs_t
            mask = pos < ohi
            tok = tl.load(OIDX + pos, mask=mask, other=0).to(tl.int64)
            q = tl.load(Q + q_base + tok[:, None] * sq_n + offs_d[None, :], mask=mask[:, None], other=0.0)
            if NORMALIZER:
                qd = tl.load(
                    QR + qr_base + tok[:, None] * sqr_n + offs_d[None, :], mask=mask[:, None], other=0.0
                ).to(tl.float32)
                den = tl.sum(qd * r[None, :], axis=1) + eps
                out = tl.dot(q, R_low) / den[:, None]
            else:
                den = tl.full([BLOCK_T], 1.0, tl.float32)
                out = tl.dot(q, R_low)
            tl.store(
                O + o_base + tok[:, None] * so_n + offs_v[None, :],
                out.to(O.dtype.element_ty),
                mask=mask[:, None],
            )
            if vt == 0:
                tl.store(DEN + row + tok, den, mask=mask)
        S_new = R
        z_new = r
        if src != 0:
            if dst != 0:
                S_new = St
                z_new = zt
                St_low = St.to(low)
                for p0 in range(rlo, rhi, BLOCK_T):
                    pos = p0 + offs_t
                    mask = pos < rhi
                    tok = tl.load(RIDX + pos, mask=mask, other=0).to(tl.int64)
                    k = tl.load(
                        K + k_base + tok[:, None] * sk_n + offs_d[None, :], mask=mask[:, None], other=0.0
                    )
                    v = tl.load(
                        V + v_base + tok[:, None] * sv_n + offs_v[None, :], mask=mask[:, None], other=0.0
                    )
                    b = tl.load(BETA + beta_base + tok, mask=mask, other=0.0).to(tl.float32)
                    kc = (k.to(tl.float32) * scale).to(low)
                    corr = (v.to(tl.float32) - tl.dot(kc, St_low)) * b[:, None]
                    S_new += tl.dot(tl.trans(kc), corr.to(low))
                    if NORMALIZER:
                        kd = (
                            tl.load(
                                KR + kr_base + tok[:, None] * skr_n + offs_d[None, :],
                                mask=mask[:, None],
                                other=0.0,
                            ).to(tl.float32)
                            * scale
                        )
                        zc = (1.0 - tl.sum(kd * zt[None, :], axis=1)) * b
                        z_new += tl.sum(kd * zc[:, None], axis=0)
        P = tl.where(dst == 1, S_new, P)
        zP = tl.where(dst == 1, z_new, zP)
        Bst = tl.where(dst == 2, S_new, Bst)
        zB = tl.where(dst == 2, z_new, zB)


@triton.jit
def _bwd_state_kernel(
    Q,
    K,
    BETA,
    DECAY,
    MODE,
    RLO,
    RHI,
    OLO,
    OHI,
    RIDX,
    OIDX,
    DO,
    DEN,
    RBARBUF,
    SBARBUF,
    STBARBUF,
    sq_b,
    sq_h,
    sq_n,
    sk_b,
    sk_h,
    sk_n,
    sdo_b,
    sdo_h,
    sdo_n,
    sbeta_b,
    sbeta_h,
    H,
    N,
    M,
    HEAD_DIM: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Reverse block walk of the value path with the persistent state's and the branch's gradients."""
    bh = tl.program_id(0)
    vt = tl.program_id(1)
    b_idx = (bh // H).to(tl.int64)
    h_idx = (bh % H).to(tl.int64)
    q_base = b_idx * sq_b + h_idx * sq_h
    k_base = b_idx * sk_b + h_idx * sk_h
    do_base = b_idx * sdo_b + h_idx * sdo_h
    beta_base = b_idx * sbeta_b + h_idx * sbeta_h
    row = bh.to(tl.int64) * N
    offs_d = tl.arange(0, HEAD_DIM)
    offs_v = vt * BLOCK_V + tl.arange(0, BLOCK_V)
    offs_t = tl.arange(0, BLOCK_T)
    tile = offs_d[:, None] * HEAD_DIM + offs_v[None, :]
    low = K.dtype.element_ty
    Pbar = tl.zeros([HEAD_DIM, BLOCK_V], dtype=tl.float32)
    Bbar = tl.zeros([HEAD_DIM, BLOCK_V], dtype=tl.float32)
    for ci in range(0, M):
        c = M - 1 - ci
        mode = tl.load(MODE + c)
        src = mode % 3
        dst = mode // 3
        rlo = tl.load(RLO + c)
        rhi = tl.load(RHI + c)
        olo = tl.load(OLO + c)
        ohi = tl.load(OHI + c)
        scale = 1.0 / tl.sqrt((HEAD_DIM * (rhi - rlo)).to(tl.float32))
        g = tl.load(DECAY + bh * M + c)
        Sbar = tl.where(dst == 1, Pbar, tl.where(dst == 2, Bbar, 0.0))
        state_off = (bh.to(tl.int64) * M + c) * HEAD_DIM * HEAD_DIM
        tl.store(SBARBUF + state_off + tile, Sbar)
        Rbar = Sbar
        for p0 in range(olo, ohi, BLOCK_T):
            pos = p0 + offs_t
            mask = pos < ohi
            tok = tl.load(OIDX + pos, mask=mask, other=0).to(tl.int64)
            q = tl.load(Q + q_base + tok[:, None] * sq_n + offs_d[None, :], mask=mask[:, None], other=0.0)
            do = tl.load(DO + do_base + tok[:, None] * sdo_n + offs_v[None, :], mask=mask[:, None], other=0.0)
            den = tl.load(DEN + row + tok, mask=mask, other=1.0)
            Rbar += tl.dot(tl.trans(q), (do.to(tl.float32) / den[:, None]).to(low))
        MS = tl.zeros([HEAD_DIM, BLOCK_V], dtype=tl.float32)
        if src != 0:
            if dst != 0:
                Sbar_low = Sbar.to(low)
                for p0 in range(rlo, rhi, BLOCK_T):
                    pos = p0 + offs_t
                    mask = pos < rhi
                    tok = tl.load(RIDX + pos, mask=mask, other=0).to(tl.int64)
                    k = tl.load(
                        K + k_base + tok[:, None] * sk_n + offs_d[None, :], mask=mask[:, None], other=0.0
                    )
                    b = tl.load(BETA + beta_base + tok, mask=mask, other=0.0).to(tl.float32)
                    kc = (k.to(tl.float32) * scale).to(low)
                    U = tl.dot(kc, Sbar_low)
                    MS += tl.dot(tl.trans(kc), (U * b[:, None]).to(low))
        tl.store(RBARBUF + state_off + tile, Rbar)
        St_bar = Rbar - MS
        tl.store(STBARBUF + state_off + tile, St_bar)
        incoming = St_bar * g
        Bbar = tl.where(dst == 2, 0.0, Bbar)
        Pbar = tl.where(
            src == 1, tl.where(dst == 1, incoming, Pbar + incoming), tl.where(dst == 1, 0.0, Pbar)
        )
        Bbar = tl.where(src == 2, Bbar + incoming, Bbar)


@triton.jit
def _bwd_token_kernel(
    K,
    V,
    BETA,
    DECAY,
    RLO,
    RHI,
    OLO,
    OHI,
    RIDX,
    OIDX,
    DO,
    DEN,
    RBUF,
    SBUF,
    RBARBUF,
    SBARBUF,
    DQ,
    DK,
    DV,
    DB,
    sk_b,
    sk_h,
    sk_n,
    sv_b,
    sv_h,
    sv_n,
    sdo_b,
    sdo_h,
    sdo_n,
    sbeta_b,
    sbeta_h,
    sdq_b,
    sdq_h,
    sdq_n,
    sdk_b,
    sdk_h,
    sdk_n,
    sdv_b,
    sdv_h,
    sdv_n,
    H,
    N,
    M,
    HEAD_DIM: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Value-path token gradients of one (batch*head, block): q over its output set, k / v / beta over its read set."""
    bh = tl.program_id(0)
    c = tl.program_id(1)
    b_idx = (bh // H).to(tl.int64)
    h_idx = (bh % H).to(tl.int64)
    k_base = b_idx * sk_b + h_idx * sk_h
    v_base = b_idx * sv_b + h_idx * sv_h
    do_base = b_idx * sdo_b + h_idx * sdo_h
    beta_base = b_idx * sbeta_b + h_idx * sbeta_h
    dq_base = b_idx * sdq_b + h_idx * sdq_h
    dk_base = b_idx * sdk_b + h_idx * sdk_h
    dv_base = b_idx * sdv_b + h_idx * sdv_h
    row = bh.to(tl.int64) * N
    offs_d = tl.arange(0, HEAD_DIM)
    offs_t = tl.arange(0, BLOCK_T)
    low = K.dtype.element_ty
    rlo = tl.load(RLO + c)
    rhi = tl.load(RHI + c)
    olo = tl.load(OLO + c)
    ohi = tl.load(OHI + c)
    scale = 1.0 / tl.sqrt((HEAD_DIM * (rhi - rlo)).to(tl.float32))
    g = tl.load(DECAY + bh * M + c)
    state_off = (bh.to(tl.int64) * M + c) * HEAD_DIM * HEAD_DIM
    for p0 in range(olo, ohi, BLOCK_T):
        pos = p0 + offs_t
        mask = pos < ohi
        tok = tl.load(OIDX + pos, mask=mask, other=0).to(tl.int64)
        den = tl.load(DEN + row + tok, mask=mask, other=1.0)
        dq = tl.zeros([BLOCK_T, HEAD_DIM], dtype=tl.float32)
        for vt in range(HEAD_DIM // BLOCK_V):
            offs_v = vt * BLOCK_V + tl.arange(0, BLOCK_V)
            tile = offs_d[:, None] * HEAD_DIM + offs_v[None, :]
            R_low = tl.load(RBUF + state_off + tile).to(low)
            do = tl.load(DO + do_base + tok[:, None] * sdo_n + offs_v[None, :], mask=mask[:, None], other=0.0)
            gbar = (do.to(tl.float32) / den[:, None]).to(low)
            dq = tl.dot(gbar, tl.trans(R_low), dq)
        tl.store(
            DQ + dq_base + tok[:, None] * sdq_n + offs_d[None, :],
            dq.to(DQ.dtype.element_ty),
            mask=mask[:, None],
        )
    for p0 in range(rlo, rhi, BLOCK_T):
        pos = p0 + offs_t
        mask = pos < rhi
        tok = tl.load(RIDX + pos, mask=mask, other=0).to(tl.int64)
        k = tl.load(K + k_base + tok[:, None] * sk_n + offs_d[None, :], mask=mask[:, None], other=0.0)
        b = tl.load(BETA + beta_base + tok, mask=mask, other=0.0).to(tl.float32)
        kc = (k.to(tl.float32) * scale).to(low)
        dk = tl.zeros([BLOCK_T, HEAD_DIM], dtype=tl.float32)
        db = tl.zeros([BLOCK_T], dtype=tl.float32)
        for vt in range(HEAD_DIM // BLOCK_V):
            offs_v = vt * BLOCK_V + tl.arange(0, BLOCK_V)
            tile = offs_d[:, None] * HEAD_DIM + offs_v[None, :]
            Rbar_low = tl.load(RBARBUF + state_off + tile).to(low)
            St_low = (tl.load(SBUF + state_off + tile) * g).to(low)
            Sbar_low = tl.load(SBARBUF + state_off + tile).to(low)
            v = tl.load(V + v_base + tok[:, None] * sv_n + offs_v[None, :], mask=mask[:, None], other=0.0)
            kR = tl.dot(kc, Rbar_low)
            dv = (b[:, None] * kR).to(DV.dtype.element_ty)
            tl.store(DV + dv_base + tok[:, None] * sdv_n + offs_v[None, :], dv, mask=mask[:, None])
            Pk = tl.dot(kc, St_low)
            U = tl.dot(kc, Sbar_low)
            dk = tl.dot(v, tl.trans(Rbar_low), dk)
            dk = tl.dot((-Pk).to(low), tl.trans(Sbar_low), dk)
            dk = tl.dot((-U).to(low), tl.trans(St_low), dk)
            db += tl.sum(v.to(tl.float32) * kR - Pk * U, axis=1)
        dk = (dk * (b * scale)[:, None]).to(DK.dtype.element_ty)
        tl.store(DK + dk_base + tok[:, None] * sdk_n + offs_d[None, :], dk, mask=mask[:, None])
        tl.store(DB + row + tok, db, mask=mask)


@triton.jit
def _bwd_z_kernel(
    QR,
    KR,
    BETA,
    DECAY,
    MODE,
    RLO,
    RHI,
    OLO,
    OHI,
    RIDX,
    OIDX,
    O,
    DO,
    DEN,
    RZBUF,
    ZBUF,
    DQR,
    DKR,
    DBZ,
    DGZ,
    sqr_b,
    sqr_h,
    sqr_n,
    skr_b,
    skr_h,
    skr_n,
    so_b,
    so_h,
    so_n,
    sdo_b,
    sdo_h,
    sdo_n,
    sbeta_b,
    sbeta_h,
    sdqr_b,
    sdqr_h,
    sdqr_n,
    sdkr_b,
    sdkr_h,
    sdkr_n,
    H,
    N,
    M,
    HEAD_DIM: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Denominator path in reverse block order with the persistent z's and the branch's gradients."""
    bh = tl.program_id(0)
    b_idx = (bh // H).to(tl.int64)
    h_idx = (bh % H).to(tl.int64)
    qr_base = b_idx * sqr_b + h_idx * sqr_h
    kr_base = b_idx * skr_b + h_idx * skr_h
    o_base = b_idx * so_b + h_idx * so_h
    do_base = b_idx * sdo_b + h_idx * sdo_h
    beta_base = b_idx * sbeta_b + h_idx * sbeta_h
    dqr_base = b_idx * sdqr_b + h_idx * sdqr_h
    dkr_base = b_idx * sdkr_b + h_idx * sdkr_h
    row = bh.to(tl.int64) * N
    offs_d = tl.arange(0, HEAD_DIM)
    offs_t = tl.arange(0, BLOCK_T)
    zPbar = tl.zeros([HEAD_DIM], dtype=tl.float32)
    zBbar = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for ci in range(0, M):
        c = M - 1 - ci
        mode = tl.load(MODE + c)
        src = mode % 3
        dst = mode // 3
        rlo = tl.load(RLO + c)
        rhi = tl.load(RHI + c)
        olo = tl.load(OLO + c)
        ohi = tl.load(OHI + c)
        scale = 1.0 / tl.sqrt((HEAD_DIM * (rhi - rlo)).to(tl.float32))
        g = tl.load(DECAY + bh * M + c)
        blk = bh.to(tl.int64) * M + c
        zbar = tl.where(dst == 1, zPbar, tl.where(dst == 2, zBbar, 0.0))
        r = tl.load(RZBUF + blk * HEAD_DIM + offs_d)
        z_prev = tl.load(ZBUF + blk * HEAD_DIM + offs_d)
        zt = z_prev * g
        rbar = zbar
        for p0 in range(olo, ohi, BLOCK_T):
            pos = p0 + offs_t
            mask = pos < ohi
            tok = tl.load(OIDX + pos, mask=mask, other=0).to(tl.int64)
            o = tl.load(O + o_base + tok[:, None] * so_n + offs_d[None, :], mask=mask[:, None], other=0.0)
            do = tl.load(DO + do_base + tok[:, None] * sdo_n + offs_d[None, :], mask=mask[:, None], other=0.0)
            den = tl.load(DEN + row + tok, mask=mask, other=1.0)
            dd = -tl.sum(do.to(tl.float32) * o.to(tl.float32), axis=1) / den
            qd = tl.load(QR + qr_base + tok[:, None] * sqr_n + offs_d[None, :], mask=mask[:, None], other=0.0)
            rbar += tl.sum(qd.to(tl.float32) * dd[:, None], axis=0)
            tl.store(
                DQR + dqr_base + tok[:, None] * sdqr_n + offs_d[None, :],
                (dd[:, None] * r[None, :]).to(DQR.dtype.element_ty),
                mask=mask[:, None],
            )
        MZ = tl.zeros([HEAD_DIM], dtype=tl.float32)
        if src != 0:
            if dst != 0:
                for p0 in range(rlo, rhi, BLOCK_T):
                    pos = p0 + offs_t
                    mask = pos < rhi
                    tok = tl.load(RIDX + pos, mask=mask, other=0).to(tl.int64)
                    kd = (
                        tl.load(
                            KR + kr_base + tok[:, None] * skr_n + offs_d[None, :],
                            mask=mask[:, None],
                            other=0.0,
                        ).to(tl.float32)
                        * scale
                    )
                    b = tl.load(BETA + beta_base + tok, mask=mask, other=0.0).to(tl.float32)
                    MZ += tl.sum(kd * (b * tl.sum(kd * zbar[None, :], axis=1))[:, None], axis=0)
        for p0 in range(rlo, rhi, BLOCK_T):
            pos = p0 + offs_t
            mask = pos < rhi
            tok = tl.load(RIDX + pos, mask=mask, other=0).to(tl.int64)
            kd = (
                tl.load(
                    KR + kr_base + tok[:, None] * skr_n + offs_d[None, :], mask=mask[:, None], other=0.0
                ).to(tl.float32)
                * scale
            )
            b = tl.load(BETA + beta_base + tok, mask=mask, other=0.0).to(tl.float32)
            a1 = tl.sum(kd * zt[None, :], axis=1)
            a2 = tl.sum(kd * zbar[None, :], axis=1)
            dkd = (
                b[:, None] * (rbar[None, :] - a1[:, None] * zbar[None, :] - a2[:, None] * zt[None, :]) * scale
            )
            dkr = dkd.to(DKR.dtype.element_ty)
            dbz = tl.sum(kd * rbar[None, :], axis=1) - a1 * a2
            tl.store(DKR + dkr_base + tok[:, None] * sdkr_n + offs_d[None, :], dkr, mask=mask[:, None])
            tl.store(DBZ + row + tok, dbz, mask=mask)
        zt_bar = rbar - MZ
        tl.store(DGZ + blk, tl.sum(zt_bar * z_prev, axis=0))
        incoming = zt_bar * g
        zBbar = tl.where(dst == 2, 0.0, zBbar)
        zPbar = tl.where(
            src == 1, tl.where(dst == 1, incoming, zPbar + incoming), tl.where(dst == 1, 0.0, zPbar)
        )
        zBbar = tl.where(src == 2, zBbar + incoming, zBbar)


class FusedTeacherForcingLayout(NamedTuple):
    """The block walk of one teacher-forcing canvas on its device: per block the mode (src + 3 dst), the read and output slices [lo, hi) of ``read_index`` / ``out_index`` (int32 tokens) and the decay's position in cat(segment decay [M], observation decay [L])."""

    mode: torch.Tensor
    read_lo: torch.Tensor
    read_hi: torch.Tensor
    out_lo: torch.Tensor
    out_hi: torch.Tensor
    read_index: torch.Tensor
    out_index: torch.Tensor
    decay_index: torch.Tensor
    num_blocks: int
    num_tokens: int


@lru_cache(maxsize=256)
def fused_teacher_forcing_layout(
    bounds: tuple, num_clean: int, sink: int, observation_spans: tuple | None, device: torch.device
) -> FusedTeacherForcingLayout:
    """Returns the block walk of ``gdn_teacher_forcing`` on the grouped [sink | clean | noisy] segments ``bounds``."""
    bounds = [int(b) for b in bounds]
    segments = len(bounds) - 1
    chunks = segments - int(num_clean)
    observation = prediction = None
    if observation_spans:
        (index, chunk_bounds) = teacher_forcing_observation_index(bounds, num_clean, observation_spans)
        observation = index.long().split(int(chunk_bounds[1]))
        (index, chunk_bounds) = teacher_forcing_prediction_index(bounds, num_clean, observation_spans)
        prediction = [
            index[lo:hi].long() for (lo, hi) in zip(chunk_bounds[:-1], chunk_bounds[1:], strict=False)
        ]
    blocks = []
    writers = 0

    def mode(src, dst):
        return src + 3 * dst

    def segment(s):
        return torch.arange(bounds[s], bounds[s + 1], dtype=torch.long)

    def write(s):
        nonlocal writers
        blocks.append((mode(PERSISTENT if writers else ZERO, PERSISTENT), segment(s), s))
        writers += 1

    for s in range(sink):
        write(s)
    for c in range(chunks):
        seg = num_clean + c
        src = PERSISTENT if writers else ZERO
        if observation is None:
            blocks.append((mode(src, ZERO), segment(seg), seg))
        else:
            blocks.append((mode(src, BRANCH), observation[c], segments + c))
            blocks.append((mode(BRANCH, ZERO), prediction[c], seg))
        if c <= chunks - 2:
            write(sink + c)
    read_index = torch.cat([blk[1] for blk in blocks])
    lo = torch.tensor([0] + [blk[1].numel() for blk in blocks]).cumsum(0)
    n_tokens = bounds[-1]

    def ints(values, dtype=torch.int32):
        return torch.as_tensor(values, dtype=dtype).to(device)

    return FusedTeacherForcingLayout(
        mode=ints([blk[0] for blk in blocks]),
        read_lo=ints(lo[:-1]),
        read_hi=ints(lo[1:]),
        out_lo=ints(lo[:-1]),
        out_hi=ints(lo[1:]),
        read_index=ints(read_index),
        out_index=ints(read_index),
        decay_index=ints([blk[2] for blk in blocks], torch.long),
        num_blocks=len(blocks),
        num_tokens=n_tokens,
    )


@lru_cache(maxsize=64)
def fused_packed_row_layout(
    windows: tuple, sink: int, observation_spans: tuple | None, device: torch.device
) -> FusedTeacherForcingLayout:
    """Returns the block walk of a packed row of teacher-forcing windows, ``windows`` = ((first token, bounds, clean segments), ...): the windows' walks one after another, every window starting from a zero state, with the decays of a window (its segments, then its observations) after those of the windows before it."""
    (fields, read_offset, decay_offset, blocks) = ({}, 0, 0, 0)
    for first_token, bounds, num_clean in windows:
        layout = fused_teacher_forcing_layout(bounds, num_clean, sink, observation_spans, device)
        shifted = {
            "mode": layout.mode,
            "read_lo": layout.read_lo + read_offset,
            "read_hi": layout.read_hi + read_offset,
            "out_lo": layout.out_lo + read_offset,
            "out_hi": layout.out_hi + read_offset,
            "read_index": layout.read_index + int(first_token),
            "out_index": layout.out_index + int(first_token),
            "decay_index": layout.decay_index + decay_offset,
        }
        for name, value in shifted.items():
            fields.setdefault(name, []).append(value)
        segments = len(bounds) - 1
        read_offset += int(layout.read_index.numel())
        decay_offset += segments + (segments - int(num_clean) if observation_spans else 0)
        blocks += layout.num_blocks
    last_token = windows[-1][0] + windows[-1][1][-1]
    return FusedTeacherForcingLayout(
        **{name: torch.cat(values) for name, values in fields.items()},
        num_blocks=blocks,
        num_tokens=int(last_token),
    )


def _bhn(tensor):
    return list(tensor.stride()[:3])


class _TeacherForcingGatedDelta(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q_num, k_num, v, q_den, k_den, beta, decay, layout, eps, normalizer):
        (B, H, N, D) = q_num.shape
        if not normalizer:
            q_den = k_den = q_num
        (q_num, k_num, v, q_den, k_den) = (
            t if t.stride(-1) == 1 else t.contiguous() for t in (q_num, k_num, v, q_den, k_den)
        )
        (beta_dtype, decay_dtype) = (beta.dtype, decay.dtype)
        beta = beta.to(q_num.dtype)
        if beta.stride(-1) != 1:
            beta = beta.contiguous()
        M = layout.num_blocks
        BH = B * H
        decay = decay.float().reshape(BH, M).contiguous()
        dev = q_num.device
        out = torch.empty((B, N, H, D), device=dev, dtype=q_num.dtype).transpose(1, 2)
        den = torch.empty((BH, N), device=dev, dtype=torch.float32)
        rbuf = torch.empty((BH, M, D, D), device=dev, dtype=torch.float32)
        sbuf = torch.empty_like(rbuf)
        rzbuf = torch.empty((BH, M, D), device=dev, dtype=torch.float32)
        zbuf = torch.empty_like(rzbuf)
        _fwd_kernel[BH, D // FWD["BLOCK_V"]](
            q_num,
            k_num,
            v,
            q_den,
            k_den,
            beta,
            decay,
            layout.mode,
            layout.read_lo,
            layout.read_hi,
            layout.out_lo,
            layout.out_hi,
            layout.read_index,
            layout.out_index,
            out,
            den,
            rbuf,
            rzbuf,
            sbuf,
            zbuf,
            *_bhn(q_num),
            *_bhn(k_num),
            *_bhn(v),
            *_bhn(q_den),
            *_bhn(k_den),
            *_bhn(out),
            beta.stride(0),
            beta.stride(1),
            H,
            N,
            M,
            float(eps),
            HEAD_DIM=HEAD_DIM,
            NORMALIZER=bool(normalizer),
            **FWD,
        )
        ctx.save_for_backward(q_num, k_num, v, q_den, k_den, beta, decay, out, den, rbuf, rzbuf, sbuf, zbuf)
        ctx.layout = layout
        ctx.shape = (B, H, N, D, M)
        ctx.dtypes = (beta_dtype, decay_dtype)
        ctx.normalizer = bool(normalizer)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (q_num, k_num, v, q_den, k_den, beta, decay, out, den, rbuf, rzbuf, sbuf, zbuf) = ctx.saved_tensors
        layout = ctx.layout
        (B, H, N, D, M) = ctx.shape
        BH = B * H
        dev = q_num.device
        if grad_out.dtype != out.dtype:
            grad_out = grad_out.to(out.dtype)
        if grad_out.stride(-1) != 1:
            grad_out = grad_out.contiguous()

        def grads(t, n):
            return torch.empty((B, n, H, D), device=dev, dtype=t.dtype).transpose(1, 2)

        (dq, dk, dv) = (grads(q_num, N), grads(k_num, N), grads(v, N))
        db = torch.empty((BH, N), device=dev, dtype=torch.float32)
        rbar = torch.empty_like(rbuf)
        sbar = torch.empty_like(sbuf)
        stbar = torch.empty_like(sbuf)
        walk = (
            layout.mode,
            layout.read_lo,
            layout.read_hi,
            layout.out_lo,
            layout.out_hi,
            layout.read_index,
            layout.out_index,
        )
        betas = (beta.stride(0), beta.stride(1))
        _bwd_state_kernel[BH, D // BWD_STATE["BLOCK_V"]](
            q_num,
            k_num,
            beta,
            decay,
            *walk,
            grad_out,
            den,
            rbar,
            sbar,
            stbar,
            *_bhn(q_num),
            *_bhn(k_num),
            *_bhn(grad_out),
            *betas,
            H,
            N,
            M,
            HEAD_DIM=HEAD_DIM,
            **BWD_STATE,
        )
        _bwd_token_kernel[BH, M](
            k_num,
            v,
            beta,
            decay,
            layout.read_lo,
            layout.read_hi,
            layout.out_lo,
            layout.out_hi,
            layout.read_index,
            layout.out_index,
            grad_out,
            den,
            rbuf,
            sbuf,
            rbar,
            sbar,
            dq,
            dk,
            dv,
            db,
            *_bhn(k_num),
            *_bhn(v),
            *_bhn(grad_out),
            *betas,
            *_bhn(dq),
            *_bhn(dk),
            *_bhn(dv),
            H,
            N,
            M,
            HEAD_DIM=HEAD_DIM,
            **BWD_TOKEN,
        )
        ddecay = (stbar * sbuf).sum((-1, -2))
        dq_den = dk_den = None
        if ctx.normalizer:
            (dq_den, dk_den) = (grads(q_den, N), grads(k_den, N))
            dbz = torch.empty((BH, N), device=dev, dtype=torch.float32)
            dgz = torch.empty((BH, M), device=dev, dtype=torch.float32)
            _bwd_z_kernel[BH,](
                q_den,
                k_den,
                beta,
                decay,
                *walk,
                out,
                grad_out,
                den,
                rzbuf,
                zbuf,
                dq_den,
                dk_den,
                dbz,
                dgz,
                *_bhn(q_den),
                *_bhn(k_den),
                *_bhn(out),
                *_bhn(grad_out),
                *betas,
                *_bhn(dq_den),
                *_bhn(dk_den),
                H,
                N,
                M,
                HEAD_DIM=HEAD_DIM,
                **BWD_Z,
            )
            db = db + dbz
            ddecay = ddecay + dgz
        dbeta = db.reshape(B, H, N).to(ctx.dtypes[0])
        ddecay = ddecay.reshape(B, H, M).to(ctx.dtypes[1])
        return (dq, dk, dv, dq_den, dk_den, dbeta, ddecay, None, None, None)


def fused_teacher_forcing_ready(features) -> bool:
    """Whether the fused pass takes these ``gdn_teacher_forcing`` features: CUDA, half precision, head dim 128."""
    q_num = features[0]
    return (
        q_num.is_cuda
        and q_num.dtype in (torch.bfloat16, torch.float16)
        and (q_num.shape[-1] == HEAD_DIM)
        and all(t.dtype == q_num.dtype for t in features[1:5])
    )


def gdn_teacher_forcing_fused(
    q_num,
    k_num,
    v,
    q_den,
    k_den,
    beta,
    chunk_decay,
    chunk_boundaries,
    num_clean: int,
    eps: float = 1e-06,
    sink: int = 0,
    observation=None,
    observation_spans: tuple | None = None,
) -> torch.Tensor:
    """``gdn_teacher_forcing`` with every token valid and no read / write gates, fused (bf16 matmul operands)."""
    spans = None
    if observation is not None:
        spans = tuple((int(offset), int(length)) for (offset, length) in observation_spans)
    layout = fused_teacher_forcing_layout(
        tuple(int(b) for b in chunk_boundaries), int(num_clean), int(sink), spans, q_num.device
    )
    decays = chunk_decay if observation is None else torch.cat([chunk_decay, observation.decay], dim=-1)
    block_decay = decays.index_select(-1, layout.decay_index)
    return _TeacherForcingGatedDelta.apply(
        q_num, k_num, v, q_den, k_den, beta, block_decay, layout, float(eps), False
    )


def gdn_packed_row_fused(
    q, k, v, beta, window_decays, windows: tuple, sink: int, observation_spans: tuple, eps: float = 1e-06
) -> torch.Tensor:
    """``gdn_teacher_forcing_fused`` for a packed row of teacher-forcing windows in one pass: the state restarts at every window. ``windows`` = ((first token, bounds, clean segments), ...); ``window_decays`` holds every window's (segment decay, observation decay)."""
    spans = tuple((int(offset), int(length)) for (offset, length) in observation_spans)
    layout = fused_packed_row_layout(windows, int(sink), spans, q.device)
    decays = torch.cat([decay for pair in window_decays for decay in pair], dim=-1)
    block_decay = decays.index_select(-1, layout.decay_index)
    return _TeacherForcingGatedDelta.apply(q, k, v, q, k, beta, block_decay, layout, float(eps), False)
