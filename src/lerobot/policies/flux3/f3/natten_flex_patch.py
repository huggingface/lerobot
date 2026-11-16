"""Local workaround for NATTEN flex-fna on GPUs without libnatten kernels (e.g. sm_120).

``torch.nn.attention.flex_attention.create_block_mask`` evaluates ``mask_mod`` over the full
Q_LEN x KV_LEN grid. For the FLUX3 video VAE (N = 42*64*128 = 344,064 tokens) that is a dense
N^2 bool tensor = 110.25 GiB (8x that when it is not compiled). Under ``compile_model=true`` the
inductor-compiled version still materializes it.

This module replaces ``create_block_mask`` *inside natten.backends.flex only* with a version that
evaluates ``mask_mod`` a few q-blocks at a time and reduces each block to partial/full flags, so
peak memory is O(group * block * N) instead of O(N^2). Masks are cached per shape. The attention
kernel itself runs compiled (inference only; backprop stays disallowed). site-packages is untouched.
"""

from __future__ import annotations

import natten
import torch
from natten.backends import flex as _flex
from torch.nn.attention.flex_attention import BlockMask

_installed = False
_orig_create_block_mask = _flex.create_block_mask
_orig_get_na_flex_mask = _flex.get_na_flex_mask
_GROUP = 4  # q-blocks evaluated at once
_cache: dict = {}


def _freeze(x):
    return tuple(_freeze(i) for i in x) if isinstance(x, (list, tuple)) else x


@torch.compiler.disable
@torch.no_grad()
def _chunked_create_block_mask(
    mask_mod, B, H, Q_LEN, KV_LEN, device="cuda", BLOCK_SIZE=128, _compile=False, **kw  # noqa: N803
):
    bq, bkv = (BLOCK_SIZE, BLOCK_SIZE) if isinstance(BLOCK_SIZE, int) else tuple(BLOCK_SIZE)
    if Q_LEN % bq or KV_LEN % bkv:
        return _orig_create_block_mask(
            mask_mod, B, H, Q_LEN, KV_LEN, device=device, BLOCK_SIZE=BLOCK_SIZE, _compile=_compile, **kw
        )
    nq, nkv = Q_LEN // bq, KV_LEN // bkv
    kv_idx = torch.arange(KV_LEN, device=device)[None, :]
    zero = torch.zeros((), dtype=torch.int32, device=device)
    partial = torch.empty(nq, nkv, dtype=torch.bool, device=device)
    full = torch.empty(nq, nkv, dtype=torch.bool, device=device)
    for s in range(0, nq, _GROUP):
        e = min(s + _GROUP, nq)
        q_idx = torch.arange(s * bq, e * bq, device=device)[:, None]
        m = mask_mod(zero, zero, q_idx, kv_idx).expand(q_idx.shape[0], KV_LEN)
        m = m.reshape(e - s, bq, nkv, bkv)
        any_, all_ = m.any(dim=(1, 3)), m.all(dim=(1, 3))
        partial[s:e], full[s:e] = any_ & ~all_, all_
        del m, any_, all_

    def to_blocks(flags):
        num = flags.sum(-1, dtype=torch.int32)
        idx = torch.argsort(~flags, dim=-1, stable=True).to(torch.int32)
        return num[None, None], idx[None, None]

    kv_num, kv_ind = to_blocks(partial)
    f_num, f_ind = to_blocks(full)
    return BlockMask.from_kv_blocks(kv_num, kv_ind, f_num, f_ind, BLOCK_SIZE=(bq, bkv), mask_mod=mask_mod)


def _cached_get_na_flex_mask(**kw):
    key = tuple(sorted((k, _freeze(v)) for k, v in kw.items()))
    mask = _cache.get(key)
    if mask is None:
        mask = _cache[key] = _orig_get_na_flex_mask(**kw)
    return mask


def install() -> None:
    global _installed
    if _installed:
        return
    natten.allow_flex_compile()
    _flex.create_block_mask = _chunked_create_block_mask
    _flex.get_na_flex_mask = _cached_get_na_flex_mask
    _installed = True
