# Vendored from runwayml/praxis-1-oss at commit 8dad1a4, original path praxis/kernels/vae_encoder_triton.py.
# Import paths are rewritten to the LeRobot package; the class PraxisPolicy of praxis.model
# is renamed PraxisModel here. Keep this file in sync with the upstream file.
"""Custom Triton kernels for the fused video autoencoder encoder.

These are custom kernels, not PyTorch built-ins. The fused encoder path in
``praxis/vae.py`` uses them. The original implementation runs the same two
kernels, so praxis runs them too to give the same bits. Both kernels work on
channels-last 5-D tensors, where the channel axis is contiguous.

* ``rmsnorm_silu`` fuses RMSNorm (``x / max(||x||, 1e-12) * sqrt(C) * gamma``)
  and SiLU. It accumulates the squared norm in float32 over channel blocks of
  ``BLOCK_C``. ``BLOCK_C`` sets the summation order, so it sets the rounding.
  ``BLOCK_M`` and the warp count do not change the bits.
* ``avg_down`` averages each 2x2 spatial window. With two time taps it writes
  the taps into adjacent output channels. Each output element adds its four
  taps in a fixed order, so no launch setting changes the bits.

The original implementation picks the launch setting with Triton autotuning by timing.
On a quiet H100 GPU it picks ``BLOCK_C = 64`` for 160 and 320 channels and
``BLOCK_C = 256`` for 640 channels. Under load it can pick another
``BLOCK_C``, and its output bits then change. Praxis does not autotune. It
uses a fixed ``BLOCK_C`` per channel count (``DEFAULT_NORM_BLOCK_C``), so its
bits do not depend on timing noise. ``Autoencoder(norm_block_c=...)`` sets
another table, for example to reproduce a run that picked another
setting.
"""

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - CPU-only installs (macOS, CPU CI)
    triton = None

    class _TritonLanguageStub:
        """Stands in for ``triton.language`` so the kernel signatures (``tl.constexpr`` annotations)
        import without Triton (macOS, CPU-only hosts). The kernels are never launched then."""

        constexpr = object()

    tl = _TritonLanguageStub()
    HAS_TRITON = False


def _jit(fn):
    """``triton.jit`` when Triton is installed; otherwise the plain function (never launched)."""
    return triton.jit(fn) if HAS_TRITON else fn


# BLOCK_C of the norm kernel per channel count. BLOCK_C sets the summation
# order. These values are the picks of the Triton autotuner on a quiet H100.
DEFAULT_NORM_BLOCK_C: dict[int, int] = {160: 64, 320: 64, 640: 256}
# BLOCK_M and num_warps per BLOCK_C. They do not change the output bits.
_NORM_LAUNCH: dict[int, tuple[int, int]] = {64: (16, 4), 128: (16, 4), 256: (8, 4)}
# Launch setting of the pooling kernel: (BLOCK_M, BLOCK_C, num_warps).
POOL_CONFIG = (8, 256, 4)
NUM_STAGES = 3


@_jit
def _rmsnorm_silu_kernel(
    x_ptr,
    y_ptr,
    g_ptr,  # float32 gamma, shape (C,)
    M,  # positions T*H*W per batch row, each a run of C elements
    x_bs,
    y_bs,  # batch strides (y can be a view into a halo buffer)
    sqrt_c,
    C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
) -> None:
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    xb = x_ptr + pid_b * x_bs
    yb = y_ptr + pid_b * y_bs
    row_x = xb + offs_m[:, None].to(tl.int64) * C
    row_y = yb + offs_m[:, None].to(tl.int64) * C

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for c0 in tl.range(0, C, BLOCK_C):
        offs_c = c0 + tl.arange(0, BLOCK_C)
        mask = mask_m[:, None] & (offs_c[None, :] < C)
        x = tl.load(row_x + offs_c[None, :], mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x * x, axis=1)

    # F.normalize semantics: x / max(||x||_2, 1e-12), then * sqrt(C), * gamma.
    denom = tl.maximum(tl.sqrt(acc), 1e-12)

    for c0 in tl.range(0, C, BLOCK_C):
        offs_c = c0 + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C
        mask = mask_m[:, None] & mask_c[None, :]
        x = tl.load(row_x + offs_c[None, :], mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + offs_c, mask=mask_c, other=0.0)
        y = ((x / denom[:, None]) * sqrt_c) * g[None, :]
        y = y * tl.sigmoid(y)
        tl.store(row_y + offs_c[None, :], y.to(y_ptr.dtype.element_ty), mask=mask)


@_jit
def _avg_down_kernel(
    x_ptr,
    y_ptr,
    M,  # output positions per batch row: TOUT * HOUT * WOUT
    HOUT,
    WOUT,
    W,  # input width
    x_bs,
    y_bs,  # batch strides
    C: tl.constexpr,
    FT: tl.constexpr,  # 1 or 2 time taps folded into channels
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
) -> None:
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    hw_out = HOUT * WOUT
    t_out = offs_m // hw_out
    rem = offs_m % hw_out
    h_out = rem // WOUT
    w_out = rem % WOUT

    xb = x_ptr + pid_b * x_bs
    yb = y_ptr + pid_b * y_bs
    h_in = 2 * h_out
    w_in = 2 * w_out
    out_row = yb + offs_m[:, None].to(tl.int64) * (C * FT)

    for j in tl.static_range(FT):
        t_in = FT * t_out + j
        base = ((t_in * (HOUT * 2) + h_in) * W + w_in).to(tl.int64)
        row00 = xb + base[:, None] * C
        row01 = row00 + C
        row10 = row00 + W.to(tl.int64) * C
        row11 = row10 + C
        for c0 in tl.range(0, C, BLOCK_C):
            offs_c = c0 + tl.arange(0, BLOCK_C)
            mask = mask_m[:, None] & (offs_c[None, :] < C)
            acc = tl.load(row00 + offs_c[None, :], mask=mask, other=0.0).to(tl.float32)
            acc += tl.load(row01 + offs_c[None, :], mask=mask, other=0.0).to(tl.float32)
            acc += tl.load(row10 + offs_c[None, :], mask=mask, other=0.0).to(tl.float32)
            acc += tl.load(row11 + offs_c[None, :], mask=mask, other=0.0).to(tl.float32)
            acc *= 0.25
            # One time tap: output channel c. Two time taps: output channel 2c + j.
            out_c = offs_c * FT + j
            tl.store(
                out_row + out_c[None, :],
                acc.to(y_ptr.dtype.element_ty),
                mask=mask,
            )


def rmsnorm_silu(x: Tensor, out: Tensor, gamma: Tensor, sqrt_c: float, block_c: int) -> None:
    """Write ``silu(rmsnorm(x))`` into ``out``.

    ``x`` and ``out`` are channels-last (B, C, T, H, W) tensors with the same
    shape. ``out`` can be a time slice of a larger buffer. ``gamma`` is a
    float32 (C,) tensor. ``block_c`` (64, 128 or 256) sets the summation order.
    """
    b, c = x.shape[0], x.shape[1]
    m = x.numel() // (b * c)
    block_m, warps = _NORM_LAUNCH[block_c]
    grid = (triton.cdiv(m, block_m), b)
    _rmsnorm_silu_kernel[grid](
        x,
        out,
        gamma,
        m,
        x.stride(0),
        out.stride(0),
        sqrt_c,
        C=c,
        BLOCK_M=block_m,
        BLOCK_C=block_c,
        num_warps=warps,
        num_stages=NUM_STAGES,
    )


def avg_down(x: Tensor, factor_t: int) -> Tensor:
    """Average 2x2 spatial windows of a channels-last (B, C, T, H, W) tensor.

    With ``factor_t == 2`` the output has 2C channels: channel ``2c + j`` holds
    the average of input frame ``2t + j``. ``T`` must be a multiple of
    ``factor_t``. The output is channels-last.
    """
    b, c, t, h, w = x.shape
    t_out, h_out, w_out = t // factor_t, h // 2, w // 2
    y = torch.empty(
        (b, c * factor_t, t_out, h_out, w_out),
        dtype=x.dtype,
        device=x.device,
        memory_format=torch.channels_last_3d,
    )
    m = t_out * h_out * w_out
    block_m, block_c, warps = POOL_CONFIG
    grid = (triton.cdiv(m, block_m), b)
    _avg_down_kernel[grid](
        x,
        y,
        m,
        h_out,
        w_out,
        w,
        x.stride(0),
        y.stride(0),
        C=c,
        FT=factor_t,
        BLOCK_M=block_m,
        BLOCK_C=block_c,
        num_warps=warps,
        num_stages=NUM_STAGES,
    )
    return y
