# Vendored from runwayml/praxis-1-oss at commit 8dad1a4, original path praxis/kernels/layers_triton.py.
# Import paths are rewritten to the LeRobot package; the class PraxisPolicy of praxis.model
# is renamed PraxisModel here. Keep this file in sync with the upstream file.
"""Custom Triton kernels for the backbone transformer layers.

These are custom kernels, not PyTorch built-ins. The original implementation
of the model runs these two kernels at inference. Their rounding differs from
the equivalent PyTorch ops, so praxis runs the same kernels to give the same
bits.

* ``swiglu_packed`` computes ``silu(gate) * up`` for the routed experts of
  the mixture-of-experts layers. It computes the SiLU in float32 with the
  Triton sigmoid, rounds it to the input dtype and multiplies by ``up``.
* ``modulate_fused`` computes ``x * scale + shift`` in float32 for the
  decoder modulation. The compiler can fuse the multiply and the add into one
  FMA, so its rounding is not the rounding of the eager ops.

Each kernel has a fixed block size and no autotuning, so the output bits do
not depend on the launch.

Both functions support autograd. The backward passes are the backward passes
of the original training code:

* ``swiglu_packed``: the eager PyTorch backward of ``F.silu(gate) * up``
  (``aten.silu_backward`` and two products in the input dtype).
* ``modulate_fused``: float32 products, rounded to the dtype of each input.
  For ``rows=True`` a Triton kernel sums ``grad`` and ``grad * x`` over the
  tokens in float32, in blocks of ``MODULATE_REDUCE_BLOCK`` tokens.
"""

from __future__ import annotations

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


SWIGLU_BLOCK = 1024
MODULATE_BLOCK = 1024
MODULATE_REDUCE_BLOCK = 128


@_jit
def _swiglu_packed_kernel(
    gu_ptr,
    out_ptr,
    gu_row_stride,
    out_row_stride,
    n_cols,
    BLOCK: tl.constexpr,
) -> None:
    """out = silu(gate) * up over a packed (rows, 2 * n_cols) buffer."""
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < n_cols
    gu = gu_ptr + row * gu_row_stride
    gate = tl.load(gu + col, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(gu + n_cols + col, mask=mask, other=0.0)
    out = (gate * tl.sigmoid(gate)).to(up.dtype) * up
    tl.store(out_ptr + row * out_row_stride + col, out, mask=mask)


@_jit
def _modulate_kernel(
    x_ptr,
    shift_ptr,
    scale_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """out = x * scale + shift in float32, for three tensors of one shape."""
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x_raw = tl.load(x_ptr + offs, mask=mask, other=0)
    shift_raw = tl.load(shift_ptr + offs, mask=mask, other=0)
    scale_raw = tl.load(scale_ptr + offs, mask=mask, other=0)
    out = x_raw.to(tl.float32) * scale_raw.to(tl.float32) + shift_raw.to(tl.float32)
    tl.store(out_ptr + offs, out.to(x_raw.dtype), mask=mask)


@_jit
def _modulate_rows_kernel(
    x_ptr,
    shift_ptr,
    scale_ptr,
    out_ptr,
    B,
    S,
    C,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """out = x * scale + shift in float32, x (B, S, C), shift/scale (B, C)."""
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    c = offs % C
    tmp = offs // C
    b = tmp // S
    bc_idx = b * C + c
    x_raw = tl.load(x_ptr + offs, mask=mask, other=0)
    shift_raw = tl.load(shift_ptr + bc_idx, mask=mask, other=0)
    scale_raw = tl.load(scale_ptr + bc_idx, mask=mask, other=0)
    out = x_raw.to(tl.float32) * scale_raw.to(tl.float32) + shift_raw.to(tl.float32)
    tl.store(out_ptr + offs, out.to(x_raw.dtype), mask=mask)


@_jit
def _modulate_bwd_kernel(
    grad_ptr,
    x_ptr,
    scale_ptr,
    dx_ptr,
    dshift_ptr,
    dscale_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """dx = grad * scale, dshift = grad, dscale = grad * x (float32 products)."""
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    grad_raw = tl.load(grad_ptr + offs, mask=mask, other=0)
    x_raw = tl.load(x_ptr + offs, mask=mask, other=0)
    scale_raw = tl.load(scale_ptr + offs, mask=mask, other=0)
    grad_f = grad_raw.to(tl.float32)
    dx = grad_f * scale_raw.to(tl.float32)
    dscale = grad_f * x_raw.to(tl.float32)
    tl.store(dx_ptr + offs, dx.to(x_raw.dtype), mask=mask)
    tl.store(dshift_ptr + offs, grad_f, mask=mask)
    tl.store(dscale_ptr + offs, dscale, mask=mask)


@_jit
def _modulate_rows_bwd_dx_kernel(
    grad_ptr,
    scale_ptr,
    dx_ptr,
    S,
    C,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """dx = grad * scale for x (B, S, C) and scale (B, C), in float32."""
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    c = offs % C
    tmp = offs // C
    b = tmp // S
    bc_idx = b * C + c
    grad_raw = tl.load(grad_ptr + offs, mask=mask, other=0)
    scale_raw = tl.load(scale_ptr + bc_idx, mask=mask, other=0)
    dx = grad_raw.to(tl.float32) * scale_raw.to(tl.float32)
    tl.store(dx_ptr + offs, dx.to(grad_raw.dtype), mask=mask)


@_jit
def _modulate_rows_bwd_reduce_kernel(
    grad_ptr,
    x_ptr,
    dshift_ptr,
    dscale_ptr,
    S,
    C,
    BLOCK_S: tl.constexpr,
) -> None:
    """dshift = sum_s grad and dscale = sum_s grad * x, for one (b, c) pair."""
    pid_bc = tl.program_id(0).to(tl.int64)
    b = pid_bc // C
    c = pid_bc % C
    acc_shift = tl.zeros((), dtype=tl.float32)
    acc_scale = tl.zeros((), dtype=tl.float32)
    s_start = 0
    while s_start < S:
        s_offs = s_start + tl.arange(0, BLOCK_S)
        s_mask = s_offs < S
        idx = ((b * S + s_offs) * C) + c
        grad = tl.load(grad_ptr + idx, mask=s_mask, other=0).to(tl.float32)
        x = tl.load(x_ptr + idx, mask=s_mask, other=0).to(tl.float32)
        acc_shift += tl.sum(grad, axis=0)
        acc_scale += tl.sum(grad * x, axis=0)
        s_start += BLOCK_S
    tl.store(dshift_ptr + pid_bc, acc_shift)
    tl.store(dscale_ptr + pid_bc, acc_scale)


@_jit
def _swiglu_packed_bwd_kernel(
    gu_ptr,
    dout_ptr,
    dgu_ptr,
    gu_row_stride,
    dout_row_stride,
    dgu_row_stride,
    n_cols,
    BLOCK: tl.constexpr,
) -> None:
    """d(silu(gate) * up) over a packed (rows, 2 * n_cols) buffer, in float32."""
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < n_cols
    gu = gu_ptr + row * gu_row_stride
    gate = tl.load(gu + col, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(gu + n_cols + col, mask=mask, other=0.0).to(tl.float32)
    dout = tl.load(dout_ptr + row * dout_row_stride + col, mask=mask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(gate)
    silu = gate * sig
    dgate = dout * up * (silu * (1 - sig) + sig)
    dup = dout * silu
    dgu = dgu_ptr + row * dgu_row_stride
    tl.store(dgu + col, dgate, mask=mask)
    tl.store(dgu + n_cols + col, dup, mask=mask)


def swiglu_packed_backward_fused(gate_up: Tensor, grad: Tensor) -> Tensor:
    """Return the gradient of ``silu(gate) * up`` with all products in float32.

    This is the backward of the routed experts in the chunked MoE dispatch
    of the original training code (one rounding to the input dtype at the
    end). On the CPU the same formula runs in eager PyTorch.
    """
    if not gate_up.is_cuda or not HAS_TRITON:
        gate, up = gate_up.chunk(2, dim=-1)
        sig = torch.sigmoid(gate)
        d_gate = grad * up * (sig * (1 + gate * (1 - sig)))
        d_up = grad * (gate * sig)
        return torch.cat([d_gate, d_up], dim=-1)
    n_cols = gate_up.shape[-1] // 2
    gu = gate_up.reshape(-1, 2 * n_cols)
    dout = grad.reshape(-1, n_cols).contiguous()
    dgu = torch.empty_like(gu)
    if gu.shape[0]:
        grid = (gu.shape[0], triton.cdiv(n_cols, SWIGLU_BLOCK))
        _swiglu_packed_bwd_kernel[grid](
            gu,
            dout,
            dgu,
            gu.stride(0),
            dout.stride(0),
            dgu.stride(0),
            n_cols,
            BLOCK=SWIGLU_BLOCK,  # ty: ignore[invalid-argument-type]
        )
    return dgu.view_as(gate_up)


def _swiglu_forward(gate_up: Tensor) -> Tensor:
    n_cols = gate_up.shape[-1] // 2
    gu = gate_up.reshape(-1, 2 * n_cols)
    out = gate_up.new_empty(gu.shape[0], n_cols)
    if gu.shape[0]:
        grid = (gu.shape[0], triton.cdiv(n_cols, SWIGLU_BLOCK))
        _swiglu_packed_kernel[grid](gu, out, gu.stride(0), out.stride(0), n_cols, BLOCK=SWIGLU_BLOCK)
    return out.view(*gate_up.shape[:-1], n_cols)


def swiglu_packed_backward(gate_up: Tensor, grad: Tensor) -> Tensor:
    """Return the gradient of ``silu(gate) * up`` with respect to ``gate_up``.

    The function runs the eager PyTorch backward of ``F.silu(gate) * up``:
    ``dgate = silu_backward(grad * up, gate)`` and ``dup = grad * silu(gate)``.
    Each product rounds to the input dtype, as in the eager backward.
    """
    gate, up = gate_up.chunk(2, dim=-1)
    dgu = torch.empty_like(gate_up)
    dgate, dup = dgu.chunk(2, dim=-1)
    torch.mul(grad, up, out=dgate)
    torch.ops.aten.silu_backward.grad_input(dgate, gate, grad_input=dgate)
    torch.ops.aten.silu.out(gate, out=dup)
    torch.mul(grad, dup, out=dup)
    return dgu


class _SwiGLUPacked(torch.autograd.Function):
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, gate_up: Tensor) -> Tensor:
        ctx.save_for_backward(gate_up)
        return _swiglu_forward(gate_up)

    @staticmethod
    def backward(  # ty: ignore[invalid-method-override]
        ctx: torch.autograd.function.FunctionCtx, grad: Tensor
    ) -> Tensor:
        (gate_up,) = ctx.saved_tensors  # ty: ignore[unresolved-attribute]
        return swiglu_packed_backward(gate_up, grad)


def swiglu_packed(gate_up: Tensor) -> Tensor:
    """Return ``silu(gate) * up`` for the two halves of a CUDA ``(rows, 2 * H)``."""
    if torch.is_grad_enabled() and gate_up.requires_grad:
        return _SwiGLUPacked.apply(gate_up)
    return _swiglu_forward(gate_up)


def _modulate_forward(x: Tensor, shift: Tensor, scale: Tensor, rows: bool) -> Tensor:
    out = torch.empty_like(x)
    n_elements = out.numel()
    if n_elements == 0:
        return out
    grid = (triton.cdiv(n_elements, MODULATE_BLOCK),)
    if not rows:
        _modulate_kernel[grid](x, shift, scale, out, n_elements, BLOCK_SIZE=MODULATE_BLOCK)
        return out
    b, s, c = x.shape
    _modulate_rows_kernel[grid](x, shift, scale, out, b, s, c, n_elements, BLOCK_SIZE=MODULATE_BLOCK)
    return out


def modulate_fused_backward(
    grad: Tensor,
    x: Tensor,
    scale: Tensor,
    shift_dtype: torch.dtype,
    rows: bool,
    needs: tuple[bool, bool, bool] = (True, True, True),
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    """Return the gradients ``(dx, dshift, dscale)`` of ``modulate_fused``.

    All tensors are contiguous CUDA tensors. ``x`` is ``(B, S, C)``. For
    ``rows=False``, ``scale`` has the shape of ``x``. For ``rows=True``,
    ``scale`` is ``(B, C)`` and the returned ``dshift`` and ``dscale`` are
    ``(B, C)``. The products run in float32. ``dx`` rounds to the dtype of
    ``x``. ``dshift`` rounds to ``shift_dtype`` and ``dscale`` to the dtype of
    ``scale``. ``needs`` selects the gradients to compute.
    """
    n_elements = grad.numel()
    grid = (triton.cdiv(n_elements, MODULATE_BLOCK),)
    if not rows:
        dx = torch.empty_like(x)
        dshift = torch.empty(x.shape, device=x.device, dtype=shift_dtype)
        dscale = torch.empty_like(scale)
        if n_elements:
            _modulate_bwd_kernel[grid](
                grad,
                x,
                scale,
                dx,
                dshift,
                dscale,
                n_elements,
                BLOCK_SIZE=MODULATE_BLOCK,
            )
        return (
            dx if needs[0] else None,
            dshift if needs[1] else None,
            dscale if needs[2] else None,
        )
    b, s, c = x.shape
    dx = dshift = dscale = None
    if needs[0]:
        dx = torch.empty_like(x)
        if n_elements:
            _modulate_rows_bwd_dx_kernel[grid](grad, scale, dx, s, c, n_elements, BLOCK_SIZE=MODULATE_BLOCK)
    if needs[1] or needs[2]:
        dshift_bc = torch.empty((b, c), device=x.device, dtype=shift_dtype)
        dscale_bc = torch.empty((b, c), device=x.device, dtype=scale.dtype)
        if b * c:
            _modulate_rows_bwd_reduce_kernel[(b * c,)](
                grad, x, dshift_bc, dscale_bc, s, c, BLOCK_S=MODULATE_REDUCE_BLOCK
            )
        dshift = dshift_bc if needs[1] else None
        dscale = dscale_bc if needs[2] else None
    return dx, dshift, dscale


class _ModulateFused(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: Tensor,
        shift: Tensor,
        scale: Tensor,
        rows: bool,
    ) -> Tensor:
        ctx.save_for_backward(x, scale)
        ctx.rows = rows  # ty: ignore[unresolved-attribute]
        ctx.shift_dtype = shift.dtype  # ty: ignore[unresolved-attribute]
        return _modulate_forward(x, shift, scale, rows)

    @staticmethod
    def backward(  # ty: ignore[invalid-method-override]
        ctx: torch.autograd.function.FunctionCtx, grad: Tensor
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, None]:
        x, scale = ctx.saved_tensors  # ty: ignore[unresolved-attribute]
        needs = tuple(ctx.needs_input_grad[:3])  # ty: ignore[unresolved-attribute]
        dx, dshift, dscale = modulate_fused_backward(
            grad.contiguous(),
            x,
            scale,
            ctx.shift_dtype,  # ty: ignore[unresolved-attribute]
            ctx.rows,  # ty: ignore[unresolved-attribute]
            needs,
        )
        return dx, dshift, dscale, None


def modulate_fused(x: Tensor, shift: Tensor, scale: Tensor, rows: bool) -> Tensor:
    """Return ``x * scale + shift`` in float32, cast to the dtype of ``x``.

    All three tensors are 3-D CUDA tensors. With ``rows=False`` they have one
    shape. With ``rows=True``, ``shift`` and ``scale`` are ``(B, 1, C)`` and
    apply to every token of ``x`` ``(B, S, C)``.
    """
    x = x.contiguous()
    if rows:
        shift, scale = shift[:, 0, :], scale[:, 0, :]
    shift, scale = shift.contiguous(), scale.contiguous()
    if torch.is_grad_enabled() and (x.requires_grad or shift.requires_grad or scale.requires_grad):
        return _ModulateFused.apply(x, shift, scale, rows)
    return _modulate_forward(x, shift, scale, rows)
