"""Optional group limits before global clipping for unsharded, synchronized gradients."""

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn


@torch.no_grad()
def clip_grad_norm_with_groups_(
    parameters: Iterable[nn.Parameter], param_groups: list[dict[str, Any]], max_norm: float
) -> tuple[Tensor, dict[str, float]]:
    """Clip selected groups, then enforce the global budget without norm overflow.

    Call after DDP synchronization and AMP unscaling. Sharded gradients are not
    supported. Inspect every group before mutating any gradient, so NaN/Inf stops
    the update instead of silently zeroing finite gradients during clipping.
    The returned norm is the original pre-clipping norm, matching PyTorch.
    """
    if not math.isfinite(max_norm) or max_norm < 0:
        raise ValueError("Global gradient limit must be finite and nonnegative")
    expected = {id(p) for p in parameters if p.grad is not None}
    seen: set[int] = set()
    groups: list[tuple[list[Tensor], Tensor, float | None, str]] = []
    for index, group in enumerate(param_groups):
        limit = group.get("grad_clip_norm")
        if limit is not None and (not math.isfinite(limit) or limit <= 0):
            raise ValueError("Group gradient limits must be finite and positive")
        grads = []
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            if id(parameter) in seen:
                raise ValueError("Gradient clipping groups must not overlap")
            seen.add(id(parameter))
            if parameter.grad.device.type not in {"cpu", "cuda"}:
                raise ValueError("Group gradient clipping requires CPU/CUDA with float64 norm support")
            grads.append(parameter.grad)
        if not grads:
            continue
        norms = torch.stack([torch.linalg.vector_norm(g, dtype=torch.float64) for g in grads])
        norm = torch.linalg.vector_norm(norms)
        groups.append((grads, norm, limit, str(group.get("name", index))))
    if seen != expected:
        raise ValueError("Gradient clipping groups must cover every parameter with a gradient")
    if not groups:
        return torch.tensor(0.0, dtype=torch.float64), {}
    norms = torch.stack([group[1] for group in groups])
    total_norm = torch.linalg.vector_norm(norms)
    if not torch.isfinite(total_norm):
        raise FloatingPointError("Nonfinite gradient detected; refusing optimizer and scheduler updates")

    # Group limits prevent an expert spike from spending the entire global budget.
    coefficients = [
        torch.clamp(limit / (norm + 1e-6), max=1.0) if limit is not None else torch.ones_like(norm)
        for _, norm, limit, _ in groups
    ]
    after_group_norm = torch.linalg.vector_norm(norms * torch.stack(coefficients))
    global_coefficient = (
        torch.clamp(max_norm / (after_group_norm + 1e-6), max=1.0)
        if max_norm > 0
        else torch.ones_like(after_group_norm)
    )
    metrics = {}
    for (grads, norm, _, name), coefficient in zip(groups, coefficients, strict=True):
        scale = coefficient * global_coefficient
        for grad in grads:
            grad.mul_(scale.to(device=grad.device, dtype=grad.dtype))
        metrics[f"grad_norm_{name}"] = norm.item()
        metrics[f"grad_scale_{name}"] = scale.item()
    return total_norm, metrics
