# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""AdamW with stochastic rounding for bfloat16 parameters, and the seed helper.

Vendored from runwayml/praxis-1-oss at commit 8dad1a4 (``praxis/train/trainer.py``): the
optimizer of the praxis fine-tune. The parameters of the checkpoint are bfloat16; a plain
rounding of ``p - lr * update`` to bfloat16 loses most updates at the learning rates of the
recipe (3e-6), so the new value is rounded at random with the expected value kept
(:func:`round_stochastic_`). The optimizer handles the dim-0 shards of FSDP2 (``DTensor``), with
or without HSDP replicas (``dp_replicate x dp_shard``: the replicas of a shard draw the same noise)
and gives every element the same noise whatever the sharding.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import numpy as np
import torch
from torch import Tensor
from torch.distributed.tensor import DTensor

__all__ = ["SR_BLOCK", "AdamW", "derive_seed", "round_stochastic_"]

SR_BLOCK = 1 << 24


def derive_seed(*keys: int) -> int:
    """Return a 64-bit seed that depends on all ``keys`` (integers >= 0).

    A CPU ``torch.Generator`` uses only the low 32 bits of a seed.
    """
    hi, lo = np.random.SeedSequence([int(k) for k in keys]).generate_state(2)
    return (int(hi) << 32) | int(lo)


def _local_flat(t: Tensor) -> tuple[Tensor, int]:
    """Return the local elements of ``t`` as a flat view, and the global flat
    index of the first local element.

    A DTensor of FSDP has dim-0 shards. Rank r of N holds the rows
    ``[r * c, (r + 1) * c)``, with ``c = ceil(rows / N)``.
    """
    if not isinstance(t, DTensor):
        return t.view(-1), 0
    local = t.to_local()
    mesh = t.device_mesh
    shard_dims = [i for i, placement in enumerate(t.placements) if placement.is_shard()]
    if len(shard_dims) != 1 or not t.placements[shard_dims[0]].is_shard(0):
        raise ValueError(
            f"expected one dim-0 shard placement (FSDP, or HSDP with replicas), got {t.placements}"
        )
    for i, placement in enumerate(t.placements):
        if i != shard_dims[0] and not placement.is_replicate():
            raise ValueError(f"expected Replicate on the other mesh dims, got {t.placements}")
    # HSDP: the parameter is sharded along one mesh dim and replicated along the others; the
    # replicas hold the same rows, so the same global offsets (and the same rounding noise).
    shard_dim = shard_dims[0]
    shards = mesh.size(shard_dim) if mesh.ndim > 1 else mesh.size()
    shard_rank = mesh.get_local_rank(shard_dim) if mesh.ndim > 1 else mesh.get_local_rank()
    rows = t.shape[0]
    per_rank = -(-rows // shards)
    start = min(shard_rank * per_rank, rows)
    if local.shape[0] != min(per_rank, rows - start):
        raise ValueError(f"unexpected shard of {local.shape[0]} rows at row {start}")
    return local.view(-1), start * math.prod(t.shape[1:])


def round_stochastic_(x: Tensor, noise: Tensor) -> Tensor:
    """Round the float32 values of ``x`` to bfloat16 values at random, in place.

    ``noise`` holds int32 values in ``[0, 2**16)``. The function adds the
    noise to the low 16 bits of each value and then clears these bits. A value
    thus moves to the next bfloat16 value away from zero with the probability
    ``low bits / 2**16``. The expected value of the result is ``x``. The
    function keeps the sign and keeps NaN values. The result is exact in
    bfloat16.
    """
    nan = x.isnan()
    bits = x.view(torch.int32)
    bits.add_(noise).bitwise_and_(-(1 << 16))
    return x.masked_fill_(nan, float("nan"))


class AdamW(torch.optim.Optimizer):
    """AdamW for bfloat16 parameters, with stochastic rounding.

    The update runs in float32 on each parameter:

        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        p = p * (1 - lr * wd) - lr / (1 - beta1**t) * m / (sqrt(v / (1 - beta2**t)) + eps)

    The optimizer keeps ``m`` and ``v`` in the dtype of the parameter. A
    bfloat16 parameter gets the new value with stochastic rounding
    (``round_stochastic_``). A plain rounding would lose most updates. A
    bfloat16 value has 8 significant bits. An update of ``lr = 5e-5`` is much
    smaller than the spacing of the values.

    The decay factor ``1 - lr * wd`` is a float32 value. When ``lr * wd`` is
    below about 3e-8, the factor is 1.0 and the weight decay has no effect.
    This is the case for ``lr`` 5e-5 or less with ``wd`` 1e-4, for example
    for the default config (3e-6 * 1e-4).

    The noise of element ``e`` of parameter ``i`` in step ``t`` comes from a
    generator with the seed ``derive_seed(seed, t, i, e // SR_BLOCK)``. Thus
    the noise does not depend on the sharding of the parameter. Parameter
    ``i`` is the position in the parameter groups, in order.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 5e-5,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
        seed: int = 0,
    ) -> None:
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)
        self.seed = int(seed)
        self._generators: dict[torch.device, torch.Generator] = {}
        # LeRobot change: the state is allocated at the first update of a parameter, not here.
        # The training loop builds the optimizer before FSDP shards the model; eager full-size
        # moments would not fit (2 x 54 GB) and would not match the sharded parameters.

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # ty: ignore[invalid-method-override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        index = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    self._update(p, index, group)
                index += 1
        return loss

    def _update(self, p: Tensor, index: int, group: dict) -> None:
        state = self.state[p]
        if not state:
            state["step"] = torch.tensor(0.0)
            state["exp_avg"] = torch.zeros_like(p)
            state["exp_avg_sq"] = torch.zeros_like(p)
        state["step"] += 1
        step = int(state["step"].item())
        lr, eps, wd = float(group["lr"]), float(group["eps"]), group["weight_decay"]
        beta1, beta2 = group["betas"]
        step_size = lr / (1 - beta1**step)
        bias_correction2_sqrt = math.sqrt(1 - beta2**step)
        assert p.grad is not None
        param, offset = _local_flat(p)
        grad, _ = _local_flat(p.grad)
        exp_avg, _ = _local_flat(state["exp_avg"])
        exp_avg_sq, _ = _local_flat(state["exp_avg_sq"])
        n, total = param.numel(), p.numel()
        if n == 0:
            return
        block = min(SR_BLOCK, total)
        for j in range(offset // block, (offset + n - 1) // block + 1):
            lo, hi = max(offset, j * block), min(offset + n, (j + 1) * block)
            sl = slice(lo - offset, hi - offset)
            g = grad[sl].float()
            m = exp_avg[sl].float().lerp(g, 1 - beta1)
            v = exp_avg_sq[sl].float().mul(beta2).addcmul_(g, g, value=1 - beta2)
            exp_avg[sl].copy_(m)
            exp_avg_sq[sl].copy_(v)
            denom = v.sqrt_().div_(bias_correction2_sqrt).add_(eps)
            new = param[sl].float().mul(1 - lr * wd).addcdiv_(m, denom, value=-step_size)
            if param.dtype == torch.bfloat16:
                gen = self._generator(param.device)
                gen.manual_seed(derive_seed(self.seed, step, index, j))
                size = min(block, total - j * block)
                noise = torch.randint(
                    0,
                    1 << 16,
                    (size,),
                    generator=gen,
                    device=param.device,
                    dtype=torch.int32,
                )
                round_stochastic_(new, noise[lo - j * block : hi - j * block])
            param[sl].copy_(new)

    def _generator(self, device: torch.device) -> torch.Generator:
        if device not in self._generators:
            self._generators[device] = torch.Generator(device=device)
        return self._generators[device]


# ---------------------------------------------------------------------------
# Text cache
# ---------------------------------------------------------------------------
