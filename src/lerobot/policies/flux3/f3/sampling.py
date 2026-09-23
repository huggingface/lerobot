# Copyright 2026 Black Forest Labs. All rights reserved.
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
# Vendored from black-forest-labs/flux-action (src/flux_action/inference/sampling.py).
"""Joint video + action denoising for policy inference.

Two samplers over a dict of noised streams (``x_video`` and ``x_<modality>``):

* ``cosmos_unipc_order2`` — the literal Cosmos UniPC (order 2, bh2, predict-x0)
  our DROID policy was evaluated with: 4 steps, shift 5, CFG 3. One denoiser
  call per step (the corrector does not re-evaluate the model), integer
  scheduler ticks passed to the model as ``tick / 1000``.
* ``euler`` — plain rectified-flow Euler on the shifted schedule, for reference.

``cfg_two_pass`` runs the unconditional (empty caption) and conditional forward
separately, exactly like the video pipeline, and combines per stream so the
video and action streams can use different guidance scales.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor

from ..utils import rational_time_shift

Samples = dict[str, Tensor]


def cfg_two_pass(
    model,
    flow: Samples,
    fixed: dict[str, Tensor],
    timesteps: dict[str, Tensor],
    ctx_uc: tuple[Tensor, Tensor],
    ctx_c: tuple[Tensor, Tensor],
    cfg_by_key: dict[str, float],
) -> Samples:
    """pred = uc + cfg * (c - uc) per flow stream. ``fixed`` = ids, cond streams, vector."""

    def forward(ctx: Tensor, ctx_ids: Tensor) -> dict[str, Tensor]:
        return model(
            **flow,
            **fixed,
            **timesteps,
            ctx=ctx,
            ctx_ids=ctx_ids,
            timesteps_ctx=torch.zeros(ctx.shape[:2], device=ctx.device, dtype=ctx.dtype),
        )

    pred_uc = forward(*ctx_uc)
    pred_c = forward(*ctx_c)
    return {k: pred_uc[k] + cfg_by_key[k] * (pred_c[k] - pred_uc[k]) for k in flow}


def euler(
    samples: Samples,
    predict_velocity: Callable[[Samples, float], Samples],
    *,
    n_steps: int,
    alpha: float,
) -> Samples:
    timesteps = rational_time_shift(torch.linspace(1.0, 0.0, n_steps + 1), alpha).tolist()
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:], strict=True):
        pred = predict_velocity(samples, t_curr)
        samples = {
            k: (samples[k].float() + (t_prev - t_curr) * pred[k].float()).to(samples[k].dtype)
            for k in samples
        }
    return samples


def cosmos_unipc_schedule(
    n_steps: int, shift: float, num_train_timesteps: int = 1000
) -> tuple[Tensor, Tensor]:
    """Cosmos' FlowUniPCMultistepScheduler grid: float sigmas for the solver, integer ticks for the model."""
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    sigma_max = np.float32(1.0 - 1.0 / num_train_timesteps).item()
    sigmas_np = np.linspace(sigma_max, 0.0, n_steps + 1).copy()[:-1]
    sigmas_np = shift * sigmas_np / (1.0 + (shift - 1.0) * sigmas_np)
    model_timesteps = torch.from_numpy((sigmas_np * num_train_timesteps).astype(np.int64))
    sigmas = torch.from_numpy(np.concatenate([sigmas_np, [0.0]]).astype(np.float32))
    return sigmas, model_timesteps


def _bh_coefficients(h: Tensor, rks: list[Tensor], order: int, *, device, dtype, corrector: bool):
    """bh2 predictor / corrector coefficients (predict_x0=True), as in Cosmos."""
    hh = -h
    h_phi_1 = torch.expm1(hh)
    h_phi_k = h_phi_1 / hh - 1.0
    b_h = torch.expm1(hh)
    rks_t = torch.stack([rk.to(device=device) for rk in rks] + [torch.ones_like(h, device=device)])
    rows, rhs = [], []
    factorial_i = 1
    for i in range(1, order + 1):
        rows.append(torch.pow(rks_t, i - 1))
        rhs.append((h_phi_k * factorial_i / b_h).to(device=device))
        factorial_i *= i + 1
        h_phi_k = h_phi_k / hh - 1.0 / factorial_i
    matrix, rhs_t = torch.stack(rows), torch.stack(rhs)
    if corrector:
        rhos = (
            torch.tensor([0.5], dtype=dtype, device=device)
            if order == 1
            else torch.linalg.solve(matrix, rhs_t).to(dtype)
        )
    elif order == 2:
        rhos = torch.tensor([0.5], dtype=dtype, device=device)  # Cosmos' simplified UniP order-2 coefficient
    else:
        rhos = None
    return h_phi_1, b_h, rhos


def cosmos_unipc_order2(
    samples: Samples,
    predict_velocity: Callable[[Samples, Tensor], Samples],
    *,
    n_steps: int,
    shift: float,
    num_train_timesteps: int = 1000,
) -> Samples:
    """Cosmos UniPC (order 2, bh2, predict-x0) over a dict of streams sharing one grid."""
    keys = list(samples)
    sigmas, model_timesteps = cosmos_unipc_schedule(n_steps, shift, num_train_timesteps)
    solver_order = 2
    model_outputs: list[Samples | None] = [None] * solver_order
    lower_order_nums = 0
    last_sample: Samples | None = None
    this_order = 1

    def alpha_sigma(s: Tensor) -> tuple[Tensor, Tensor]:
        return 1.0 - s, s

    def lam(s: Tensor) -> Tensor:
        a, sg = alpha_sigma(s)
        return torch.log(a) - torch.log(sg)

    for step, tick in enumerate(model_timesteps):
        velocity = predict_velocity(samples, tick)
        sigma_cur = sigmas[step]
        x0 = {k: samples[k].float() - sigma_cur * velocity[k].float() for k in keys}

        if step > 0 and last_sample is not None:  # UniC corrector, using the previous step's order
            order_c = this_order
            sigma_t, sigma_s0 = sigmas[step], sigmas[step - 1]
            alpha_t, sig_t = alpha_sigma(sigma_t)
            _, sig_s0 = alpha_sigma(sigma_s0)
            h = lam(sigma_t) - lam(sigma_s0)
            rks, hist = [], []
            prev_x0 = model_outputs[-1]
            for i in range(1, order_c):
                older = model_outputs[-(i + 1)]
                rk = (lam(sigmas[step - (i + 1)]) - lam(sigma_s0)) / h
                rks.append(rk)
                hist.append({k: (older[k] - prev_x0[k]) / rk for k in keys})
            ex = samples[keys[0]]
            h_phi_1, b_h, rhos_c = _bh_coefficients(
                h, rks, order_c, device=ex.device, dtype=torch.float32, corrector=True
            )
            corrected = {}
            for k in keys:
                base = sig_t / sig_s0 * last_sample[k].float() - alpha_t * h_phi_1 * prev_x0[k]
                residual = sum(rhos_c[i] * d[k] for i, d in enumerate(hist)) if hist else 0.0
                corrected[k] = (base - alpha_t * b_h * (residual + rhos_c[-1] * (x0[k] - prev_x0[k]))).to(
                    samples[k].dtype
                )
            samples = corrected

        model_outputs[0] = model_outputs[1]
        model_outputs[1] = x0  # history holds the pre-correction x0, as in Cosmos

        remaining = len(model_timesteps) - step
        this_order = min(min(solver_order, remaining), lower_order_nums + 1)
        last_sample = samples

        sigma_t, sigma_s0 = sigmas[step + 1], sigmas[step]  # UniP predictor
        alpha_t, sig_t = alpha_sigma(sigma_t)
        _, sig_s0 = alpha_sigma(sigma_s0)
        h = lam(sigma_t) - lam(sigma_s0)
        rks, hist = [], []
        latest = model_outputs[-1]
        for i in range(1, this_order):
            older = model_outputs[-(i + 1)]
            rk = (lam(sigmas[step - i]) - lam(sigma_s0)) / h
            rks.append(rk)
            hist.append({k: (older[k] - latest[k]) / rk for k in keys})
        ex = samples[keys[0]]
        h_phi_1, b_h, rhos_p = _bh_coefficients(
            h, rks, this_order, device=ex.device, dtype=torch.float32, corrector=False
        )
        predicted = {}
        for k in keys:
            base = sig_t / sig_s0 * samples[k].float() - alpha_t * h_phi_1 * latest[k]
            if hist:
                base = base - alpha_t * b_h * sum(rhos_p[i] * d[k] for i, d in enumerate(hist))
            predicted[k] = base.to(samples[k].dtype)
        samples = predicted
        if lower_order_nums < solver_order:
            lower_order_nums += 1
    return samples
