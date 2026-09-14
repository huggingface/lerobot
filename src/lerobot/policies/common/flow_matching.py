#!/usr/bin/env python

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

"""Flow-matching sampling primitives shared across policies.

Canonical versions of the beta-distributed timestep sampler, the noise sampler and the
forward-Euler denoising loop (with its real-time-chunking hook) that the openpi-derived
policies (pi0, pi05, smolvla, eo1) historically each carried a copy of, plus the knobs
needed by the policies that diverge from that convention (groot, evo1, wall_x).

The samplers deliberately return *untransformed* draws: every adopter's distribution,
dtype and order of casting versus transformation is decided at its own call site, because
those choices are baked into released checkpoints. All functions are stateless; adopting
them does not affect checkpoints.
"""

from collections.abc import Callable
from functools import cache
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor

if TYPE_CHECKING:
    from lerobot.policies.rtc.modeling_rtc import RTCProcessor

BetaSampler = Callable[[float, float, int], Tensor]
"""Draws ``bsize`` raw ``Beta(alpha, beta)`` samples as ``(alpha, beta, bsize) -> (bsize,)``."""


@cache
def _beta_distribution(alpha: float, beta: float) -> "torch.distributions.Beta":
    """Cached ``Beta(alpha, beta)`` whose concentrations are pinned to CPU.

    Beta sampling goes through ``_sample_dirichlet``, which is unimplemented on MPS, so the
    draw happens on CPU and the result is moved afterwards. The concentrations are built
    with an explicit ``device="cpu"`` rather than the ambient default device: because this
    function is cached, a single construction inside a ``torch.device(...)`` context would
    otherwise be reused for every later call.
    """
    alpha_t = torch.tensor(alpha, device="cpu", dtype=torch.float32)
    beta_t = torch.tensor(beta, device="cpu", dtype=torch.float32)
    return torch.distributions.Beta(alpha_t, beta_t, validate_args=False)


def device_beta_sampler(device) -> BetaSampler:
    """Build a ``sample_beta`` hook that draws from ``device``'s RNG instead of the CPU one.

    Only for callers whose released behavior depends on the device-side RNG stream (wall_x).
    Drawing on CPU and moving the result yields different numbers for the same seed, so such
    callers cannot use the default CPU path without changing what their checkpoints see.
    """

    def sample(alpha: float, beta: float, bsize: int) -> Tensor:
        dist = torch.distributions.Beta(
            torch.tensor(alpha, dtype=torch.float32, device=device),
            torch.tensor(beta, dtype=torch.float32, device=device),
            validate_args=False,
        )
        return dist.sample([bsize])

    return sample


def sample_beta(
    alpha: float,
    beta: float,
    bsize: int,
    device,
    *,
    dtype: torch.dtype | None = None,
    sampler: BetaSampler | None = None,
) -> Tensor:  # see openpi
    """Draw raw ``Beta(alpha, beta)`` samples of shape ``(bsize,)``.

    No affine transform is applied. Callers whose recipe transforms the sample (complement,
    scale, clamp) apply it themselves: the order of those operations relative to the dtype
    cast is policy-specific and numerically load-bearing, so it cannot be folded in here.

    Args:
        device: Device the sample is moved to.
        dtype: Cast applied together with the device move. ``None`` (default) keeps the
            sampler's float32, the openpi convention. Pass the action dtype to reproduce a
            ``.to(device, dtype=...)`` that precedes the caller's own transform (groot).
        sampler: Optional hook replacing the cached CPU distribution, for callers that must
            preserve a different RNG stream (see ``device_beta_sampler``).
    """
    if sampler is not None:
        raw = sampler(alpha, beta, bsize)
    else:
        raw = _beta_distribution(alpha, beta).sample((bsize,))
    if dtype is None:
        return raw.to(device)
    return raw.to(device, dtype=dtype)


def sample_noise(
    shape,
    device,
    *,
    dtype: torch.dtype = torch.float32,
    distribution: Literal["normal", "uniform"] = "normal",
) -> Tensor:
    """The flow-matching ``x_1`` noise sample.

    Args:
        dtype: Defaults to float32, so existing openpi-derived callers
            (pi0/pi05/smolvla/eo1, wall_x) are unchanged. Pass the action dtype when the
            policy samples directly in it (groot, evo1) instead of casting afterwards.
        distribution: ``"normal"`` (default) draws standard-normal noise, matching
            ``torch.randn``. ``"uniform"`` draws from ``[-1, 1)`` as ``rand * 2 - 1``
            (evo1).
    """
    if distribution == "normal":
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=dtype,
            device=device,
        )
    if distribution == "uniform":
        return torch.rand(shape, dtype=dtype, device=device) * 2 - 1
    raise ValueError(f"Unknown noise distribution {distribution!r}, expected 'normal' or 'uniform'")


def sample_time_beta(
    bsize: int,
    device,
    *,
    alpha: float,
    beta: float,
    scale: float = 1.0,
    offset: float = 0.0,
) -> Tensor:
    """Beta-distributed flow-matching timesteps: ``Beta(alpha, beta) * scale + offset``.

    ``scale`` and ``offset`` default to the identity transform (``scale=1.0, offset=0.0``),
    i.e. the raw Beta sample in float32. The ``scale=0.999, offset=0.001`` endpoint offsets
    are the pi-family recipe, which pi0/pi05/smolvla/eo1 pass in from their configs; they
    are not a default of this helper.
    """
    time_beta = sample_beta(alpha, beta, bsize, device)
    time = time_beta * scale + offset
    return time.to(dtype=torch.float32, device=device)


def euler_integrate(
    denoise_fn: Callable[[Tensor, Tensor], Tensor],
    noise: Tensor,
    num_steps: int,
    *,
    rtc_processor: "RTCProcessor | None" = None,
    rtc_enabled: bool = False,
    inference_delay: int | None = None,
    prev_chunk_left_over: Tensor | None = None,
    execution_horizon: int | None = None,
    hard_prefix: Tensor | None = None,
    hard_prefix_mask: Tensor | None = None,
) -> Tensor:
    """Forward-Euler integration of a velocity field from t=1 (noise) to t=0 (actions).

    This is the openpi sampling loop: ``dt = -1/num_steps``, ``time = 1.0 + step*dt``,
    ``x_t <- x_t + dt * v_t``, with the optional real-time-chunking (RTC) guidance hook
    wrapping the velocity computation and debug tracking after each step.

    Args:
        denoise_fn: Computes the velocity ``v_t`` from ``(x_t, time_tensor)`` where
            ``time_tensor`` is a float32 tensor of shape ``(batch_size,)``. The returned
            velocity must have the same shape and dtype as ``x_t``.
        noise: Initial sample ``x_1`` of shape ``(batch_size, ...)``.
        num_steps: Number of Euler steps.
        rtc_processor: Optional RTC processor. Debug tracking fires whenever it is set and
            has debugging enabled, even if RTC guidance itself is disabled (this mirrors
            the historical per-policy loops).
        rtc_enabled: Whether to route the velocity computation through
            ``rtc_processor.denoise_step`` (requires ``rtc_processor``).
        inference_delay: RTC guidance parameter, forwarded verbatim.
        prev_chunk_left_over: RTC guidance parameter, forwarded verbatim.
        execution_horizon: RTC guidance parameter, forwarded verbatim.
        hard_prefix: Optional clean action prefix to clamp throughout denoising.
        hard_prefix_mask: Boolean mask selecting the values clamped from ``hard_prefix``.
    """
    bsize = noise.shape[0]
    device = noise.device

    dt = -1.0 / num_steps
    x_t = noise
    for step in range(num_steps):
        time = 1.0 + step * dt
        time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

        if hard_prefix is not None:
            if hard_prefix_mask is None:
                raise ValueError("hard_prefix_mask is required when hard_prefix is provided")
            x_t = torch.where(hard_prefix_mask, hard_prefix, x_t)
            time_tensor = time_tensor[:, None].expand(bsize, x_t.shape[1]).clone()
            time_tensor[hard_prefix_mask[..., 0]] = 0.0

        def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
            return denoise_fn(input_x_t, current_timestep)

        if rtc_enabled:
            v_t = rtc_processor.denoise_step(
                x_t=x_t,
                prev_chunk_left_over=prev_chunk_left_over,
                inference_delay=inference_delay,
                time=time,
                original_denoise_step_partial=denoise_step_partial_call,
                execution_horizon=execution_horizon,
            )
        else:
            v_t = denoise_step_partial_call(x_t)

        x_t = x_t + dt * v_t

        if hard_prefix is not None:
            x_t = torch.where(hard_prefix_mask, hard_prefix, x_t)

        if rtc_processor is not None and rtc_processor.is_debug_enabled():
            rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

    return x_t
