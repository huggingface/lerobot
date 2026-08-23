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

Canonical versions of the beta-distributed timestep sampler and the forward-Euler
denoising loop (with its real-time-chunking hook) that the openpi-derived policies
(pi0, pi05, smolvla, eo1) historically each carried a copy of. All functions are
stateless; adopting them does not affect checkpoints.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from lerobot.policies.rtc.modeling_rtc import RTCProcessor


def sample_beta(alpha: float, beta: float, bsize: int, device) -> Tensor:  # see openpi (exact copy)
    # Beta sampling uses _sample_dirichlet which isn't implemented for MPS, so sample on CPU
    alpha_t = torch.tensor(alpha, dtype=torch.float32)
    beta_t = torch.tensor(beta, dtype=torch.float32)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,)).to(device)


def sample_noise(shape, device) -> Tensor:
    """Standard-normal float32 noise, the flow-matching x_1 sample."""
    return torch.normal(
        mean=0.0,
        std=1.0,
        size=shape,
        dtype=torch.float32,
        device=device,
    )


def sample_time_beta(bsize: int, device, *, alpha: float, beta: float, scale: float, offset: float) -> Tensor:
    """Beta-distributed flow-matching timesteps: ``Beta(alpha, beta) * scale + offset`` (openpi convention)."""
    time_beta = sample_beta(alpha, beta, bsize, device)
    time = time_beta * scale + offset
    return time.to(dtype=torch.float32, device=device)


def staircase_time(delay: int, horizon: int, *, device, dtype: torch.dtype = torch.float32) -> Tensor:
    """Per-position flow timestep of the piR2 staircase (arXiv 2607.26055, Eq. 3).

    The paper states the schedule as three regions with tau=1 clean and tau=0 noise. Under this
    module's convention (t=1 noise, t=0 clean) the clean front and the pure-noise tail are just
    where the interior ramp saturates, so a single clamped ramp reproduces all three.

    Args:
        delay: Number of in-flight actions ``d``; also the number of slots emitted per call.
        horizon: Chunk length ``H``.

    Returns:
        Shape ``(horizon,)``: ``0`` over the clean front, a linear ramp of slope
        ``1 / (H - 2d)`` across the interior, and ``1`` over the pure-noise tail.
    """
    positions = torch.arange(horizon, device=device, dtype=dtype)
    interior = max(horizon - 2 * delay, 1)
    return ((positions - delay) / interior).clamp(0.0, 1.0)


def staircase_substep(
    denoise_fn: Callable[[Tensor, Tensor], Tensor],
    x_t: Tensor,
    delay: int,
    *,
    noise: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """One piR2 inference call: advance the buffer, emit ``delay`` actions, slide (Eq. 4, Fig. 2).

    A single velocity evaluation shifts the whole schedule right by ``delay`` positions, which
    carries positions ``[delay, 2 * delay)`` all the way to clean. Those are handed to the robot,
    the buffer slides forward, and ``delay`` fresh-noise slots are appended, leaving the schedule
    identical to the one this call started from.

    Args:
        denoise_fn: Maps ``(x_t, per_position_time)`` to a velocity of the same shape as ``x_t``.
        x_t: Buffer of shape ``(batch, horizon, action_dim)`` sitting at ``staircase_time(delay, ...)``.
        delay: Measured inference delay in control steps; must be at least 1 to emit anything.
        noise: Optional tail noise of shape ``(batch, delay, action_dim)``, for deterministic tests.

    Returns:
        ``(emitted, next_buffer)`` where ``emitted`` is ``(batch, delay, action_dim)`` of finished
        actions and ``next_buffer`` has the same shape as ``x_t``.
    """
    if x_t.ndim != 3:
        raise ValueError(f"Expected a (batch, horizon, action_dim) buffer, got {tuple(x_t.shape)}")
    horizon = x_t.shape[1]
    if delay < 1:
        raise ValueError(
            f"staircase_substep emits `delay` actions per call, so delay must be >= 1, got {delay}"
        )
    if 2 * delay > horizon:
        raise ValueError(f"delay ({delay}) must satisfy 2 * delay <= horizon ({horizon})")

    time = staircase_time(delay, horizon, device=x_t.device, dtype=x_t.dtype)
    # Shifting the schedule right by `delay` is what defines the per-position advance: each
    # position inherits the noise level of the one `delay` slots behind it, and the front of the
    # ramp lands on zero.
    shifted_index = (torch.arange(horizon, device=x_t.device) - delay).clamp(min=0)
    dt = time[shifted_index] - time

    v_t = denoise_fn(x_t, time.unsqueeze(0).expand(x_t.shape[0], horizon))
    x_t = x_t + dt[None, :, None] * v_t

    emitted = x_t[:, delay : 2 * delay]
    if noise is None:
        noise = sample_noise((x_t.shape[0], delay, x_t.shape[2]), x_t.device).to(dtype=x_t.dtype)
    next_buffer = torch.cat([x_t[:, delay:], noise], dim=1)
    return emitted, next_buffer


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
