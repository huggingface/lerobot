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
    """
    bsize = noise.shape[0]
    device = noise.device

    dt = -1.0 / num_steps
    x_t = noise
    for step in range(num_steps):
        time = 1.0 + step * dt
        time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

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

        if rtc_processor is not None and rtc_processor.is_debug_enabled():
            rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

    return x_t
