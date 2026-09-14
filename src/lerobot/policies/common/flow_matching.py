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
(pi0, pi05, smolvla, eo1) and the forward-convention policies (evo1, groot, wall_x)
historically each carried a copy of. All functions are stateless; adopting them does
not affect checkpoints.
"""

import enum
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from lerobot.policies.rtc.modeling_rtc import RTCProcessor


class FlowConvention(str, enum.Enum):
    """Which end of the ``[0, 1]`` schedule holds the noise.

    Both conventions describe the same family of probability paths, they only disagree on
    the direction of time (and therefore on the sign of the velocity field). A checkpoint is
    tied to the convention it was trained with, so this is a property of the policy, never a
    tunable.
    """

    NOISE_AT_ONE = "noise_at_one"
    """openpi (pi0, pi05, smolvla, eo1): ``x_1`` is noise, integrate ``t: 1 -> 0``.

    ``dt = -1/num_steps``, ``time = 1.0 + step * dt``, and the clean sample is
    ``x_0 = x_t - time * v_t``.
    """

    NOISE_AT_ZERO = "noise_at_zero"
    """groot / evo1 / wall_x: ``x_0`` is noise, integrate ``t: 0 -> 1``.

    ``dt = +1/num_steps``, ``time = step * dt``, and the clean sample is
    ``x_1 = x_t + (1 - time) * v_t``.
    """


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
    num_steps: int | None = None,
    *,
    convention: FlowConvention = FlowConvention.NOISE_AT_ONE,
    time_grid: Tensor | None = None,
    velocity_scale: Tensor | None = None,
    rtc_processor: "RTCProcessor | None" = None,
    rtc_enabled: bool = False,
    inference_delay: int | None = None,
    prev_chunk_left_over: Tensor | None = None,
    execution_horizon: int | None = None,
    hard_prefix: Tensor | None = None,
    hard_prefix_mask: Tensor | None = None,
) -> Tensor:
    """Forward-Euler integration of a flow-matching velocity field between noise and actions.

    The loop is ``x_t <- x_t + dt * v_t`` over a uniform schedule whose direction is set by
    ``convention`` (see :class:`FlowConvention`), with an optional real-time-chunking (RTC)
    guidance hook wrapping the velocity computation and debug tracking after each step.

    RTC conventions: :class:`~lerobot.policies.rtc.modeling_rtc.RTCProcessor` is written
    against ``NOISE_AT_ONE``. Under ``NOISE_AT_ZERO`` this function hands it ``1 - time`` and
    negates the velocity both entering and leaving the processor, so the guided step is
    correct in the caller's own convention. ``rtc_processor.track`` therefore always reports
    ``time`` and ``v_t`` in the ``NOISE_AT_ONE`` convention, whatever the caller's, which
    keeps the entries logged here consistent with those logged inside ``denoise_step``.

    Args:
        denoise_fn: Computes the velocity ``v_t`` from ``(x_t, time_tensor)`` where
            ``time_tensor`` is a float32 tensor of shape ``(batch_size,)``. The returned
            velocity must have the same shape and dtype as ``x_t``.
        noise: Initial sample of shape ``(batch_size, ...)``: ``x_1`` under ``NOISE_AT_ONE``,
            ``x_0`` under ``NOISE_AT_ZERO``.
        num_steps: Number of Euler steps. Required unless ``time_grid`` is given.
        convention: Which end of the schedule holds the noise.
        time_grid: Optional explicit schedule of ``num_steps + 1`` sample times, replacing the
            uniform one. Step ``k`` evaluates the velocity at ``time_grid[k]`` and advances by
            ``time_grid[k + 1] - time_grid[k]``, so non-uniform schedules are supported.
        velocity_scale: Optional per-element weighting of the update, applied as
            ``x_t + (dt * v_t) * velocity_scale``. Used to freeze or ramp part of the chunk.
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
    noise_at_one = convention is FlowConvention.NOISE_AT_ONE

    if time_grid is not None:
        if time_grid.ndim != 1 or time_grid.shape[0] < 2:
            raise ValueError(f"time_grid must be 1-D with at least 2 entries, got {tuple(time_grid.shape)}")
        if num_steps is not None and num_steps != time_grid.shape[0] - 1:
            raise ValueError(
                f"num_steps={num_steps} conflicts with a time_grid of {time_grid.shape[0]} entries"
            )
        num_steps = time_grid.shape[0] - 1
    elif num_steps is None:
        raise ValueError("euler_integrate requires either num_steps or time_grid")

    # Clean tokens sit at the far end of the schedule from the noise.
    clean_time = 0.0 if noise_at_one else 1.0
    uniform_dt = (-1.0 if noise_at_one else 1.0) / num_steps

    x_t = noise
    for step in range(num_steps):
        if time_grid is None:
            time = (1.0 + step * uniform_dt) if noise_at_one else step * uniform_dt
            dt = uniform_dt
        else:
            time = float(time_grid[step])
            dt = time_grid[step + 1] - time_grid[step]
        time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

        if hard_prefix is not None:
            if hard_prefix_mask is None:
                raise ValueError("hard_prefix_mask is required when hard_prefix is provided")
            x_t = torch.where(hard_prefix_mask, hard_prefix, x_t)
            time_tensor = time_tensor[:, None].expand(bsize, x_t.shape[1]).clone()
            time_tensor[hard_prefix_mask[..., 0]] = clean_time

        def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
            return denoise_fn(input_x_t, current_timestep)

        def flipped_denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
            return -denoise_fn(input_x_t, current_timestep)

        # Time and velocity as RTCProcessor expects them, i.e. in the NOISE_AT_ONE convention.
        rtc_time = time if noise_at_one else 1.0 - time

        if rtc_enabled:
            v_t = rtc_processor.denoise_step(
                x_t=x_t,
                prev_chunk_left_over=prev_chunk_left_over,
                inference_delay=inference_delay,
                time=rtc_time,
                original_denoise_step_partial=(
                    denoise_step_partial_call if noise_at_one else flipped_denoise_step_partial_call
                ),
                execution_horizon=execution_horizon,
            )
            if not noise_at_one:
                v_t = -v_t
        else:
            v_t = denoise_step_partial_call(x_t)

        update = dt * v_t
        if velocity_scale is not None:
            update = update * velocity_scale
        x_t = x_t + update

        if hard_prefix is not None:
            x_t = torch.where(hard_prefix_mask, hard_prefix, x_t)

        if rtc_processor is not None and rtc_processor.is_debug_enabled():
            rtc_processor.track(time=rtc_time, x_t=x_t, v_t=v_t if noise_at_one else -v_t)

    return x_t
