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
policies (pi0, pi05, smolvla, eo1) and the forward-convention policies (evo1, groot,
wall_x) historically each carried a copy of.

The samplers deliberately return *untransformed* draws: every adopter's distribution,
dtype and order of casting versus transformation is decided at its own call site, because
those choices are baked into released checkpoints. All functions are stateless; adopting
them does not affect checkpoints.
"""

import enum
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor

if TYPE_CHECKING:
    from lerobot.policies.rtc.modeling_rtc import RTCProcessor

BetaSampler = Callable[[float, float, int], Tensor]
"""Draws ``bsize`` raw ``Beta(alpha, beta)`` samples as ``(alpha, beta, bsize) -> (bsize,)``."""


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


def _beta_distribution(alpha: float, beta: float) -> "torch.distributions.Beta":
    """``Beta(alpha, beta)`` whose concentrations are pinned to CPU.

    Beta sampling goes through ``_sample_dirichlet``, which is unimplemented on MPS and meta,
    so the draw happens on CPU and the result is moved afterwards. The concentrations are
    built with an explicit ``device="cpu"`` rather than the ambient default device, so a
    call inside a ``torch.device(...)`` context still samples on CPU.
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
        sampler: Optional hook replacing the CPU distribution, for callers that must
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
    denoise_fn: Callable[..., Tensor],
    noise: Tensor,
    num_steps: int | None = None,
    *,
    convention: FlowConvention = FlowConvention.NOISE_AT_ONE,
    step_aware: bool = False,
    time_grid: Tensor | None = None,
    velocity_scale: Tensor | None = None,
    rtc_processor: "RTCProcessor | None" = None,
    rtc_enabled: bool = False,
    inference_delay: int | None = None,
    prev_chunk_left_over: Tensor | None = None,
    execution_horizon: int | None = None,
    hard_prefix: Tensor | None = None,
    hard_prefix_mask: Tensor | None = None,
    precompute_times: bool = False,
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
        step_aware: Call ``denoise_fn`` as ``denoise_fn(x_t, time_tensor, step)`` with the
            integer index of the current Euler step. Policies whose network input is a discrete
            function of that index (groot's timestep buckets, evo1's positional-encoding lookup)
            take it from here; recovering it from ``time_tensor`` would both round-trip through
            the host once per step and risk landing in a different bucket.
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
        precompute_times: Create the float32 timestep schedule on the target device once,
            avoiding a scalar tensor allocation/host transfer in each denoising step.
            Uses the same Python scalar arithmetic as the historical loop.
    """
    bsize = noise.shape[0]
    device = noise.device
    if not isinstance(convention, FlowConvention):
        raise TypeError(f"convention must be a FlowConvention, got {convention!r}")
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

    def velocity(x_t: Tensor, time_tensor: Tensor, step: int) -> Tensor:
        return denoise_fn(x_t, time_tensor, step) if step_aware else denoise_fn(x_t, time_tensor)

    guidance = rtc_processor if rtc_enabled else None
    if rtc_enabled and guidance is None:
        raise ValueError("rtc_processor is required when rtc_enabled is True")

    def uniform_time(step: int) -> float:
        return (1.0 + step * uniform_dt) if noise_at_one else step * uniform_dt

    times = (
        torch.tensor([uniform_time(step) for step in range(num_steps)], dtype=torch.float32, device=device)
        if precompute_times and time_grid is None
        else None
    )
    x_t = noise
    for step in range(num_steps):
        if time_grid is None:
            dt = uniform_dt
            time_tensor = (
                times[step].expand(bsize)
                if times is not None
                else torch.full((bsize,), uniform_time(step), dtype=torch.float32, device=device)
            )
        else:
            # Slice the grid rather than reading it back: `float(time_grid[step])` would
            # synchronise with the accelerator once per step.
            dt = time_grid[step + 1] - time_grid[step]
            time_tensor = time_grid[step].to(dtype=torch.float32, device=device).expand(bsize)

        if hard_prefix is not None:
            if hard_prefix_mask is None:
                raise ValueError("hard_prefix_mask is required when hard_prefix is provided")
            x_t = torch.where(hard_prefix_mask, hard_prefix, x_t)
            time_tensor = time_tensor[:, None].expand(bsize, x_t.shape[1]).clone()
            time_tensor[hard_prefix_mask[..., 0]] = clean_time

        def denoise_step_partial_call(
            input_x_t: Tensor, current_timestep: Tensor = time_tensor, current_step: int = step
        ) -> Tensor:
            return velocity(input_x_t, current_timestep, current_step)

        def flipped_denoise_step_partial_call(
            input_x_t: Tensor, current_timestep: Tensor = time_tensor, current_step: int = step
        ) -> Tensor:
            return -velocity(input_x_t, current_timestep, current_step)

        needs_rtc_time = rtc_enabled or (rtc_processor is not None and rtc_processor.is_debug_enabled())
        if needs_rtc_time:
            # RTCProcessor takes a plain float in the NOISE_AT_ONE convention. On the explicit-grid
            # path this is the one place the schedule has to come back to the host.
            host_time = uniform_time(step) if time_grid is None else float(time_grid[step])
            rtc_time = host_time if noise_at_one else 1.0 - host_time

        if guidance is not None:
            v_t = guidance.denoise_step(
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
