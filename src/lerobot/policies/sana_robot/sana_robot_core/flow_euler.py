# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

# Vendored from the SANA-Robot code release (https://github.com/NVlabs/Sana/tree/robot) at commit 1e7d6cc51,
# original path sana_robot/deploy/sampling/flow_euler.py. Import paths are rewritten to the LeRobot package;
# the real-time chunking guidance of SANA-Robot-base is not vendored.
# Keep this file in sync with the upstream file.

from __future__ import annotations

import functools

import torch
from diffusers import FlowMatchEulerDiscreteScheduler

TRAIN_TIMESTEPS = 1000


def session_verb(method):
    """Run a session method in inference mode, under bf16 autocast when the session's policy has bf16 weights."""

    @functools.wraps(method)
    def run(self, *arguments, **keyword_arguments):
        policy = self.policy
        device_type = next(policy.parameters()).device.type
        with (
            torch.inference_mode(),
            torch.autocast(device_type, dtype=torch.bfloat16, enabled=policy.dtype == torch.bfloat16),
        ):
            return method(self, *arguments, **keyword_arguments)

    return run


def sample_dtype(policy) -> torch.dtype:
    """Return the dtype the sampler keeps its samples in: float32, or the policy's when that is wider."""
    return torch.promote_types(policy.dtype, torch.float32)


def flow_schedule(steps: int, shift: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the timesteps [steps] of a Flow-Euler sampler with the given shift and its noise levels [steps + 1], the last one 0."""
    scheduler = FlowMatchEulerDiscreteScheduler(shift=float(shift))
    scheduler.set_timesteps(int(steps), device=device)
    return (scheduler.timesteps, scheduler.sigmas)


def euler_step(
    sample: torch.Tensor, flow: torch.Tensor, timestep: torch.Tensor, next_level: torch.Tensor
) -> torch.Tensor:
    """Move a sample from the noise level of ``timestep`` to ``next_level`` along the predicted flow."""
    dtype = torch.promote_types(sample.dtype, torch.float32)
    level = timestep / TRAIN_TIMESTEPS
    return (sample.to(dtype) + (next_level - level) * flow.to(dtype)).to(sample.dtype)


def draw_noise(shape, generator, device, dtype) -> torch.Tensor:
    """Draw Gaussian noise on the device."""
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)


def host_geometry(data_info: dict) -> dict:
    """Return a copy of ``data_info`` with the view geometry on the CPU and the view count and the frame stride as ints: the policy reads them on the host at every forward."""
    info = dict(data_info)
    for name in ("view_latent_shape", "view_slot_ids"):
        if isinstance(info.get(name), torch.Tensor):
            info[name] = info[name].detach().cpu()
    for name in ("view_count", "video_frame_stride"):
        if isinstance(info.get(name), torch.Tensor):
            info[name] = int(info[name].reshape(-1)[0])
    return info
