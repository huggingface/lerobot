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

"""Server-scoped Real-Time Chunking (RTC) guidance for async inference.

This intentionally does NOT depend on `lerobot.policies.rtc.*`.

It provides a minimal interface compatible with the flow-policy sampling code paths
that expect `rtc_processor.denoise_step(...)`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor


@dataclass(frozen=True)
class AsyncRTCConfig:
    enabled: bool = False
    prefix_attention_schedule: str = "linear"
    max_guidance_weight: float = 10.0
    execution_horizon: int = 10


class AsyncRTCProcessor:
    """RTC-style prefix guidance wrapper around an existing denoiser.

    The call signature matches what PI0/PI05/SmolVLA flow models use in their sampling loop.
    """

    def __init__(self, cfg: AsyncRTCConfig, *, postprocess: Callable[[Tensor], Tensor] | None = None):
        self.cfg = cfg
        self._postprocess = postprocess

    def is_debug_enabled(self) -> bool:
        return False

    def track(self, **_kwargs) -> None:
        return

    def denoise_step(
        self,
        x_t: Tensor,
        prev_chunk_left_over: Tensor | None,
        inference_delay: int | None,
        time: float | Tensor,
        original_denoise_step_partial: Callable[[Tensor], Tensor],
        execution_horizon: int | None = None,
    ) -> Tensor:
        # No guidance if disabled or missing prefix / delay.
        if not self.cfg.enabled or prev_chunk_left_over is None or inference_delay is None:
            return original_denoise_step_partial(x_t)

        tau = 1 - time  # match existing RTC convention (inverted time)

        x_t_local = x_t.clone().detach()

        squeezed = False
        if x_t_local.ndim < 3:
            x_t_local = x_t_local.unsqueeze(0)
            squeezed = True

        prev = prev_chunk_left_over
        if prev.ndim < 3:
            prev = prev.unsqueeze(0)

        if execution_horizon is None:
            execution_horizon = self.cfg.execution_horizon

        # If the previous chunk is shorter than the configured horizon, clamp.
        if execution_horizon > prev.shape[1]:
            execution_horizon = prev.shape[1]

        batch_size, chunk_t, chunk_a = x_t_local.shape

        # Pad prefix to (B, T, A) for broadcasting and loss computation.
        if prev.shape[1] < chunk_t or prev.shape[2] < chunk_a:
            padded = torch.zeros(batch_size, chunk_t, chunk_a, device=x_t_local.device, dtype=x_t_local.dtype)
            padded[:, : prev.shape[1], : prev.shape[2]] = prev.to(device=x_t_local.device, dtype=x_t_local.dtype)
            prev = padded
        else:
            prev = prev[:, :chunk_t, :chunk_a].to(device=x_t_local.device, dtype=x_t_local.dtype)

        weights_1d = self._get_prefix_weights(inference_delay, execution_horizon, chunk_t).to(x_t_local.device)
        weights = weights_1d.unsqueeze(0).unsqueeze(-1)  # (1, T, 1)

        # We need gradients for the correction term (and optional postprocess), but we do NOT want
        # to build a backward graph through the denoiser/model parameters.
        with torch.enable_grad():
            with torch.no_grad():
                v_t = original_denoise_step_partial(x_t_local)

            x_t_local.requires_grad_(True)

            # Match policy-side convention: x1_t = x_t - time * v_t
            time_tensor = torch.as_tensor(time, device=x_t_local.device, dtype=x_t_local.dtype)
            x1_t = x_t_local - time_tensor * v_t.detach()

            # If we're guiding in executable-action space, apply the (differentiable) postprocessor here.
            # This is used when the client only provides frozen actions in executable space.
            x1_t_for_loss = x1_t
            if self._postprocess is not None:
                x1_t_for_loss = self._postprocess(x1_t_for_loss)

            err = (prev - x1_t_for_loss) * weights
            correction = torch.autograd.grad(x1_t_for_loss, x_t_local, err.detach(), retain_graph=False)[0]

        max_guidance_weight = torch.as_tensor(self.cfg.max_guidance_weight, device=x_t_local.device)
        tau_tensor = torch.as_tensor(tau, device=x_t_local.device, dtype=x_t_local.dtype)
        squared_one_minus_tau = (1 - tau_tensor) ** 2
        inv_r2 = (squared_one_minus_tau + tau_tensor**2) / (squared_one_minus_tau)
        c = torch.nan_to_num((1 - tau_tensor) / tau_tensor, posinf=max_guidance_weight)
        guidance_weight = torch.nan_to_num(c * inv_r2, posinf=max_guidance_weight)
        guidance_weight = torch.minimum(guidance_weight, max_guidance_weight)

        result = v_t - guidance_weight * correction
        if squeezed:
            result = result.squeeze(0)
        return result

    def _get_prefix_weights(self, start: int, end: int, total: int) -> Tensor:
        start = int(start)
        end = int(end)
        total = int(total)

        start = min(start, end)
        schedule = (self.cfg.prefix_attention_schedule or "linear").lower()

        if schedule == "zeros":
            weights = torch.zeros(total)
            weights[: min(start, total)] = 1.0
            return weights
        if schedule == "ones":
            weights = torch.ones(total)
            weights[max(end, 0) :] = 0.0
            return weights

        lin = self._linweights(start, end, total)
        if schedule == "exp":
            lin = lin * torch.expm1(lin).div(math.e - 1)

        weights = self._add_trailing_zeros(lin, total, end)
        weights = self._add_leading_ones(weights, start, total)
        return weights

    @staticmethod
    def _linweights(start: int, end: int, total: int) -> Tensor:
        skip_steps_at_end = max(total - end, 0)
        linspace_steps = total - skip_steps_at_end - start
        if end <= start or linspace_steps <= 0:
            return torch.tensor([])
        return torch.linspace(1, 0, linspace_steps + 2)[1:-1]

    @staticmethod
    def _add_trailing_zeros(weights: Tensor, total: int, end: int) -> Tensor:
        zeros_len = total - end
        if zeros_len <= 0:
            return weights
        return torch.cat([weights, torch.zeros(zeros_len)])

    @staticmethod
    def _add_leading_ones(weights: Tensor, start: int, total: int) -> Tensor:
        ones_len = min(start, total)
        if ones_len <= 0:
            return weights
        return torch.cat([torch.ones(ones_len), weights])


