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
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class AsyncRTCConfig:
    """Configuration for async RTC guidance.

    Attributes:
        enabled: Whether RTC guidance is enabled.
        prefix_attention_schedule: Schedule for prefix attention weights (zeros|ones|linear|exp).
        max_guidance_weight: Maximum guidance weight for clamping. If None, uses
            num_flow_matching_steps (Alex Soare optimization).
        sigma_d: Prior variance σ_d for the guidance weight formula. Lower values (e.g., 0.2)
            give stronger guidance and smoother transitions. Default 1.0 matches original RTC.
            Reference: https://alexander-soare.github.io/robotics/2025/08/05/smooth-as-butter-robot-policies.html
        full_trajectory_alignment: If True, skip gradient computation and use error directly
            as correction. Faster and smoother when distance between chunks is small.
    """

    enabled: bool = False
    prefix_attention_schedule: str = "linear"
    max_guidance_weight: float | None = None  # None = use num_flow_matching_steps (Alex Soare opt)
    sigma_d: float = 1.0  # Prior variance (0.2 = stronger guidance, 1.0 = original RTC)
    full_trajectory_alignment: bool = False  # Skip gradient for faster/smoother transitions


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
        overlap_end: int | None = None,
        num_flow_matching_steps: int | None = None,
        # Backwards compat: policies pass execution_horizon
        execution_horizon: int | None = None,
    ) -> Tensor:
        """RTC guidance wrapper around an existing denoiser.

        Args:
            x_t: Current noisy sample tensor.
            prev_chunk_left_over: Previous chunk for inpainting guidance.
            inference_delay: Latency in action steps (d).
            time: Current denoising timestep (1 = noise, 0 = clean).
            original_denoise_step_partial: Callable that computes base velocity given x_t.
            overlap_end: Where soft masking region ends (H - d). If None, computed from
                chunk size and inference_delay.
            num_flow_matching_steps: Number of flow matching steps. Used as max_guidance_weight
                when cfg.max_guidance_weight is None (Alex Soare optimization).
            execution_horizon: Deprecated alias for overlap_end (for policy compatibility).

        Returns:
            Guided velocity tensor.
        """
        # No guidance if disabled or missing prefix / delay.
        if not self.cfg.enabled or prev_chunk_left_over is None or inference_delay is None:
            return original_denoise_step_partial(x_t)

        # Backwards compat: use execution_horizon if overlap_end not provided
        if overlap_end is None and execution_horizon is not None:
            overlap_end = execution_horizon

        tau = 1 - time  # match existing RTC convention (inverted time)

        x_t_local = x_t.clone().detach()

        squeezed = False
        if x_t_local.ndim < 3:
            x_t_local = x_t_local.unsqueeze(0)
            squeezed = True

        prev = prev_chunk_left_over
        if prev.ndim < 3:
            prev = prev.unsqueeze(0)

        batch_size, chunk_t, chunk_a = x_t_local.shape
        prev_a = prev.shape[2]

        # Compute overlap_end if not provided: H - d (maximum soft masking)
        if overlap_end is None:
            overlap_end = chunk_t - inference_delay

        # Clamp overlap_end to available prefix length
        prefix_len = prev.shape[1]
        overlap_end = min(overlap_end, prefix_len)

        # With server-side zero-padding to max_action_dim, dimensions should now match.
        # Log at debug level if they still differ (shouldn't happen after the fix).
        if prev_a != chunk_a:
            import logging

            logging.getLogger(__name__).debug(
                "RTC dimension mismatch: prev_a=%d, chunk_a=%d",
                prev_a,
                chunk_a,
            )

        # Determine target action dimension: when postprocess is used, comparison happens
        # in executable action space (prev's dimension), not raw model space.
        target_a = prev_a if self._postprocess is not None else chunk_a

        # Pad prefix temporal dimension to match chunk_t, but keep action dimension as target_a.
        if prev.shape[1] < chunk_t:
            padded = torch.zeros(
                batch_size, chunk_t, target_a, device=x_t_local.device, dtype=x_t_local.dtype
            )
            padded[:, : prev.shape[1], :] = prev.to(device=x_t_local.device, dtype=x_t_local.dtype)
            prev = padded
        else:
            prev = prev[:, :chunk_t, :target_a].to(device=x_t_local.device, dtype=x_t_local.dtype)

        # Build weights: frozen [0, d), soft mask [d, overlap_end), fresh [overlap_end, H)
        weights_1d = self._get_prefix_weights(inference_delay, overlap_end, chunk_t).to(x_t_local.device)
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

            # Compute correction term
            # If full_trajectory_alignment is enabled, skip gradient and use error directly.
            # This is faster and smoother when distance between chunks is small.
            if self.cfg.full_trajectory_alignment:
                correction = err
            else:
                correction = torch.autograd.grad(x1_t_for_loss, x_t_local, err.detach(), retain_graph=False)[
                    0
                ]

        # Alex Soare optimization: Use num_flow_matching_steps as max_guidance_weight if not set.
        # Reference: https://alexander-soare.github.io/robotics/2025/08/05/smooth-as-butter-robot-policies.html
        # The number of flow matching steps can be used as β without hyperparameter tuning.
        max_gw = self.cfg.max_guidance_weight
        if max_gw is None:
            max_gw = float(num_flow_matching_steps) if num_flow_matching_steps is not None else 10.0
        max_guidance_weight = torch.as_tensor(max_gw, device=x_t_local.device)

        tau_tensor = torch.as_tensor(tau, device=x_t_local.device, dtype=x_t_local.dtype)
        squared_one_minus_tau = (1 - tau_tensor) ** 2

        # Alex Soare's formula with prior variance σ_d:
        # Original RTC: inv_r2 = ((1-τ)² + τ²) / (1-τ)²
        # With σ_d:     inv_r2 = ((1-τ)² + τ² * σ_d²) / ((1-τ)² * σ_d²)
        # Lower σ_d (e.g., 0.2) = narrower prior = stronger guidance = smoother transitions
        prior_variance = torch.as_tensor(self.cfg.sigma_d**2, device=x_t_local.device, dtype=x_t_local.dtype)
        inv_r2 = (squared_one_minus_tau + tau_tensor**2 * prior_variance) / (
            squared_one_minus_tau * prior_variance
        )

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
