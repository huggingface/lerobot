#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
Real-Time Chunking (RTC) implementation for LeRobot.

Based on Physical Intelligence's Kinetix implementation:
https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/main/src/model.py#L214
"""

import logging
import math

import torch
from torch import Tensor

from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.rtc.configuration_rtc import RTCConfig

logger = logging.getLogger(__name__)


class RTCProcessor:
    """Real-Time Chunking processor for action chunking policies.

    This class implements RTC techniques including velocity calculation,
    prefix attention, and adaptive chunk processing.
    """

    def __init__(
        self,
        rtc_config: RTCConfig,
        verbose: bool = False,
    ):
        """Initialize RTC processor.

        Args:
            rtc_config (RTCConfig): Configuration holding RTC parameters used by
                the processor, including for example:
                - execution_horizon: number of timesteps used to build prefix weights
                - prefix_attention_schedule: strategy for prefix weights
                  (ZEROS, ONES, LINEAR, EXP)
                - max_guidance_weight: upper bound applied to the guidance scale
            verbose (bool): Enable verbose debug logging.
        """
        self.rtc_config = rtc_config
        self.verbose = verbose

    @staticmethod
    def _tensor_stats(tensor: Tensor, name: str = "tensor") -> str:
        """Generate readable statistics string for a tensor.

        Args:
            tensor: Input tensor
            name: Name to display

        Returns:
            Formatted string with shape and statistics
        """
        if tensor is None:
            return f"{name}: None"

        stats = (
            f"{name}: shape={tuple(tensor.shape)}, "
            f"dtype={tensor.dtype}, "
            f"device={tensor.device}, "
            f"min={tensor.min().item():.4f}, "
            f"max={tensor.max().item():.4f}, "
            f"mean={tensor.mean().item():.4f}, "
            f"std={tensor.std().item():.4f}"
        )
        return stats

    def denoise_step(
        self,
        x_t,
        prev_chunk_left_over,
        inference_delay,
        time,
        original_denoise_step_partial,
        execution_horizon=None,
    ) -> Tensor:
        """RTC guidance wrapper around an existing denoiser.

        This method wraps an original denoising callable that only takes ``x_t`` and
        returns a base denoised velocity ``v_t``. It then applies Real-Time Chunking
        (RTC) prefix guidance using the leftover prefix from the previous chunk.

        Args:
            x_t (Tensor): Current latent/state to denoise. Shape ``(B, T, A)`` or ``(T, A)``.
            prev_chunk_left_over (Tensor | None): Unexecuted prefix from the previous
                chunk. Shape ``(B, T_prev, A)`` or ``(T_prev, A)``. If ``None``, no guidance
                is applied and the method returns ``v_t`` from the original denoiser.
            inference_delay (int): Number of timesteps from the prefix to use for guidance.
            time (float | Tensor): Scalar in [0, 1] indicating normalized time. Must be
                broadcastable with ``x_t``.
            original_denoise_step_partial (Callable[[Tensor], Tensor]): Callable that
                computes the base denoised velocity given only ``x_t``.
            execution_horizon (int | None): Horizon used to build prefix weights. If
                ``None``, defaults to ``self.rtc_config.execution_horizon``.

        Returns:
            Tensor: Guided velocity with the same shape as ``v_t``.

        Notes:
            - If inputs are 2D, a batch dimension is temporarily added and removed at the end.
            - If ``prev_chunk_left_over`` is shorter than the current chunk length ``T``, it is
              right-padded with zeros to match ``T``.
            - Prefix weights are constructed via ``get_prefix_weights(inference_delay, execution_horizon, T)``
              and broadcast to ``(B, T, A)``.
            - Guidance correction is computed via autograd using ``x1_t = x_t + time * v_t`` and
              ``error = (prev_chunk_left_over - x1_t) * weights``.
            - The final guidance weight is clamped by ``max_guidance_weight`` from the config.

        Reference:
            https://www.physicalintelligence.company/download/real_time_chunking.pdf
        """

        # In the original implementation, the time goes from 0 to 1 and
        # In our implementation, the time goes from 1 to 0
        # So we need to invert the time
        tau = 1 - time

        x_t = x_t.clone().detach().requires_grad_(True)

        if self.verbose:
            logger.info("=" * 80)
            logger.info(
                f"RTC denoise_step: time={time:.4f} (tau={tau:.4f}), inference_delay={inference_delay}"
            )
            logger.info(self._tensor_stats(x_t, "x_t"))

        if prev_chunk_left_over is None:
            if self.verbose:
                logger.info("No prev_chunk_left_over - skipping guidance (first step)")
            # First step, no guidance
            return original_denoise_step_partial(x_t)

        squeezed = False
        if len(x_t.shape) < 3:
            # Add batch dimension
            x_t = x_t.unsqueeze(0)
            squeezed = True

        if len(prev_chunk_left_over.shape) < 3:
            # Add batch dimension
            prev_chunk_left_over = prev_chunk_left_over.unsqueeze(0)

        if execution_horizon is None:
            execution_horizon = self.rtc_config.execution_horizon

        # If the previous action chunk is to short then it doesn't make sense to use long execution horizon
        # because there is nothing to merge
        original_execution_horizon = execution_horizon
        if execution_horizon > prev_chunk_left_over.shape[1]:
            execution_horizon = prev_chunk_left_over.shape[1]
            if self.verbose and execution_horizon != original_execution_horizon:
                logger.info(
                    f"Adjusted execution_horizon: {original_execution_horizon} -> {execution_horizon} "
                    f"(limited by prev_chunk size)"
                )

        batch_size = x_t.shape[0]
        action_chunk_size = x_t.shape[1]
        action_dim = x_t.shape[2]

        if self.verbose:
            logger.info(
                f"Batch size: {batch_size}, Chunk size: {action_chunk_size}, Action dim: {action_dim}"
            )
            logger.info(self._tensor_stats(prev_chunk_left_over, "prev_chunk_left_over (before padding)"))

        if prev_chunk_left_over.shape[1] < action_chunk_size or prev_chunk_left_over.shape[2] < action_dim:
            # We need to pad the left over chunk with zeros
            if self.verbose:
                logger.info(
                    f"Padding prev_chunk_left_over from {tuple(prev_chunk_left_over.shape)} "
                    f"to ({batch_size}, {action_chunk_size}, {action_dim})"
                )
            padded = torch.zeros(batch_size, action_chunk_size, action_dim).to(x_t.device)
            padded[:, : prev_chunk_left_over.shape[1], : prev_chunk_left_over.shape[2]] = prev_chunk_left_over
            prev_chunk_left_over = padded

        assert prev_chunk_left_over.shape == x_t.shape, (
            "The padded previous chunk must be the same size as the input tensor"
        )
        weights = self.get_prefix_weights(inference_delay, execution_horizon, action_chunk_size).to(
            x_t.device
        )

        if self.verbose:
            logger.info(
                f"Prefix weights: schedule={self.rtc_config.prefix_attention_schedule}, "
                f"start={inference_delay}, end={execution_horizon}, total={action_chunk_size}"
            )
            logger.info(self._tensor_stats(weights, "weights (1D)"))

        # Reshape weights to match the tensor dimensions (batch, time, action_dim)
        # weights is shape (action_chunk_size,) and needs to be (1, action_chunk_size, 1)
        weights = weights.unsqueeze(0).unsqueeze(-1)  # Add batch and action dimensions

        with torch.enable_grad():
            v_t = original_denoise_step_partial(x_t)

            if self.verbose:
                logger.info(self._tensor_stats(v_t, "v_t (original velocity)"))

            # In the original implementation, the time goes from 0 to 1 and x_1t calculates
            # as velocity * (1 - time). https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/main/src/model.py#L234
            # Our integration runs from time=1 -> 0, so we still want the step magnitude
            # to scale with (1 - time) to avoid overly large corrections at the start.
            x1_t = x_t - tau * v_t

            error = (prev_chunk_left_over - x1_t) * weights

            print(
                "Error calculation: prev_chunk_left_over: ",
                prev_chunk_left_over[:, :1, :6],
                " x1_t: ",
                x1_t[:, :1, :6],
            )

            if self.verbose:
                logger.info(self._tensor_stats(x1_t, "x1_t (predicted next state)"))
                logger.info(self._tensor_stats(prev_chunk_left_over, "prev_chunk_left_over (target)"))
                logger.info(self._tensor_stats(error, "error (weighted difference)"))

            correction = torch.autograd.grad(x1_t, x_t, error, retain_graph=True)[0]

        # print("error: ", error[0, :3, :6], weights)

        max_guidance_weight = torch.as_tensor(self.rtc_config.max_guidance_weight)

        squared_one_minus_tau = (1 - tau) ** 2
        inv_r2 = (squared_one_minus_tau + tau**2) / (squared_one_minus_tau)
        c = torch.nan_to_num((1 - tau) / tau, posinf=max_guidance_weight)
        guidance_weight = torch.nan_to_num(c * inv_r2, posinf=max_guidance_weight)
        guidance_weight = torch.minimum(guidance_weight, max_guidance_weight)

        if self.verbose:
            logger.info(
                f"Guidance calculation: squared_one_minus_tau={squared_one_minus_tau:.4f}, "
                f"inv_r2={inv_r2:.4f}, c={c:.4f}"
            )
            logger.info(f"guidance_weight: {guidance_weight:.4f} (max={max_guidance_weight:.4f})")
            logger.info(self._tensor_stats(correction, "correction"))

        result = v_t - guidance_weight * correction

        # Remove the batch dimension if it was added
        if squeezed:
            result = result.squeeze(0)

        return result

    def get_prefix_weights(self, start, end, total):
        start = min(start, end)

        if self.rtc_config.prefix_attention_schedule == RTCAttentionSchedule.ZEROS:
            weights = torch.zeros(total)
            weights[:start] = 1.0
        elif self.rtc_config.prefix_attention_schedule == RTCAttentionSchedule.ONES:
            weights = torch.ones(total)
            weights[end:] = 0.0
        elif self.rtc_config.prefix_attention_schedule == RTCAttentionSchedule.LINEAR:
            lin_weights = self._linweights(start, end, total)
            weights = self._add_trailing_zeros(lin_weights, total, end)
            weights = self._add_leading_ones(weights, start, total)
        elif self.rtc_config.prefix_attention_schedule == RTCAttentionSchedule.EXP:
            lin_weights = self._linweights(start, end, total)
            lin_weights = lin_weights * torch.expm1(lin_weights).div(math.e - 1)
            weights = self._add_trailing_zeros(lin_weights, total, end)
            weights = self._add_leading_ones(weights, start, total)

        return weights

    def _linweights(self, start, end, total):
        skip_steps_at_end = max(total - end, 0)

        linspace_steps = total - skip_steps_at_end - start

        if end <= start or linspace_steps <= 0:
            return torch.tensor([])

        return torch.linspace(1, 0, linspace_steps + 2)[1:-1]

    def _add_trailing_zeros(self, weights, total, end):
        zeros_len = total - end

        if zeros_len <= 0:
            return weights

        zeros = torch.zeros(zeros_len)
        return torch.cat([weights, zeros])

    def _add_leading_ones(self, weights, start, total):
        ones_len = min(start, total)

        if ones_len <= 0:
            return weights

        ones = torch.ones(ones_len)
        return torch.cat([ones, weights])
