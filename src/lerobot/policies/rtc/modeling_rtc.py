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

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.debug_handler import DebugHandler

logger = logging.getLogger(__name__)

# Optional import for gradient visualization
try:
    from torchviz import make_dot

    TORCHVIZ_AVAILABLE = True
except ImportError:
    TORCHVIZ_AVAILABLE = False
    logger.debug("torchviz not available - gradient visualization disabled")


def plot_waypoints(axs, chunk, start_from: int = 0, color: str | None = None, label: str | None = None):
    chunk = chunk[0].cpu().numpy()
    # Limit to 6 action dimensions to match number of subplots
    num_dims = min(chunk.shape[-1], 6)
    for j in range(num_dims):
        axs[j].plot(
            np.arange(start_from, start_from + chunk.shape[0]),
            chunk[:, j],
            color=color,
            label=label,
        )
        axs[j].set_ylabel("Joint angle", fontsize=14)
        axs[j].grid()
        plt.tick_params(labelsize=14)
        axs[j].legend(loc="upper right", fontsize=14)
        if j == 2:
            axs[j].set_xlabel("Step #", fontsize=16)


class RTCProcessor:
    """Real-Time Chunking processor for action chunking policies.

    This class implements RTC techniques including velocity calculation,
    prefix attention, and adaptive chunk processing.
    """

    def __init__(
        self,
        rtc_config: RTCConfig,
        verbose: bool = False,
        visualize_gradients: bool = False,
        viz_output_dir: str = ".",
    ):
        """Initialize RTC processor.

        Args:
            rtc_config (RTCConfig): Configuration holding RTC parameters used by
                the processor, including for example:
                - execution_horizon: number of timesteps used to build prefix weights
                - prefix_attention_schedule: strategy for prefix weights
                  (ZEROS, ONES, LINEAR, EXP)
                - max_guidance_weight: upper bound applied to the guidance scale
                - debug: whether to collect debug information
                - debug_maxlen: sliding window size for debug information
            verbose (bool): Enable verbose debug logging.
            visualize_gradients (bool): Enable gradient visualization using torchviz.
            viz_output_dir (str): Directory to save gradient visualizations.
        """
        self.rtc_config = rtc_config
        self.verbose = verbose
        self.visualize_gradients = visualize_gradients and TORCHVIZ_AVAILABLE
        self.viz_output_dir = viz_output_dir
        self._viz_counter = 0

        # Initialize tracker
        self.tracker = DebugHandler(
            enabled=rtc_config.debug,
            maxlen=rtc_config.debug_maxlen,
        )

        if visualize_gradients and not TORCHVIZ_AVAILABLE:
            logger.warning(
                "visualize_gradients=True but torchviz is not installed. "
                "Install it with: uv pip install torchviz graphviz"
            )

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

    def _visualize_correction_graph(self, correction, x_t, v_t, x1_t, err, time, weights, prev_chunk):
        """Visualize the computational graph for the correction term.

        Args:
            correction: The correction gradient tensor
            x_t: Current latent/state
            v_t: Velocity from denoiser
            x1_t: Denoised prediction (x_t - time * v_t)
            err: Weighted error term
            time: Time parameter
            weights: Prefix attention weights
            prev_chunk: Previous chunk leftover

        Saves two PNG files:
            1. rtc_correction_forward_graph_<counter>.png - Shows forward computation to err
            2. rtc_correction_gradient_graph_<counter>.png - Shows gradient computation
        """
        if not TORCHVIZ_AVAILABLE:
            return

        import os

        os.makedirs(self.viz_output_dir, exist_ok=True)

        # Visualize the forward graph leading to the error term
        try:
            dot_forward = make_dot(
                err.mean(),
                params={
                    "x_t": x_t,
                    "v_t": v_t,
                    "x1_t": x1_t,
                    "prev_chunk": prev_chunk,
                    "weights": weights,
                },
                show_attrs=True,
                show_saved=True,
            )
            dot_forward.format = "png"
            forward_path = os.path.join(
                self.viz_output_dir, f"rtc_correction_forward_graph_{self._viz_counter}"
            )
            dot_forward.render(forward_path, cleanup=True)
            logger.info(f"Forward graph saved to {forward_path}.png")
        except Exception as e:
            logger.warning(f"Failed to create forward graph: {e}")

        # Visualize the correction gradient itself
        try:
            dot_correction = make_dot(
                correction.mean(),
                params={
                    "x_t": x_t,
                    "correction": correction,
                },
                show_attrs=True,
                show_saved=True,
            )
            dot_correction.format = "png"
            correction_path = os.path.join(
                self.viz_output_dir, f"rtc_correction_gradient_graph_{self._viz_counter}"
            )
            dot_correction.render(correction_path, cleanup=True)
            logger.info(f"Correction gradient graph saved to {correction_path}.png")
        except Exception as e:
            logger.warning(f"Failed to create correction gradient graph: {e}")

        self._viz_counter += 1

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

        x_t = x_t.clone().detach()

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

        weights = (
            self.get_prefix_weights(inference_delay, execution_horizon, action_chunk_size)
            .to(x_t.device)
            .unsqueeze(0)
            .unsqueeze(-1)
        )

        with torch.enable_grad():
            v_t = original_denoise_step_partial(x_t)
            x_t.requires_grad_(True)

            x1_t = x_t - time * v_t  # noqa: N806
            err = (prev_chunk_left_over - x1_t) * weights
            grad_outputs = err.clone().detach()
            correction = torch.autograd.grad(x1_t, x_t, grad_outputs, retain_graph=False)[0]

        # Visualize correction gradient graph if enabled
        if self.visualize_gradients:
            self._visualize_correction_graph(
                correction=correction,
                x_t=x_t,
                v_t=v_t,
                x1_t=x1_t,
                err=err,
                time=time,
                weights=weights,
                prev_chunk=prev_chunk_left_over,
            )

        max_guidance_weight = torch.as_tensor(self.rtc_config.max_guidance_weight)
        squared_one_minus_tau = (1 - tau) ** 2
        inv_r2 = (squared_one_minus_tau + tau**2) / (squared_one_minus_tau)
        c = torch.nan_to_num((1 - tau) / tau, posinf=max_guidance_weight)
        guidance_weight = torch.nan_to_num(c * inv_r2, posinf=max_guidance_weight)
        guidance_weight = torch.minimum(guidance_weight, max_guidance_weight)

        result = v_t - guidance_weight * correction

        # Record debug information if enabled
        self.tracker.record_step(
            x_t=x_t,
            v_t=v_t,
            x1_t=x1_t,
            correction=correction,
            err=err,
            weights=weights,
            guidance_weight=guidance_weight,
            time=time,
            inference_delay=inference_delay,
            execution_horizon=execution_horizon,
            prev_chunk_shape=tuple(prev_chunk_left_over.shape) if prev_chunk_left_over is not None else None,
        )

        # Remove the batch dimension if it was added
        if squeezed:
            result = result.squeeze(0)
            correction = correction.squeeze(0)
            x1_t = x1_t.squeeze(0)
            err = err.squeeze(0)

        return result, correction, x1_t, err

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
