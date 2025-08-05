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

import math

import torch
from torch import Tensor

from lerobot.configs.types import RTCAttentionSchedule
from lerobot.policies.rtc_config import RTCConfig


class RTCProcessor:
    """Real-Time Chunking processor for action chunking policies.

    This class implements RTC techniques including velocity calculation,
    prefix attention, and adaptive chunk processing.
    """

    def __init__(
        self,
        rtc_config: RTCConfig,
    ):
        """Initialize RTC processor.

        Args:
            chunk_size: Size of action chunks
            soft_mask_length: Number of actions to soft mask in overlap regions
            beta: Maximum guidance weight for prefix attention
            prefix_attention_schedule: Schedule for prefix attention weights ("linear", "exp", "constant")
            device: PyTorch device for computations
        """
        self.rtc_config = rtc_config

        # Cache for previous chunk
        self.previous_chunk: Tensor | None = None
        self.chunk_step = 0

    def denoise_step(self, noise, prev_chunk, latency_delay, dt, v_t) -> Tensor:
        """Denoise the noise to get the next action.
        Real-time chunking (RTC) denoising.

        Reference:
        https://www.physicalintelligence.company/download/real_time_chunking.pdf
        """
        if prev_chunk is None:
            # First step, no guidance
            return v_t

        x1_t = noise + dt * v_t

        # Prepare the previous chunk for guidance.
        # Rotate the second (time) dimension left by `t` steps and pad the right with zeros.
        # Keep the unexecuted part, pad the remainder with zeros.
        # pad = torch.zeros_like(self.prev_chunk[:, :rtc_t])
        # A_prev = torch.cat([self.prev_chunk[:, rtc_t:], pad], dim=1)  # noqa: N806

        # weights = self.get_prefix_weights(
        #     inference_delay, prefix_attention_horizon, self.action_chunk_size, prefix_attention_schedule
        # )

        with torch.enable_grad():
            error = (x1_t - noise).clone().detach()
            grad_output = torch.autograd.grad(x1_t, noise, error, retain_graph=True)[0]

        return v_t + self.rtc_config.max_guidance_weight * grad_output

        # H = self.config.chunk_size  # noqa: N806

        # total_start = time.perf_counter()
        # denoise_time = 0
        # grad_time = 0

        # # s is the number of steps in the end to not blend with the previous chunk
        # if rtc_soft_mask_length == -1:
        #     rtc_s = rtc_t
        # else:
        #     rtc_s = max(0, H - rtc_d - rtc_soft_mask_length)

        # # Prepare the guidance mask
        # W = make_soft_mask(rtc_d, rtc_s, H, device)  # noqa: N806
        # # broadcast to (B,H,M)
        # W_row = W[None, :, None]  # noqa: N806

        # # A^0  ~ 𝒩(0,I)
        # A_tau = noise  # noqa: N806
        # t = torch.tensor(1.0, device=device)

        # while t >= -dt / 2:
        #     tau = 1 - t  # tau goes from 0 to 1, to be consistent with the paper
        #     # ΠGDM guidance
        #     denoise_start = time.perf_counter()
        #     v_pi = -self.denoise_step(prefix_pad_masks, past_key_values, A_tau, t.expand(bsize))
        #     denoise_time += time.perf_counter() - denoise_start

        #     grad_start = time.perf_counter()
        #     A_tau.requires_grad_(True)
        #     with torch.enable_grad():
        #         # Â¹_tau   Eq. 3
        #         A_hat = A_tau + (1 - tau) * v_pi  # noqa: N806
        #         err = (A_prev - A_hat) * W_row
        #         grad_outputs = err.clone().detach()
        #         g = torch.autograd.grad(A_hat, A_tau, grad_outputs, retain_graph=True)[0]
        #     grad_time += time.perf_counter() - grad_start

        #     r_sq = (1 - tau) ** 2 / (tau**2 + (1 - tau) ** 2)  # Eq. 4
        #     scale = min(self.config.inference_rtc_beta, (1 - tau) / (tau * r_sq))  # Eq.2
        #     # integration step  Eq. 1
        #     A_tau = A_tau - dt * (v_pi + scale * g)  # noqa: N806
        #     # stop grads before next step
        #     A_tau = A_tau.detach()  # noqa: N806

        #     if self.config.inference_rtc_debug:
        #         # For debugging. This makes the code slower
        #         A_tau_d_err = (A_prev[:, :rtc_d] - A_tau[:, :rtc_d]).norm()  # noqa: N806
        #         print(
        #             f"[RTC Debug] t={t.item():.2f} tau={tau.item():.2f} err[:,:rtc_d].norm()={err[:, :rtc_d].norm().item():.2f} A_tau_d_err={A_tau_d_err.item():.2f} scale={scale:.2f} g.norm()={g.norm().item():.2f}"
        #         )

        #     t += dt

        # # sanity check: the first d steps of A_prev should be the similar to the first d steps of A_tau because of masking
        # A_tau_d_err = (A_prev[:, :rtc_d] - A_tau[:, :rtc_d]).norm()  # noqa: N806
        # if A_tau_d_err > 0.5:
        #     print(
        #         f"WARNING: [RTC] The first {rtc_d=} steps of the new chunk are too different from the previous chunk. This may result in jerky motion. {A_tau_d_err=}"
        #     )

        # total_time = time.perf_counter() - total_start
        # if self.config.inference_rtc_debug:
        #     print(
        #         f"[RTC Debug] Denoising total time: {total_time:.2f}s | Denoise: {denoise_time:.2f}s | Grad: {grad_time:.2f}s | {rtc_t=} {rtc_d=} {rtc_s=}"
        #     )

        # self.prev_chunk = A_tau
        # return A_tau

    def get_prefix_weights(self, start, end, total):
        start = min(start, end)

        if self.rtc_config.prefix_attention_schedule == RTCAttentionSchedule.ZEROS:
            weights = torch.zeros(total)
            weights[:start] = 1.0
        elif self.rtc_config.prefix_attention_schedule == RTCAttentionSchedule.ONES:
            weights = torch.ones(total)
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
