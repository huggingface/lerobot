# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Reusable adapter for policies with fixed-step iterative sampling.

Bakes an unrolled N-step denoising / flow-matching loop into a single ONNX
graph. Subclasses encode the per-step model call and per-step update rule,
plus pre-computed schedule constants registered as buffers.

Used by Diffusion DDIM (in this PR) and intended as the base class for
follow-up implementations of multi_task_dit, GR00T, and flow-matching VLAs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn


class IterativeDenoisingAdapter(nn.Module, ABC):
    """ABC for ONNX export of fixed-N-step iterative samplers.

    Subclass contract:

    1. Register any schedule tensors (alphas, betas, sigmas, ...) as buffers
       in ``__init__`` so they become ONNX constants.
    2. Implement ``_call_model(sample, step_idx, *cond) -> Tensor`` with the
       per-step model forward.
    3. Implement ``_step(sample, model_output, step_idx) -> Tensor`` with the
       pure-tensor update rule (no Python branching on shapes).

    The number of steps is fixed at construction time; the loop is unrolled
    by Python, so the resulting ONNX graph has exactly ``num_steps`` model
    calls baked in.
    """

    def __init__(self, num_steps: int) -> None:
        super().__init__()
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")
        self.num_steps = num_steps

    @abstractmethod
    def _call_model(self, sample: Tensor, step_idx: int, *cond: Tensor) -> Tensor:
        """One model forward at the given step. Implementations may consult
        buffers (e.g. timestep tensors) keyed by ``step_idx``."""

    @abstractmethod
    def _step(self, sample: Tensor, model_output: Tensor, step_idx: int) -> Tensor:
        """Scheduler update from ``sample`` and ``model_output`` at ``step_idx``."""

    def forward(self, noise: Tensor, *cond: Tensor) -> Tensor:
        """Run the unrolled denoising loop.

        Args:
            noise: Initial sample, e.g. Gaussian noise.
            *cond: Additional conditioning tensors forwarded to ``_call_model``.

        Returns:
            The denoised sample after ``num_steps`` iterations.
        """
        sample = noise
        for i in range(self.num_steps):
            model_output = self._call_model(sample, i, *cond)
            sample = self._step(sample, model_output, i)
        return sample
