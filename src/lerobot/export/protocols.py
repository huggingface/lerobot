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
"""Export protocols for policy classes.

This module defines protocols that policies can implement to provide
clean, self-contained export logic. Instead of hardcoded wrappers in
the exporter, each policy provides its own export modules.

Protocol hierarchy:
- ExportableSinglePhase: For single-pass policies (ACT, VQ-BeT) that output actions directly
- ExportableIterative: For iterative policies (Diffusion) with denoise step pattern
- ExportableTwoPhase: For VLA policies (PI0, SmolVLA) with encode + denoise pattern

Each protocol enables policies to encapsulate their own export logic, making
the export system extensible without modifying the exporter itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from torch import Tensor, nn


@dataclass
class SinglePhaseExportConfig:
    """Configuration for single-phase (single-pass) export.

    Used by policies like ACT and VQ-BeT that produce actions in one forward pass.
    """

    chunk_size: int
    action_dim: int
    n_action_steps: int | None = None


@dataclass
class IterativeExportConfig:
    """Configuration for iterative (denoising) export.

    Used by policies like Diffusion that iteratively refine actions.
    """

    horizon: int
    action_dim: int
    num_inference_steps: int
    scheduler_type: str = "ddpm"


@dataclass
class TwoPhaseExportConfig:
    """Configuration for two-phase (VLA) export.

    Captures architecture-specific information needed to export a two-phase
    policy and reconstruct the KV cache at runtime.
    """

    num_layers: int
    num_kv_heads: int
    head_dim: int

    chunk_size: int
    action_dim: int
    state_dim: int
    num_steps: int

    state_in_denoise: bool = True

    input_mapping: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class ExportableTwoPhase(Protocol):
    """Protocol for two-phase policies (VLAs like PI0, SmolVLA).

    Two-phase policies use:
    1. Encoder phase: Process images/language/state → KV cache
    2. Denoise phase: Iteratively denoise actions using cached KV

    Implement this protocol to enable clean ONNX export without
    external wrapper classes.
    """

    def get_two_phase_export_config(self) -> TwoPhaseExportConfig:
        """Return export configuration with architecture details."""
        ...

    def get_encoder_module(self, num_images: int = 1) -> nn.Module:
        """Return an nn.Module for the encoder phase.

        The module should:
        - Accept flattened inputs: (image_0, ..., img_mask_0, ..., lang_tokens, lang_masks, state)
        - Output: (prefix_pad_mask, past_key_0, past_value_0, ...)

        Args:
            num_images: Number of image inputs to expect.

        Returns:
            nn.Module ready for ONNX export.
        """
        ...

    def get_denoise_module(self) -> nn.Module:
        """Return an nn.Module for a single denoise step.

        The module should:
        - Accept: (state?, x_t, timestep, prefix_pad_mask, past_key_0, ...)
        - Output: v_t (velocity prediction)

        Whether state is an input depends on state_in_denoise in config.

        Returns:
            nn.Module ready for ONNX export.
        """
        ...

    def prepare_encoder_inputs(
        self,
        example_batch: dict[str, Tensor],
    ) -> tuple[tuple[Tensor, ...], list[str], int, dict[str, str]]:
        """Prepare inputs for encoder ONNX tracing.

        Args:
            example_batch: Example observation batch.

        Returns:
            Tuple of:
            - input_tensors: Tuple of tensors for tracing
            - input_names: Names for each input
            - num_images: Number of image inputs
            - input_mapping: Maps observation keys to ONNX input names
        """
        ...

    def prepare_denoise_inputs(
        self,
        prefix_len: int,
        device,
    ) -> tuple[tuple[Tensor, ...], list[str]]:
        """Prepare inputs for denoise step ONNX tracing.

        Args:
            prefix_len: Length of the prefix sequence from encoder.
            device: Device to create tensors on.

        Returns:
            Tuple of (input_tensors, input_names)
        """
        ...


def is_two_phase_exportable(policy) -> bool:
    """Check if a policy implements ExportableTwoPhase."""
    return isinstance(policy, ExportableTwoPhase)


@runtime_checkable
class ExportableSinglePhase(Protocol):
    """Protocol for single-phase policies (ACT, VQ-BeT).

    Single-phase policies produce an action chunk in one forward pass.
    Implement this protocol to provide custom export logic.
    """

    def get_single_phase_export_config(self) -> SinglePhaseExportConfig:
        """Return export configuration with action dimensions."""
        ...

    def get_forward_module(self) -> nn.Module:
        """Return an nn.Module for the forward pass.

        The module should:
        - Accept observation inputs as positional args
        - Output: actions [B, chunk_size, action_dim]

        Returns:
            nn.Module ready for ONNX export.
        """
        ...

    def prepare_forward_inputs(
        self,
        example_batch: dict[str, Tensor],
    ) -> tuple[tuple[Tensor, ...], list[str], list[str]]:
        """Prepare inputs for ONNX tracing.

        Args:
            example_batch: Example observation batch.

        Returns:
            Tuple of (input_tensors, input_names, output_names)
        """
        ...


@runtime_checkable
class ExportableIterative(Protocol):
    """Protocol for iterative denoising policies (Diffusion).

    Iterative policies refine actions over multiple denoising steps.
    Implement this protocol to provide custom export logic.
    """

    def get_iterative_export_config(self) -> IterativeExportConfig:
        """Return export configuration with denoising parameters."""
        ...

    def get_denoise_module(self) -> nn.Module:
        """Return an nn.Module for a single denoise step.

        The module should:
        - Accept: (*observations, x_t, timestep)
        - Output: v_t (velocity/noise prediction) [B, horizon, action_dim]

        Returns:
            nn.Module ready for ONNX export.
        """
        ...

    def prepare_denoise_inputs(
        self,
        example_batch: dict[str, Tensor],
    ) -> tuple[tuple[Tensor, ...], list[str], list[str]]:
        """Prepare inputs for denoise step ONNX tracing.

        Should include x_t and timestep in addition to observations.

        Args:
            example_batch: Example observation batch.

        Returns:
            Tuple of (input_tensors, input_names, output_names)
        """
        ...


def is_single_phase_exportable(policy) -> bool:
    """Check if a policy implements ExportableSinglePhase."""
    return isinstance(policy, ExportableSinglePhase)


def is_iterative_exportable(policy) -> bool:
    """Check if a policy implements ExportableIterative."""
    return isinstance(policy, ExportableIterative)
