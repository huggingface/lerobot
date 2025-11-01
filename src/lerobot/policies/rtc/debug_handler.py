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

"""Debug information handler for Real-Time Chunking (RTC)."""

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass
class DebugStep:
    """Container for debug information from a single denoising step.

    Attributes:
        step_idx (int): Step index/counter.
        x_t (Tensor | None): Current latent/state tensor.
        v_t (Tensor | None): Velocity from denoiser.
        x1_t (Tensor | None): Denoised prediction (x_t - time * v_t).
        correction (Tensor | None): Correction gradient tensor.
        err (Tensor | None): Weighted error term.
        weights (Tensor | None): Prefix attention weights.
        guidance_weight (float | Tensor | None): Applied guidance weight.
        time (float | Tensor | None): Time parameter.
        inference_delay (int | None): Inference delay parameter.
        execution_horizon (int | None): Execution horizon parameter.
        metadata (dict[str, Any]): Additional metadata.
    """

    step_idx: int = 0
    x_t: Tensor | None = None
    v_t: Tensor | None = None
    x1_t: Tensor | None = None
    correction: Tensor | None = None
    err: Tensor | None = None
    weights: Tensor | None = None
    guidance_weight: float | Tensor | None = None
    time: float | Tensor | None = None
    inference_delay: int | None = None
    execution_horizon: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_tensors: bool = False) -> dict[str, Any]:
        """Convert debug step to dictionary.

        Args:
            include_tensors (bool): If True, include tensor values. If False, only include
                tensor statistics (shape, mean, std, min, max).

        Returns:
            Dictionary representation of the debug step.
        """
        result = {
            "step_idx": self.step_idx,
            "guidance_weight": (
                self.guidance_weight.item()
                if isinstance(self.guidance_weight, Tensor)
                else self.guidance_weight
            ),
            "time": self.time.item() if isinstance(self.time, Tensor) else self.time,
            "inference_delay": self.inference_delay,
            "execution_horizon": self.execution_horizon,
            "metadata": self.metadata.copy(),
        }

        # Add tensor information
        tensor_fields = ["x_t", "v_t", "x1_t", "correction", "err", "weights"]
        for field_name in tensor_fields:
            tensor = getattr(self, field_name)
            if tensor is not None:
                if include_tensors:
                    result[field_name] = tensor.detach().cpu()
                else:
                    result[f"{field_name}_stats"] = {
                        "shape": tuple(tensor.shape),
                        "mean": tensor.mean().item(),
                        "std": tensor.std().item(),
                        "min": tensor.min().item(),
                        "max": tensor.max().item(),
                    }

        return result


class Tracker:
    """Collects and manages debug information for RTC processing.

    This tracker stores debug information from recent denoising steps in a sliding window,
    allowing inspection of intermediate values, tensors, and statistics.

    Args:
        enabled (bool): Whether debug collection is enabled.
        maxlen (int | None): Optional sliding window size. If provided, only the
            most recent ``maxlen`` debug steps are kept. If ``None``, keeps all.
    """

    def __init__(self, enabled: bool = False, maxlen: int = 100):
        self.enabled = enabled
        self._steps = deque(maxlen=maxlen) if enabled else None
        self._step_counter = 0

    def reset(self) -> None:
        """Clear all recorded debug information."""
        if self.enabled and self._steps is not None:
            self._steps.clear()
        self._step_counter = 0

    def record_step(
        self,
        x_t: Tensor | None = None,
        v_t: Tensor | None = None,
        x1_t: Tensor | None = None,
        correction: Tensor | None = None,
        err: Tensor | None = None,
        weights: Tensor | None = None,
        guidance_weight: float | Tensor | None = None,
        time: float | Tensor | None = None,
        inference_delay: int | None = None,
        execution_horizon: int | None = None,
        update_last: bool = False,
        **metadata,
    ) -> None:
        """Record debug information from a denoising step.

        Args:
            x_t (Tensor | None): Current latent/state tensor.
            v_t (Tensor | None): Velocity from denoiser.
            x1_t (Tensor | None): Denoised prediction.
            correction (Tensor | None): Correction gradient tensor.
            err (Tensor | None): Weighted error term.
            weights (Tensor | None): Prefix attention weights.
            guidance_weight (float | Tensor | None): Applied guidance weight.
            time (float | Tensor | None): Time parameter.
            inference_delay (int | None): Inference delay parameter.
            execution_horizon (int | None): Execution horizon parameter.
            update_last (bool): If True, update the most recent step instead of creating a new one.
                Only updates fields that are not None.
            **metadata: Additional metadata to store.
        """
        if not self.enabled:
            return

        # Update existing step if requested
        if update_last and len(self._steps) > 0:
            last_step = self._steps[-1]
            # Only update fields that are provided (not None)
            if x_t is not None:
                last_step.x_t = x_t.detach().clone()
            if v_t is not None:
                last_step.v_t = v_t.detach().clone()
            if x1_t is not None:
                last_step.x1_t = x1_t.detach().clone()
            if correction is not None:
                last_step.correction = correction.detach().clone()
            if err is not None:
                last_step.err = err.detach().clone()
            if weights is not None:
                last_step.weights = weights.detach().clone()
            if guidance_weight is not None:
                last_step.guidance_weight = guidance_weight
            if time is not None:
                last_step.time = time
            if inference_delay is not None:
                last_step.inference_delay = inference_delay
            if execution_horizon is not None:
                last_step.execution_horizon = execution_horizon
            if metadata:
                last_step.metadata.update(metadata)
            return

        # Create new step
        step = DebugStep(
            step_idx=self._step_counter,
            x_t=x_t.detach().clone() if x_t is not None else None,
            v_t=v_t.detach().clone() if v_t is not None else None,
            x1_t=x1_t.detach().clone() if x1_t is not None else None,
            correction=correction.detach().clone() if correction is not None else None,
            err=err.detach().clone() if err is not None else None,
            weights=weights.detach().clone() if weights is not None else None,
            guidance_weight=guidance_weight,
            time=time,
            inference_delay=inference_delay,
            execution_horizon=execution_horizon,
            metadata=metadata,
        )

        self._steps.append(step)
        self._step_counter += 1

    def get_recent_steps(self, n: int = 1) -> list[DebugStep]:
        """Get the n most recent debug steps.

        Args:
            n (int): Number of recent steps to retrieve.

        Returns:
            List of DebugStep objects (may be empty if disabled or no steps recorded).
        """
        if not self.enabled or self._steps is None:
            return []

        return list(self._steps)[-n:]

    def get_all_steps(self) -> list[DebugStep]:
        """Get all recorded debug steps.

        Returns:
            List of all DebugStep objects (may be empty if disabled).
        """
        if not self.enabled or self._steps is None:
            return []

        return list(self._steps)

    def get_step_stats_summary(self) -> dict[str, Any]:
        """Get summary statistics across all recorded steps.

        Returns:
            Dictionary containing aggregate statistics.
        """
        if not self.enabled or self._steps is None or len(self._steps) == 0:
            return {"enabled": self.enabled, "total_steps": 0}

        # Aggregate statistics
        corrections = [s.correction for s in self._steps if s.correction is not None]
        errors = [s.err for s in self._steps if s.err is not None]
        guidance_weights = [s.guidance_weight for s in self._steps if s.guidance_weight is not None]

        summary = {
            "enabled": self.enabled,
            "total_steps": len(self._steps),
            "step_counter": self._step_counter,
        }

        if corrections:
            correction_norms = torch.stack([c.norm().item() for c in corrections])
            summary["correction_norms"] = {
                "mean": correction_norms.mean().item(),
                "std": correction_norms.std().item(),
                "min": correction_norms.min().item(),
                "max": correction_norms.max().item(),
            }

        if errors:
            error_norms = torch.stack([e.norm().item() for e in errors])
            summary["error_norms"] = {
                "mean": error_norms.mean().item(),
                "std": error_norms.std().item(),
                "min": error_norms.min().item(),
                "max": error_norms.max().item(),
            }

        if guidance_weights:
            gw_tensor = torch.tensor([gw.item() if isinstance(gw, Tensor) else gw for gw in guidance_weights])
            summary["guidance_weights"] = {
                "mean": gw_tensor.mean().item(),
                "std": gw_tensor.std().item(),
                "min": gw_tensor.min().item(),
                "max": gw_tensor.max().item(),
            }

        return summary

    def export_to_dict(self, include_tensors: bool = False) -> dict[str, Any]:
        """Export all debug information to a dictionary.

        Args:
            include_tensors (bool): If True, include full tensor values. If False,
                only include tensor statistics.

        Returns:
            Dictionary containing all debug information.
        """
        if not self.enabled or self._steps is None:
            return {"enabled": False, "steps": []}

        return {
            "enabled": True,
            "total_steps": len(self._steps),
            "step_counter": self._step_counter,
            "steps": [step.to_dict(include_tensors=include_tensors) for step in self._steps],
        }

    def __len__(self) -> int:
        """Return the number of recorded debug steps."""
        if not self.enabled or self._steps is None:
            return 0
        return len(self._steps)

    @staticmethod
    def tensor_stats(tensor: Tensor, name: str = "tensor") -> str:
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
