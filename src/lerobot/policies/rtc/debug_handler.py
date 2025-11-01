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

    This tracker stores debug information from recent denoising steps in a dictionary,
    using time as the key for efficient lookups and updates.

    Args:
        enabled (bool): Whether debug collection is enabled.
        maxlen (int | None): Optional sliding window size. If provided, only the
            most recent ``maxlen`` debug steps are kept. If ``None``, keeps all.
    """

    def __init__(self, enabled: bool = False, maxlen: int = 100):
        self.enabled = enabled
        self._steps = {} if enabled else None  # Dictionary with time as key
        self._maxlen = maxlen
        self._step_counter = 0

    def reset(self) -> None:
        """Clear all recorded debug information."""
        if self.enabled and self._steps is not None:
            self._steps.clear()
        self._step_counter = 0

    def track(
        self,
        time: float | Tensor,
        x_t: Tensor | None = None,
        v_t: Tensor | None = None,
        x1_t: Tensor | None = None,
        correction: Tensor | None = None,
        err: Tensor | None = None,
        weights: Tensor | None = None,
        guidance_weight: float | Tensor | None = None,
        inference_delay: int | None = None,
        execution_horizon: int | None = None,
        **metadata,
    ) -> None:
        """Track debug information for a denoising step at a given time.

        If a step with the given time already exists, it will be updated with the new data.
        Otherwise, a new step will be created. Only non-None fields are updated/set.

        Args:
            time (float | Tensor): Time parameter - used as the key to identify the step.
            x_t (Tensor | None): Current latent/state tensor.
            v_t (Tensor | None): Velocity from denoiser.
            x1_t (Tensor | None): Denoised prediction.
            correction (Tensor | None): Correction gradient tensor.
            err (Tensor | None): Weighted error term.
            weights (Tensor | None): Prefix attention weights.
            guidance_weight (float | Tensor | None): Applied guidance weight.
            inference_delay (int | None): Inference delay parameter.
            execution_horizon (int | None): Execution horizon parameter.
            **metadata: Additional metadata to store.
        """
        if not self.enabled:
            return

        # Convert time to float and round to avoid float precision issues
        time_value = time.item() if isinstance(time, Tensor) else time
        time_key = round(time_value, 6)  # Use rounded time as dictionary key

        # Check if step with this time already exists
        if time_key in self._steps:
            # Update existing step with non-None fields
            existing_step = self._steps[time_key]
            if x_t is not None:
                existing_step.x_t = x_t.detach().clone()
            if v_t is not None:
                existing_step.v_t = v_t.detach().clone()
            if x1_t is not None:
                existing_step.x1_t = x1_t.detach().clone()
            if correction is not None:
                existing_step.correction = correction.detach().clone()
            if err is not None:
                existing_step.err = err.detach().clone()
            if weights is not None:
                existing_step.weights = weights.detach().clone()
            if guidance_weight is not None:
                existing_step.guidance_weight = guidance_weight
            if inference_delay is not None:
                existing_step.inference_delay = inference_delay
            if execution_horizon is not None:
                existing_step.execution_horizon = execution_horizon
            if metadata:
                existing_step.metadata.update(metadata)
        else:
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
                time=time_value,
                inference_delay=inference_delay,
                execution_horizon=execution_horizon,
                metadata=metadata,
            )

            # Add to dictionary
            self._steps[time_key] = step
            self._step_counter += 1

            # Enforce maxlen if set
            if self._maxlen is not None and len(self._steps) > self._maxlen:
                # Remove oldest entry (first key in dict - Python 3.7+ preserves insertion order)
                oldest_key = next(iter(self._steps))
                del self._steps[oldest_key]

    def get_recent_steps(self, n: int = 1) -> list[DebugStep]:
        """Get the n most recent debug steps.

        Args:
            n (int): Number of recent steps to retrieve.

        Returns:
            List of DebugStep objects (may be empty if disabled or no steps recorded).
        """
        if not self.enabled or self._steps is None:
            return []

        # Get all values and return the last n
        all_steps = list(self._steps.values())
        return all_steps[-n:]

    def get_all_steps(self) -> list[DebugStep]:
        """Get all recorded debug steps.

        Returns:
            List of all DebugStep objects (may be empty if disabled).
        """
        if not self.enabled or self._steps is None:
            return []

        return list(self._steps.values())

    def get_step_stats_summary(self) -> dict[str, Any]:
        """Get summary statistics across all recorded steps.

        Returns:
            Dictionary containing aggregate statistics.
        """
        if not self.enabled or self._steps is None or len(self._steps) == 0:
            return {"enabled": self.enabled, "total_steps": 0}

        # Aggregate statistics from dictionary values
        corrections = [s.correction for s in self._steps.values() if s.correction is not None]
        errors = [s.err for s in self._steps.values() if s.err is not None]
        guidance_weights = [s.guidance_weight for s in self._steps.values() if s.guidance_weight is not None]

        summary = {
            "enabled": self.enabled,
            "total_steps": len(self._steps),
            "step_counter": self._step_counter,
        }

        if corrections:
            correction_norms = torch.tensor([c.norm().item() for c in corrections])
            summary["correction_norms"] = {
                "mean": correction_norms.mean().item(),
                "std": correction_norms.std().item(),
                "min": correction_norms.min().item(),
                "max": correction_norms.max().item(),
            }

        if errors:
            error_norms = torch.tensor([e.norm().item() for e in errors])
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
            "steps": [step.to_dict(include_tensors=include_tensors) for step in self._steps.values()],
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
