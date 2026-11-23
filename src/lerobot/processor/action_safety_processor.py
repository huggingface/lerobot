#!/usr/bin/env python

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
"""Runtime safety guardrail processor step for robot and policy actions."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey

from .pipeline import ProcessorStep, ProcessorStepRegistry

NonFinitePolicy = Literal["raise", "zero", "previous"]
BoundSpec = float | list[float] | dict[str, float] | None

_VALID_NON_FINITE_POLICIES: tuple[NonFinitePolicy, ...] = ("raise", "zero", "previous")


def _validate_bound_spec(name: str, spec: BoundSpec, *, non_negative: bool = False) -> None:
    """Validate that a scalar, sequence, or dict bound specification contains finite numbers."""
    if spec is None:
        return

    if isinstance(spec, bool):
        raise TypeError(f"{name} must be a float, list[float], dict[str, float], or None, got bool")

    if isinstance(spec, int | float):
        val = float(spec)
        if not math.isfinite(val):
            raise ValueError(f"{name} must be finite, got {spec}")
        if non_negative and val < 0.0:
            raise ValueError(f"{name} must be >= 0, got {spec}")
        return

    if isinstance(spec, list):
        if len(spec) == 0:
            raise ValueError(f"{name} list must not be empty")
        for idx, item in enumerate(spec):
            if isinstance(item, bool) or not isinstance(item, int | float):
                raise TypeError(f"{name}[{idx}] must be a number, got {type(item).__name__}")
            val = float(item)
            if not math.isfinite(val):
                raise ValueError(f"{name}[{idx}] must be finite, got {item}")
            if non_negative and val < 0.0:
                raise ValueError(f"{name}[{idx}] must be >= 0, got {item}")
        return

    if isinstance(spec, dict):
        for key, item in spec.items():
            if not isinstance(key, str):
                raise TypeError(f"{name} keys must be strings, got {type(key).__name__}")
            if isinstance(item, bool) or not isinstance(item, int | float):
                raise TypeError(f"{name}[{key!r}] must be a number, got {type(item).__name__}")
            val = float(item)
            if not math.isfinite(val):
                raise ValueError(f"{name}[{key!r}] must be finite, got {item}")
            if non_negative and val < 0.0:
                raise ValueError(f"{name}[{key!r}] must be >= 0, got {item}")
        return

    raise TypeError(
        f"{name} must be a float, list[float], dict[str, float], or None, got {type(spec).__name__}"
    )


def _validate_min_max_compatibility(action_min: BoundSpec, action_max: BoundSpec) -> None:
    """Validate that action_min and action_max have compatible types, lengths, and ordering."""
    if action_min is None or action_max is None:
        return

    if isinstance(action_min, list) and isinstance(action_max, dict):
        raise ValueError("action_min (list) and action_max (dict) cannot mix sequence and dict specs")
    if isinstance(action_min, dict) and isinstance(action_max, list):
        raise ValueError("action_min (dict) and action_max (list) cannot mix dict and sequence specs")

    if isinstance(action_min, int | float) and isinstance(action_max, int | float):
        if float(action_min) > float(action_max):
            raise ValueError(f"action_min ({action_min}) must be <= action_max ({action_max})")
    elif isinstance(action_min, list) and isinstance(action_max, list):
        if len(action_min) != len(action_max):
            raise ValueError(
                f"action_min length ({len(action_min)}) must match action_max length ({len(action_max)})"
            )
        for idx, (lo, hi) in enumerate(zip(action_min, action_max, strict=True)):
            if float(lo) > float(hi):
                raise ValueError(f"action_min[{idx}] ({lo}) must be <= action_max[{idx}] ({hi})")
    elif isinstance(action_min, list) and isinstance(action_max, int | float):
        for idx, lo in enumerate(action_min):
            if float(lo) > float(action_max):
                raise ValueError(f"action_min[{idx}] ({lo}) must be <= action_max ({action_max})")
    elif isinstance(action_min, int | float) and isinstance(action_max, list):
        for idx, hi in enumerate(action_max):
            if float(action_min) > float(hi):
                raise ValueError(f"action_min ({action_min}) must be <= action_max[{idx}] ({hi})")
    elif isinstance(action_min, dict) and isinstance(action_max, dict):
        for key in action_min.keys() & action_max.keys():
            if float(action_min[key]) > float(action_max[key]):
                raise ValueError(
                    f"action_min[{key!r}] ({action_min[key]}) must be <= action_max[{key!r}] ({action_max[key]})"
                )
    elif isinstance(action_min, dict) and isinstance(action_max, int | float):
        for key, lo in action_min.items():
            if float(lo) > float(action_max):
                raise ValueError(f"action_min[{key!r}] ({lo}) must be <= action_max ({action_max})")
    elif isinstance(action_min, int | float) and isinstance(action_max, dict):
        for key, hi in action_max.items():
            if float(action_min) > float(hi):
                raise ValueError(f"action_min ({action_min}) must be <= action_max[{key!r}] ({hi})")


def _resolve_tensor_bound(
    spec: BoundSpec,
    *,
    name: str,
    action: torch.Tensor,
) -> torch.Tensor | None:
    """Convert a scalar or list bound into a tensor matching ``action``'s last dimension, dtype, and device."""
    if spec is None:
        return None
    if isinstance(spec, dict):
        raise ValueError(f"{name} is a dict, which is only supported for dict robot actions, not tensors")
    if isinstance(spec, int | float):
        return torch.tensor(float(spec), dtype=action.dtype, device=action.device)
    if action.ndim == 0 or len(spec) != action.shape[-1]:
        action_dim = 0 if action.ndim == 0 else action.shape[-1]
        raise ValueError(f"{name} length ({len(spec)}) does not match action last dimension ({action_dim})")
    return torch.tensor(spec, dtype=action.dtype, device=action.device)


def _resolve_numpy_bound(
    spec: BoundSpec,
    *,
    name: str,
    action: np.ndarray,
) -> np.ndarray | float | None:
    """Convert a scalar or list bound into a NumPy scalar/array matching ``action``'s last dimension."""
    if spec is None:
        return None
    if isinstance(spec, dict):
        raise ValueError(
            f"{name} is a dict, which is only supported for dict robot actions, not numpy arrays"
        )
    if isinstance(spec, int | float):
        return float(spec)
    if action.ndim == 0 or len(spec) != action.shape[-1]:
        action_dim = 0 if action.ndim == 0 else action.shape[-1]
        raise ValueError(f"{name} length ({len(spec)}) does not match action last dimension ({action_dim})")
    return np.asarray(spec, dtype=action.dtype if np.issubdtype(action.dtype, np.floating) else np.float64)


def _resolve_dict_bound(spec: BoundSpec, *, name: str, key: str) -> float | None:
    """Resolve a scalar or per-key dict bound for a single ``RobotAction`` key."""
    if spec is None:
        return None
    if isinstance(spec, list):
        raise ValueError(f"{name} is a list, which is only supported for tensor/array actions, not dicts")
    if isinstance(spec, int | float):
        return float(spec)
    return float(spec[key]) if key in spec else None


@dataclass
@ProcessorStepRegistry.register(name="action_safety_processor")
class ActionSafetyProcessorStep(ProcessorStep):
    """Runtime safety guardrails for policy and robot actions.

    Applies three composable, opt-in protections in order to the transition's action
    (``torch.Tensor``, ``np.ndarray``, or ``dict[str, float]``):

    1. **Finite-value guard** (``check_finite``, ``on_non_finite``): Detects ``NaN``/``Inf`` values
       and either raises a ``ValueError``, replaces them with ``0.0``, or falls back to the last
       safe action.
    2. **Slew-rate / step-delta limiter** (``max_delta``): Bounds step-to-step changes so
       ``|a_t - a_{t-1}| <= max_delta``. Call :meth:`reset` at episode boundaries to clear the
       recorded previous action.
    3. **Range clamping** (``action_min``, ``action_max``): Clamps action values into
       ``[action_min, action_max]``. Applied last so output bounds are always guaranteed.

    Attributes:
        check_finite: Whether to check action values for ``NaN`` and ``Inf``. Defaults to ``True``.
        on_non_finite: Strategy when non-finite values are found: ``"raise"`` (raise ``ValueError``),
            ``"zero"`` (replace non-finite entries with ``0.0``), or ``"previous"`` (replace with the
            corresponding entry from the previous safe action, or ``0.0`` on the first step).
        action_min: Optional lower bound(s). Accepts a scalar ``float``, a per-dimension
            ``list[float]`` (for tensor/array actions), or a per-joint ``dict[str, float]`` (for
            dict robot actions).
        action_max: Optional upper bound(s), using the same formats as ``action_min``.
        max_delta: Optional maximum allowed absolute change per step, using the same formats as
            ``action_min``. All values must be ``>= 0``.
    """

    check_finite: bool = True
    on_non_finite: NonFinitePolicy = "raise"
    action_min: BoundSpec = None
    action_max: BoundSpec = None
    max_delta: BoundSpec = None

    _prev_action: torch.Tensor | np.ndarray | dict[str, Any] | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.on_non_finite not in _VALID_NON_FINITE_POLICIES:
            raise ValueError(
                f"on_non_finite must be one of {_VALID_NON_FINITE_POLICIES}, got {self.on_non_finite!r}"
            )
        _validate_bound_spec("action_min", self.action_min)
        _validate_bound_spec("action_max", self.action_max)
        _validate_bound_spec("max_delta", self.max_delta, non_negative=True)
        _validate_min_max_compatibility(self.action_min, self.action_max)

    def reset(self) -> None:
        """Clear the recorded previous action so the next step starts a fresh episode."""
        self._prev_action = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Apply configured safety checks and limits to the transition's action."""
        transition = transition.copy()
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition

        if isinstance(action, torch.Tensor):
            transition[TransitionKey.ACTION] = self._process_tensor(action)
        elif isinstance(action, np.ndarray):
            transition[TransitionKey.ACTION] = self._process_numpy(action)
        elif isinstance(action, dict):
            transition[TransitionKey.ACTION] = self._process_dict(action)
        else:
            raise TypeError(
                f"ActionSafetyProcessorStep expects action to be torch.Tensor, np.ndarray, or dict, "
                f"got {type(action).__name__}"
            )

        return transition

    def _process_tensor(self, action: torch.Tensor) -> torch.Tensor:
        out = action.clone()

        prev_tensor: torch.Tensor | None = None
        if self._prev_action is not None:
            if not isinstance(self._prev_action, torch.Tensor):
                raise ValueError(
                    "Action type changed between steps (expected torch.Tensor). "
                    "Call reset() at episode boundaries."
                )
            if self._prev_action.shape != out.shape:
                raise ValueError(
                    f"Action shape changed from {tuple(self._prev_action.shape)} to {tuple(out.shape)}. "
                    "Call reset() when action shape changes."
                )
            prev_tensor = self._prev_action.to(device=out.device, dtype=out.dtype)

        if self.check_finite and (out.is_floating_point() or out.is_complex()):
            finite_mask = torch.isfinite(out)
            if not bool(torch.all(finite_mask)):
                if self.on_non_finite == "raise":
                    n_bad = int((~finite_mask).sum().item())
                    raise ValueError(
                        f"Non-finite action values detected ({n_bad} of {out.numel()} entries, "
                        f"shape={tuple(out.shape)})."
                    )
                fallback = (
                    prev_tensor
                    if self.on_non_finite == "previous" and prev_tensor is not None
                    else torch.zeros_like(out)
                )
                out = torch.where(finite_mask, out, fallback)

        max_delta = _resolve_tensor_bound(self.max_delta, name="max_delta", action=out)
        if max_delta is not None and prev_tensor is not None:
            out = torch.clamp(out, min=prev_tensor - max_delta, max=prev_tensor + max_delta)

        min_bound = _resolve_tensor_bound(self.action_min, name="action_min", action=out)
        max_bound = _resolve_tensor_bound(self.action_max, name="action_max", action=out)
        if min_bound is not None or max_bound is not None:
            out = torch.clamp(out, min=min_bound, max=max_bound)

        self._prev_action = out.detach().clone()
        return out

    def _process_numpy(self, action: np.ndarray) -> np.ndarray:
        out = action.copy()

        prev_array: np.ndarray | None = None
        if self._prev_action is not None:
            if not isinstance(self._prev_action, np.ndarray):
                raise ValueError(
                    "Action type changed between steps (expected np.ndarray). "
                    "Call reset() at episode boundaries."
                )
            if self._prev_action.shape != out.shape:
                raise ValueError(
                    f"Action shape changed from {self._prev_action.shape} to {out.shape}. "
                    "Call reset() when action shape changes."
                )
            prev_array = self._prev_action

        if self.check_finite and np.issubdtype(out.dtype, np.floating):
            finite_mask = np.isfinite(out)
            if not bool(np.all(finite_mask)):
                if self.on_non_finite == "raise":
                    n_bad = int(np.sum(~finite_mask))
                    raise ValueError(
                        f"Non-finite action values detected ({n_bad} of {out.size} entries, "
                        f"shape={out.shape})."
                    )
                fallback = (
                    prev_array
                    if self.on_non_finite == "previous" and prev_array is not None
                    else np.zeros_like(out)
                )
                out = np.where(finite_mask, out, fallback)

        max_delta = _resolve_numpy_bound(self.max_delta, name="max_delta", action=out)
        if max_delta is not None and prev_array is not None:
            out = np.clip(out, prev_array - max_delta, prev_array + max_delta).astype(out.dtype, copy=False)

        min_bound = _resolve_numpy_bound(self.action_min, name="action_min", action=out)
        max_bound = _resolve_numpy_bound(self.action_max, name="action_max", action=out)
        if min_bound is not None or max_bound is not None:
            out = np.clip(out, min_bound, max_bound).astype(out.dtype, copy=False)

        self._prev_action = out.copy()
        return out

    def _process_dict(self, action: dict[str, Any]) -> dict[str, Any]:
        prev_dict: dict[str, Any] | None = None
        if self._prev_action is not None:
            if not isinstance(self._prev_action, dict):
                raise ValueError(
                    "Action type changed between steps (expected dict). Call reset() at episode boundaries."
                )
            prev_dict = self._prev_action

        out: dict[str, Any] = {}
        for key, raw_value in action.items():
            if isinstance(raw_value, bool) or not isinstance(raw_value, int | float | np.number):
                out[key] = raw_value
                continue

            val = float(raw_value)
            prev_val: float | None = None
            if prev_dict is not None and key in prev_dict:
                raw_prev = prev_dict[key]
                if not isinstance(raw_prev, bool) and isinstance(raw_prev, int | float | np.number):
                    prev_val = float(raw_prev)

            if self.check_finite and not math.isfinite(val):
                if self.on_non_finite == "raise":
                    raise ValueError(f"Non-finite action value detected for key {key!r}: {raw_value}")
                val = prev_val if self.on_non_finite == "previous" and prev_val is not None else 0.0

            delta = _resolve_dict_bound(self.max_delta, name="max_delta", key=key)
            if delta is not None and prev_val is not None:
                val = max(prev_val - delta, min(prev_val + delta, val))

            lo = _resolve_dict_bound(self.action_min, name="action_min", key=key)
            hi = _resolve_dict_bound(self.action_max, name="action_max", key=key)
            if lo is not None and val < lo:
                val = lo
            if hi is not None and val > hi:
                val = hi

            out[key] = val

        self._prev_action = dict(out)
        return out

    def get_config(self) -> dict[str, Any]:
        """Return JSON-serializable configuration for pipeline persistence."""
        return {
            "check_finite": self.check_finite,
            "on_non_finite": self.on_non_finite,
            "action_min": self.action_min,
            "action_max": self.action_max,
            "max_delta": self.max_delta,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Preserve input feature specifications unchanged."""
        return features
