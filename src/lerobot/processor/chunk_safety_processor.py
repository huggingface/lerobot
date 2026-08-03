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

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import PolicyAction

from .pipeline import PolicyActionProcessorStep, ProcessorStepRegistry

logger = logging.getLogger(__name__)


def _resolve_limit(
    limit: float | Sequence[float] | None, action_dim: int, device: torch.device, dtype: torch.dtype
) -> Tensor | None:
    if limit is None:
        return None
    if isinstance(limit, (int, float)):
        return torch.full((action_dim,), float(limit), device=device, dtype=dtype)
    limit_t = torch.as_tensor(limit, device=device, dtype=dtype)
    if limit_t.shape[-1] != action_dim:
        raise ValueError(f"Limit has length {limit_t.shape[-1]}, expected action_dim={action_dim}.")
    return limit_t


def clamp_action_chunk(
    actions: Tensor,
    max_relative_target: float | Sequence[float] | None = None,
    max_jerk: float | Sequence[float] | None = None,
    min_action: float | Sequence[float] | None = None,
    max_action: float | Sequence[float] | None = None,
) -> tuple[Tensor, bool]:
    """Clamp a predicted action (or action chunk) for safety, per-dim or scalar.

    Args:
        actions: (B, T, action_dim) chunk, or (B, action_dim) single action.
        max_relative_target: Max |actions[t] - actions[t-1]|. None disables this check.
        max_jerk: Max |delta[t] - delta[t-1]|. None disables this check.
        min_action: Absolute lower bound. None disables this side of the bounds check.
        max_action: Absolute upper bound. None disables this side of the bounds check.
    """
    action_dim = actions.shape[-1]
    device, dtype = actions.device, actions.dtype
    out = actions
    was_clamped = False

    lo = _resolve_limit(min_action, action_dim, device, dtype)
    hi = _resolve_limit(max_action, action_dim, device, dtype)
    if lo is not None or hi is not None:
        clamp_lo = (
            lo if lo is not None else torch.full_like(out[..., 0, :] if out.ndim > 1 else out, -torch.inf)
        )
        clamp_hi = (
            hi if hi is not None else torch.full_like(out[..., 0, :] if out.ndim > 1 else out, torch.inf)
        )
        clamped = out.clamp(min=clamp_lo, max=clamp_hi)
        if not torch.equal(clamped, out):
            was_clamped = True
        out = clamped

    has_time_dim = out.ndim == 3 and out.shape[1] >= 2
    if has_time_dim and (max_relative_target is not None or max_jerk is not None):
        out = out.clone()
        rel_limit = _resolve_limit(max_relative_target, action_dim, device, dtype)
        jerk_limit = _resolve_limit(max_jerk, action_dim, device, dtype)
        prev_delta = None
        for t in range(1, out.shape[1]):
            delta = out[:, t, :] - out[:, t - 1, :]
            clamped_delta = delta.clamp(min=-rel_limit, max=rel_limit) if rel_limit is not None else delta
            if jerk_limit is not None and prev_delta is not None:
                jerk = clamped_delta - prev_delta
                clamped_jerk = jerk.clamp(min=-jerk_limit, max=jerk_limit)
                clamped_delta = prev_delta + clamped_jerk
            if not torch.equal(clamped_delta, delta):
                was_clamped = True
            out[:, t, :] = out[:, t - 1, :] + clamped_delta
            prev_delta = clamped_delta

    return out, was_clamped


@ProcessorStepRegistry.register("chunk_safety_processor")
@dataclass
class ChunkSafetyProcessorStep(PolicyActionProcessorStep):
    """Checks a predicted action chunk as a sequence, not just action-by-action.

    Clamps only the values that violate a limit, leaving the rest of the
    chunk untouched. Each of its three checks — absolute bounds,
    discontinuity, jerk — is independent and off by default.

    Attributes:
        enabled: Whether to apply any of the checks below.
        max_relative_target: Max ``|delta|`` between consecutive chunk
            steps, scalar or per-dim. ``None`` disables this check.
        max_jerk: Max ``|change in delta|`` between consecutive steps,
            scalar or per-dim. ``None`` disables this check.
        min_action: Absolute per-dim (or scalar) lower bound. ``None``
            disables this side of the bounds check.
        max_action: Absolute per-dim (or scalar) upper bound. ``None``
            disables this side of the bounds check.
    """

    enabled: bool = True
    max_relative_target: float | list[float] | None = None
    max_jerk: float | list[float] | None = None
    min_action: float | list[float] | None = None
    max_action: float | list[float] | None = None

    def action(self, action: PolicyAction) -> PolicyAction:
        if not self.enabled:
            return action
        if (
            self.max_relative_target is None
            and self.max_jerk is None
            and self.min_action is None
            and self.max_action is None
        ):
            return action

        clamped, was_clamped = clamp_action_chunk(
            action,
            max_relative_target=self.max_relative_target,
            max_jerk=self.max_jerk,
            min_action=self.min_action,
            max_action=self.max_action,
        )
        if was_clamped:
            logger.warning("ChunkSafetyProcessorStep clamped an out-of-bounds predicted action chunk.")
        return clamped

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_relative_target": self.max_relative_target,
            "max_jerk": self.max_jerk,
            "min_action": self.min_action,
            "max_action": self.max_action,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
