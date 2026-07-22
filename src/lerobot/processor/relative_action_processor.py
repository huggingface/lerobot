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

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_STATE

from .delta_action_processor import MapDeltaActionToRobotActionStep, MapTensorToDeltaActionDictStep
from .pipeline import ProcessorStep, ProcessorStepRegistry

# Re-export for backward compatibility
__all__ = [
    "MapDeltaActionToRobotActionStep",
    "MapTensorToDeltaActionDictStep",
    "RelativeActionsProcessorStep",
    "AbsoluteActionsProcessorStep",
    "to_relative_actions",
    "to_absolute_actions",
]


def _state_base_offset(
    state: Tensor,
    mask_t: Tensor,
    state_index_map: Sequence[int] | None,
) -> Tensor:
    """Select the per-action-dim reference state used as the relative base.

    By default the base is ``state[..., :action_dim]``, which assumes the first
    ``action_dim`` state channels are positionally aligned with the action (i.e.
    ``state[i]`` is the quantity that ``action[i]`` controls). That assumption
    breaks whenever ``observation.state`` carries anything other than a prefix
    copy of the action -- interleaved position/velocity, force/torque channels,
    end-effector pose, contact flags, etc.

    ``state_index_map`` makes the alignment explicit: ``state_index_map[i]`` is
    the state column that action dim ``i`` is relative to. It provides one entry
    per action dim; only the first ``mask_t.shape[0]`` entries are used, matching
    the dims that ``mask`` actually converts (``mask`` may be shorter than
    ``action_dim``).
    """
    dims = mask_t.shape[0]
    if state_index_map is not None:
        idx = torch.as_tensor(list(state_index_map)[:dims], dtype=torch.long, device=state.device)
        base = state.index_select(-1, idx)
    else:
        base = state[..., :dims]
    return base * mask_t


def to_relative_actions(
    actions: Tensor,
    state: Tensor,
    mask: Sequence[bool],
    state_index_map: Sequence[int] | None = None,
) -> Tensor:
    """Convert absolute actions to relative: relative = action - state (for masked dims).

    Args:
        actions: (B, T, action_dim) or (B, action_dim).
        state: (B, state_dim). Broadcast across time dimension.
        mask: Which dims to convert. Can be shorter than action_dim.
        state_index_map: Optional length-``action_dim`` list mapping each action
            dim to the state column it is relative to. When ``None`` (default),
            the first ``action_dim`` state channels are used (prefix alignment).
    """
    mask_t = torch.tensor(mask, dtype=actions.dtype, device=actions.device)
    # Align state to the same device/dtype as actions. _last_state is cached before
    # DeviceProcessorStep moves the transition, so it can be on CPU while actions are on CUDA.
    if state.device != actions.device or state.dtype != actions.dtype:
        state = state.to(device=actions.device, dtype=actions.dtype)
    dims = mask_t.shape[0]
    state_offset = _state_base_offset(state, mask_t, state_index_map)
    if actions.ndim == 3:
        state_offset = state_offset.unsqueeze(-2)
    actions = actions.clone()
    actions[..., :dims] -= state_offset
    return actions


def to_absolute_actions(
    actions: Tensor,
    state: Tensor,
    mask: Sequence[bool],
    state_index_map: Sequence[int] | None = None,
) -> Tensor:
    """Convert relative actions back to absolute: absolute = relative + state (for masked dims).

    Args:
        actions: (B, T, action_dim) or (B, action_dim).
        state: (B, state_dim). Broadcast across time dimension.
        mask: Which dims to convert. Can be shorter than action_dim.
        state_index_map: Optional length-``action_dim`` list mapping each action
            dim to the state column it is relative to. When ``None`` (default),
            the first ``action_dim`` state channels are used (prefix alignment).
    """
    mask_t = torch.tensor(mask, dtype=actions.dtype, device=actions.device)
    # Align state to the same device/dtype as actions. _last_state is cached before
    # DeviceProcessorStep moves the transition, so it can be on CPU while actions are on CUDA.
    if state.device != actions.device or state.dtype != actions.dtype:
        state = state.to(device=actions.device, dtype=actions.dtype)
    dims = mask_t.shape[0]
    state_offset = _state_base_offset(state, mask_t, state_index_map)
    if actions.ndim == 3:
        state_offset = state_offset.unsqueeze(-2)
    actions = actions.clone()
    actions[..., :dims] += state_offset
    return actions


@ProcessorStepRegistry.register("relative_actions_processor")
@dataclass
class RelativeActionsProcessorStep(ProcessorStep):
    """Converts absolute actions to relative actions (action -= state) for masked dimensions.

    Mirrors OpenPI's DeltaActions transform. Applied during preprocessing so the model
    trains on relative offsets instead of absolute positions.
    Caches the last seen state so a paired AbsoluteActionsProcessorStep can reverse
    the conversion during postprocessing.

    Attributes:
        enabled: Whether to apply the relative conversion.
        exclude_joints: Joint names to keep absolute (not converted to relative).
        action_names: Action dimension names from dataset metadata, used to build
            the mask from exclude_joints. If None, all dims are converted.
        state_action_index_map: Optional explicit mapping where entry ``i`` is the
            ``observation.state`` column that action dim ``i`` is relative to. Use
            this when the state is not a prefix-aligned copy of the action, e.g.
            interleaved ``[pos, vel]`` per joint, or a state augmented with
            force/torque, pose, or contact channels. When ``None`` (default), the
            legacy ``state[:action_dim]`` prefix behavior is used, so existing
            configs are unchanged.
    """

    enabled: bool = False
    exclude_joints: list[str] = field(default_factory=list)
    action_names: list[str] | None = None
    state_action_index_map: list[int] | None = None
    _last_state: torch.Tensor | None = field(default=None, init=False, repr=False)

    def _build_mask(self, action_dim: int) -> list[bool]:
        if not self.exclude_joints or self.action_names is None:
            return [True] * action_dim

        exclude_tokens = [str(name).lower() for name in self.exclude_joints if name]
        if not exclude_tokens:
            return [True] * action_dim

        mask = []
        for name in self.action_names[:action_dim]:
            action_name = str(name).lower()
            is_excluded = any(token == action_name or token in action_name for token in exclude_tokens)
            mask.append(not is_excluded)

        if len(mask) < action_dim:
            mask.extend([True] * (action_dim - len(mask)))

        return mask

    def _resolve_state_index_map(self, action_dim: int, state_dim: int) -> list[int] | None:
        """Resolve which state column each action dim is relative to.

        Returns the explicit ``state_action_index_map`` when set (validated), else
        ``None`` for the legacy ``state[:action_dim]`` prefix behavior. Raises on an
        obviously inconsistent map so a misaligned config fails loudly instead of
        silently producing wrong offsets.
        """
        if self.state_action_index_map is None:
            return None

        idx_map = list(self.state_action_index_map)
        if len(idx_map) != action_dim:
            raise ValueError(
                f"state_action_index_map has length {len(idx_map)} but action_dim is "
                f"{action_dim}; it must provide one state index per action dim."
            )
        if any(not (0 <= j < state_dim) for j in idx_map):
            raise ValueError(
                f"state_action_index_map contains out-of-range indices for state_dim={state_dim}: {idx_map}."
            )
        return idx_map

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(OBS_STATE) if observation else None

        # Always cache state for the paired AbsoluteActionsProcessorStep
        if state is not None:
            self._last_state = state

        if not self.enabled:
            return transition

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None or state is None:
            return new_transition

        mask = self._build_mask(action.shape[-1])
        state_index_map = self._resolve_state_index_map(action.shape[-1], state.shape[-1])
        new_transition[TransitionKey.ACTION] = to_relative_actions(action, state, mask, state_index_map)
        return new_transition

    def get_cached_state(self) -> torch.Tensor | None:
        """Return the cached ``observation.state`` used as the reference point for relative/absolute action conversions."""
        return self._last_state

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "exclude_joints": self.exclude_joints,
            "action_names": self.action_names,
            "state_action_index_map": self.state_action_index_map,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("absolute_actions_processor")
@dataclass
class AbsoluteActionsProcessorStep(ProcessorStep):
    """Converts relative actions back to absolute actions (action += state) for all dimensions.

    Mirrors OpenPI's AbsoluteActions transform. Applied during postprocessing so
    predicted relative offsets are converted back to absolute positions for execution.
    Reads the cached state from its paired RelativeActionsProcessorStep.

    Attributes:
        enabled: Whether to apply the absolute conversion.
        relative_step: Reference to the paired RelativeActionsProcessorStep that caches state.
    """

    enabled: bool = False
    relative_step: RelativeActionsProcessorStep | None = field(default=None, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition

        if self.relative_step is None:
            raise RuntimeError(
                "AbsoluteActionsProcessorStep requires a paired RelativeActionsProcessorStep "
                "but relative_step is None. Ensure relative_step is set when constructing the postprocessor."
            )

        cached_state = self.relative_step.get_cached_state()
        if cached_state is None:
            raise RuntimeError(
                "AbsoluteActionsProcessorStep requires state from RelativeActionsProcessorStep "
                "but no state has been cached. Ensure the preprocessor runs before the postprocessor."
            )

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            return new_transition

        mask = self.relative_step._build_mask(action.shape[-1])
        state_index_map = self.relative_step._resolve_state_index_map(
            action.shape[-1], cached_state.shape[-1]
        )
        new_transition[TransitionKey.ACTION] = to_absolute_actions(
            action, cached_state, mask, state_index_map
        )
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
