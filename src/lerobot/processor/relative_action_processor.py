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

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_STATE

from .delta_action_processor import MapDeltaActionToRobotActionStep, MapTensorToDeltaActionDictStep
from .pipeline import ProcessorStep, ProcessorStepRegistry

# Re-export for backward compatibility
__all__ = [
    "MapDeltaActionToRobotActionStep",
    "MapTensorToDeltaActionDictStep",
    "DeriveStateFromActionStep",
    "RelativeActionsProcessorStep",
    "AbsoluteActionsProcessorStep",
    "RelativeStateProcessorStep",
    "to_relative_actions",
    "to_absolute_actions",
    "to_relative_state",
]


def to_relative_actions(actions: Tensor, state: Tensor, mask: Sequence[bool]) -> Tensor:
    """Convert absolute actions to relative: relative = action - state (for masked dims).

    Args:
        actions: (B, T, action_dim) or (B, action_dim).
        state: (B, state_dim). Broadcast across time dimension.
        mask: Which dims to convert. Can be shorter than action_dim.
    """
    mask_t = torch.tensor(mask, dtype=actions.dtype, device=actions.device)
    dims = mask_t.shape[0]
    # Align state to the same device/dtype as actions. _last_state is cached before
    # DeviceProcessorStep moves the transition, so it can be on CPU while actions are on CUDA.
    if state.device != actions.device or state.dtype != actions.dtype:
        state = state.to(device=actions.device, dtype=actions.dtype)
    state_offset = state[..., :dims] * mask_t
    if actions.ndim == 3:
        state_offset = state_offset.unsqueeze(-2)
    actions = actions.clone()
    actions[..., :dims] -= state_offset
    return actions


def to_absolute_actions(actions: Tensor, state: Tensor, mask: Sequence[bool]) -> Tensor:
    """Convert relative actions back to absolute: absolute = relative + state (for masked dims).

    Args:
        actions: (B, T, action_dim) or (B, action_dim).
        state: (B, state_dim). Broadcast across time dimension.
        mask: Which dims to convert. Can be shorter than action_dim.
    """
    mask_t = torch.tensor(mask, dtype=actions.dtype, device=actions.device)
    dims = mask_t.shape[0]
    # Align state to the same device/dtype as actions. _last_state is cached before
    # DeviceProcessorStep moves the transition, so it can be on CPU while actions are on CUDA.
    if state.device != actions.device or state.dtype != actions.dtype:
        state = state.to(device=actions.device, dtype=actions.dtype)
    state_offset = state[..., :dims] * mask_t
    if actions.ndim == 3:
        state_offset = state_offset.unsqueeze(-2)
    actions = actions.clone()
    actions[..., :dims] += state_offset
    return actions


@ProcessorStepRegistry.register("derive_state_from_action_processor")
@dataclass
class DeriveStateFromActionStep(ProcessorStep):
    """Derives 2-step observation.state from the action chunk (UMI-style).

    Expects action with one extra leading timestep: [B, chunk_size+1, D]
    from action_delta_indices = [-1, 0, 1, ..., chunk_size-1].
    Extracts [action[t-1], action[t]] as state and strips the extra timestep.
    No-op during inference (state comes from robot).
    """

    enabled: bool = False

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition
        action = transition.get(TransitionKey.ACTION)
        if action is None or action.ndim < 3:
            return transition
        new_transition = transition.copy()
        new_obs = dict(new_transition.get(TransitionKey.OBSERVATION, {}))
        new_obs[OBS_STATE] = action[..., :2, :]
        new_transition[TransitionKey.ACTION] = action[..., 1:, :]
        new_transition[TransitionKey.OBSERVATION] = new_obs
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("delta_actions_processor")
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
    """

    enabled: bool = False
    exclude_joints: list[str] = field(default_factory=list)
    action_names: list[str] | None = None
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

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {})
        raw_state = observation.get(OBS_STATE) if observation else None

        # When state_delta_indices loads multi-timestep state [B, n_obs, D],
        # use only the current (last) timestep for relative action conversion.
        if raw_state is not None:
            state = raw_state[..., -1, :] if raw_state.ndim >= 3 else raw_state
        else:
            state = None

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
        new_transition[TransitionKey.ACTION] = to_relative_actions(action, state, mask)
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "exclude_joints": self.exclude_joints,
            "action_names": self.action_names,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def to_relative_state(state: Tensor, mask: Sequence[bool]) -> Tensor:
    """Convert multi-timestep absolute state to relative (offset from current timestep).

    Each timestep becomes: ``state[..., t, :] - state[..., -1, :]`` for masked dims.
    The last (current) timestep becomes zeros for masked dims.

    Args:
        state: (..., n_obs, state_dim) — last timestep is the reference (current).
        mask: Which dims to convert. Can be shorter than state_dim.
    """
    mask_t = torch.tensor(mask, dtype=state.dtype, device=state.device)
    dims = mask_t.shape[0]
    current = state[..., -1:, :]  # (..., 1, state_dim)
    state = state.clone()
    state[..., :dims] -= current[..., :dims] * mask_t
    return state


@ProcessorStepRegistry.register("relative_state_processor")
@dataclass
class RelativeStateProcessorStep(ProcessorStep):
    """Converts observation.state to relative (offset from current timestep).

    UMI-style relative proprioception: each state timestep is expressed as
    an offset from the current EE pose, providing velocity information.

    During training (multi-timestep input from ``state_delta_indices``):
        ``state[..., t, :] -= state[..., -1, :]`` — subtract current from all.

    During inference (single timestep): buffers the previous state and stacks
    ``[previous, current]`` before applying the relative conversion, producing
    the same ``[n_obs, D]`` shape the model expects.

    Attributes:
        enabled: Whether to apply the relative conversion.
        exclude_joints: Joint/dim names to keep absolute.
        state_names: State dimension names from dataset metadata.
    """

    enabled: bool = False
    exclude_joints: list[str] = field(default_factory=list)
    state_names: list[str] | None = None
    _previous_state: torch.Tensor | None = field(default=None, init=False, repr=False)

    def _build_mask(self, state_dim: int) -> list[bool]:
        if not self.exclude_joints or self.state_names is None:
            return [True] * state_dim

        exclude_tokens = [str(name).lower() for name in self.exclude_joints if name]
        if not exclude_tokens:
            return [True] * state_dim

        mask = []
        for name in self.state_names[:state_dim]:
            state_name = str(name).lower()
            is_excluded = any(token == state_name or token in state_name for token in exclude_tokens)
            mask.append(not is_excluded)

        if len(mask) < state_dim:
            mask.extend([True] * (state_dim - len(mask)))

        return mask

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition

        observation = transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(OBS_STATE) if observation else None

        if state is None:
            return transition

        new_transition = transition.copy()
        new_obs = dict(new_transition.get(TransitionKey.OBSERVATION, {}))
        mask = self._build_mask(state.shape[-1])

        if state.ndim >= 3:
            # [B, n_obs, D] — multi-timestep (training with state_delta_indices)
            relative = to_relative_state(state, mask)
            new_obs[OBS_STATE] = relative.flatten(start_dim=-2)  # [B, n_obs*D]
        elif state.ndim == 2:
            # [B, D] — single timestep (inference): buffer previous and stack
            current = state
            if self._previous_state is None:
                self._previous_state = current.clone()
            prev = self._previous_state
            if prev.device != current.device or prev.dtype != current.dtype:
                prev = prev.to(device=current.device, dtype=current.dtype)
            stacked = torch.stack([prev, current], dim=-2)  # [B, 2, D]
            relative = to_relative_state(stacked, mask)
            new_obs[OBS_STATE] = relative.flatten(start_dim=-2)  # [B, 2*D]
            self._previous_state = current.clone()

        new_transition[TransitionKey.OBSERVATION] = new_obs
        return new_transition

    def reset(self) -> None:
        """Reset the state buffer. Call at episode boundaries during inference."""
        self._previous_state = None

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "exclude_joints": self.exclude_joints,
            "state_names": self.state_names,
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

        if self.relative_step._last_state is None:
            raise RuntimeError(
                "AbsoluteActionsProcessorStep requires state from RelativeActionsProcessorStep "
                "but no state has been cached. Ensure the preprocessor runs before the postprocessor."
            )

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            return new_transition

        mask = self.relative_step._build_mask(action.shape[-1])
        new_transition[TransitionKey.ACTION] = to_absolute_actions(
            action, self.relative_step._last_state, mask
        )
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
