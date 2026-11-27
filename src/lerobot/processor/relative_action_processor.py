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
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.utils.constants import ACTION_ANCHOR, OBS_STATE

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


def to_relative_actions(actions: Tensor, state: Tensor, mask: Sequence[bool]) -> Tensor:
    """Convert absolute actions to relative: relative = action - state (for masked dims).

    Args:
        actions: (B, T, action_dim) or (B, action_dim).
        state: (B, state_dim), or (B, T_obs, state_dim) for a temporally stacked
            observation (collapsed to the current frame). Broadcast across time dimension.
        mask: Which dims to convert. Can be shorter than action_dim.
    """
    mask_t = torch.tensor(mask, dtype=actions.dtype, device=actions.device)
    dims = mask_t.shape[0]
    # Align state to the same device/dtype as actions. _last_state is cached before
    # DeviceProcessorStep moves the transition, so it can be on CPU while actions are on CUDA.
    if state.device != actions.device or state.dtype != actions.dtype:
        state = state.to(device=actions.device, dtype=actions.dtype)
    # A temporally stacked observation (a policy whose ``training_sampler`` window spans several
    # frames, e.g. VLA-JEPA or LingBot-VA) hands over a (B, T_obs, state_dim) state. The reference
    # is the CURRENT frame -- delta 0, i.e. index 0 -- so collapse to it and let the offset
    # broadcast over the action horizon. pi0/pi05 pass a 2D (B, state_dim) state and are unaffected.
    if state.ndim == 3:
        state = state[:, 0]
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
        state: (B, state_dim), or (B, T_obs, state_dim) for a temporally stacked
            observation (collapsed to the current frame). Broadcast across time dimension.
        mask: Which dims to convert. Can be shorter than action_dim.
    """
    mask_t = torch.tensor(mask, dtype=actions.dtype, device=actions.device)
    dims = mask_t.shape[0]
    # Align state to the same device/dtype as actions. _last_state is cached before
    # DeviceProcessorStep moves the transition, so it can be on CPU while actions are on CUDA.
    if state.device != actions.device or state.dtype != actions.dtype:
        state = state.to(device=actions.device, dtype=actions.dtype)
    # A temporally stacked observation (a policy whose ``training_sampler`` window spans several
    # frames, e.g. VLA-JEPA or LingBot-VA) hands over a (B, T_obs, state_dim) state. The reference
    # is the CURRENT frame -- delta 0, i.e. index 0 -- so collapse to it and let the offset
    # broadcast over the action horizon. pi0/pi05 pass a 2D (B, state_dim) state and are unaffected.
    if state.ndim == 3:
        state = state[:, 0]
    state_offset = state[..., :dims] * mask_t
    if actions.ndim == 3:
        state_offset = state_offset.unsqueeze(-2)
    actions = actions.clone()
    actions[..., :dims] += state_offset
    return actions


def shift_actions(actions: Tensor, anchor: Tensor, *, sign: int) -> Tensor:
    """Add ``sign * anchor`` to the leading dimensions of ``(B, A)`` or ``(B, T, A)`` actions."""
    anchor = anchor.to(device=actions.device, dtype=actions.dtype)
    dims = min(actions.shape[-1], anchor.shape[-1])
    offset = anchor[..., :dims]
    if actions.ndim == 3:
        offset = offset.unsqueeze(-2)
    shifted = actions.clone()
    shifted[..., :dims] += sign * offset
    return shifted


@ProcessorStepRegistry.register("relative_actions_processor")
@dataclass
class RelativeActionsProcessorStep(ProcessorStep):
    """Converts absolute actions to relative actions (action -= state) for masked dimensions.

    Mirrors OpenPI's DeltaActions transform. Applied during preprocessing so the model
    trains on relative offsets instead of absolute positions. Writes the offset it subtracts,
    the masked current state, under ``action_anchor`` in the complementary data, where the
    paired AbsoluteActionsProcessorStep reads it back during postprocessing.

    Attributes:
        enabled: Whether to apply the relative conversion.
        exclude_joints: Joint names to keep absolute (not converted to relative).
        action_names: Action dimension names from dataset metadata, used to build
            the mask from exclude_joints. If None, all dims are converted.
    """

    enabled: bool = False
    exclude_joints: list[str] = field(default_factory=list)
    action_names: list[str] | None = None

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

    def current_state(self, state: torch.Tensor) -> torch.Tensor:
        """The state frame actions are relative to: frame 0 of a stacked ``(B, T, S)`` state."""
        return state[:, 0] if state.ndim == 3 else state

    def anchor(self, state: torch.Tensor) -> torch.Tensor:
        """The offset subtracted from actions: the current state on the converted dimensions."""
        current = self.current_state(state)
        mask = torch.tensor(self._build_mask(current.shape[-1]), dtype=current.dtype, device=current.device)
        return current * mask

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        state = observation.get(OBS_STATE)
        if not isinstance(state, torch.Tensor):
            return transition
        anchor = self.anchor(state)
        new_transition = transition.copy()
        complementary = dict(new_transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        complementary[ACTION_ANCHOR] = anchor
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            return new_transition
        if not isinstance(action, torch.Tensor):
            raise ValueError(f"RelativeActionsProcessorStep expects a tensor action, got {type(action)}")
        new_transition[TransitionKey.ACTION] = shift_actions(action, anchor, sign=-1)
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


@ProcessorStepRegistry.register("absolute_actions_processor")
@dataclass
class AbsoluteActionsProcessorStep(ProcessorStep):
    """Converts relative actions back to absolute actions (action += anchor).

    Mirrors OpenPI's AbsoluteActions transform. Applied during postprocessing so
    predicted relative offsets are converted back to absolute positions for execution.
    Reads the ``action_anchor`` its paired RelativeActionsProcessorStep wrote into the batch.

    Attributes:
        enabled: Whether to apply the absolute conversion.
    """

    enabled: bool = False

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition
        anchor = (transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}).get(ACTION_ANCHOR)
        if anchor is None:
            raise RuntimeError(
                "AbsoluteActionsProcessorStep needs the action_anchor a RelativeActionsProcessorStep writes; "
                "postprocess the batch the preprocessor and the policy returned."
            )
        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            return new_transition
        if not isinstance(action, torch.Tensor):
            raise ValueError(f"AbsoluteActionsProcessorStep expects a tensor action, got {type(action)}")
        new_transition[TransitionKey.ACTION] = shift_actions(action, anchor, sign=1)
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
