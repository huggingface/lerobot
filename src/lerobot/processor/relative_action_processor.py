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

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.rotation import (
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_rotate,
    quaternion_to_rotvec,
    rotvec_to_quaternion,
)

from .delta_action_processor import MapDeltaActionToRobotActionStep, MapTensorToDeltaActionDictStep
from .pipeline import PolicyProcessorPipeline, ProcessorStep, ProcessorStepRegistry

if TYPE_CHECKING:  # a runtime import would be circular: ``pretrained`` imports from this package
    from lerobot.policies.pretrained import PreTrainedPolicy

# Re-export for backward compatibility
__all__ = [
    "MapDeltaActionToRobotActionStep",
    "MapTensorToDeltaActionDictStep",
    "RelativeActionsProcessorStep",
    "AbsoluteActionsProcessorStep",
    "bind_relative_anchor",
    "to_relative_actions",
    "to_absolute_actions",
]


def to_relative_se3_pose(target_pose: Tensor, reference_pose: Tensor) -> Tensor:
    """Encode a pose as ``inv(T_reference) @ T_target``.

    Poses use ``[x, y, z, rx, ry, rz]`` with an axis-angle rotation vector.
    The relative translation is therefore expressed in the reference EE frame.
    """
    if target_pose.shape[-1] != 6 or reference_pose.shape[-1] != 6:
        raise ValueError("SE(3) poses must have six values: xyz followed by a rotation vector")
    reference_quaternion = rotvec_to_quaternion(reference_pose[..., 3:])
    target_quaternion = rotvec_to_quaternion(target_pose[..., 3:])
    inverse_reference_quaternion = quaternion_conjugate(reference_quaternion)
    relative_translation = quaternion_rotate(
        inverse_reference_quaternion, target_pose[..., :3] - reference_pose[..., :3]
    )
    relative_quaternion = quaternion_multiply(inverse_reference_quaternion, target_quaternion)
    return torch.cat((relative_translation, quaternion_to_rotvec(relative_quaternion)), dim=-1)


def to_absolute_se3_pose(relative_pose: Tensor, reference_pose: Tensor) -> Tensor:
    """Decode a pose with ``T_target = T_reference @ T_relative``."""
    if relative_pose.shape[-1] != 6 or reference_pose.shape[-1] != 6:
        raise ValueError("SE(3) poses must have six values: xyz followed by a rotation vector")
    reference_quaternion = rotvec_to_quaternion(reference_pose[..., 3:])
    relative_quaternion = rotvec_to_quaternion(relative_pose[..., 3:])
    target_translation = reference_pose[..., :3] + quaternion_rotate(
        reference_quaternion, relative_pose[..., :3]
    )
    target_quaternion = quaternion_multiply(reference_quaternion, relative_quaternion)
    return torch.cat((target_translation, quaternion_to_rotvec(target_quaternion)), dim=-1)


def _resolve_se3_pose_groups(
    pose_groups: Sequence[Sequence[int]] | None,
    mask: Sequence[bool],
    action_dim: int,
    state_dim: int,
) -> list[list[int]]:
    """Validate ``se3_pose_groups`` against the action layout and return them as lists.

    Each group is six consecutive-or-not action indices laid out as ``[x, y, z, rx, ry, rz]``
    with an axis-angle rotation vector, and must be inside the relative mask -- a pose the
    policy keeps absolute has nothing to compose against.

    The same indices address the state, which is the pose the actions are composed against, so
    they must also fit inside it.
    """
    if not pose_groups:
        return []
    resolved: list[list[int]] = []
    seen: set[int] = set()
    for group in pose_groups:
        indices = [int(i) for i in group]
        if len(indices) != 6:
            raise ValueError(
                f"An SE(3) pose group needs six indices (xyz + rotation vector), got {len(indices)}"
            )
        for index in indices:
            if not 0 <= index < action_dim:
                raise ValueError(f"SE(3) pose index {index} is outside the action of width {action_dim}")
            if index >= state_dim:
                raise ValueError(
                    f"SE(3) pose index {index} is outside the state of width {state_dim}. The pose "
                    "group addresses both the action and the state it is composed against, so the "
                    "state must carry the same pose at the same indices."
                )
            if index in seen:
                raise ValueError(f"Action index {index} appears in more than one SE(3) pose group")
            if index < len(mask) and not mask[index]:
                raise ValueError(
                    f"Action index {index} is excluded from the relative conversion, so it cannot "
                    "be part of an SE(3) pose group"
                )
            seen.add(index)
        resolved.append(indices)
    return resolved


def to_relative_actions(
    actions: Tensor,
    state: Tensor,
    mask: Sequence[bool],
    se3_pose_groups: Sequence[Sequence[int]] | None = None,
) -> Tensor:
    """Convert absolute actions to relative: relative = action - state (for masked dims).

    Dimensions listed in ``se3_pose_groups`` are composed as ``inv(T_state) @ T_action``
    instead. Subtracting rotation components is only meaningful for scalar quantities such
    as joint angles; on an end-effector pose it is wrong, because rotations compose by
    multiplication.

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
    # A temporally stacked observation (a policy whose ``observation_delta_indices`` spans several
    # frames, e.g. VLA-JEPA or LingBot-VA) hands over a (B, T_obs, state_dim) state. The reference
    # is the CURRENT frame -- delta 0, i.e. index 0 -- so collapse to it and let the offset
    # broadcast over the action horizon. pi0/pi05 pass a 2D (B, state_dim) state and are unaffected.
    if state.ndim == 3:
        state = state[:, 0]
    groups = _resolve_se3_pose_groups(se3_pose_groups, mask, actions.shape[-1], state.shape[-1])
    component_mask = mask_t.clone()
    for group in groups:
        component_mask[group] = 0
    state_offset = state[..., :dims] * component_mask
    if actions.ndim == 3:
        state_offset = state_offset.unsqueeze(-2)
        reference = state.unsqueeze(-2)
    else:
        reference = state
    actions = actions.clone()
    actions[..., :dims] -= state_offset
    for group in groups:
        actions[..., group] = to_relative_se3_pose(actions[..., group], reference[..., group])
    return actions


def to_absolute_actions(
    actions: Tensor,
    state: Tensor,
    mask: Sequence[bool],
    se3_pose_groups: Sequence[Sequence[int]] | None = None,
) -> Tensor:
    """Convert relative actions back to absolute: absolute = relative + state (for masked dims).

    Dimensions listed in ``se3_pose_groups`` are composed as ``T_state @ T_relative``,
    inverting what :func:`to_relative_actions` did to them.

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
    # A temporally stacked observation (a policy whose ``observation_delta_indices`` spans several
    # frames, e.g. VLA-JEPA or LingBot-VA) hands over a (B, T_obs, state_dim) state. The reference
    # is the CURRENT frame -- delta 0, i.e. index 0 -- so collapse to it and let the offset
    # broadcast over the action horizon. pi0/pi05 pass a 2D (B, state_dim) state and are unaffected.
    if state.ndim == 3:
        state = state[:, 0]
    groups = _resolve_se3_pose_groups(se3_pose_groups, mask, actions.shape[-1], state.shape[-1])
    component_mask = mask_t.clone()
    for group in groups:
        component_mask[group] = 0
    state_offset = state[..., :dims] * component_mask
    if actions.ndim == 3:
        state_offset = state_offset.unsqueeze(-2)
        reference = state.unsqueeze(-2)
    else:
        reference = state
    actions = actions.clone()
    for group in groups:
        actions[..., group] = to_absolute_se3_pose(actions[..., group], reference[..., group])
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
        se3_pose_groups: Action index groups laid out as ``[x, y, z, rx, ry, rz]`` with an
            axis-angle rotation vector, composed in SE(3) rather than subtracted. Empty by
            default, which keeps the component-wise behaviour for every dimension.
    """

    enabled: bool = False
    exclude_joints: list[str] = field(default_factory=list)
    action_names: list[str] | None = None
    se3_pose_groups: list[list[int]] = field(default_factory=list)
    _last_state: torch.Tensor | None = field(default=None, init=False, repr=False)
    _count_queued_actions: Callable[[], int] | None = field(default=None, init=False, repr=False)

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
        state = observation.get(OBS_STATE) if observation else None

        # Cache state for the paired AbsoluteActionsProcessorStep -- but hold it for as long
        # as the policy is still serving the chunk that was generated against it. A fresh
        # chunk re-anchors on the tick the queue runs dry.
        if state is not None and not self._chunk_in_flight():
            self._last_state = state

        if not self.enabled:
            return transition

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None or state is None:
            return new_transition
        if not isinstance(action, torch.Tensor):
            raise ValueError(f"RelativeActionsProcessorStep expects a tensor action, got {type(action)}")

        mask = self._build_mask(action.shape[-1])
        new_transition[TransitionKey.ACTION] = to_relative_actions(action, state, mask, self.se3_pose_groups)
        return new_transition

    def reset(self) -> None:
        self._last_state = None

    def _chunk_in_flight(self) -> bool:
        """Whether the policy still holds actions generated against the cached anchor."""
        return self._count_queued_actions is not None and self._count_queued_actions() > 0

    def bind_action_queue(self, count_queued_actions: Callable[[], int] | None) -> None:
        self._count_queued_actions = count_queued_actions

    def get_cached_state(self) -> torch.Tensor | None:
        """Return the cached ``observation.state`` used as the reference point for relative/absolute action conversions."""
        return self._last_state

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "exclude_joints": self.exclude_joints,
            "action_names": self.action_names,
            "se3_pose_groups": self.se3_pose_groups,
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
        if not isinstance(action, torch.Tensor):
            raise ValueError(f"AbsoluteActionsProcessorStep expects a tensor action, got {type(action)}")

        mask = self.relative_step._build_mask(action.shape[-1])
        new_transition[TransitionKey.ACTION] = to_absolute_actions(
            action, cached_state, mask, self.relative_step.se3_pose_groups
        )
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def bind_relative_anchor(
    policy: "PreTrainedPolicy", pipeline: PolicyProcessorPipeline[Any, Any]
) -> RelativeActionsProcessorStep | None:
    """Let ``pipeline``'s relative-action step hold a chunk's anchor until the chunk drains.

    Call once wherever a policy and its preprocessor are built together; a disabled step counts
    as absent. Returns the step that was bound, or ``None`` if the pipeline has no enabled one.
    """
    step = next(
        (
            s
            for s in getattr(pipeline, "steps", ())
            if isinstance(s, RelativeActionsProcessorStep) and s.enabled
        ),
        None,
    )
    if step is not None:
        step.bind_action_queue(policy.count_queued_actions)
    return step
