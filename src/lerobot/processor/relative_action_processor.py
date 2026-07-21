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
    "to_relative_se3_pose",
    "to_absolute_se3_pose",
]


def _rotvec_to_quaternion(rotvec: Tensor) -> Tensor:
    angle = torch.linalg.vector_norm(rotvec, dim=-1, keepdim=True)
    angle_sq = angle.square()
    small_scale = 0.5 - angle_sq / 48.0 + angle_sq.square() / 3840.0
    scale = torch.where(angle > 1e-6, torch.sin(angle / 2.0) / angle.clamp_min(1e-12), small_scale)
    return torch.cat((torch.cos(angle / 2.0), rotvec * scale), dim=-1)


def _quaternion_to_rotvec(quaternion: Tensor) -> Tensor:
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1e-12)
    quaternion = quaternion * torch.where(quaternion[..., :1] < 0, -1.0, 1.0)
    vector = quaternion[..., 1:]
    sin_half_angle = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half_angle, quaternion[..., :1].clamp_min(0.0))
    small_scale = 2.0 + sin_half_angle.square() / 3.0
    scale = torch.where(
        sin_half_angle > 1e-6,
        angle / sin_half_angle.clamp_min(1e-12),
        small_scale,
    )
    return vector * scale


def _quaternion_multiply(left: Tensor, right: Tensor) -> Tensor:
    left_w, left_xyz = left[..., :1], left[..., 1:]
    right_w, right_xyz = right[..., :1], right[..., 1:]
    return torch.cat(
        (
            left_w * right_w - (left_xyz * right_xyz).sum(dim=-1, keepdim=True),
            left_w * right_xyz + right_w * left_xyz + torch.linalg.cross(left_xyz, right_xyz, dim=-1),
        ),
        dim=-1,
    )


def _quaternion_conjugate(quaternion: Tensor) -> Tensor:
    return torch.cat((quaternion[..., :1], -quaternion[..., 1:]), dim=-1)


def _quaternion_rotate(quaternion: Tensor, vector: Tensor) -> Tensor:
    quaternion_xyz = quaternion[..., 1:]
    uv = torch.linalg.cross(quaternion_xyz, vector, dim=-1)
    uuv = torch.linalg.cross(quaternion_xyz, uv, dim=-1)
    return vector + 2.0 * (quaternion[..., :1] * uv + uuv)


def to_relative_se3_pose(target_pose: Tensor, reference_pose: Tensor) -> Tensor:
    """Encode a pose as ``inv(T_reference) @ T_target``.

    Poses use ``[x, y, z, rx, ry, rz]`` with an axis-angle rotation vector.
    The relative translation is therefore expressed in the reference EE frame.
    """
    if target_pose.shape[-1] != 6 or reference_pose.shape[-1] != 6:
        raise ValueError("SE(3) poses must have six values: xyz followed by a rotation vector")
    reference_quaternion = _rotvec_to_quaternion(reference_pose[..., 3:])
    target_quaternion = _rotvec_to_quaternion(target_pose[..., 3:])
    inverse_reference_quaternion = _quaternion_conjugate(reference_quaternion)
    relative_translation = _quaternion_rotate(
        inverse_reference_quaternion, target_pose[..., :3] - reference_pose[..., :3]
    )
    relative_quaternion = _quaternion_multiply(inverse_reference_quaternion, target_quaternion)
    return torch.cat((relative_translation, _quaternion_to_rotvec(relative_quaternion)), dim=-1)


def to_absolute_se3_pose(relative_pose: Tensor, reference_pose: Tensor) -> Tensor:
    """Decode a pose with ``T_target = T_reference @ T_relative``."""
    if relative_pose.shape[-1] != 6 or reference_pose.shape[-1] != 6:
        raise ValueError("SE(3) poses must have six values: xyz followed by a rotation vector")
    reference_quaternion = _rotvec_to_quaternion(reference_pose[..., 3:])
    relative_quaternion = _rotvec_to_quaternion(relative_pose[..., 3:])
    target_translation = reference_pose[..., :3] + _quaternion_rotate(
        reference_quaternion, relative_pose[..., :3]
    )
    target_quaternion = _quaternion_multiply(reference_quaternion, relative_quaternion)
    return torch.cat((target_translation, _quaternion_to_rotvec(target_quaternion)), dim=-1)


def _broadcast_reference(actions: Tensor, state: Tensor) -> Tensor:
    if state.device != actions.device or state.dtype != actions.dtype:
        state = state.to(device=actions.device, dtype=actions.dtype)
    if actions.ndim == state.ndim + 1:
        state = state.unsqueeze(-2)
    return state


def _validate_se3_pose_groups(
    pose_representation: str,
    se3_pose_groups: Sequence[Sequence[int]] | None,
    mask: Sequence[bool],
    action_dim: int,
) -> list[list[int]]:
    if pose_representation not in {"componentwise", "se3"}:
        raise ValueError(
            f"Unsupported pose_representation={pose_representation!r}; expected 'componentwise' or 'se3'"
        )
    if pose_representation == "componentwise":
        return []
    if not se3_pose_groups:
        raise ValueError("pose_representation='se3' requires at least one six-index se3_pose_group")

    normalized_groups: list[list[int]] = []
    used_indices: set[int] = set()
    for raw_group in se3_pose_groups:
        group = [int(index) for index in raw_group]
        if len(group) != 6:
            raise ValueError(f"Each SE(3) pose group must contain six indices, got {group}")
        if len(set(group)) != 6 or any(index < 0 or index >= action_dim for index in group):
            raise ValueError(f"Invalid SE(3) pose group for action_dim={action_dim}: {group}")
        if any(index >= len(mask) for index in group):
            raise ValueError(f"SE(3) pose group lies outside the relative mask: {group}")
        if used_indices.intersection(group):
            raise ValueError(f"SE(3) pose groups must not overlap: {group}")
        group_mask = [bool(mask[index]) for index in group]
        if any(group_mask) and not all(group_mask):
            raise ValueError(f"An SE(3) pose group must be wholly relative or wholly absolute: {group}")
        used_indices.update(group)
        if all(group_mask):
            normalized_groups.append(group)
    return normalized_groups


def to_relative_actions(
    actions: Tensor,
    state: Tensor,
    mask: Sequence[bool],
    *,
    pose_representation: str = "componentwise",
    se3_pose_groups: Sequence[Sequence[int]] | None = None,
) -> Tensor:
    """Convert absolute actions to a configured relative representation.

    Component-wise mode computes ``action - state``. SE(3) mode computes
    ``inv(T_state) @ T_action`` for each configured pose group.

    Args:
        actions: (B, T, action_dim) or (B, action_dim).
        state: (B, state_dim). Broadcast across time dimension.
        mask: Which dims to convert. Can be shorter than action_dim.
    """
    groups = _validate_se3_pose_groups(pose_representation, se3_pose_groups, mask, actions.shape[-1])
    mask_t = torch.tensor(mask, dtype=actions.dtype, device=actions.device)
    dims = mask_t.shape[0]
    # Align state to the same device/dtype as actions. _last_state is cached before
    # DeviceProcessorStep moves the transition, so it can be on CPU while actions are on CUDA.
    state = _broadcast_reference(actions, state)
    component_mask = mask_t.clone()
    for group in groups:
        component_mask[group] = 0
    state_offset = state[..., :dims] * component_mask
    actions = actions.clone()
    actions[..., :dims] -= state_offset
    for group in groups:
        actions[..., group] = to_relative_se3_pose(actions[..., group], state[..., group])
    return actions


def to_absolute_actions(
    actions: Tensor,
    state: Tensor,
    mask: Sequence[bool],
    *,
    pose_representation: str = "componentwise",
    se3_pose_groups: Sequence[Sequence[int]] | None = None,
) -> Tensor:
    """Convert relative actions back to absolute actions.

    Component-wise mode computes ``relative + state``. SE(3) mode computes
    ``T_state @ T_relative`` for each configured pose group.

    Args:
        actions: (B, T, action_dim) or (B, action_dim).
        state: (B, state_dim). Broadcast across time dimension.
        mask: Which dims to convert. Can be shorter than action_dim.
    """
    groups = _validate_se3_pose_groups(pose_representation, se3_pose_groups, mask, actions.shape[-1])
    mask_t = torch.tensor(mask, dtype=actions.dtype, device=actions.device)
    dims = mask_t.shape[0]
    # Align state to the same device/dtype as actions. _last_state is cached before
    # DeviceProcessorStep moves the transition, so it can be on CPU while actions are on CUDA.
    state = _broadcast_reference(actions, state)
    component_mask = mask_t.clone()
    for group in groups:
        component_mask[group] = 0
    state_offset = state[..., :dims] * component_mask
    actions = actions.clone()
    actions[..., :dims] += state_offset
    for group in groups:
        actions[..., group] = to_absolute_se3_pose(actions[..., group], state[..., group])
    return actions


@ProcessorStepRegistry.register("relative_actions_processor")
@dataclass
class RelativeActionsProcessorStep(ProcessorStep):
    """Converts absolute actions to the configured relative representation.

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
    pose_representation: str = "componentwise"
    se3_pose_groups: list[list[int]] = field(default_factory=list)
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
        state = observation.get(OBS_STATE) if observation else None

        # State history has shape (B, H, D). Relative actions are referenced to
        # the newest proprioceptive state, not the whole history tensor.
        reference_state = state[:, -1] if state is not None and state.ndim == 3 else state

        # Always cache state for the paired AbsoluteActionsProcessorStep
        if reference_state is not None:
            self._last_state = reference_state

        if not self.enabled:
            return transition

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None or reference_state is None:
            return new_transition

        mask = self._build_mask(action.shape[-1])
        new_transition[TransitionKey.ACTION] = to_relative_actions(
            action,
            reference_state,
            mask,
            pose_representation=self.pose_representation,
            se3_pose_groups=self.se3_pose_groups,
        )
        return new_transition

    def get_cached_state(self) -> torch.Tensor | None:
        """Return the cached ``observation.state`` used as the reference point for relative/absolute action conversions."""
        return self._last_state

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "exclude_joints": self.exclude_joints,
            "action_names": self.action_names,
            "pose_representation": self.pose_representation,
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

        mask = self.relative_step._build_mask(action.shape[-1])
        new_transition[TransitionKey.ACTION] = to_absolute_actions(
            action,
            cached_state,
            mask,
            pose_representation=self.relative_step.pose_representation,
            se3_pose_groups=self.relative_step.se3_pose_groups,
        )
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
