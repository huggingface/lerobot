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
from lerobot.utils.constants import ACTION, OBS_STATE

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
    "to_relative_se3_pose_6d",
    "to_absolute_se3_pose_6d",
    "rotation_6d_to_rotvec",
    "rotvec_to_rotation_6d",
    "relative_action_output_dim",
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


def _quaternion_to_matrix(quaternion: Tensor) -> Tensor:
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = quaternion.unbind(-1)
    two_s = 2.0
    return torch.stack(
        (
            1.0 - two_s * (y * y + z * z),
            two_s * (x * y - z * w),
            two_s * (x * z + y * w),
            two_s * (x * y + z * w),
            1.0 - two_s * (x * x + z * z),
            two_s * (y * z - x * w),
            two_s * (x * z - y * w),
            two_s * (y * z + x * w),
            1.0 - two_s * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def _matrix_to_quaternion(matrix: Tensor) -> Tensor:
    """Convert proper rotation matrices to normalized ``[w, x, y, z]`` quaternions."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Rotation matrices must have shape (..., 3, 3), got {matrix.shape}")

    m00 = matrix[..., 0, 0]
    m01 = matrix[..., 0, 1]
    m02 = matrix[..., 0, 2]
    m10 = matrix[..., 1, 0]
    m11 = matrix[..., 1, 1]
    m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]
    m21 = matrix[..., 2, 1]
    m22 = matrix[..., 2, 2]

    # Each row is a quaternion candidate scaled by the magnitude of its
    # best-conditioned component. Selecting the largest component avoids the
    # trace singularity at rotations close to pi.
    q_abs = torch.sqrt(
        torch.clamp(
            torch.stack(
                (
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ),
                dim=-1,
            ),
            min=0.0,
        )
    )
    quat_by_rijk = torch.stack(
        (
            torch.stack((q_abs[..., 0].square(), m21 - m12, m02 - m20, m10 - m01), dim=-1),
            torch.stack((m21 - m12, q_abs[..., 1].square(), m10 + m01, m02 + m20), dim=-1),
            torch.stack((m02 - m20, m10 + m01, q_abs[..., 2].square(), m12 + m21), dim=-1),
            torch.stack((m10 - m01, m02 + m20, m12 + m21, q_abs[..., 3].square()), dim=-1),
        ),
        dim=-2,
    )
    candidates = quat_by_rijk / (2.0 * q_abs[..., :, None].clamp_min(0.1))
    best = torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4).to(dtype=matrix.dtype)
    quaternion = (candidates * best[..., :, None]).sum(dim=-2)
    return quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1e-12)


def rotvec_to_rotation_6d(rotvec: Tensor) -> Tensor:
    """Encode an axis-angle rotation as two rotation-matrix columns."""
    matrix = _quaternion_to_matrix(_rotvec_to_quaternion(rotvec))
    return torch.cat((matrix[..., :, 0], matrix[..., :, 1]), dim=-1)


def rotation_6d_to_rotvec(rotation_6d: Tensor) -> Tensor:
    """Decode two predicted 3-D vectors into an axis-angle rotation.

    Gram-Schmidt orthonormalization follows the continuous 6-D rotation
    representation. Degenerate predictions fail closed instead of producing an
    invalid physical rotation.
    """
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f"6-D rotations must have six values, got {rotation_6d.shape}")
    first = rotation_6d[..., :3]
    second = rotation_6d[..., 3:]
    first_norm = torch.linalg.vector_norm(first, dim=-1, keepdim=True)
    first_unit = first / first_norm.clamp_min(1e-12)
    second_orthogonal = second - (first_unit * second).sum(dim=-1, keepdim=True) * first_unit
    second_norm = torch.linalg.vector_norm(second_orthogonal, dim=-1, keepdim=True)
    if bool(torch.any(first_norm <= 1e-8)) or bool(torch.any(second_norm <= 1e-8)):
        raise ValueError("Cannot decode a degenerate 6-D rotation prediction")
    second_unit = second_orthogonal / second_norm
    third_unit = torch.linalg.cross(first_unit, second_unit, dim=-1)
    matrix = torch.stack((first_unit, second_unit, third_unit), dim=-1)
    return _quaternion_to_rotvec(_matrix_to_quaternion(matrix))


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


def to_relative_se3_pose_6d(target_pose: Tensor, reference_pose: Tensor) -> Tensor:
    """Encode ``inv(T_reference) @ T_target`` as xyz plus continuous 6-D rotation."""
    relative_pose = to_relative_se3_pose(target_pose, reference_pose)
    return torch.cat((relative_pose[..., :3], rotvec_to_rotation_6d(relative_pose[..., 3:])), dim=-1)


def to_absolute_se3_pose_6d(relative_pose: Tensor, reference_pose: Tensor) -> Tensor:
    """Decode xyz plus continuous 6-D rotation with ``T_target = T_reference @ T_relative``."""
    if relative_pose.shape[-1] != 9:
        raise ValueError("6-D encoded SE(3) poses must have nine values: xyz plus rotation-6D")
    relative_rotvec_pose = torch.cat(
        (relative_pose[..., :3], rotation_6d_to_rotvec(relative_pose[..., 3:])), dim=-1
    )
    return to_absolute_se3_pose(relative_rotvec_pose, reference_pose)


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
    if pose_representation not in {"componentwise", "se3", "se3_6d"}:
        raise ValueError(
            f"Unsupported pose_representation={pose_representation!r}; expected "
            "'componentwise', 'se3', or 'se3_6d'"
        )
    if pose_representation == "componentwise":
        return []
    if not se3_pose_groups:
        raise ValueError(
            f"pose_representation={pose_representation!r} requires at least one six-index se3_pose_group"
        )

    normalized_groups: list[list[int]] = []
    used_indices: set[int] = set()
    for raw_group in se3_pose_groups:
        group = [int(index) for index in raw_group]
        if len(group) != 6:
            raise ValueError(f"Each SE(3) pose group must contain six indices, got {group}")
        if len(set(group)) != 6 or any(index < 0 or index >= action_dim for index in group):
            raise ValueError(f"Invalid SE(3) pose group for action_dim={action_dim}: {group}")
        if pose_representation == "se3_6d" and group != list(range(group[0], group[0] + 6)):
            raise ValueError("se3_6d pose groups must contain six contiguous ascending indices")
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


def relative_action_output_dim(
    source_dim: int,
    pose_representation: str,
    se3_pose_groups: Sequence[Sequence[int]] | None,
) -> int:
    """Return the model-space action width for a source action width."""
    if pose_representation != "se3_6d":
        return source_dim
    groups = se3_pose_groups or []
    return source_dim + 3 * len(groups)


def _expand_se3_6d_actions(
    actions: Tensor,
    state: Tensor,
    groups: Sequence[Sequence[int]],
) -> Tensor:
    group_by_start = {group[0]: list(group) for group in groups}
    grouped_indices = {index for group in groups for index in group}
    parts: list[Tensor] = []
    for index in range(actions.shape[-1]):
        group = group_by_start.get(index)
        if group is not None:
            parts.append(to_relative_se3_pose_6d(actions[..., group], state[..., group]))
        elif index not in grouped_indices:
            parts.append(actions[..., index : index + 1])
    return torch.cat(parts, dim=-1)


def _collapse_se3_6d_actions(
    actions: Tensor,
    state: Tensor,
    mask: Sequence[bool],
    groups: Sequence[Sequence[int]],
) -> Tensor:
    source_dim = len(mask)
    expected_dim = relative_action_output_dim(source_dim, "se3_6d", groups)
    if actions.shape[-1] != expected_dim:
        raise ValueError(
            f"Expected se3_6d action width {expected_dim} for source width {source_dim}, "
            f"got {actions.shape[-1]}"
        )
    group_by_start = {group[0]: list(group) for group in groups}
    grouped_indices = {index for group in groups for index in group}
    parts: list[Tensor] = []
    cursor = 0
    for index in range(source_dim):
        group = group_by_start.get(index)
        if group is not None:
            parts.append(to_absolute_se3_pose_6d(actions[..., cursor : cursor + 9], state[..., group]))
            cursor += 9
        elif index not in grouped_indices:
            value = actions[..., cursor : cursor + 1]
            if mask[index]:
                value = value + state[..., index : index + 1]
            parts.append(value)
            cursor += 1
    if cursor != actions.shape[-1]:
        raise RuntimeError(f"Consumed {cursor} action values from width {actions.shape[-1]}")
    return torch.cat(parts, dim=-1)


def to_relative_actions(
    actions: Tensor,
    state: Tensor,
    mask: Sequence[bool],
    *,
    pose_representation: str = "componentwise",
    se3_pose_groups: Sequence[Sequence[int]] | None = None,
) -> Tensor:
    """Convert absolute actions to a configured relative representation.

    Component-wise mode computes ``action - state``. SE(3) modes compute
    ``inv(T_state) @ T_action`` for each configured pose group. ``se3_6d``
    replaces each three-value relative rotation vector with its continuous
    six-value encoding, increasing the output width by three per pose group.

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
    if pose_representation == "se3_6d":
        return _expand_se3_6d_actions(actions, state, groups)
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
    source_dim = len(mask)
    groups = _validate_se3_pose_groups(pose_representation, se3_pose_groups, mask, source_dim)
    state = _broadcast_reference(actions, state)
    if pose_representation == "se3_6d":
        return _collapse_se3_6d_actions(actions, state, mask, groups)

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
    _last_mask: list[bool] | None = field(default=None, init=False, repr=False)

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
            self._last_mask = self._build_mask(reference_state.shape[-1])

        if not self.enabled:
            return transition

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None or reference_state is None:
            return new_transition

        mask = self._last_mask or self._build_mask(action.shape[-1])
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

    def get_cached_mask(self) -> list[bool] | None:
        """Return the source-space mask cached with the latest state."""
        return self._last_mask

    def reset(self) -> None:
        """Drop the inference reference so it cannot leak between sessions."""
        self._last_state = None
        self._last_mask = None

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
        if not self.enabled or self.pose_representation != "se3_6d":
            return features
        transformed = {feature_type: dict(feature_group) for feature_type, feature_group in features.items()}
        for feature_group in transformed.values():
            action_feature = feature_group.get(ACTION)
            if action_feature is None:
                continue
            source_dim = len(self.action_names) if self.action_names is not None else action_feature.shape[-1]
            model_dim = relative_action_output_dim(source_dim, self.pose_representation, self.se3_pose_groups)
            if action_feature.shape[-1] == source_dim:
                feature_group[ACTION] = PolicyFeature(
                    type=action_feature.type,
                    shape=(model_dim,),
                )
            elif action_feature.shape[-1] != model_dim:
                raise ValueError(
                    f"Expected source/model action width {source_dim}/{model_dim}, "
                    f"got {action_feature.shape[-1]}"
                )
        return transformed


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

        mask = self.relative_step.get_cached_mask()
        if mask is None:
            mask = self.relative_step._build_mask(cached_state.shape[-1])
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
