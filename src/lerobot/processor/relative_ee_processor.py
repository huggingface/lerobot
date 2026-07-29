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

"""Processor steps for fixed-reference relative end-effector action chunks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE

from .pipeline import ProcessorStep, ProcessorStepRegistry


def axis_angle_to_matrix(axis_angle: Tensor) -> Tensor:
    """Convert axis-angle vectors to rotation matrices."""
    theta_sq = (axis_angle * axis_angle).sum(dim=-1, keepdim=True)
    theta = theta_sq.sqrt()
    small = theta_sq < 1e-8
    theta_sq_safe = theta_sq.clamp_min(1e-16)
    theta_safe = theta.clamp_min(1e-8)
    sin_over_theta = torch.where(
        small,
        1 - theta_sq / 6 + theta_sq * theta_sq / 120,
        torch.sin(theta) / theta_safe,
    )
    one_minus_cos_over_theta_sq = torch.where(
        small,
        0.5 - theta_sq / 24 + theta_sq * theta_sq / 720,
        (1 - torch.cos(theta)) / theta_sq_safe,
    )

    x, y, z = axis_angle.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1).reshape(*axis_angle.shape[:-1], 3, 3)
    identity = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    return (
        identity
        + sin_over_theta.unsqueeze(-1) * skew
        + one_minus_cos_over_theta_sq.unsqueeze(-1) * (skew @ skew)
    )


def _sqrt_positive_part(value: Tensor) -> Tensor:
    return torch.sqrt(torch.clamp(value, min=0))


def matrix_to_axis_angle(matrix: Tensor) -> Tensor:
    """Convert rotation matrices to canonical axis-angle vectors."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Rotation matrices must have shape [..., 3, 3], got {tuple(matrix.shape)}")

    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    quaternion_abs = _sqrt_positive_part(
        torch.stack(
            [
                1 + m00 + m11 + m22,
                1 + m00 - m11 - m22,
                1 - m00 + m11 - m22,
                1 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )
    quaternion_candidates = torch.stack(
        [
            torch.stack([quaternion_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, quaternion_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, quaternion_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, quaternion_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    candidates = quaternion_candidates / (2 * quaternion_abs).clamp_min(0.1).unsqueeze(-1)
    best = quaternion_abs.argmax(dim=-1)
    gather_index = best[..., None, None].expand(*best.shape, 1, 4)
    quaternion = candidates.gather(dim=-2, index=gather_index).squeeze(-2)
    quaternion = torch.where(quaternion[..., :1] < 0, -quaternion, quaternion)

    vector = quaternion[..., 1:]
    vector_norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    half_angle = torch.atan2(vector_norm, quaternion[..., :1])
    angle = 2 * half_angle
    small = angle.abs() < 1e-6
    sin_half_over_angle = torch.where(
        small,
        0.5 - angle * angle / 48,
        torch.sin(half_angle) / angle.clamp_min(1e-8),
    )
    return vector / sin_half_over_angle.clamp_min(1e-8)


def matrix_to_rotation_6d(matrix: Tensor) -> Tensor:
    """Return the first two rows of a rotation matrix."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Rotation matrices must have shape [..., 3, 3], got {tuple(matrix.shape)}")
    return matrix[..., :2, :].clone().reshape(*matrix.shape[:-2], 6)


def rotation_6d_to_matrix(rotation_6d: Tensor) -> Tensor:
    """Convert the row-based 6D rotation representation to a matrix."""
    if rotation_6d.shape[-1] != 6:
        raise ValueError(f"6D rotations must have shape [..., 6], got {tuple(rotation_6d.shape)}")
    row0 = torch.nn.functional.normalize(rotation_6d[..., :3], dim=-1)
    row1 = rotation_6d[..., 3:] - (row0 * rotation_6d[..., 3:]).sum(dim=-1, keepdim=True) * row0
    row1 = torch.nn.functional.normalize(row1, dim=-1)
    row2 = torch.cross(row0, row1, dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def absolute_ee_to_relative(reference: Tensor, target: Tensor) -> Tensor:
    """Express absolute 7D EE targets in a reference EE frame as 10D poses."""
    if reference.shape[-1] != 7 or target.shape[-1] != 7:
        raise ValueError(
            "Relative EE requires 7D [xyz, axis-angle, gripper] poses; "
            f"got reference={reference.shape[-1]}D and target={target.shape[-1]}D"
        )
    reference_rotation = axis_angle_to_matrix(reference[..., 3:6])
    target_rotation = axis_angle_to_matrix(target[..., 3:6])
    reference_rotation_t = reference_rotation.transpose(-2, -1)
    relative_rotation = reference_rotation_t @ target_rotation
    relative_translation = (
        reference_rotation_t @ (target[..., :3] - reference[..., :3]).unsqueeze(-1)
    ).squeeze(-1)
    return torch.cat(
        [relative_translation, matrix_to_rotation_6d(relative_rotation), target[..., 6:7]], dim=-1
    )


def relative_ee_to_absolute(relative: Tensor, reference: Tensor) -> Tensor:
    """Convert relative 10D EE poses to absolute 7D EE targets."""
    if relative.shape[-1] != 10 or reference.shape[-1] != 7:
        raise ValueError(
            "Relative EE requires 10D actions and a 7D reference pose; "
            f"got action={relative.shape[-1]}D and reference={reference.shape[-1]}D"
        )
    reference_rotation = axis_angle_to_matrix(reference[..., 3:6])
    absolute_rotation = reference_rotation @ rotation_6d_to_matrix(relative[..., 3:9])
    absolute_translation = reference[..., :3] + (
        reference_rotation @ relative[..., :3].unsqueeze(-1)
    ).squeeze(-1)
    return torch.cat(
        [absolute_translation, matrix_to_axis_angle(absolute_rotation), relative[..., 9:10]], dim=-1
    )


def to_relative_ee_actions(actions: Tensor, state: Tensor) -> Tensor:
    """Convert absolute action chunks using one reference pose per batch item."""
    if actions.shape[-1] != 7 or state.shape[-1] != 7:
        raise ValueError(
            "Relative EE requires 7D [xyz, axis-angle, gripper] input; "
            f"got action={actions.shape[-1]}D and state={state.shape[-1]}D"
        )
    if state.ndim == 3:
        state = state[:, -1]
    state = state.to(device=actions.device, dtype=actions.dtype)
    if actions.ndim == 3:
        state = state.unsqueeze(1).expand(*actions.shape[:-1], 7)
    return absolute_ee_to_relative(state, actions)


def to_absolute_ee_actions(actions: Tensor, state: Tensor) -> Tensor:
    """Convert relative action chunks back to absolute EE targets."""
    if actions.shape[-1] != 10 or state.shape[-1] != 7:
        raise ValueError(
            "Relative EE requires 10D actions and a 7D reference state; "
            f"got action={actions.shape[-1]}D and state={state.shape[-1]}D"
        )
    if state.ndim == 3:
        state = state[:, -1]
    state = state.to(device=actions.device, dtype=actions.dtype)
    if actions.ndim == 3:
        state = state.unsqueeze(1).expand(*actions.shape[:-1], 7)
    return relative_ee_to_absolute(actions, state)


def to_relative_ee_state(state: Tensor) -> Tensor:
    """Convert ``[previous, current]`` absolute poses to a flattened 20D state."""
    if state.ndim != 3 or state.shape[-2:] != (2, 7):
        raise ValueError(f"Relative EE state must have shape [batch, 2, 7], got {tuple(state.shape)}")
    current = state[:, -1:].expand_as(state)
    return absolute_ee_to_relative(current, state).flatten(start_dim=-2)


@ProcessorStepRegistry.register("relative_ee_derive_state")
@dataclass
class RelativeEEDeriveStateStep(ProcessorStep):
    """Derive ``[previous, current]`` state from the leading two actions."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None or action.ndim < 3:
            return transition
        if action.shape[-2] < 2:
            raise ValueError("Relative EE training requires at least two queried action poses")
        result = transition.copy()
        observation = dict(result.get(TransitionKey.OBSERVATION) or {})
        observation[OBS_STATE] = action[..., :2, :]
        result[TransitionKey.OBSERVATION] = observation
        result[TransitionKey.ACTION] = action[..., 1:, :]

        complementary = dict(result.get(TransitionKey.COMPLEMENTARY_DATA, {}))
        for container in (result, complementary):
            padding = container.get("action_is_pad")
            if isinstance(padding, Tensor) and padding.ndim >= 2:
                container["action_is_pad"] = padding[..., 1:]
        if complementary:
            result[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return result

    def transform_features(self, features):
        return features


@ProcessorStepRegistry.register("relative_ee_actions")
@dataclass
class RelativeEEActionsStep(ProcessorStep):
    """Convert absolute action chunks to fixed-base relative EE chunks."""

    _last_state: Tensor | None = field(default=None, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {})
        raw_state = observation.get(OBS_STATE) if observation else None
        state = raw_state[..., -1, :] if raw_state is not None and raw_state.ndim >= 3 else raw_state
        if state is not None:
            self._last_state = state.detach().clone()
        action = transition.get(TransitionKey.ACTION)
        if action is None or state is None:
            return transition
        result = transition.copy()
        result[TransitionKey.ACTION] = to_relative_ee_actions(action, state)
        return result

    def get_cached_state(self) -> Tensor | None:
        return self._last_state

    def reset(self) -> None:
        self._last_state = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        result = {feature_type: dict(values) for feature_type, values in features.items()}
        action_features = result.get(PipelineFeatureType.ACTION, {})
        if ACTION in action_features:
            feature = action_features[ACTION]
            action_features[ACTION] = PolicyFeature(type=feature.type, shape=(10,))
        return result


@ProcessorStepRegistry.register("relative_ee_state")
@dataclass
class RelativeEEStateStep(ProcessorStep):
    """Convert raw EE observations to a two-pose, current-relative state."""

    _previous_state: Tensor | None = field(default=None, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(OBS_STATE) if observation else None
        if state is None:
            return transition
        if state.ndim == 2:
            previous = state if self._previous_state is None else self._previous_state.to(state)
            state_pair = torch.stack([previous, state], dim=1)
            self._previous_state = state.detach().clone()
        elif state.ndim == 3 and state.shape[1] == 2:
            state_pair = state
        else:
            raise ValueError(f"Relative EE state must be [B, 7] or [B, 2, 7], got {tuple(state.shape)}")
        result = transition.copy()
        result_observation = dict(observation)
        result_observation[OBS_STATE] = to_relative_ee_state(state_pair)
        result[TransitionKey.OBSERVATION] = result_observation
        return result

    def reset(self) -> None:
        self._previous_state = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        result = {feature_type: dict(values) for feature_type, values in features.items()}
        observation_features = result.get(PipelineFeatureType.OBSERVATION, {})
        if OBS_STATE in observation_features:
            feature = observation_features[OBS_STATE]
            observation_features[OBS_STATE] = PolicyFeature(type=feature.type, shape=(20,))
        return result


@ProcessorStepRegistry.register("absolute_ee_actions")
@dataclass
class AbsoluteEEActionsStep(ProcessorStep):
    """Convert model-relative actions back to absolute EE targets."""

    relative_step: RelativeEEActionsStep | None = field(default=None, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if self.relative_step is None:
            raise RuntimeError("AbsoluteEEActionsStep requires a paired RelativeEEActionsStep")
        state = self.relative_step.get_cached_state()
        if state is None:
            raise RuntimeError("Relative EE postprocessing requires the preprocessor to run first")
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        result = transition.copy()
        result[TransitionKey.ACTION] = to_absolute_ee_actions(action, state)
        return result

    def get_config(self) -> dict[str, Any]:
        return {}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        result = {feature_type: dict(values) for feature_type, values in features.items()}
        action_features = result.get(PipelineFeatureType.ACTION, {})
        if ACTION in action_features:
            feature = action_features[ACTION]
            action_features[ACTION] = PolicyFeature(type=feature.type, shape=(7,))
        return result
