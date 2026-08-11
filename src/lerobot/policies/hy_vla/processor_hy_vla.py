#!/usr/bin/env python

# Copyright 2026 HuggingFace Inc. team. All rights reserved.
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

"""Serializable geometry and mean/std processors for Hy-VLA."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenderMessagesStep,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_hy_vla import HyVLAConfig

DUAL_NATIVE_DIM = 16
DUAL_HY_DIM = 20
LEFT_NATIVE = slice(0, 8)
RIGHT_NATIVE = slice(8, 16)
LEFT_HY = slice(0, 10)
RIGHT_HY = slice(10, 20)


def _safe_norm(value: Tensor, eps: float = 1e-8) -> Tensor:
    return torch.linalg.vector_norm(value, dim=-1, keepdim=True).clamp_min(eps)


def normalize_quaternion_xyzw(quaternion: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalize an ``(..., 4)`` xyzw quaternion tensor."""

    if quaternion.shape[-1] != 4:
        raise ValueError(f"Expected xyzw quaternion with last dimension 4, got {quaternion.shape}")
    return quaternion / _safe_norm(quaternion, eps)


def quaternion_xyzw_to_matrix(quaternion: Tensor) -> Tensor:
    """Convert normalized or unnormalized xyzw quaternions to rotation matrices."""

    x, y, z, w = normalize_quaternion_xyzw(quaternion).unbind(-1)
    two = quaternion.new_tensor(2.0)
    matrix = torch.stack(
        (
            1 - two * (y * y + z * z),
            two * (x * y - z * w),
            two * (x * z + y * w),
            two * (x * y + z * w),
            1 - two * (x * x + z * z),
            two * (y * z - x * w),
            two * (x * z - y * w),
            two * (y * z + x * w),
            1 - two * (x * x + y * y),
        ),
        dim=-1,
    )
    return matrix.reshape(*quaternion.shape[:-1], 3, 3)


def matrix_to_quaternion_xyzw(matrix: Tensor) -> Tensor:
    """Convert proper rotation matrices to canonicalized xyzw quaternions.

    The implementation uses the numerically stable four-candidate construction
    used by PyTorch3D and canonicalizes the sign to ``w >= 0``.
    """

    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrices (..., 3, 3), got {matrix.shape}")
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    q_abs = torch.sqrt(
        torch.clamp(
            torch.stack(
                (
                    1 + m00 + m11 + m22,
                    1 + m00 - m11 - m22,
                    1 - m00 + m11 - m22,
                    1 - m00 - m11 + m22,
                ),
                dim=-1,
            ),
            min=0,
        )
    )
    # Candidate order is w, x, y, z. Divide by twice the largest component.
    candidates = torch.stack(
        (
            torch.stack((q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01), dim=-1),
            torch.stack((m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20), dim=-1),
            torch.stack((m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21), dim=-1),
            torch.stack((m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2), dim=-1),
        ),
        dim=-2,
    )
    candidates = candidates / (2 * q_abs[..., :, None].clamp_min(1e-8))
    best = q_abs.argmax(dim=-1)
    gather_index = best[..., None, None].expand(*best.shape, 1, 4)
    quaternion_wxyz = torch.gather(candidates, -2, gather_index).squeeze(-2)
    quaternion_xyzw = quaternion_wxyz[..., (1, 2, 3, 0)]
    quaternion_xyzw = normalize_quaternion_xyzw(quaternion_xyzw)
    return torch.where(quaternion_xyzw[..., 3:4] < 0, -quaternion_xyzw, quaternion_xyzw)


def rotation_matrix_to_6d(matrix: Tensor) -> Tensor:
    """Flatten the first two rotation-matrix rows to ``(..., 6)``."""

    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrices (..., 3, 3), got {matrix.shape}")
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def rotation_6d_to_matrix(rotation_6d: Tensor, eps: float = 1e-8) -> Tensor:
    """Recover a proper matrix from the first-two-rows 6D representation."""

    if rotation_6d.shape[-1] != 6:
        raise ValueError(f"Expected rotation-6D with last dimension 6, got {rotation_6d.shape}")
    first, second = rotation_6d[..., :3], rotation_6d[..., 3:]
    row0 = first / _safe_norm(first, eps)
    second_orthogonal = second - (row0 * second).sum(dim=-1, keepdim=True) * row0
    row1 = second_orthogonal / _safe_norm(second_orthogonal, eps)
    row2 = torch.linalg.cross(row0, row1, dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def rotation_6d_to_author_absolute_matrix(rotation_6d: Tensor, eps: float = 1e-8) -> Tensor:
    """Match the author's SciPy projection used for absolute RoboTwin actions."""

    if rotation_6d.shape[-1] != 6:
        raise ValueError(f"Expected rotation-6D with last dimension 6, got {rotation_6d.shape}")
    row0 = rotation_6d[..., :3]
    row1 = rotation_6d[..., 3:]
    row2 = torch.linalg.cross(
        row0 / _safe_norm(row0, eps),
        row1 / _safe_norm(row1, eps),
        dim=-1,
    )
    row2 = row2 / _safe_norm(row2, eps)
    matrix = torch.stack((row0, row1, row2), dim=-2)
    left, _, right = torch.linalg.svd(matrix)
    projected = left @ right
    determinant = torch.linalg.det(projected)
    correction = torch.ones(*determinant.shape, 3, device=matrix.device, dtype=matrix.dtype)
    correction[..., -1] = torch.where(determinant < 0, -1, 1)
    return (left * correction.unsqueeze(-2)) @ right


def _reorder_quaternion(pose: Tensor, source: str, destination: str) -> Tensor:
    if source == destination:
        return pose
    if {source, destination} != {"xyzw", "wxyz"}:
        raise ValueError(f"Quaternion order must be 'xyzw' or 'wxyz', got {source!r} -> {destination!r}")
    result = pose.clone()
    if source == "wxyz":
        result[..., 3:7] = pose[..., (4, 5, 6, 3)]
        result[..., 11:15] = pose[..., (12, 13, 14, 11)]
    else:
        result[..., 3:7] = pose[..., (6, 3, 4, 5)]
        result[..., 11:15] = pose[..., (14, 11, 12, 13)]
    return result


def dual_native_to_hy(pose: Tensor, quaternion_order: str = "xyzw") -> Tensor:
    """Convert a dual-arm 16D PosQuat pose to Hy-VLA's 20D representation."""

    if pose.shape[-1] != DUAL_NATIVE_DIM:
        raise ValueError(f"Expected dual-arm native pose dimension 16, got {pose.shape}")
    pose = _reorder_quaternion(pose, quaternion_order, "xyzw")
    left_matrix = quaternion_xyzw_to_matrix(pose[..., 3:7])
    right_matrix = quaternion_xyzw_to_matrix(pose[..., 11:15])
    return torch.cat(
        (
            pose[..., :3],
            rotation_matrix_to_6d(left_matrix),
            pose[..., 7:8],
            pose[..., 8:11],
            rotation_matrix_to_6d(right_matrix),
            pose[..., 15:16],
        ),
        dim=-1,
    )


def dual_hy_to_native(pose: Tensor, quaternion_order: str = "xyzw") -> Tensor:
    """Inverse of :func:`dual_native_to_hy`."""

    if pose.shape[-1] != DUAL_HY_DIM:
        raise ValueError(f"Expected Hy dual-arm pose dimension 20, got {pose.shape}")
    xyzw = torch.cat(
        (
            pose[..., :3],
            matrix_to_quaternion_xyzw(rotation_6d_to_matrix(pose[..., 3:9])),
            pose[..., 9:10],
            pose[..., 10:13],
            matrix_to_quaternion_xyzw(rotation_6d_to_matrix(pose[..., 13:19])),
            pose[..., 19:20],
        ),
        dim=-1,
    )
    return _reorder_quaternion(xyzw, "xyzw", quaternion_order)


def dual_hy_absolute_to_native(pose: Tensor, quaternion_order: str = "xyzw") -> Tensor:
    """Decode absolute tokens using the author's non-orthogonal SciPy projection."""

    if pose.shape[-1] != DUAL_HY_DIM:
        raise ValueError(f"Expected Hy dual-arm pose dimension 20, got {pose.shape}")
    xyzw = torch.cat(
        (
            pose[..., :3],
            matrix_to_quaternion_xyzw(rotation_6d_to_author_absolute_matrix(pose[..., 3:9])),
            pose[..., 9:10],
            pose[..., 10:13],
            matrix_to_quaternion_xyzw(rotation_6d_to_author_absolute_matrix(pose[..., 13:19])),
            pose[..., 19:20],
        ),
        dim=-1,
    )
    return _reorder_quaternion(xyzw, "xyzw", quaternion_order)


def _pose_to_transform(single_arm_pose_xyzw: Tensor) -> Tensor:
    transform = torch.zeros(
        *single_arm_pose_xyzw.shape[:-1],
        4,
        4,
        dtype=single_arm_pose_xyzw.dtype,
        device=single_arm_pose_xyzw.device,
    )
    transform[..., :3, :3] = quaternion_xyzw_to_matrix(single_arm_pose_xyzw[..., 3:7])
    transform[..., :3, 3] = single_arm_pose_xyzw[..., :3]
    transform[..., 3, 3] = 1
    return transform


def _relative_single_arm(sequence_xyzw: Tensor, reference_xyzw: Tensor) -> Tensor:
    reference_transform = _pose_to_transform(reference_xyzw)
    sequence_transform = _pose_to_transform(sequence_xyzw)
    rotation = reference_transform[..., :3, :3].transpose(-1, -2)
    translation = -(rotation @ reference_transform[..., :3, 3:4]).squeeze(-1)
    inverse = torch.zeros_like(reference_transform)
    inverse[..., :3, :3] = rotation
    inverse[..., :3, 3] = translation
    inverse[..., 3, 3] = 1
    relative = inverse.unsqueeze(-3) @ sequence_transform
    return torch.cat(
        (relative[..., :3, 3], rotation_matrix_to_6d(relative[..., :3, :3])),
        dim=-1,
    )


def dual_relative_to_reference(sequence: Tensor, reference: Tensor, quaternion_order: str = "xyzw") -> Tensor:
    """Encode a dual-arm native sequence relative to an explicit reference.

    ``sequence`` is ``(..., T, 16)`` and ``reference`` is ``(..., 16)``.
    The output is ``(..., T, 20)`` with each arm expressed in its reference
    wrist frame; grippers remain absolute and left/right order is preserved.
    """

    sequence = _reorder_quaternion(sequence, quaternion_order, "xyzw")
    reference = _reorder_quaternion(reference, quaternion_order, "xyzw")
    left = _relative_single_arm(sequence[..., :7], reference[..., :7])
    right = _relative_single_arm(sequence[..., 8:15], reference[..., 8:15])
    return torch.cat((left, sequence[..., 7:8], right, sequence[..., 15:16]), dim=-1)


def dual_relative_to_first(sequence: Tensor, quaternion_order: str = "xyzw") -> Tensor:
    """Encode every action relative to the first frame of its chunk."""

    return dual_relative_to_reference(sequence, sequence[..., 0, :], quaternion_order)


def dual_relative_to_current(sequence: Tensor, current: Tensor, quaternion_order: str = "xyzw") -> Tensor:
    """Encode every action relative to the current observation."""

    return dual_relative_to_reference(sequence, current, quaternion_order)


def _decode_relative_single_arm(relative: Tensor, reference_xyzw: Tensor) -> Tensor:
    reference_transform = _pose_to_transform(reference_xyzw)
    delta = torch.zeros(*relative.shape[:-1], 4, 4, dtype=relative.dtype, device=relative.device)
    delta[..., :3, :3] = rotation_6d_to_matrix(relative[..., 3:9])
    delta[..., :3, 3] = relative[..., :3]
    delta[..., 3, 3] = 1
    absolute = reference_transform.unsqueeze(-3) @ delta
    return torch.cat(
        (absolute[..., :3, 3], matrix_to_quaternion_xyzw(absolute[..., :3, :3])),
        dim=-1,
    )


def dual_relative_to_native(relative: Tensor, reference: Tensor, quaternion_order: str = "xyzw") -> Tensor:
    """Decode a 20D relative chunk against a dual-arm native reference."""

    if relative.shape[-1] != DUAL_HY_DIM:
        raise ValueError(f"Expected relative action dimension 20, got {relative.shape}")
    reference_xyzw = _reorder_quaternion(reference, quaternion_order, "xyzw")
    left = _decode_relative_single_arm(relative[..., :9], reference_xyzw[..., :7])
    right = _decode_relative_single_arm(relative[..., 10:19], reference_xyzw[..., 8:15])
    result = torch.cat(
        (left, relative[..., 9:10], right, relative[..., 19:20]),
        dim=-1,
    )
    return _reorder_quaternion(result, "xyzw", quaternion_order)


_ROBO_TO_UMI_WORLD = torch.tensor(((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
_ROBO_TO_UMI_LOCAL = torch.tensor(((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))


def transform_robotwin_umi(
    pose: Tensor,
    *,
    inverse: bool = False,
    convert_gripper: bool = False,
    quaternion_order: str = "xyzw",
) -> Tensor:
    """Transform dual-arm native RoboTwin coordinates to/from UMI coordinates."""

    xyzw = _reorder_quaternion(pose, quaternion_order, "xyzw")
    world = _ROBO_TO_UMI_WORLD.to(device=pose.device, dtype=pose.dtype)
    local = _ROBO_TO_UMI_LOCAL.to(device=pose.device, dtype=pose.dtype)
    if inverse:
        world, local = world.transpose(-1, -2), local.transpose(-1, -2)
    result = xyzw.clone()
    for pos_slice, quat_slice, grip_index in (
        (slice(0, 3), slice(3, 7), 7),
        (slice(8, 11), slice(11, 15), 15),
    ):
        result[..., pos_slice] = (world @ xyzw[..., pos_slice].unsqueeze(-1)).squeeze(-1)
        rotation = quaternion_xyzw_to_matrix(xyzw[..., quat_slice])
        result[..., quat_slice] = matrix_to_quaternion_xyzw(world @ rotation @ local)
        if convert_gripper:
            result[..., grip_index] = (
                1 - xyzw[..., grip_index] / 90 if inverse else (1 - xyzw[..., grip_index]) * 90
            )
    return _reorder_quaternion(result, "xyzw", quaternion_order)


def pad_with_mask(value: Tensor, dimension: int) -> tuple[Tensor, Tensor]:
    """Right-pad the last dimension and return an equally shaped validity mask."""

    if value.shape[-1] > dimension:
        raise ValueError(f"Cannot pad dimension {value.shape[-1]} to smaller size {dimension}")
    output = value.new_zeros(*value.shape[:-1], dimension)
    mask = torch.zeros_like(output, dtype=torch.bool)
    output[..., : value.shape[-1]] = value
    mask[..., : value.shape[-1]] = True
    return output, mask


def mean_std_normalize(value: Tensor, mean: Tensor, std: Tensor, eps: float = 1e-8) -> Tensor:
    """Mean/std normalize, mapping zero-variance dimensions deterministically to zero."""

    mean, std = mean.to(value), std.to(value)
    valid = std.abs() > eps
    return torch.where(valid, (value - mean) / torch.where(valid, std, torch.ones_like(std)), 0)


def mean_std_unnormalize(value: Tensor, mean: Tensor, std: Tensor, eps: float = 1e-8) -> Tensor:
    """Inverse of :func:`mean_std_normalize`; constant dimensions return their mean."""

    dtype = torch.promote_types(value.dtype, torch.promote_types(mean.dtype, std.dtype))
    value = value.to(dtype=dtype)
    mean = mean.to(device=value.device, dtype=dtype)
    std = std.to(device=value.device, dtype=dtype)
    valid = std.abs() > eps
    return torch.where(valid, value * std + mean, mean)


def _identity_stats(horizon: int) -> dict[str, Tensor]:
    return {
        "qpos_mean": torch.zeros(DUAL_HY_DIM),
        "qpos_std": torch.ones(DUAL_HY_DIM),
        "action_mean": torch.zeros(horizon, DUAL_HY_DIM),
        "action_std": torch.ones(horizon, DUAL_HY_DIM),
    }


def _as_tensor_stats(stats: dict[str, Any] | None, horizon: int) -> dict[str, Tensor]:
    output = _identity_stats(horizon)
    if stats is None:
        return output
    aliases = {"act_mean": "action_mean", "act_std": "action_std"}
    for key, value in stats.items():
        destination = aliases.get(key, key)
        if destination in {
            "qpos_mean",
            "qpos_std",
            "action_mean",
            "action_std",
            "action_mean_abs",
            "action_std_abs",
        }:
            output[destination] = torch.as_tensor(value)
    return output


class _HyVLAStatsMixin:
    qpos_mean: Tensor | None
    qpos_std: Tensor | None
    action_mean: Tensor | None
    action_std: Tensor | None
    action_mean_abs: Tensor | None
    action_std_abs: Tensor | None

    def set_stats(self, stats: dict[str, Any]) -> None:
        for name in (
            "qpos_mean",
            "qpos_std",
            "action_mean",
            "action_std",
            "action_mean_abs",
            "action_std_abs",
        ):
            value = stats.get(name)
            setattr(self, name, None if value is None else torch.as_tensor(value))

    def state_dict(self) -> dict[str, Tensor]:
        return {
            name: value
            for name in (
                "qpos_mean",
                "qpos_std",
                "action_mean",
                "action_std",
                "action_mean_abs",
                "action_std_abs",
            )
            if (value := getattr(self, name)) is not None
        }

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        self.set_stats(state)

    def _require_stats(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        values = (self.qpos_mean, self.qpos_std, self.action_mean, self.action_std)
        if any(value is None for value in values):
            raise RuntimeError("Hy-VLA processor normalization state was not loaded.")
        return values  # type: ignore[return-value]


@ProcessorStepRegistry.register(name="hy_vla_encode_v1")
@dataclass
class HyVLAEncodeStep(_HyVLAStatsMixin, ProcessorStep):
    """Convert native dual-arm data into the normalized released representation."""

    embodiment: str = "umi_dual_arm"
    action_representation: str = "relative"
    relative_convention: str = "first_frame"
    quaternion_order: str = "xyzw"
    coordinate_transform: str = "identity"
    convert_gripper: bool = False
    max_state_dim: int = 32
    max_action_dim: int = 32
    model_action_dim: int = DUAL_HY_DIM
    epsilon: float = 1e-8
    qpos_mean: Tensor | None = field(default=None, repr=False)
    qpos_std: Tensor | None = field(default=None, repr=False)
    action_mean: Tensor | None = field(default=None, repr=False)
    action_std: Tensor | None = field(default=None, repr=False)
    action_mean_abs: Tensor | None = field(default=None, repr=False)
    action_std_abs: Tensor | None = field(default=None, repr=False)
    latest_native_reference: Tensor | None = field(default=None, init=False, repr=False)

    def get_config(self) -> dict[str, Any]:
        return {
            "embodiment": self.embodiment,
            "action_representation": self.action_representation,
            "relative_convention": self.relative_convention,
            "quaternion_order": self.quaternion_order,
            "coordinate_transform": self.coordinate_transform,
            "convert_gripper": self.convert_gripper,
            "max_state_dim": self.max_state_dim,
            "max_action_dim": self.max_action_dim,
            "model_action_dim": self.model_action_dim,
            "epsilon": self.epsilon,
        }

    def _to_model_frame(self, value: Tensor) -> Tensor:
        if self.coordinate_transform == "robotwin_to_umi":
            return transform_robotwin_umi(
                value,
                convert_gripper=self.convert_gripper,
                quaternion_order=self.quaternion_order,
            )
        return value

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = dict(transition.get(TransitionKey.OBSERVATION) or {})
        state = observation.get(OBS_STATE)
        if not isinstance(state, Tensor):
            raise ValueError("Hy-VLA requires a tensor observation.state.")
        qpos_mean, qpos_std, action_mean, action_std = self._require_stats()

        if state.shape[-1] == DUAL_NATIVE_DIM:
            self.latest_native_reference = state.detach().clone()
            model_state = dual_native_to_hy(
                self._to_model_frame(state), quaternion_order=self.quaternion_order
            )
        elif state.shape[-1] in {DUAL_HY_DIM, self.max_state_dim}:
            model_state = state[..., :DUAL_HY_DIM]
            self.latest_native_reference = None
        else:
            raise ValueError(f"Hy-VLA cannot map state shape {state.shape} for {self.embodiment}.")
        model_state = mean_std_normalize(model_state, qpos_mean, qpos_std, self.epsilon)
        model_state, state_mask = pad_with_mask(model_state, self.max_state_dim)
        padded_state = model_state
        observation[OBS_STATE] = padded_state
        observation[f"{OBS_STATE}.mask"] = state_mask
        transition[TransitionKey.OBSERVATION] = observation

        action = transition.get(TransitionKey.ACTION)
        if action is not None:
            if not isinstance(action, Tensor):
                raise ValueError("Hy-VLA requires tensor actions.")
            if action.ndim == 2 and state.ndim == 2 and state.shape[0] == 1:
                # The generic batch step intentionally treats only 1D actions
                # as single actions. Hy-VLA's single training sample is a TD
                # chunk, so add its batch dimension explicitly.
                action = action.unsqueeze(0)
            if action.shape[-1] == DUAL_NATIVE_DIM:
                native_action = self._to_model_frame(action)
                if self.relative_convention == "first_frame":
                    relative = dual_relative_to_first(native_action, quaternion_order=self.quaternion_order)
                else:
                    current = self._to_model_frame(self.latest_native_reference)
                    relative = dual_relative_to_current(
                        native_action, current, quaternion_order=self.quaternion_order
                    )
                normalized = mean_std_normalize(relative, action_mean, action_std, self.epsilon)
                if self.action_representation == "relative_absolute":
                    if self.action_mean_abs is None or self.action_std_abs is None:
                        raise RuntimeError("relative_absolute action stats were not loaded.")
                    absolute = dual_native_to_hy(native_action, quaternion_order=self.quaternion_order)
                    normalized_abs = mean_std_normalize(
                        absolute, self.action_mean_abs, self.action_std_abs, self.epsilon
                    )
                    normalized = torch.cat((normalized, normalized_abs), dim=-2)
            elif action.shape[-1] in {DUAL_HY_DIM, self.max_action_dim}:
                normalized = action[..., :DUAL_HY_DIM]
            else:
                raise ValueError(f"Hy-VLA does not map action shape {action.shape} for {self.embodiment}.")
            padded_action, action_mask = pad_with_mask(normalized, self.max_action_dim)
            transition[TransitionKey.ACTION] = padded_action
            complementary = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
            complementary[f"{ACTION}.mask"] = action_mask
            transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def reset(self) -> None:
        self.latest_native_reference = None


def _blend_native_pose(relative: Tensor, absolute: Tensor, weight: float) -> Tensor:
    """Blend dual native poses, using the exact half-way slerp shortcut for quaternions."""

    output = torch.lerp(relative, absolute, weight)
    for quat_slice in (slice(3, 7), slice(11, 15)):
        first = relative[..., quat_slice]
        second = absolute[..., quat_slice]
        second = torch.where((first * second).sum(-1, keepdim=True) < 0, -second, second)
        if weight == 0.5:
            quaternion = first + second
        else:
            dot = (first * second).sum(-1, keepdim=True).clamp(-1, 1)
            theta = torch.acos(dot)
            sin_theta = torch.sin(theta)
            linear = (1 - weight) * first + weight * second
            spherical = (
                torch.sin((1 - weight) * theta) / sin_theta.clamp_min(1e-8) * first
                + torch.sin(weight * theta) / sin_theta.clamp_min(1e-8) * second
            )
            quaternion = torch.where(sin_theta.abs() < 1e-6, linear, spherical)
        output[..., quat_slice] = quaternion / torch.linalg.vector_norm(
            quaternion, dim=-1, keepdim=True
        ).clamp_min(1e-8)
    return output


@ProcessorStepRegistry.register(name="hy_vla_decode_v1")
@dataclass
class HyVLADecodeStep(_HyVLAStatsMixin, ProcessorStep):
    """Inverse-normalize and decode queued Hy-VLA actions to native dual-arm poses."""

    embodiment: str = "umi_dual_arm"
    physical_horizon: int = 50
    execution_horizon: int = 50
    action_representation: str = "relative"
    action_decode_mode: str = "relative"
    blend_weight: float = 0.5
    quaternion_order: str = "xyzw"
    coordinate_transform: str = "identity"
    convert_gripper: bool = False
    model_action_dim: int = DUAL_HY_DIM
    epsilon: float = 1e-8
    qpos_mean: Tensor | None = field(default=None, repr=False)
    qpos_std: Tensor | None = field(default=None, repr=False)
    action_mean: Tensor | None = field(default=None, repr=False)
    action_std: Tensor | None = field(default=None, repr=False)
    action_mean_abs: Tensor | None = field(default=None, repr=False)
    action_std_abs: Tensor | None = field(default=None, repr=False)
    encoder: HyVLAEncodeStep | None = field(default=None, repr=False)
    _reference: Tensor | None = field(default=None, init=False, repr=False)
    _index: int = field(default=0, init=False, repr=False)

    def get_config(self) -> dict[str, Any]:
        return {
            "embodiment": self.embodiment,
            "physical_horizon": self.physical_horizon,
            "execution_horizon": self.execution_horizon,
            "action_representation": self.action_representation,
            "action_decode_mode": self.action_decode_mode,
            "blend_weight": self.blend_weight,
            "quaternion_order": self.quaternion_order,
            "coordinate_transform": self.coordinate_transform,
            "convert_gripper": self.convert_gripper,
            "model_action_dim": self.model_action_dim,
            "epsilon": self.epsilon,
        }

    def _select_reference(self, action: Tensor) -> Tensor:
        if self._index == 0:
            if self.encoder is None or self.encoder.latest_native_reference is None:
                raise RuntimeError(
                    "Hy-VLA relative action decoding requires a preceding native 16D "
                    "state and a connected preprocessor."
                )
            self._reference = self.encoder.latest_native_reference.to(action)
            if self.coordinate_transform == "robotwin_to_umi":
                self._reference = transform_robotwin_umi(
                    self._reference,
                    convert_gripper=self.convert_gripper,
                    quaternion_order=self.quaternion_order,
                )
        if self._reference is None:
            raise RuntimeError("Hy-VLA decoder reference is unavailable.")
        return self._reference

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, Tensor):
            raise ValueError("Hy-VLA postprocessor expects a tensor policy action.")
        squeeze_batch = action.ndim == 1
        if squeeze_batch:
            action = action.unsqueeze(0)
        _, _, action_mean, action_std = self._require_stats()
        stats_index = min(self._index, action_mean.shape[-2] - 1)

        if self.action_representation == "relative_absolute":
            if action.shape[-1] != 2 * DUAL_HY_DIM:
                raise ValueError(
                    "A relative_absolute Hy-VLA policy must emit paired 40D rel/abs tokens, "
                    f"got {action.shape}."
                )
            relative_token, absolute_token = action[..., :DUAL_HY_DIM], action[..., DUAL_HY_DIM:]
        else:
            relative_token, absolute_token = action[..., :DUAL_HY_DIM], None

        reference = self._select_reference(action)
        relative = mean_std_unnormalize(
            relative_token, action_mean[stats_index], action_std[stats_index], self.epsilon
        )
        reference = reference.to(relative)
        relative_native = dual_relative_to_native(
            relative.unsqueeze(-2), reference, quaternion_order=self.quaternion_order
        ).squeeze(-2)
        decoded = relative_native

        if absolute_token is not None and self.action_decode_mode in {"absolute", "blend"}:
            if self.action_mean_abs is None or self.action_std_abs is None:
                raise RuntimeError("Absolute Hy-VLA normalization stats were not loaded.")
            absolute = mean_std_unnormalize(
                absolute_token,
                self.action_mean_abs[stats_index],
                self.action_std_abs[stats_index],
                self.epsilon,
            )
            absolute_native = dual_hy_absolute_to_native(absolute, quaternion_order=self.quaternion_order)
            decoded = (
                absolute_native
                if self.action_decode_mode == "absolute"
                else _blend_native_pose(relative_native, absolute_native, self.blend_weight)
            )

        if self.coordinate_transform == "robotwin_to_umi":
            decoded = transform_robotwin_umi(
                decoded,
                inverse=True,
                convert_gripper=self.convert_gripper,
                quaternion_order=self.quaternion_order,
            )
        if squeeze_batch:
            decoded = decoded.squeeze(0)
        transition[TransitionKey.ACTION] = decoded
        self._index = (self._index + 1) % self.execution_horizon
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def reset(self) -> None:
        self._reference = None
        self._index = 0


def reconnect_hy_vla_processors(
    preprocessor: PolicyProcessorPipeline, postprocessor: PolicyProcessorPipeline
) -> None:
    """Reconnect the decoder's non-serializable reference to the encoder."""

    encoder = next((step for step in preprocessor.steps if isinstance(step, HyVLAEncodeStep)), None)
    decoder = next((step for step in postprocessor.steps if isinstance(step, HyVLADecodeStep)), None)
    if decoder is not None:
        decoder.encoder = encoder


def make_hy_vla_pre_post_processors(
    config: HyVLAConfig,
    dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    norm_stats: dict[str, Any] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build serializable Hy-VLA processors.

    Checkpoint preparation passes the author's normalization data through
    ``norm_stats``. Runtime reloads the resulting safe tensor state and never
    needs to unpickle normalization data.
    """

    if norm_stats is None and dataset_stats:
        # This intentionally accepts only the author-style schema. Generic
        # per-feature LeRobot stats cannot reproduce time-dependent action stats.
        norm_stats = dataset_stats if "qpos_mean" in dataset_stats else None  # type: ignore[assignment]
    stats = _as_tensor_stats(norm_stats, config.physical_action_horizon)
    encoder = HyVLAEncodeStep(
        embodiment=config.embodiment,
        action_representation=config.action_representation,
        relative_convention=config.relative_convention,
        quaternion_order=config.native_quaternion_order,
        coordinate_transform=config.coordinate_transform,
        convert_gripper=config.convert_gripper,
        max_state_dim=config.max_state_dim,
        max_action_dim=config.max_action_dim,
        model_action_dim=config.model_action_dim,
        epsilon=config.zero_variance_epsilon,
    )
    decoder = HyVLADecodeStep(
        embodiment=config.embodiment,
        physical_horizon=config.physical_action_horizon,
        execution_horizon=config.execution_horizon,
        action_representation=config.action_representation,
        action_decode_mode=config.action_decode_mode,
        blend_weight=config.relative_absolute_blend_weight,
        quaternion_order=config.native_quaternion_order,
        coordinate_transform=config.coordinate_transform,
        convert_gripper=config.convert_gripper,
        model_action_dim=config.model_action_dim,
        epsilon=config.zero_variance_epsilon,
        encoder=encoder,
    )
    encoder.set_stats(stats)
    decoder.set_stats(stats)
    steps = make_default_policy_processor_steps(config, dataset_stats=None)
    preprocessor, postprocessor = make_policy_processor_pipelines(
        input_steps=[
            RenderMessagesStep(config.recipe, render_training=False),
            steps.rename_observations,
            steps.add_batch_dim,
            encoder,
            steps.to_device,
        ],
        output_steps=[decoder, steps.to_cpu],
    )
    return preprocessor, postprocessor


__all__ = [
    "DUAL_HY_DIM",
    "DUAL_NATIVE_DIM",
    "HyVLADecodeStep",
    "HyVLAEncodeStep",
    "dual_hy_absolute_to_native",
    "dual_hy_to_native",
    "dual_native_to_hy",
    "dual_relative_to_current",
    "dual_relative_to_first",
    "dual_relative_to_native",
    "matrix_to_quaternion_xyzw",
    "make_hy_vla_pre_post_processors",
    "mean_std_normalize",
    "mean_std_unnormalize",
    "pad_with_mask",
    "quaternion_xyzw_to_matrix",
    "reconnect_hy_vla_processors",
    "rotation_6d_to_author_absolute_matrix",
    "rotation_6d_to_matrix",
    "rotation_matrix_to_6d",
    "transform_robotwin_umi",
]
