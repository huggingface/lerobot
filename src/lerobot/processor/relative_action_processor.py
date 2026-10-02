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
    "to_relative_poses",
    "to_absolute_poses",
]

RELATIVE_ACTION_MODES = ("subtract", "pose")
ROTATION_NAMES = {"axis_angle": ["ax", "ay", "az"], "rot6d": [f"r6d_{i}" for i in range(6)]}


def _decode_rotation(rot: Tensor, rotation_format: str) -> Tensor:
    """(..., 3) axis-angle or (..., 6) rot6d (first two matrix rows) -> (..., 3, 3) rotation matrices."""
    if rotation_format == "rot6d":
        b1 = torch.nn.functional.normalize(rot[..., :3], dim=-1)
        b2 = rot[..., 3:] - (b1 * rot[..., 3:]).sum(-1, keepdim=True) * b1
        b2 = torch.nn.functional.normalize(b2, dim=-1)
        return torch.stack([b1, b2, torch.cross(b1, b2, dim=-1)], dim=-2)
    angle = rot.norm(dim=-1, keepdim=True)
    x, y, z = (rot / angle.clamp_min(1e-12)).unbind(-1)
    zero = torch.zeros_like(x)
    k = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1).reshape(*rot.shape[:-1], 3, 3)
    eye = torch.eye(3, dtype=rot.dtype, device=rot.device)
    return eye + torch.sin(angle)[..., None] * k + (1 - torch.cos(angle))[..., None] * (k @ k)


def _encode_rotation(matrix: Tensor, rotation_format: str) -> Tensor:
    """Inverse of ``_decode_rotation``."""
    if rotation_format == "rot6d":
        return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)
    m = matrix.reshape(*matrix.shape[:-2], 9).unbind(-1)
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = m
    q_abs = (
        torch.stack(
            [1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22], dim=-1
        )
        .clamp_min(0)
        .sqrt()
    )
    # Each row is a (w, x, y, z) quaternion estimate; keep the best-conditioned one.
    candidates = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    ) / (2 * q_abs[..., None].clamp_min(0.1))
    best = q_abs.argmax(dim=-1)[..., None, None].expand(*q_abs.shape[:-1], 1, 4)
    quat = candidates.gather(-2, best).squeeze(-2)
    quat = torch.where(quat[..., :1] < 0, -quat, quat)
    w, xyz = quat[..., :1], quat[..., 1:]
    sin_half = xyz.norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(sin_half, w)
    scale = torch.where(sin_half > 1e-8, angle / sin_half.clamp_min(1e-8), 2 / w.clamp_min(1e-8))
    return xyz * scale


def _split_pose_reference(reference: Tensor, actions: Tensor, rotation_format: str) -> tuple[Tensor, Tensor]:
    reference = reference.to(device=actions.device, dtype=actions.dtype)
    if reference.ndim == 3:
        reference = reference[:, 0]
    position, rotation = reference[..., :3], _decode_rotation(reference[..., 3:], rotation_format)
    if actions.ndim == 3:
        position, rotation = position.unsqueeze(-2), rotation.unsqueeze(-3)
    return position, rotation


def to_relative_poses(
    actions: Tensor, reference: Tensor, pos_idx: list[int], rot_idx: list[int], rotation_format: str
) -> Tensor:
    """Express the pose dims of ``actions`` in the reference frame: R_refᵀ(p - p_ref), R_refᵀR.

    Args:
        actions: (B, T, action_dim) or (B, action_dim).
        reference: (B, 3 + rot_dim) laid out as [position, rotation], or (B, T_obs, 3 + rot_dim)
            collapsed to the current frame.
        pos_idx / rot_idx: Action dims holding the position (3) and rotation (3 or 6).
        rotation_format: "axis_angle" or "rot6d", shared by actions and reference.
    """
    p_ref, r_ref = _split_pose_reference(reference, actions, rotation_format)
    r_ref_t = r_ref.transpose(-1, -2)
    out = actions.clone()
    out[..., pos_idx] = (r_ref_t @ (actions[..., pos_idx] - p_ref).unsqueeze(-1)).squeeze(-1)
    rotation = r_ref_t @ _decode_rotation(actions[..., rot_idx], rotation_format)
    out[..., rot_idx] = _encode_rotation(rotation, rotation_format)
    return out


def to_absolute_poses(
    actions: Tensor, reference: Tensor, pos_idx: list[int], rot_idx: list[int], rotation_format: str
) -> Tensor:
    """Inverse of ``to_relative_poses``: p_ref + R_ref p, R_ref R."""
    p_ref, r_ref = _split_pose_reference(reference, actions, rotation_format)
    out = actions.clone()
    out[..., pos_idx] = (r_ref @ actions[..., pos_idx].unsqueeze(-1)).squeeze(-1) + p_ref
    rotation = r_ref @ _decode_rotation(actions[..., rot_idx], rotation_format)
    out[..., rot_idx] = _encode_rotation(rotation, rotation_format)
    return out


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
    # A temporally stacked observation (a policy whose ``observation_delta_indices`` spans several
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
    # A temporally stacked observation (a policy whose ``observation_delta_indices`` spans several
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
        mode: "subtract" (action - state on masked dims) or "pose" (SE(3) composition on
            the named pose dims, every other dim stays absolute; exclude_joints is ignored).
        position_names: Pose mode, the 3 position dims (exact, case-insensitive match).
        rotation_names: Pose mode, the rotation dims; None uses ``ROTATION_NAMES[rotation_format]``.
        rotation_format: Pose mode, "axis_angle" or "rot6d".
        reference_key: Pose mode, observation key holding the reference pose ([position, rotation]
            or laid out like the action). It is removed from the observation unless it is
            ``observation.state``. None uses the chunk's first action during training and
            leaves predictions relative at inference.
    """

    enabled: bool = False
    exclude_joints: list[str] = field(default_factory=list)
    action_names: list[str] | None = None
    mode: str = "subtract"
    position_names: list[str] = field(default_factory=lambda: ["x", "y", "z"])
    rotation_names: list[str] | None = None
    rotation_format: str = "axis_angle"
    reference_key: str | None = None
    _last_state: torch.Tensor | None = field(default=None, init=False, repr=False)
    _count_queued_actions: Callable[[], int] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in RELATIVE_ACTION_MODES:
            raise ValueError(f"mode must be one of {RELATIVE_ACTION_MODES}, got {self.mode!r}")
        if self.rotation_format not in ROTATION_NAMES:
            raise ValueError(
                f"rotation_format must be one of {list(ROTATION_NAMES)}, got {self.rotation_format!r}"
            )

    def _pose_indices(self) -> tuple[list[int], list[int]]:
        rotation_names = self.rotation_names or ROTATION_NAMES[self.rotation_format]
        if len(self.position_names) != 3 or len(rotation_names) != len(ROTATION_NAMES[self.rotation_format]):
            raise ValueError(
                f"Pose mode needs 3 position names and {len(ROTATION_NAMES[self.rotation_format])} "
                f"{self.rotation_format} rotation names, got {self.position_names} / {rotation_names}"
            )
        if self.action_names is None:
            raise ValueError("Pose mode needs action_names to locate the pose dims")
        names = [str(name).lower() for name in self.action_names]
        try:
            return (
                [names.index(name.lower()) for name in self.position_names],
                [names.index(name.lower()) for name in rotation_names],
            )
        except ValueError as e:
            raise ValueError(
                f"Pose names {self.position_names} / {rotation_names} not all in action names {self.action_names}"
            ) from e

    def _pose_reference(self, state: Tensor) -> Tensor:
        pos_idx, rot_idx = self._pose_indices()
        if state.shape[-1] == len(pos_idx) + len(rot_idx):
            return state
        return state[..., pos_idx + rot_idx]

    def stats_signature(self, chunk_size: int) -> dict[str, Any]:
        """Settings the relative action stats depend on, stored next to ``stats.json``."""
        if self.mode == "pose":
            return {
                "mode": self.mode,
                "chunk_size": chunk_size,
                "position_names": list(self.position_names),
                "rotation_names": list(self.rotation_names or ROTATION_NAMES[self.rotation_format]),
                "rotation_format": self.rotation_format,
                "reference_key": self.reference_key,
            }
        return {"mode": self.mode, "chunk_size": chunk_size, "exclude_joints": list(self.exclude_joints)}

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
        anchor_key = self.reference_key if self.mode == "pose" else OBS_STATE
        state = observation.get(anchor_key) if observation and anchor_key else None

        # Cache state for the paired AbsoluteActionsProcessorStep -- but hold it for as long
        # as the policy is still serving the chunk that was generated against it. A fresh
        # chunk re-anchors on the tick the queue runs dry.
        if state is not None and not self._chunk_in_flight():
            self._last_state = state

        if not self.enabled:
            return transition

        new_transition = transition.copy()
        if self.mode == "pose" and observation and anchor_key not in (None, OBS_STATE):
            new_transition[TransitionKey.OBSERVATION] = {
                k: v for k, v in observation.items() if k != anchor_key
            }
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            return new_transition
        if not isinstance(action, torch.Tensor):
            raise ValueError(f"RelativeActionsProcessorStep expects a tensor action, got {type(action)}")

        if self.mode == "pose":
            pos_idx, rot_idx = self._pose_indices()
            if state is not None:
                reference = self._pose_reference(state)
            elif action.ndim == 3:
                reference = action[:, 0, pos_idx + rot_idx]
            else:
                return new_transition
            new_transition[TransitionKey.ACTION] = to_relative_poses(
                action, reference, pos_idx, rot_idx, self.rotation_format
            )
            return new_transition

        if state is None:
            return new_transition
        mask = self._build_mask(action.shape[-1])
        new_transition[TransitionKey.ACTION] = to_relative_actions(action, state, mask)
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
            "mode": self.mode,
            "position_names": self.position_names,
            "rotation_names": self.rotation_names,
            "rotation_format": self.rotation_format,
            "reference_key": self.reference_key,
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
        step = self.relative_step
        # Without a reference pose, predictions stay relative; the robot side chains them.
        if step.mode == "pose" and step.reference_key is None:
            return transition

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

        if step.mode == "pose":
            pos_idx, rot_idx = step._pose_indices()
            new_transition[TransitionKey.ACTION] = to_absolute_poses(
                action, step._pose_reference(cached_state), pos_idx, rot_idx, step.rotation_format
            )
            return new_transition
        mask = step._build_mask(action.shape[-1])
        new_transition[TransitionKey.ACTION] = to_absolute_actions(action, cached_state, mask)
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
