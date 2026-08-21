from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from .action_semantics import LiberoSafetyActionSemantics


@dataclass
class TrajectoryGeometryTarget:
    translation_goal: Tensor
    approach_direction: Tensor
    translation_magnitude: Tensor
    rotation_goal: Tensor | None
    gripper_transition: Tensor
    valid_mask: Tensor


class TrajectoryGeometryTargetBuilder:
    def __init__(self, action_semantics=None, motion_threshold=1e-5, require_physical_scale=True):
        self.action_semantics = action_semantics or LiberoSafetyActionSemantics()
        self.motion_threshold = motion_threshold
        self.require_physical_scale = require_physical_scale

    def build(
        self,
        action_chunk: Tensor,
        state: Tensor,
        action_stats: Mapping[str, Tensor] | None,
        padding_mask: Tensor | None,
    ):
        del state  # Reserved for future robot-frame transforms; no target is inferred from state.
        physical = self.action_semantics.denormalize_actions(action_chunk, action_stats)
        scale_valid = physical is not None
        if physical is None:
            physical = action_chunk
        valid_steps = torch.ones(action_chunk.shape[:2], dtype=torch.bool, device=action_chunk.device)
        if padding_mask is not None:
            valid_steps &= ~padding_mask.bool()

        translation = self.action_semantics.translation_delta(physical) * valid_steps.unsqueeze(-1)
        goal = translation.sum(dim=1)
        magnitude = goal.norm(dim=-1, keepdim=True)
        has_steps = valid_steps.any(dim=1, keepdim=True)

        gripper = self.action_semantics.gripper_command(physical)
        first_index = valid_steps.float().argmax(dim=1)
        last_index = valid_steps.long().sum(dim=1).clamp_min(1) - 1
        batch_index = torch.arange(action_chunk.shape[0], device=action_chunk.device)
        transition = (gripper[batch_index, last_index] - gripper[batch_index, first_index]).unsqueeze(-1)

        finite = (
            torch.isfinite(goal).all(dim=-1, keepdim=True)
            & torch.isfinite(magnitude)
            & torch.isfinite(transition)
        )
        meaningful = (magnitude > self.motion_threshold) | (transition.abs() > self.motion_threshold)
        valid = has_steps & finite & meaningful
        if self.require_physical_scale and not scale_valid:
            valid &= False
        direction_valid = valid & (magnitude > self.motion_threshold)
        direction = torch.where(
            direction_valid, torch.nn.functional.normalize(goal, dim=-1), torch.zeros_like(goal)
        )

        # OSC_POSE rotation deltas are axis-angle increments. Plain vector summation is not
        # a valid general composition, so rotation supervision is disabled in this integration.
        rotation_goal = None
        return TrajectoryGeometryTarget(goal, direction, magnitude, rotation_goal, transition, valid)
