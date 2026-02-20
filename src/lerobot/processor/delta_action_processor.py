#!/usr/bin/env python

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

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.utils.constants import OBS_STATE

from .core import EnvTransition, PolicyAction, RobotAction, TransitionKey
from .pipeline import ActionProcessorStep, ProcessorStep, ProcessorStepRegistry, RobotActionProcessorStep


def to_delta_actions(actions: Tensor, state: Tensor, mask: Sequence[bool]) -> Tensor:
    """Convert absolute actions to delta: delta = action - state (for masked dims).

    Args:
        actions: (B, T, action_dim) or (B, action_dim).
        state: (B, state_dim). Broadcast across time dimension.
        mask: Which dims to convert. Can be shorter than action_dim.
    """
    mask_t = torch.tensor(mask, dtype=actions.dtype, device=actions.device)
    dims = mask_t.shape[0]
    state_offset = state[..., :dims] * mask_t
    if actions.ndim == 3:
        state_offset = state_offset.unsqueeze(-2)
    actions = actions.clone()
    actions[..., :dims] -= state_offset
    return actions


def to_absolute_actions(actions: Tensor, state: Tensor, mask: Sequence[bool]) -> Tensor:
    """Convert delta actions back to absolute: absolute = delta + state (for masked dims).

    Args:
        actions: (B, T, action_dim) or (B, action_dim).
        state: (B, state_dim). Broadcast across time dimension.
        mask: Which dims to convert. Can be shorter than action_dim.
    """
    mask_t = torch.tensor(mask, dtype=actions.dtype, device=actions.device)
    dims = mask_t.shape[0]
    state_offset = state[..., :dims] * mask_t
    if actions.ndim == 3:
        state_offset = state_offset.unsqueeze(-2)
    actions = actions.clone()
    actions[..., :dims] += state_offset
    return actions


@ProcessorStepRegistry.register("map_tensor_to_delta_action_dict")
@dataclass
class MapTensorToDeltaActionDictStep(ActionProcessorStep):
    """
    Maps a flat action tensor from a policy to a structured delta action dictionary.

    This step is typically used after a policy outputs a continuous action vector.
    It decomposes the vector into named components for delta movements of the
    end-effector (x, y, z) and optionally the gripper.

    Attributes:
        use_gripper: If True, assumes the 4th element of the tensor is the
                     gripper action.
    """

    use_gripper: bool = True

    def action(self, action: PolicyAction) -> RobotAction:
        if not isinstance(action, PolicyAction):
            raise ValueError("Only PolicyAction is supported for this processor")

        if action.dim() > 1:
            action = action.squeeze(0)

        # TODO (maractingi): add rotation
        delta_action = {
            "delta_x": action[0].item(),
            "delta_y": action[1].item(),
            "delta_z": action[2].item(),
        }
        if self.use_gripper:
            delta_action["gripper"] = action[3].item()
        return delta_action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for axis in ["x", "y", "z"]:
            features[PipelineFeatureType.ACTION][f"delta_{axis}"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        if self.use_gripper:
            features[PipelineFeatureType.ACTION]["gripper"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )
        return features


@ProcessorStepRegistry.register("map_delta_action_to_robot_action")
@dataclass
class MapDeltaActionToRobotActionStep(RobotActionProcessorStep):
    """
    Maps delta actions from teleoperators to robot target actions for inverse kinematics.

    This step converts a dictionary of delta movements (e.g., from a gamepad)
    into a target action format that includes an "enabled" flag and target
    end-effector positions. It also handles scaling and noise filtering.

    Attributes:
        position_scale: A factor to scale the delta position inputs.
        noise_threshold: The magnitude below which delta inputs are considered noise
                         and do not trigger an "enabled" state.
    """

    # Scale factors for delta movements
    position_scale: float = 1.0
    noise_threshold: float = 1e-3  # 1 mm threshold to filter out noise

    def action(self, action: RobotAction) -> RobotAction:
        # NOTE (maractingi): Action can be a dict from the teleop_devices or a tensor from the policy
        # TODO (maractingi): changing this target_xyz naming convention from the teleop_devices
        delta_x = action.pop("delta_x")
        delta_y = action.pop("delta_y")
        delta_z = action.pop("delta_z")
        gripper = action.pop("gripper")

        # Determine if the teleoperator is actively providing input
        # Consider enabled if any significant movement delta is detected
        position_magnitude = (delta_x**2 + delta_y**2 + delta_z**2) ** 0.5  # Use Euclidean norm for position
        enabled = position_magnitude > self.noise_threshold  # Small threshold to avoid noise

        # Scale the deltas appropriately
        scaled_delta_x = delta_x * self.position_scale
        scaled_delta_y = delta_y * self.position_scale
        scaled_delta_z = delta_z * self.position_scale

        # For gamepad/keyboard, we don't have rotation input, so set to 0
        # These could be extended in the future for more sophisticated teleoperators
        target_wx = 0.0
        target_wy = 0.0
        target_wz = 0.0

        # Update action with robot target format
        action = {
            "enabled": enabled,
            "target_x": scaled_delta_x,
            "target_y": scaled_delta_y,
            "target_z": scaled_delta_z,
            "target_wx": target_wx,
            "target_wy": target_wy,
            "target_wz": target_wz,
            "gripper_vel": float(gripper),
        }

        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for axis in ["x", "y", "z", "gripper"]:
            features[PipelineFeatureType.ACTION].pop(f"delta_{axis}", None)

        for feat in ["enabled", "target_x", "target_y", "target_z", "target_wx", "target_wy", "target_wz"]:
            features[PipelineFeatureType.ACTION][f"{feat}"] = PolicyFeature(
                type=FeatureType.ACTION, shape=(1,)
            )

        return features


@ProcessorStepRegistry.register("delta_actions_processor")
@dataclass
class DeltaActionsProcessorStep(ProcessorStep):
    """Converts absolute actions to delta actions (action -= state) for masked dimensions.

    Mirrors OpenPI's DeltaActions transform. Applied during preprocessing so the model
    trains on relative offsets instead of absolute positions.
    Caches the last seen state so a paired AbsoluteActionsProcessorStep can reverse
    the conversion during postprocessing.

    Attributes:
        enabled: Whether to apply the delta conversion.
        exclude_joints: Joint names to keep absolute (not converted to delta).
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
        exclude = set(self.exclude_joints)
        return [n not in exclude for n in self.action_names]

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
        new_transition[TransitionKey.ACTION] = to_delta_actions(action, state, mask)
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "exclude_joints": self.exclude_joints}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("absolute_actions_processor")
@dataclass
class AbsoluteActionsProcessorStep(ProcessorStep):
    """Converts delta actions back to absolute actions (action += state) for all dimensions.

    Mirrors OpenPI's AbsoluteActions transform. Applied during postprocessing so
    predicted deltas are converted back to absolute positions for execution.
    Reads the cached state from its paired DeltaActionsProcessorStep.

    Attributes:
        enabled: Whether to apply the absolute conversion.
        delta_step: Reference to the paired DeltaActionsProcessorStep that caches state.
    """

    enabled: bool = False
    delta_step: DeltaActionsProcessorStep | None = field(default=None, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition

        if self.delta_step is None:
            raise RuntimeError(
                "AbsoluteActionsProcessorStep requires a paired DeltaActionsProcessorStep "
                "but delta_step is None. Ensure delta_step is set when constructing the postprocessor."
            )

        if self.delta_step._last_state is None:
            raise RuntimeError(
                "AbsoluteActionsProcessorStep requires state from DeltaActionsProcessorStep "
                "but no state has been cached. Ensure the preprocessor runs before the postprocessor."
            )

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            return new_transition

        mask = self.delta_step._build_mask(action.shape[-1])
        new_transition[TransitionKey.ACTION] = to_absolute_actions(
            action, self.delta_step._last_state, mask
        )
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
