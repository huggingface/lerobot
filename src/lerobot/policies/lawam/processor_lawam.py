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

"""LeRobot preprocessing and postprocessing pipelines for LaWAM."""

from __future__ import annotations

from typing import Any

import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    EnvTransition,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TransitionKey,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import (
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_lawam import LaWAMConfig


@ProcessorStepRegistry.register(name="lawam_clip_actions")
class LaWAMClipActionsProcessorStep(ProcessorStep):
    """Clamp normalized actions to the range expected by LaWAM."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Clamp an action transition to the normalized interval."""
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        transition = dict(transition)
        transition[TransitionKey.ACTION] = action.clamp(-1.0, 1.0)
        return transition

    def transform_features(self, features):
        """Preserve feature declarations because clipping does not change shape."""
        return features

    def get_config(self) -> dict[str, Any]:
        """Return the serializable processor configuration."""
        return {}


@ProcessorStepRegistry.register(name="lawam_pre_snap_gripper")
class LaWAMPreSnapGripperProcessorStep(ProcessorStep):
    """Snap the normalized gripper channel to binary values before unnormalizing."""

    def __init__(self, gripper_dim: int = 6, threshold: float = 0.5):
        self.gripper_dim = gripper_dim
        self.threshold = threshold

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Snap the configured gripper channel when it is present."""
        action = transition.get(TransitionKey.ACTION)
        if action is None or action.shape[-1] <= self.gripper_dim:
            return transition
        transition = dict(transition)
        snapped = action.clone()
        snapped[..., self.gripper_dim] = (snapped[..., self.gripper_dim] >= self.threshold).float()
        transition[TransitionKey.ACTION] = snapped
        return transition

    def transform_features(self, features):
        """Preserve feature declarations because snapping does not change shape."""
        return features

    def get_config(self) -> dict[str, Any]:
        """Return the gripper index and threshold for serialization."""
        return {"gripper_dim": self.gripper_dim, "threshold": self.threshold}


@ProcessorStepRegistry.register(name="lawam_binarize_gripper")
class LaWAMBinarizeGripperProcessorStep(ProcessorStep):
    """Map the emitted gripper channel to the LIBERO minus-one/plus-one convention."""

    def __init__(self, gripper_dim: int = 6, threshold: float = 0.5):
        self.gripper_dim = gripper_dim
        self.threshold = threshold

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Binarize the configured gripper channel when it is present."""
        action = transition.get(TransitionKey.ACTION)
        if action is None or action.shape[-1] <= self.gripper_dim:
            return transition
        transition = dict(transition)
        binarized = action.clone()
        binarized[..., self.gripper_dim] = (
            2.0 * (binarized[..., self.gripper_dim] > self.threshold).float() - 1.0
        )
        transition[TransitionKey.ACTION] = binarized
        return transition

    def transform_features(self, features):
        """Preserve feature declarations because binarization does not change shape."""
        return features

    def get_config(self) -> dict[str, Any]:
        """Return the gripper index and threshold for serialization."""
        return {"gripper_dim": self.gripper_dim, "threshold": self.threshold}


@ProcessorStepRegistry.register(name="lawam_libero_state")
class LaWAMLiberoStateProcessorStep(ProcessorStep):
    """Convert LIBERO's eight-value observation state to the trained state layout."""

    def __init__(self, target_state_dim: int):
        self.target_state_dim = int(target_state_dim)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Drop LIBERO's redundant gripper state value when a seven-value state is expected."""
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is None or OBS_STATE not in observation:
            return transition

        state = torch.as_tensor(observation[OBS_STATE])
        if state.shape[-1] == self.target_state_dim:
            return transition
        if state.shape[-1] != 8 or self.target_state_dim != 7:
            return transition

        transition = dict(transition)
        observation = dict(observation)
        observation[OBS_STATE] = torch.cat((state[..., :6], state[..., -1:]), dim=-1)
        transition[TransitionKey.OBSERVATION] = observation
        return transition

    def get_config(self) -> dict[str, Any]:
        """Return the target state width for serialization."""
        return {"target_state_dim": self.target_state_dim}

    def transform_features(self, features):
        """Preserve declared features because runtime statistics define the target width."""
        return features


def _stats_feature_dim(stats: dict[str, Any]) -> int | None:
    """Infer a feature width from the first available statistics tensor."""
    for key in ("min", "max", "mean", "std", "q01", "q99"):
        values = stats.get(key)
        if values is not None:
            return int(torch.as_tensor(values).numel())
    return None


def make_lawam_pre_post_processors(
    config: LaWAMConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build LaWAM input normalization and action postprocessing pipelines."""
    features = {**config.input_features, **config.output_features}
    state_stats = dataset_stats.get(OBS_STATE) if dataset_stats is not None else None
    state_stats_dim = _stats_feature_dim(state_stats) if state_stats is not None else None
    if state_stats_dim is not None and OBS_STATE not in features:
        features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(state_stats_dim,))

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
    ]
    if state_stats_dim is not None:
        input_steps.append(LaWAMLiberoStateProcessorStep(target_state_dim=state_stats_dim))
    input_steps.append(
        NormalizerProcessorStep(
            features=features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        )
    )

    output_steps: list[ProcessorStep] = []
    if config.clip_normalized_actions:
        output_steps.append(LaWAMClipActionsProcessorStep())
    if config.pre_snap_gripper_action:
        output_steps.append(
            LaWAMPreSnapGripperProcessorStep(
                gripper_dim=config.gripper_dim,
                threshold=config.gripper_threshold,
            )
        )
    output_steps.append(
        UnnormalizerProcessorStep(
            features=features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        )
    )
    if config.binarize_gripper_action:
        output_steps.append(
            LaWAMBinarizeGripperProcessorStep(
                gripper_dim=config.gripper_dim,
                threshold=config.gripper_threshold,
            )
        )
    output_steps.append(DeviceProcessorStep(device="cpu"))

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
