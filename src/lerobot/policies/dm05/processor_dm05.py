#!/usr/bin/env python

# Copyright 2026 Dexmal and HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    AddBatchDimensionProcessorStep,
    ComplementaryDataProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
    make_policy_processor_pipelines,
)
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_dm05 import DM05Config
from .constants import ACTION_REFERENCE_OFFSET
from .conversion_dm05 import DM05ProcessorArtifactsStep, DM05StateBinsProcessorStep
from .stats_validation_dm05 import (
    dm05_prepare_stats_command,
    dm05_stats_complete,
    validate_dm05_relative_action_stats,
)

_ACTION_PROBE_KIND = "_dm05_action_probe_kind"


@dataclass
@ProcessorStepRegistry.register(name="dm05_task_processor")
class DM05TaskProcessor(ComplementaryDataProcessorStep):
    """Normalize the task prompt field expected by DM05 tokenization."""

    default_task: str = "Execute the robot action."

    def complementary_data(self, complementary_data: dict[str, Any]) -> dict[str, Any]:
        """Normalize missing or blank task prompts in complementary data."""
        if (task := complementary_data.get("task")) is None:
            return {**complementary_data, "task": self.default_task}

        if isinstance(task, str):
            return {**complementary_data, "task": task.strip() or self.default_task}
        if isinstance(task, list):
            return {
                **complementary_data,
                "task": [str(item).strip() or self.default_task for item in task],
            }
        return complementary_data

    def get_config(self) -> dict[str, Any]:
        """Return the serializable processor-step configuration."""
        return {"default_task": self.default_task}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Leave policy feature metadata unchanged."""
        return features


@dataclass
@ProcessorStepRegistry.register(name="dm05_action_reference_probe_processor")
class DM05ActionReferenceProbeProcessorStep(ProcessorStep):
    """Append state/zero probes used to measure normalized state displacement."""

    action_dim: int

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Attach temporary action probes before normalization."""
        observation = transition.get(TransitionKey.OBSERVATION)
        state = observation.get(OBS_STATE) if isinstance(observation, dict) else None
        if state is None:
            return transition

        state = torch.as_tensor(state)
        if state.ndim != 2:
            raise ValueError(f"DM05 expects batched state [B,D], got {tuple(state.shape)}.")

        action = transition.get(TransitionKey.ACTION)
        kind = "none"
        if action is not None:
            action = torch.as_tensor(action)
            self.action_dim = int(action.shape[-1])
            if action.ndim == state.ndim:
                action = action.unsqueeze(-2)
                kind = "single"
            elif action.ndim == state.ndim + 1:
                kind = "chunk"
            else:
                raise ValueError(f"DM05 expects action [B,D] or [B,T,D], got {tuple(action.shape)}.")
            if action.shape[0] != state.shape[0]:
                raise ValueError("DM05 state and action batch dimensions must match.")

        # The probes are only consumed by relative-action inference. Absolute
        # policies may legitimately expose fewer state than action dimensions.
        if state.shape[-1] < self.action_dim:
            return transition
        reference = state[..., : self.action_dim]
        probes = torch.stack((reference, torch.zeros_like(reference)), dim=-2)

        result = transition.copy()
        result[TransitionKey.ACTION] = probes if action is None else torch.cat((action, probes), dim=-2)
        complementary = dict(result.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        complementary[_ACTION_PROBE_KIND] = kind
        result[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return result

    def get_config(self) -> dict[str, Any]:
        """Return the action dimension used for serialized probes."""
        return {"action_dim": self.action_dim}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Leave policy feature metadata unchanged."""
        return features


@dataclass
@ProcessorStepRegistry.register(name="dm05_action_reference_extract_processor")
class DM05ActionReferenceExtractProcessorStep(ProcessorStep):
    """Extract the normalized reference offset and remove the temporary probes."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Convert temporary probes into the relative-action reference offset."""
        complementary = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        kind = complementary.pop(_ACTION_PROBE_KIND, None)
        if kind is None:
            return transition

        action = transition.get(TransitionKey.ACTION)
        if action is None or action.ndim < 3 or action.shape[-2] < 2:
            raise ValueError("DM05 action reference probes are missing after normalization.")

        result = transition.copy()
        complementary[ACTION_REFERENCE_OFFSET] = (action[..., -2, :] - action[..., -1, :]).float()
        result[TransitionKey.COMPLEMENTARY_DATA] = complementary
        action = action[..., :-2, :]
        if kind == "none":
            result[TransitionKey.ACTION] = None
        elif kind == "single":
            result[TransitionKey.ACTION] = action.squeeze(-2)
        else:
            result[TransitionKey.ACTION] = action
        return result

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Leave policy feature metadata unchanged."""
        return features


def make_dm05_pre_post_processors(
    config: DM05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build the LeRobot processor pipeline for the OpenDM adapter."""

    config.validate_features()
    if not dm05_stats_complete(config, dataset_stats):
        if not config.use_relative_actions:
            raise ValueError("DM05 absolute-action training requires standard LeRobot dataset statistics.")
        command = dm05_prepare_stats_command(config, getattr(config, "_runtime_dataset_meta", None))
        raise ValueError(
            "DM05 relative-action training requires matching state-relative action statistics. "
            f"Run `{command}` before training."
        )
    validate_dm05_relative_action_stats(config, dataset_stats)

    # OpenDM normalizes only numeric state/action fields.
    normalizer = NormalizerProcessorStep(
        features={
            OBS_STATE: config.input_features[OBS_STATE],
            ACTION: config.output_features[ACTION],
        },
        norm_map=config.normalization_mapping,
        stats=dataset_stats,
        normalize_observation_keys={OBS_STATE},
        eps=1e-6,
    )
    unnormalizer = UnnormalizerProcessorStep(
        features=config.output_features,
        norm_map=config.normalization_mapping,
        stats=dataset_stats,
        eps=1e-6,
    )
    relative_actions = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=config.relative_exclude_joints,
        action_names=config.action_feature_names,
    )
    action_dim = int(config.output_features[ACTION].shape[-1])
    processor_source = config.processor_name_or_path or config.pretrained_name_or_path
    if not processor_source:
        raise ValueError("DM05 requires processor_name_or_path when creating a new processor pipeline.")
    return make_policy_processor_pipelines(
        input_steps=[
            RenameObservationsProcessorStep(rename_map={}),
            AddBatchDimensionProcessorStep(),
            DM05TaskProcessor(),
            DM05ActionReferenceProbeProcessorStep(action_dim=action_dim),
            relative_actions,
            normalizer,
            DM05ActionReferenceExtractProcessorStep(),
            DM05StateBinsProcessorStep(),
            DeviceProcessorStep(device=config.device),
            DM05ProcessorArtifactsStep(processor_name_or_path=processor_source),
        ],
        output_steps=[
            DeviceProcessorStep(device="cpu", float_dtype="float32"),
            unnormalizer,
            AbsoluteActionsProcessorStep(
                enabled=config.use_relative_actions,
                relative_step=relative_actions,
            ),
        ],
    )
