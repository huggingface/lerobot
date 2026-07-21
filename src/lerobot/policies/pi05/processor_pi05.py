#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
    policy_action_to_transition,
    to_relative_actions,
    transition_to_policy_action,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_pi05 import PI05Config


@ProcessorStepRegistry.register(name="pi05_state_from_action_processor_step")
@dataclass
class Pi05StateFromActionProcessorStep(ProcessorStep):
    """Synthesize proprioception from absolute actions in state-less datasets.

    The dataset loader supplies ``history_steps - 1`` actions before the normal
    target chunk. Those leading samples and action(t) become state history; only
    the leading samples are then removed from the action targets.
    """

    enabled: bool = False
    history_steps: int = 1
    _inference_history: torch.Tensor | None = field(default=None, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition

        observation = transition.get(TransitionKey.OBSERVATION, {})
        observed_state = observation.get(OBS_STATE)
        if observed_state is not None:
            # At inference the robot normally provides only the current state and
            # there is no action target. Build a rolling history in the processor.
            if transition.get(TransitionKey.ACTION) is None and observed_state.ndim == 2:
                if self._inference_history is None:
                    self._inference_history = observed_state.unsqueeze(1).repeat(1, self.history_steps, 1)
                else:
                    self._inference_history = torch.cat(
                        [self._inference_history[:, 1:], observed_state.unsqueeze(1)], dim=1
                    )
                new_transition = transition.copy()
                new_observation = dict(observation)
                new_observation[OBS_STATE] = self._inference_history.clone()
                new_transition[TransitionKey.OBSERVATION] = new_observation
                return new_transition
            return transition

        action = transition.get(TransitionKey.ACTION)
        if action is None:
            raise ValueError("Cannot synthesize PI0.5 state without action")
        if action.ndim != 3:
            raise ValueError(f"Expected batched action chunks with shape (B, T, D), got {action.shape}")
        if action.shape[1] < self.history_steps:
            raise ValueError(
                f"Action chunk has {action.shape[1]} steps, fewer than history_steps={self.history_steps}"
            )

        new_transition = transition.copy()
        new_observation = dict(observation)
        state = action[:, : self.history_steps].clone()
        if self.history_steps == 1:
            state = state[:, 0]
        new_observation[OBS_STATE] = state
        new_transition[TransitionKey.OBSERVATION] = new_observation
        new_transition[TransitionKey.ACTION] = action[:, self.history_steps - 1 :]
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "history_steps": self.history_steps}

    def reset(self) -> None:
        self._inference_history = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="pi05_flatten_state_history_processor_step")
@dataclass
class Pi05FlattenStateHistoryProcessorStep(ProcessorStep):
    """Optionally relativize raw state history, then flatten it for PI0.5."""

    history_steps: int = 1
    max_state_dim: int = 32
    relative: bool = False
    exclude_joints: list[str] = field(default_factory=list)
    state_names: list[str] | None = None
    pose_representation: str = "componentwise"
    se3_pose_groups: list[list[int]] = field(default_factory=list)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for PI05")
        if self.history_steps == 1 and state.ndim == 2:
            state = state.unsqueeze(1)
        if state.ndim != 3 or state.shape[1] != self.history_steps:
            raise ValueError(
                f"Expected state history with shape (B, {self.history_steps}, D), got {state.shape}"
            )

        flattened_dim = state.shape[1] * state.shape[2]
        if flattened_dim > self.max_state_dim:
            raise ValueError(
                f"Flattened state history has {flattened_dim} dimensions, above max_state_dim={self.max_state_dim}"
            )

        processed_state = state.clone()
        if self.relative:
            mask_step = RelativeActionsProcessorStep(
                enabled=True,
                exclude_joints=self.exclude_joints,
                action_names=self.state_names,
            )
            processed_state = to_relative_actions(
                state,
                state[:, -1],
                mask_step._build_mask(state.shape[-1]),
                pose_representation=self.pose_representation,
                se3_pose_groups=self.se3_pose_groups,
            )

        new_transition = transition.copy()
        new_observation = dict(observation)
        new_observation[OBS_STATE] = processed_state.flatten(start_dim=1)
        new_transition[TransitionKey.OBSERVATION] = new_observation
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "history_steps": self.history_steps,
            "max_state_dim": self.max_state_dim,
            "relative": self.relative,
            "exclude_joints": self.exclude_joints,
            "state_names": self.state_names,
            "pose_representation": self.pose_representation,
            "se3_pose_groups": self.se3_pose_groups,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        transformed = deepcopy(features)
        for feature_group in transformed.values():
            state_feature = feature_group.get(OBS_STATE)
            if state_feature is not None and self.history_steps > 1:
                state_dim = state_feature.shape[-1] * self.history_steps
                feature_group[OBS_STATE] = PolicyFeature(type=state_feature.type, shape=(state_dim,))
        return transformed


@ProcessorStepRegistry.register(name="pi05_prepare_state_tokenizer_processor_step")
@dataclass
class Pi05PrepareStateTokenizerProcessorStep(ProcessorStep):
    """
    Processor step to prepare the state and tokenize the language input.
    """

    max_state_dim: int = 32
    task_key: str = "task"

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for PI05")
        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")

        # TODO: check if this necessary
        state = deepcopy(state)

        # State should already be normalized to [-1, 1] by the NormalizerProcessorStep that runs before this step
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        state_np = state.cpu().numpy()
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        full_prompts = []
        for i, task in enumerate(tasks):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, discretized_states[i]))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            full_prompts.append(full_prompt)

        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts
        # Normalize state to [-1, 1] range if needed (assuming it's already normalized by normalizer processor step!!)
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This step does not alter the feature definitions.
        """
        return features


def make_pi05_pre_post_processors(
    config: PI05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the PI0 policy.

    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match pretrained configurations.
    2. Normalizing input and output features based on dataset statistics.
    3. Adding a batch dimension.
    4. Appending a newline character to the task description for tokenizer compatibility.
    5. Tokenizing the text prompt using the PaliGemma tokenizer.
    6. Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving data to the CPU.
    2. Unnormalizing the output features to their original scale.

    Args:
        config: The configuration object for the PI0 policy.
        dataset_stats: A dictionary of statistics for normalization.
        preprocessor_kwargs: Additional arguments for the pre-processor pipeline.
        postprocessor_kwargs: Additional arguments for the post-processor pipeline.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    relative_step = RelativeActionsProcessorStep(
        enabled=config.use_relative_actions,
        exclude_joints=getattr(config, "relative_exclude_joints", []),
        action_names=getattr(config, "action_feature_names", None),
        pose_representation=config.relative_pose_representation,
        se3_pose_groups=config.relative_se3_pose_groups,
    )

    # OpenPI order: raw → relative → normalize → model → unnormalize → absolute
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),  # To mimic the same processor as pretrained one
        AddBatchDimensionProcessorStep(),
        Pi05StateFromActionProcessorStep(
            enabled=config.state_from_action,
            history_steps=config.proprioception_history_steps,
        ),
        relative_step,
        Pi05FlattenStateHistoryProcessorStep(
            history_steps=config.proprioception_history_steps,
            max_state_dim=config.max_state_dim,
            relative=config.use_relative_state_history,
            exclude_joints=config.relative_state_exclude_joints,
            state_names=config.action_feature_names,
            pose_representation=config.relative_pose_representation,
            se3_pose_groups=config.relative_se3_pose_groups,
        ),
        # NOTE: NormalizerProcessorStep MUST come before Pi05PrepareStateTokenizerProcessorStep
        # because the tokenizer step expects normalized state in [-1, 1] range for discretization
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        Pi05PrepareStateTokenizerProcessorStep(max_state_dim=config.max_state_dim),
        TokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        AbsoluteActionsProcessorStep(enabled=config.use_relative_actions, relative_step=relative_step),
        DeviceProcessorStep(device="cpu"),
    ]

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
