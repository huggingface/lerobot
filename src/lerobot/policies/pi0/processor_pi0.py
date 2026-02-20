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

from typing import Any

import torch

from lerobot.configs.types import NormalizationMode, PipelineFeatureType, PolicyFeature
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.processor import (
    Action32DecodeProcessorStep,
    AddBatchDimensionProcessorStep,
    ComplementaryDataProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    StateAction32AdapterProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


@ProcessorStepRegistry.register(name="pi0_new_line_processor")
class Pi0NewLineProcessor(ComplementaryDataProcessorStep):
    """
    Ensures that the task description string ends with a newline character.

    This processing step is required for compatibility with the PaliGemma tokenizer,
    which expects a newline at the end of the text prompt. It handles both single
    strings and lists of strings for the 'task' key in complementary data.
    """

    def complementary_data(self, complementary_data):
        """
        Adds a newline to the 'task' field if it doesn't already have one.

        Args:
            complementary_data: A dictionary that may contain a 'task' key with a
                                string or list of strings.

        Returns:
            A new dictionary with the modified 'task' field.
        """
        if "task" not in complementary_data:
            return complementary_data

        task = complementary_data["task"]
        if task is None:
            return complementary_data

        new_complementary_data = dict(complementary_data)

        # Handle both string and list of strings
        if isinstance(task, str):
            # Single string: add newline if not present
            if not task.endswith("\n"):
                new_complementary_data["task"] = f"{task}\n"
        elif isinstance(task, list) and all(isinstance(t, str) for t in task):
            # List of strings: add newline to each if not present
            new_complementary_data["task"] = [t if t.endswith("\n") else f"{t}\n" for t in task]
        # If task is neither string nor list of strings, leave unchanged

        return new_complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This step does not alter the feature definitions.

        Args:
            features: The input feature dictionary.

        Returns:
            The unchanged feature dictionary.
        """
        return features


def make_pi0_pre_post_processors(
    config: PI0Config,
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
    norm_map = dict(config.normalization_mapping)
    adapter_cfg = config.state_action_32_adapter
    if adapter_cfg.enabled and adapter_cfg.disable_builtin_normalizer_for_state_action:
        norm_map["STATE"] = NormalizationMode.IDENTITY
        norm_map["ACTION"] = NormalizationMode.IDENTITY

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map=config.observation_rename_map),
        AddBatchDimensionProcessorStep(),
        Pi0NewLineProcessor(),  # Add newlines before tokenization for PaliGemma
        TokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        DeviceProcessorStep(device=config.device),
    ]

    if adapter_cfg.enabled:
        input_steps.append(
            StateAction32AdapterProcessorStep(
                enabled=adapter_cfg.enabled,
                mode=adapter_cfg.mode,
                target_state_dim=adapter_cfg.target_state_dim,
                target_action_dim=adapter_cfg.target_action_dim,
                raw_state_dim=adapter_cfg.raw_state_dim,
                raw_action_dim=adapter_cfg.raw_action_dim,
                state_index_map=list(adapter_cfg.state_index_map),
                action_index_map=list(adapter_cfg.action_index_map),
                projection_init=adapter_cfg.projection_init,
                projection_seed=adapter_cfg.projection_seed,
                apply_mean_std_normalization=adapter_cfg.apply_mean_std_normalization,
                dataset_stats=dataset_stats,
                gripper_enabled=adapter_cfg.gripper_enabled,
                gripper_raw_index_in_state=adapter_cfg.gripper_raw_index_in_state,
                gripper_raw_index_in_action=adapter_cfg.gripper_raw_index_in_action,
                gripper_conversion_type=adapter_cfg.gripper_conversion_type,
                gripper_raw_open_value=adapter_cfg.gripper_raw_open_value,
                gripper_raw_closed_value=adapter_cfg.gripper_raw_closed_value,
                gripper_pi0_open_value=adapter_cfg.gripper_pi0_open_value,
                gripper_pi0_closed_value=adapter_cfg.gripper_pi0_closed_value,
            )
        )

    input_steps.append(
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=norm_map,
            stats=dataset_stats,
        )
    )

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=norm_map,
            stats=dataset_stats,
        ),
    ]

    if adapter_cfg.enabled and adapter_cfg.decode_action_to_raw:
        output_steps.append(
            Action32DecodeProcessorStep(
                enabled=True,
                mode=adapter_cfg.mode,
                target_action_dim=adapter_cfg.target_action_dim,
                raw_action_dim=adapter_cfg.raw_action_dim,
                action_index_map=list(adapter_cfg.action_index_map),
                projection_init=adapter_cfg.projection_init,
                projection_seed=adapter_cfg.projection_seed,
                gripper_enabled=adapter_cfg.gripper_enabled,
                gripper_raw_index_in_action=adapter_cfg.gripper_raw_index_in_action,
                gripper_conversion_type=adapter_cfg.gripper_conversion_type,
                gripper_raw_open_value=adapter_cfg.gripper_raw_open_value,
                gripper_raw_closed_value=adapter_cfg.gripper_raw_closed_value,
                gripper_pi0_open_value=adapter_cfg.gripper_pi0_open_value,
                gripper_pi0_closed_value=adapter_cfg.gripper_pi0_closed_value,
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
