#!/usr/bin/env python

# Copyright 2024 Nicklas Hansen, Xiaolong Wang, Hao Su,
# and The HuggingFace Inc. team. All rights reserved.
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

from lerobot.policies.tdmpc.configuration_tdmpc import TDMPCConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME


def make_tdmpc_pre_post_processors(
    config: TDMPCConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the TDMPC policy.

    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match pretrained configurations.
    2. Normalizing input and output features based on dataset statistics.
    3. Adding a batch dimension.
    4. Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving data to the CPU.
    2. Unnormalizing the output features to their original scale.

    Args:
        config: The configuration object for the TDMPC policy.
        dataset_stats: A dictionary of statistics for normalization.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
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
