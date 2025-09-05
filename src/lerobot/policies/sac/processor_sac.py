#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team.
# All rights reserved.
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

import torch

from lerobot.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    ProcessorKwargs,
    RenameProcessorStep,
    UnnormalizerProcessorStep,
)


def make_sac_pre_post_processors(
    config: SACConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    preprocessor_kwargs: ProcessorKwargs | None = None,
    postprocessor_kwargs: ProcessorKwargs | None = None,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """
    Constructs pre-processor and post-processor pipelines for the SAC policy.

    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match pretrained configurations.
    2. Normalizing input and output features based on dataset statistics.
    3. Adding a batch dimension.
    4. Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving data to the CPU.
    2. Unnormalizing the output features to their original scale.

    Args:
        config: The configuration object for the SAC policy.
        dataset_stats: A dictionary of statistics for normalization.
        preprocessor_kwargs: Additional arguments for the pre-processor pipeline.
        postprocessor_kwargs: Additional arguments for the post-processor pipeline.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """
    if preprocessor_kwargs is None:
        preprocessor_kwargs = {}
    if postprocessor_kwargs is None:
        postprocessor_kwargs = {}

    input_steps = [
        RenameProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]
    output_steps = [
        DeviceProcessorStep(device="cpu"),
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
    ]
    return (
        PolicyProcessorPipeline(
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
            **preprocessor_kwargs,
        ),
        PolicyProcessorPipeline(
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            **postprocessor_kwargs,
        ),
    )
