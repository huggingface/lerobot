#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
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
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    ProcessorKwargs,
    RenameProcessorStep,
    UnnormalizerProcessorStep,
)


def make_act_pre_post_processors(
    config: ACTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    preprocessor_kwargs: ProcessorKwargs | None = None,
    postprocessor_kwargs: ProcessorKwargs | None = None,
) -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Creates the pre- and post-processing pipelines for the ACT policy.

    The pre-processing pipeline handles normalization, batching, and device placement for the model inputs.
    The post-processing pipeline handles unnormalization and moves the model outputs back to the CPU.

    Args:
        config (ACTConfig): The ACT policy configuration object.
        dataset_stats (dict[str, dict[str, torch.Tensor]] | None): A dictionary containing dataset
            statistics (e.g., mean and std) used for normalization. Defaults to None.
        preprocessor_kwargs (ProcessorKwargs | None): Extra keyword arguments to pass to the
            preprocessor pipeline's constructor. Defaults to None.
        postprocessor_kwargs (ProcessorKwargs | None): Extra keyword arguments to pass to the
            postprocessor pipeline's constructor. Defaults to None.

    Returns:
        tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]: A tuple containing the
        pre-processor pipeline and the post-processor pipeline.
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
            device=config.device,
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
